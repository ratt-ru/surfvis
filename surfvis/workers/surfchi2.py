#!/usr/bin/env python
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
# mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from surfvis.workers.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('svis')
log = pyscilog.get_logger('SURFCHI2')
import time
import fsspec

import os
import sys
import numpy as np
import xarray as xr
import dask
import dask.array as da
from surfvis.utils import surfchisq, surfchisq_plot
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_from_storage_table as xds_from_table
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
# might make for cooler histograms but doesn't work out of the box
from astropy.visualization import hist
from pathlib import Path
import concurrent.futures as cf

from scabha.schema_utils import clickify_parameters
from surfvis.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.surfchi2)
def surfchi2(**kw):
    '''
    Per baseline chi squared plots 
    '''
    opts = OmegaConf.create(kw)


    print('Input Options:')
    for key, value in opts.items():
        print('     %25s = %s' % (key, value), file=log)

    if opts.dataout == '':
        opts.dataout = os.getcwd() + '/chi2'

    if os.path.isdir(opts.dataout):
        print(f"Removing existing {opts.dataout} folder")
        os.system(f"rm -r {opts.dataout}")

    if opts.imagesout == '':
        opts.imagesout = os.getcwd() + '/chi2'

    if os.path.isdir(opts.imagesout):
        print(f"Removing existing {opts.imagesout} folder")
        os.system(f"rm -r {opts.imagesout}")

    # Some error trapping
    msname = opts.ms.rstrip('/')

    from multiprocessing.pool import ThreadPool
    dask.config.set(pool=ThreadPool(opts.nthreads))

    # chunking info
    schema = {}
    schema[opts.fcol] = {'dims': ('chan', 'corr')}
    xds = xds_from_ms(msname,
                      chunks={'row': -1},
                      columns=['TIME', opts.fcol],
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'],
                      table_schema=schema)

    chunks = []
    rbin_idx = []
    rbin_counts = []
    tbin_idx = []
    tbin_counts = []
    fbin_idx = []
    fbin_counts = []
    t0s = []
    tfs = []
    for ds in xds:
        time = ds.TIME.values
        ut, counts = np.unique(time, return_counts=True)
        if opts.ntimes in [0, -1]:
            utpc = ut.size
        else:
            utpc = opts.ntimes
        row_chunks = [np.sum(counts[i:i+utpc])
                       for i in range(0, ut.size, utpc)]

        nchan = ds.chan.size
        if opts.nfreqs in [0, -1]:
            opts.nfreqs = nchan

        # list per ds
        chunks.append({'row': tuple(row_chunks), 'chan': opts.nfreqs})

        ridx = np.zeros(len(row_chunks))
        ridx[1:] = np.cumsum(row_chunks)[0:-1]
        rbin_idx.append(ridx.astype(int))
        rbin_counts.append(row_chunks)

        ntime = ut.size
        tidx = np.arange(0, ntime, utpc)
        tbin_idx.append(tidx.astype(int))
        tidx2 = np.append(tidx, ntime)
        tcounts = tidx2[1:] - tidx2[0:-1]
        tbin_counts.append(tcounts)

        t0 = ut[tidx]
        t0s.append(t0)
        tf = ut[tidx + tcounts -1]
        tfs.append(tf)

        fidx = np.arange(0, nchan, opts.nfreqs)
        fbin_idx.append(fidx)
        fidx2 = np.append(fidx, nchan)
        fcounts = fidx2[1:] - fidx2[0:-1]
        fbin_counts.append(fcounts)

    schema = {}
    schema[opts.rcol] = {'dims': ('chan', 'corr')}
    schema[opts.wcol] = {'dims': ('chan', 'corr')}
    schema[opts.fcol] = {'dims': ('chan', 'corr')}

    xds = xds_from_ms(msname,
                      columns=[opts.rcol, opts.wcol, opts.fcol,'ANTENNA1', 'ANTENNA2', 'TIME'],
                      chunks=chunks,
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'],
                      table_schema=schema)
    if opts.use_corrs is None:
        print('Using only diagonal correlations')
        if len(xds[0].corr) > 1:
            use_corrs = [0, -1]
        else:
            use_corrs = [0]
    else:
        use_corrs = list(map(int, opts.use_corrs.split(',')))
        print(f"Using correlations {use_corrs}")
    ncorr = len(use_corrs)

    chi2s = {}
    counts = {}
    futures = []
    foldername = opts.imagesout.rstrip('/')
    with cf.ProcessPoolExecutor(max_workers=opts.nthreads) as executor:
        for i, ds in enumerate(xds):
            field = ds.FIELD_ID
            spw = ds.DATA_DESC_ID
            scan = ds.SCAN_NUMBER

            basename = foldername + f'/field{field}' + f'/spw{spw}'+ f'/scan{scan}/'
            odir = Path(basename).resolve()
            odir.mkdir(parents=True, exist_ok=True)

            ntime = tbin_idx[i].size
            nfreq = fbin_idx[i].size
            ncorr = len(use_corrs)
            for t in range(ntime):
                for f in range(nfreq):
                    for c in range(ncorr):
                        t0 = tbin_idx[i][t]
                        tf = t0 + tbin_counts[i][t]
                        chan0 = fbin_idx[i][f]
                        chanf = chan0 + fbin_counts[i][f]
                        row0 = rbin_idx[i][t]
                        rowf = rbin_idx[i][t] + rbin_counts[i][t]
                        Inu = slice(chan0, chanf)
                        Irow = slice(row0, rowf)
                        dso = ds[{'row': Irow, 'chan': Inu}]
                        # import ipdb; ipdb.set_trace()
                        dso = dso.sel(corr=use_corrs)
                        resid = dso.get(opts.rcol).data
                        if opts.wcol == 'SIGMA_SPECTRUM':
                            weight = 1.0/dso.get(opts.wcol).data**2
                        else:
                            weight = dso.get(opts.wcol).data
                        flag = dso.get(opts.fcol).data
                        ant1 = dso.ANTENNA1.data
                        ant2 = dso.ANTENNA2.data
                        t0 = tbin_idx[i][t]
                        tf = t0 + tbin_counts[i][t]
                        chan0 = fbin_idx[i][f]
                        chanf = chan0 + fbin_counts[i][f]
                        fut = executor.submit(surfchisq_plot, resid, weight, flag, ant1, ant2,
                                              field, spw, scan,
                                              basename + f't{t}_f{f}_c{c}.png',
                                              f't {t0}-{tf}, chan {chan0}-{chanf}, corr {c}')
                        futures.append(fut)

            # to reduce over time, freq and corr at the end
            nant = np.maximum(ant1.compute().max(), ant2.compute().max()) + 1
            chi2s[f'field{field}_spw{spw}_scan{scan}'] = np.zeros((nant, nant), dtype=float)
            counts[f'field{field}_spw{spw}_scan{scan}'] = np.zeros((nant, nant), dtype=float)
            print(f"Submitted field{field}_spw{spw}_scan{scan}")

        # reduce per scan
        num_completed = 0
        num_futures = len(futures)
        for fut in cf.as_completed(futures):
            num_completed += 1
            print(f"\rProcessing: {num_completed}/{num_futures}", end='', flush=True)
            try:
                field, spw, scan, chi2, count = fut.result()
                chi2s[f'field{field}_spw{spw}_scan{scan}'] += chi2
                counts[f'field{field}_spw{spw}_scan{scan}'] += count
            except Exception as e:
                raise e

    # LB - is it worth doing this in parallel?
    print("Plotting per scan")
    for key, val in chi2s.items():
        field, spw, scan = key.split('_')
        field = field.strip('field')
        spw = spw.strip('spw')
        scan = scan.strip('scan')
        count = counts[key]
        chi2_dof = np.zeros_like(val)
        chi2_dof[count>0] = val[count>0]/count[count>0]
        chi2_dof[count<=0] = np.nan

        basename = foldername + f'/field{field}' + f'/spw{spw}'+ f'/scan{scan}/'
        makeplot(chi2_dof, basename + f'combined.png',
                f'scan {scan}.png')

def makeplot(data, name, subt):
    nant, _ = data.shape
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(data, cmap='inferno')
    ax.set_xticks(np.arange(0, nant, 2))
    ax.set_yticks(np.arange(nant))
    ax.tick_params(axis='both', which='major',
                      length=1, width=1, labelsize=4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.2)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=6, pad=0.1)

    rax = divider.append_axes("right", size="50%", pad=0.025)
    x = data[~ np.isnan(data)]
    if x.any():
        hist(x, bins='scott', ax=rax, histtype='stepfilled',
            alpha=0.5, density=False)
        rax.set_yticks([])
        rax.tick_params(axis='y', which='both',
                        bottom=False, top=False,
                        labelbottom=False)
        rax.tick_params(axis='x', which='both',
                        length=1, width=1, labelsize=8)

    fig.suptitle(subt, fontsize=20)
    plt.savefig(name, dpi=250)
    plt.close(fig)
