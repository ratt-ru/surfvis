#!/usr/bin/env python
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
# mpl.use('Agg')
from surfvis.main import cli
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import click
from omegaconf import OmegaConf
import os
import sys
import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from surfvis.utils import surfchisq
from daskms import xds_from_ms, xds_from_table
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
# might make for cooler histograms but doesn't work out of the box
from astropy.visualization import hist
import pyscilog
pyscilog.init('surfvis')
log = pyscilog.get_logger('SURFCHI2')


from scabha.schema_utils import clickify_parameters
from surfvis.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.surfchi2["inputs"].keys():
    defaults[key] = schema.surfchi2["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.surfchi2)
def surfchi2(**kw):
    defaults.update(kw)
    opts = OmegaConf.create(defaults)

    if opts.dataout == '':
        opts.dataout = os.getcwd() + '/chi2'

    if os.path.isdir(opts.dataout):
        print(f"Removing existing {opts.dataout} folder")
        os.system(f"rm -r {opts.dataout}")

    msname = opts.ms.rstrip('/')

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
        rbin_idx.append(da.from_array(ridx.astype(int), chunks=1))
        rbin_counts.append(da.from_array(row_chunks, chunks=1))

        ntime = ut.size
        tidx = np.arange(0, ntime, utpc)
        tbin_idx.append(da.from_array(tidx.astype(int), chunks=1))
        tidx2 = np.append(tidx, ntime)
        tcounts = tidx2[1:] - tidx2[0:-1]
        tbin_counts.append(da.from_array(tcounts, chunks=1))

        t0 = ut[tidx]
        t0s.append(da.from_array(t0, chunks=1))
        tf = ut[tidx + tcounts -1]
        tfs.append(da.from_array(tf, chunks=1))

        fidx = np.arange(0, nchan, opts.nfreqs)
        fbin_idx.append(da.from_array(fidx, chunks=1))
        fidx2 = np.append(fidx, nchan)
        fcounts = fidx2[1:] - fidx2[0:-1]
        fbin_counts.append(da.from_array(fcounts, chunks=1))

    schema = {}
    schema[opts.rcol] = {'dims': ('chan', 'corr')}
    schema[opts.wcol] = {'dims': ('chan', 'corr')}
    schema[opts.fcol] = {'dims': ('chan', 'corr')}

    xds = xds_from_ms(msname,
                        columns=[opts.rcol, opts.wcol, opts.fcol,
                                'ANTENNA1', 'ANTENNA2', 'TIME'],
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

    out_ds = []
    idts = []
    for i, ds in enumerate(xds):
        ds = ds.sel(corr=use_corrs)

        resid = ds.get(opts.rcol).data
        weight = ds.get(opts.wcol).data
        flag = ds.get(opts.fcol).data
        ant1 = ds.ANTENNA1.data
        ant2 = ds.ANTENNA2.data

        # ncorr = resid.shape[0]

        # time = ds.TIME.values
        # utime = np.unique(time)

        # spw = xds_from_table(msname + '::SPECTRAL_WINDOW')
        # freq = spw[0].CHAN_FREQ.values

        field = ds.FIELD_ID
        ddid = ds.DATA_DESC_ID
        scan = ds.SCAN_NUMBER

        tmp = surfchisq(resid, weight, flag, ant1, ant2,
                        rbin_idx[i], rbin_counts[i],
                        fbin_idx[i], fbin_counts[i])

        d = xr.Dataset(
            data_vars={'data': (('time', 'freq', 'corr', 'p', 'q', '2'), tmp),
                       'fbin_idx': (('freq'), fbin_idx[i]),
                       'fbin_counts': (('freq'), fbin_counts[i]),
                       'tbin_idx': (('time'), tbin_idx[i]),
                       'tbin_counts': (('time'), tbin_counts[i])},
            attrs = {'FIELD_ID': ds.FIELD_ID,
                     'DATA_DESC_ID': ds.DATA_DESC_ID,
                     'SCAN_NUMBER': ds.SCAN_NUMBER},
            # coords={'time': (('time'), utime),
            # 		'freq': (('freq'), freq),
            # 		'corr': (('corr'), np.arange(ncorr))}
        )

        idt = f'::F{ds.FIELD_ID}_D{ds.DATA_DESC_ID}_S{ds.SCAN_NUMBER}'
        out_ds.append(xds_to_zarr(d, opts.dataout + idt))
        idts.append(idt)


    with ProgressBar():
        dask.compute(out_ds)

    # primitive plotting
    if opts.imagesout is not None:
        foldername = opts.imagesout.rstrip('/')
        if not os.path.isdir(foldername):
                os.system('mkdir '+ foldername)

        for idt in idts:
            xds = xds_from_zarr(opts.dataout + idt)
            for ds in xds:
                field = ds.FIELD_ID
                if not os.path.isdir(foldername + f'/field{field}'):
                    os.system('mkdir '+ foldername + f'/field{field}')

                spw = ds.DATA_DESC_ID
                if not os.path.isdir(foldername + f'/field{field}' + f'/spw{spw}'):
                    os.system('mkdir '+ foldername + f'/field{field}' + f'/spw{spw}')

                scan = ds.SCAN_NUMBER
                if not os.path.isdir(foldername + f'/field{field}' + f'/spw{spw}' + f'/scan{scan}'):
                    os.system('mkdir '+ foldername + f'/field{field}' + f'/spw{spw}'+ f'/scan{scan}')

                tmp = ds.data.values
                tbin_idx = ds.tbin_idx.values
                tbin_counts = ds.tbin_counts.values
                fbin_idx = ds.fbin_idx.values
                fbin_counts = ds.fbin_counts.values

                ntime, nfreq, ncorr, _, _, _ = tmp.shape

                basename = foldername + f'/field{field}' + f'/spw{spw}'+ f'/scan{scan}/'
                if len(os.listdir(basename)):
                    print(f"Removing contents of {basename} folder")
                    os.system(f'rm {basename}*.png')
                for t in range(ntime):
                    for f in range(nfreq):
                        for c in range(ncorr):
                            chi2 = tmp[t, f, c, :, :, 0]
                            N = tmp[t, f, c, :, :, 1]
                            chi2_dof = np.zeros_like(chi2)
                            chi2_dof[N>0] = chi2[N>0]/N[N>0]
                            chi2_dof[N==0] = np.nan
                            t0 = tbin_idx[t]
                            tf = tbin_idx[t] + tbin_counts[t]
                            chan0 = fbin_idx[f]
                            chanf = fbin_idx[f] + fbin_counts[f]
                            makeplot(chi2_dof, basename + f't{t}_f{f}_c{c}.png',
                                     f't {t0}-{tf}, chan {chan0}-{chanf}, corr {c}')

                        # reduce over corr
                        chi2 = np.nansum(tmp[t, f, (0, -1), :, :, 0], axis=0)
                        N = np.nansum(tmp[t, f, (0, -1), :, :, 1], axis=0)
                        chi2_dof = np.zeros_like(chi2)
                        chi2_dof[N>0] = chi2[N>0]/N[N>0]
                        chi2_dof[N==0] = np.nan
                        t0 = tbin_idx[t]
                        tf = tbin_idx[t] + tbin_counts[t]
                        chan0 = fbin_idx[f]
                        chanf = fbin_idx[f] + fbin_counts[f]
                        makeplot(chi2_dof, basename + f't{t}_f{f}.png',
                                 f't {t0}-{tf}, chan {chan0}-{chanf}')

                    # reduce over freq
                    chi2 = np.nansum(tmp[t, :, (0, -1), :, :, 0], axis=(0,1))
                    N = np.nansum(tmp[t, :, (0, -1), :, :, 1], axis=(0,1))
                    chi2_dof = np.zeros_like(chi2)
                    chi2_dof[N>0] = chi2[N>0]/N[N>0]
                    chi2_dof[N==0] = np.nan
                    t0 = tbin_idx[t]
                    tf = tbin_idx[t] + tbin_counts[t]
                    makeplot(chi2_dof, basename + f't{t}.png',
                             f't {t0}-{tf}')

                # now the entire scan
                chi2 = np.nansum(tmp[:, :, (0, -1), :, :, 0], axis=(0,1,2))
                N = np.nansum(tmp[:, :, (0, -1), :, :, 1], axis=(0,1,2))
                chi2_dof = np.zeros_like(chi2)
                chi2_dof[N>0] = chi2[N>0]/N[N>0]
                chi2_dof[N==0] = np.nan
                makeplot(chi2_dof, basename + f'scan.png',
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
