# flake8: noqa
import os
import sys
from contextlib import ExitStack
from surfvis.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('svis')
log = pyscilog.get_logger('PHASEBALL')
import time
import fsspec

from scabha.schema_utils import clickify_parameters
from surfvis.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.phaseball)
def phaseball(**kw):
    '''
    Plot phase balls
    '''
    opts = OmegaConf.create(kw)

    if '://' in opts.output_folder:
        protocol = opts.output_folder.split('://')[0]
        prefix = f'{protocol}://'
    else:
        protocol = 'file'
        prefix = ''

    fs = fsspec.filesystem(protocol)
    basedir = fs.expand_path('/'.join(opts.output_folder.split('/')))[0]
    if not fs.exists(basedir):
        fs.makedirs(basedir)

    opts.output_folder = basedir

    import psutil
    nthreads = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if opts.nthreads is None:
        opts.nthreads = nthreads//2
        ncpu = ncpu//2

    # if opts.product.upper() not in ["I","Q", "U", "V"]:
    #     raise NotImplementedError(f"Product {opts.product} not yet supported")

    OmegaConf.set_struct(opts, True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(basedir)}/phaseball_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    from surfvis import set_envs
    set_envs(opts.nthreads, ncpu)

    # with ExitStack() as stack:
    import dask
    dask.config.set(**{'array.slicing.split_large_chunks': False})

    ti = time.time()
    _phaseball(**opts)

    print(f"All done after {time.time() - ti}s", file=log)


def _phaseball(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import dask.array as da
    import dask.dataframe as dd
    import datashader
    import datashader.transfer_functions as tf
    from datashader.colors import Greys9, viridis
    import holoviews as hv
    from typing import Optional, Tuple, Union
    import colorcet
    from daskms import xds_from_storage_ms as xds_from_ms
    import xarray as xr
    import matplotlib.pyplot as plt
    hv.extension('bokeh')

    columns = [opts.flag_column, 'FLAG_ROW',
               'ANTENNA1', 'ANTENNA2', 'TIME', opts.column]

    group_cols = ['FIELD_ID', 'DATA_DESC_ID']
    if not opts.combine_scans:
        group_cols.append('SCAN_NUMBER')

    xdsi = xds_from_ms(opts.ms,
                       group_cols=group_cols,
                       columns=columns)
    xds = []
    for ds in xdsi:
        fid = ds.FIELD_ID
        ddid = ds.DATA_DESC_ID
        if (opts.fields is not None) and (fid not in opts.fields):
            continue
        if (opts.ddids is not None) and (ddid not in opts.ddids):
            continue
        if ('SCAN_NUMBER' in ds) and (opts.scans is not None) and (scanid not in opts.scans):
            continue
        xds.append(ds)

    cvs = datashader.Canvas(plot_width=200, plot_height=200)
    aggregator = datashader.count()

    ncorr = xds[0].sizes['corr']
    if ncorr == 4:
        corrs = {'00': 0, '01': 1, '10': 2, '11': 3}
        try:
            assert len(opts.corrs) < 5
        except:
            raise RuntimeError('Provided too many corrs for MS')
    elif ncorr == 2:
        corrs = {'00': 0, '11': 1}
        try:
            assert len(opts.corrs) < 3
        except:
            raise RuntimeError('Provided too many corrs for MS')
    elif ncorr == 1:
        corrs = {'00': 0}
        try:
            assert len(opts.corrs) < 2
        except:
            raise RuntimeError('Provided too many corrs for MS')
    else:
        raise RuntimeError('Invalid number of correlations in MS')

    def filter_data(data, flag):

        data = data[~flag]
        xvals = data.real
        yvals = data.imag

        # we need the size and chunking information to create the dask dataframe
        return xvals.compute_chunk_sizes(), yvals.compute_chunk_sizes()

    imgs = []
    for ds in xds:
        # make sure autocorrs are flagged
        ant1 = ds.ANTENNA1.data
        ant2 = ds.ANTENNA2.data
        frow = frow = ds.FLAG_ROW.data | (ant1 == ant2)
        for corr, c in corrs.items():
            if corr not in opts.corrs:
                continue
            dsc = ds.sel({'corr': c})
            data = getattr(dsc, opts.column).data
            flag = getattr(dsc, opts.flag_column).data
            flag = da.logical_or(flag, frow[:, None])
            xvals, yvals = filter_data(data, flag)
            data_vars = {
            'x' : (('rowchan',), xvals),
            'y' : (('rowchan',), yvals)
            }
            dsn = xr.Dataset(data_vars)
            ddf = dsn.to_dask_dataframe()
            agg = cvs.points(ddf, 'x', 'y', aggregator)
            title = f'F{ds.FIELD_ID}_D{ds.DATA_DESC_ID}'
            if not opts.combine_scans:
                title += f'_S{ds.SCAN_NUMBER}'
            title += f'_C{corr}'
            img = hv.Image(tf.shade(agg, cmap=colorcet.fire)).opts(
                title=title
            )
            imgs.append(img)

    ms = opts.ms.split('/')[-1]
    oname = opts.output_folder + f'/{ms}_{opts.column}_' + '_'.join(opts.corrs) + '.html'
    layout = hv.Layout(imgs).cols(len(opts.corrs))
    hv.save(layout, oname)
