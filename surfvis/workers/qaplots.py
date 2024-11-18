# flake8: noqa
import os
import sys
from contextlib import ExitStack
from surfvis.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('INIT')
import time
import fsspec

from scabha.schema_utils import clickify_parameters
from surfvis.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.qaplots)
def qaplots(**kw):
    '''
    Quality assurance plots
    '''
    opts = OmegaConf.create(kw)

    if '://' in opts.output_folder:
        protocol = opts.utput_folder.split('://')[0]
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
    logname = f'{str(basedir)}/qaplots_{timestamp}.log'
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
    _qaplots(**opts)

    print(f"All done after {time.time() - ti}s", file=log)


def _qaplots(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import dask
    import dask.array as da
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

    xcol, xptype = opts.xcolumn.split(':')
    ycol, yptype = opts.ycolumn.split(':')

    columns = [opts.flag_column, 'FLAG_ROW',
               'ANTENNA1', 'ANTENNA2', 'TIME', xcol]
    if ycol != xcol:
        columns.append(ycol)

    xds = xds_from_ms(opts.ms,
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'],
                      columns=columns)
    x_col = xcol+xptype
    y_col = ycol+yptype

    cvs = datashader.Canvas(plot_width=800, plot_height=800)
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

    def filter_data(xdata, ydata, flag, xptype, yptype):
        if xptype == 'real':
            xvals = xdata.real
        elif xptype == 'imag':
            xvals = xdata.imag
        elif xptype == 'amp':
            xvals = da.abs(xdata)
        elif xptype == 'phase':
            xvals = da.angle(xdata)
        else:
            raise RuntimeError('Invalid xptype')

        if yptype == 'real':
            yvals = ydata.real
        elif yptype == 'imag':
            yvals = ydata.imag
        elif yptype == 'amp':
            yvals = da.abs(ydata)
        elif yptype == 'phase':
            yvals = da.angle(ydata)
        else:
            raise RuntimeError('Invalid xptype')

        xvals = xvals[~flag]
        yvals = yvals[~flag]

        # we need the size and chunking information to create the dask dataframe
        return xvals.compute_chunk_sizes(), yvals.compute_chunk_sizes()


    imgs = {}
    for i, ds in enumerate(xds):
        imgs[i] = {}
        for corr, c in corrs.items():
            if corr not in opts.corrs:
                continue
            dsc = ds.sel({'corr': c})
            xdata = getattr(dsc, xcol).data
            ydata = getattr(dsc, ycol).data
            flag = getattr(dsc, opts.flag_column).data
            # make sure autocorrs are flagged
            ant1 = dsc.ANTENNA1.data
            ant2 = dsc.ANTENNA2.data
            frow = frow = dsc.FLAG_ROW.data | (ant1 == ant2)
            # combine flag and frow
            flag = da.logical_or(flag, frow[:, None])
            xvals, yvals = filter_data(xdata, ydata, flag, xptype, yptype)
            data_vars = {
            x_col : (('rowchan',), xvals),
            y_col : (('rowchan',), yvals)
            }
            # TODO- create Dask dataframe directly
            dsn = xr.Dataset(data_vars)
            ddf = dsn.to_dask_dataframe()
            # import ipdb; ipdb.set_trace()
            # Create aggregate array
            agg = cvs.points(ddf, x_col, y_col, aggregator)
            imgs[i][corr] = tf.shade(agg, cmap=colorcet.fire)


    imgs = dask.compute(imgs)[0]
    # import ipdb; ipdb.set_trace()
    # create subplots for each ds and corr
    nds = len(xds)
    nc = len(opts.corrs)
    # import ipdb; ipdb.set_trace()
    fig = plt.figure(figsize=(6*nc, 6*nds))

    for i in range(nds):
        for c, corr in enumerate(opts.corrs):
            ax = fig.add_subplot(nds, nc, i*nc + c + 1)
            img = imgs[i][corr]
            # rgb = hv.RGB(hv.operation.datashader.shade.uint32_to_uint8_xr(img))
            # ax.imshow(agg.values, cmap='Purples')
            ax.imshow(img.data, cmap='Purples')

    oname = opts.output_folder + f'/{x_col}_{y_col}_' + '_'.join(opts.corrs) + '.jpeg'
    plt.savefig(oname, dpi=250)

