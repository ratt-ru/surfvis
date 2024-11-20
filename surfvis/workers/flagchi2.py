#!/usr/bin/env python

import os
import sys
import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from surfvis.utils import flagchisq
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_from_storage_table as xds_from_table
from daskms import xds_to_storage_table as xds_to_table

from surfvis.workers.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('svis')
log = pyscilog.get_logger('FLAGCHI2')
import time
import fsspec


from scabha.schema_utils import clickify_parameters
from surfvis.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.flagchi2)
def flagchi2(**kw):
    opts = OmegaConf.create(kw)

    print('Input Options:')
    for key, value in opts.items():
        print('     %25s = %s' % (key, value), file=log)

    msname = opts.ms.rstrip('/')

    from multiprocessing.pool import ThreadPool
    dask.config.set(pool=ThreadPool(opts.nthreads))

    schema = {}
    schema[opts.rcol] = {'dims': ('chan', 'corr')}
    schema[opts.wcol] = {'dims': ('chan', 'corr')}
    schema[opts.fcol] = {'dims': ('chan', 'corr')}

    xds = xds_from_ms(msname,
                      columns=[opts.rcol, opts.wcol, opts.fcol,
                              'ANTENNA1', 'ANTENNA2'],
                      chunks={'row': opts.nrows, 'chan': opts.nfreqs},
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'],
                      table_schema=schema)

    if opts.use_corrs is None:
        print('Using only diagonal correlations')
        if len(xds[0].corr) > 1:
            use_corrs = [0, -1]
        else:
            use_corrs = [0]
    else:
        use_corrs = tuple(map(int, opts.use_corrs.split(',')))
        print(f"Using correlations {use_corrs}")

    if opts.respect_ants is not None:
        rants = list(map(int, opts.respect_ants.split(',')))
    else:
        rants = []

    out_data = []
    for i, ds in enumerate(xds):
        resid = ds.get(opts.rcol).data
        if opts.wcol == 'SIGMA_SPECTRUM':
            weight = 1.0/ds.get(opts.wcol).data**2
        else:
            weight = ds.get(opts.wcol).data
        flag = ds.get(opts.fcol).data
        ant1 = ds.ANTENNA1.data
        ant2 = ds.ANTENNA2.data

        uflag = flagchisq(resid, weight, flag, ant1, ant2,
                          use_corrs=tuple(use_corrs),
                          flag_above=opts.flag_above,
                          respect_ants=tuple(rants))

        out_ds = ds.assign(**{opts.fcol: (("row", "chan", "corr"), uflag)})

        # update FLAG_ROW
        flag_row = da.all(uflag.rechunk({1:-1, 2:-1}), axis=(1,2))

        out_ds = out_ds.assign(**{'FLAG_ROW': (("row",), flag_row)})

        out_data.append(out_ds)

    writes = xds_to_table(out_data, msname,
                          columns=[opts.fcol, 'FLAG_ROW'],
                          rechunk=True)

    with ProgressBar():
        dask.compute(writes)
