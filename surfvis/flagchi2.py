#!/usr/bin/env python
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
# mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from optparse import OptionParser
import os
import sys
import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from surfvis.utils import flagchisq
from daskms import xds_from_ms, xds_from_table, xds_to_table


def create_parser():
    parser = OptionParser(usage='%prog [options] msname')
    parser.add_option('--rcol', default='RESIDUAL',
                      help='Residual column (default = RESIDUAL)')
    parser.add_option('--scol',
                      help='sigma column to use. Overides wcol if provided.')
    parser.add_option('--wcol', default='WEIGHT_SPECTRUM',
                      help='Weight column (default = WEIGHT_SPECTRUM)')
    parser.add_option('--fcol', default='FLAG',
                      help='Flag column (default = FLAG)')
    parser.add_option('--sigma', default=3, type=float,
                      help='chisq threshold (default = 3)')
    parser.add_option('--nthreads', default=4, type=int,
                      help='Number of dask threads to use')
    parser.add_option('--nrows', default=10000, type=int,
                      help='Number of rows in each chunk (default=10000)')
    parser.add_option('--nfreqs', default=128, type=int,
                      help='Number of frequencies in a chunk (default=128)')
    parser.add_option("--use-corrs", type=str,
                      help='Comma seprated list of correlations to use (do not use spaces)')
    return parser

def main():
    (options,args) = create_parser().parse_args()

    # Some error trapping
    if len(args) != 1:
        print('Please specify a single Measurement set to flag.')
        sys.exit(-1)
    else:
        msname = args[0].rstrip('/')

    schema = {}
    schema[options.rcol] = {'dims': ('chan', 'corr')}
    schema[options.fcol] = {'dims': ('chan', 'corr')}
    columns = [options.rcol, options.fcol]
    if options.scol is not None:
        schema[options.scol] = {'dims': ('chan', 'corr')}
        columns.append(options.scol)
    else:
        schema[options.wcol] = {'dims': ('chan', 'corr')}
        columns.append(options.wcol)


    xds = xds_from_ms(msname,
                      columns=columns,
                      chunks={'row': options.nrows, 'chan': options.nfreqs},
                      group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'],
                      table_schema=schema)

    if options.use_corrs is None:
        print('Using only diagonal correlations')
        if len(xds[0].corr) > 1:
            use_corrs = [0, -1]
        else:
            use_corrs = [0]
    else:
        use_corrs = list(map(int, options.use_corrs.split(',')))
        print(f"Using correlations {use_corrs}")

    out_data = []
    for i, ds in enumerate(xds):
        resid = ds.get(options.rcol).data
        if options.scol is not None:
            sigma = ds.get(options.scol).data
            weight = 1/sigma**2
        else:
            weight = ds.get(options.wcol).data
        flag = ds.get(options.fcol).data

        uflag = flagchisq(resid, weight, flag, tuple(use_corrs), sigma=options.sigma)

        out_ds = ds.assign(**{options.fcol: (("row", "chan", "corr"), uflag)})
        out_data.append(out_ds)

    writes = xds_to_table(out_data, msname, columns=[options.fcol])

    with ProgressBar():
        dask.compute(writes)
