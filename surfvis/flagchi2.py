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
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_from_storage_table as xds_from_table
from daskms import xds_to_storage_table as xds_to_table


def create_parser():
    parser = OptionParser(usage='%prog [options] msname')
    parser.add_option('--rcol', default='RESIDUAL',
                      help='Residual column (default = RESIDUAL)')
    parser.add_option('--wcol', default='WEIGHT_SPECTRUM',
                      help='Weight column (default = WEIGHT_SPECTRUM). '
                      'The special value SIGMA_SPECTRUM can be passed to '
                      'initialise the weights as 1/sigma**2')
    parser.add_option('--fcol', default='FLAG',
                      help='Flag column (default = FLAG)')
    parser.add_option('--flag-above', default=3, type=float,
                      help='flag data with chisq above this value (default = 3)')
    parser.add_option('--unflag-below', default=1.15, type=float,
                      help='unflag data with chisq below this value (default = 1.15)')
    parser.add_option('--nthreads', default=4, type=int,
                      help='Number of dask threads to use')
    parser.add_option('--nrows', default=10000, type=int,
                      help='Number of rows in each chunk (default=10000)')
    parser.add_option('--nfreqs', default=128, type=int,
                      help='Number of frequencies in a chunk (default=128)')
    parser.add_option("--use-corrs", type=str,
                      help='Comma seprated list of correlations to use (do not use spaces)')
    parser.add_option("--respect-ants", type=str,
                      help='Comma seprated list of antennas to respect (do not use spaces)')
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
    schema[options.wcol] = {'dims': ('chan', 'corr')}
    schema[options.fcol] = {'dims': ('chan', 'corr')}

    xds = xds_from_ms(msname,
                      columns=[options.rcol, options.wcol, options.fcol,
                              'ANTENNA1', 'ANTENNA2'],
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
        use_corrs = tuple(map(int, options.use_corrs.split(',')))
        print(f"Using correlations {use_corrs}")

    if options.respect_ants is not None:
        rants = list(map(int, options.respect_ants.split(',')))
    else:
        rants = []

    out_data = []
    for i, ds in enumerate(xds):
        resid = ds.get(options.rcol).data
        if options.wcol == 'SIGMA_SPECTRUM':
            weight = 1.0/ds.get(options.wcol).data**2
        else:
            weight = ds.get(options.wcol).data
        flag = ds.get(options.fcol).data
        ant1 = ds.ANTENNA1.data
        ant2 = ds.ANTENNA2.data

        # import pdb; pdb.set_trace()

        uflag = flagchisq(resid, weight, flag, ant1, ant2,
                          use_corrs=tuple(use_corrs),
                          flag_above=options.flag_above,
                          unflag_below=options.unflag_below,
                          respect_ants=tuple(rants))

        out_ds = ds.assign(**{options.fcol: (("row", "chan", "corr"), uflag)})

        # update FLAG_ROW
        flag_row = da.all(uflag.rechunk({1:-1, 2:-1}), axis=(1,2))

        out_ds = ds.assign(**{'FLAG_ROW': (("row",), flag_row)})

        out_data.append(out_ds)

    writes = xds_to_table(out_data, msname,
                          columns=[options.fcol, 'FLAG_ROW'],
                          rechunk=True)

    with ProgressBar():
        dask.compute(writes)
