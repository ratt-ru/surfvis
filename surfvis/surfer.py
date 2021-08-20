#!/usr/bin/env python
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from optparse import OptionParser
import os
import sys
import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from surfvis.utils import chisq
from daskms import xds_from_ms, xds_from_table

# COMMAND LINE OPTIONS
def create_parser():
	parser = OptionParser(usage='%prog [options] msname')
	parser.add_option('--rcol', default='RESIDUAL',
					  help='Residual column (default = RESIDUAL)')
	parser.add_option('--wcol', default='WEIGHT_SPECTRUM',
					  help='Weight column (default = WEIGHT_SPECTRUM)')
	parser.add_option('--fcol', default='FLAG',
					  help='Flag column (default = FLAG)')
	parser.add_option('--outdir', default='',
					  help='Output folder to store plots')
	parser.add_option('--nthreads', default=4, type=int,
					  help='Number of dask threads to use')
	parser.add_option('--ntimes', default=-1, type=int,
				      help='Number of unique times in each chunk.')
	parser.add_option('--nfreqs', default=128, type=int,
					  help='Number of frequencies in a chunk.')
	return parser

def main():
	(options,args) = create_parser().parse_args()

	if options.outdir == '':
		options.outdir = os.getcwd() + '/stats'

	foldername = options.outdir.rstrip('/')
	if not os.path.isdir(foldername):
		os.system('mkdir '+ foldername)

	# Some error trapping
	if len(args) != 1:
		print('Please specify a single Measurement Set to plot.')
		sys.exit(-1)
	else:
		msname = args[0].rstrip('/')

	# chunking info
	xds = xds_from_ms(msname,
					  chunks={'row': -1},
					  columns=['TIME', 'FLAG'],
					  group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])

	chunks = []
	tbin_idx = []
	tbin_counts = []
	fbin_idx = []
	fbin_counts = []
	for ds in xds:
		time = ds.TIME.values
		ut, counts = np.unique(time, return_counts=True)
		if options.ntimes in [0, -1]:
			options.ntimes = ut.size
		utpc = options.ntimes
		row_chunks = [np.sum(counts[i:i+utpc])
                 	  for i in range(0, ut.size, utpc)]

		# list per ds
		chunks.append({'row': tuple(row_chunks), 'chan': options.nfreqs})

		tidx = np.zeros(len(row_chunks))
		tidx[1:] = np.cumsum(row_chunks)[0:-1]
		tbin_idx.append(da.from_array(tidx.astype(int), chunks=1))
		tbin_counts.append(da.from_array(row_chunks, chunks=1))

		nchan = ds.chan.size
		fidx = np.arange(0, nchan, options.nfreqs)
		fbin_idx.append(da.from_array(fidx, chunks=1))
		fidx2 = np.append(fidx, nchan)
		fcounts = fidx2[1:] - fidx2[0:-1]
		fbin_counts.append(da.from_array(fcounts, chunks=1))

	xds = xds_from_ms(msname,
					  columns=[options.rcol, options.wcol, options.fcol,
					  		   'ANTENNA1', 'ANTENNA2'],
					  chunks=chunks,
					  group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])

	chisqs = []
	for i, ds in enumerate(xds):
		field = ds.FIELD_ID
		if not os.path.isdir(foldername + f'/field{field}'):
			os.system('mkdir '+ foldername + f'/field{field}')

		spw = ds.DATA_DESC_ID
		if not os.path.isdir(foldername + f'/field{field}' + f'/spw{spw}'):
			os.system('mkdir '+ foldername + f'/field{field}' + f'/spw{spw}')

		scan = ds.SCAN_NUMBER
		if not os.path.isdir(foldername + f'/field{field}' + f'/spw{spw}' + f'/scan{scan}'):
			os.system('mkdir '+ foldername + f'/field{field}' + f'/spw{spw}'+ f'/scan{scan}')


		resid = ds.get(options.rcol).data
		weight = ds.get(options.wcol).data
		flag = ds.get(options.fcol).data
		ant1 = ds.ANTENNA1.data
		ant2 = ds.ANTENNA2.data

		chisqs.append(
			chisq(resid, weight, flag, ant1, ant2,
				  tbin_idx[i], tbin_counts[i],
				  fbin_idx[i], fbin_counts[i])
		)

	with ProgressBar():
		chisqs = dask.compute(chisqs)


	# plt.figure(scan, figsize=(5,5))
	# plt.imshow(chi2_dof, cmap='inferno', interpolation=None)
	# plt.colorbar()
	# plt.title(f'scan {scan} chi2dof')
	# plt.savefig(basename + '/chi2dof.png', dpi=500)
	# plt.close()

	# plt.figure(scan, figsize=(5,5))
	# plt.imshow(wsums, cmap='inferno', interpolation=None)
	# plt.colorbar()
	# plt.title(f'scan {scan} wsum')
	# plt.savefig(basename + '/wsum.png', dpi=500)
	# plt.close()

	# chi2_dof_mean = np.nanmean(chi2_dof_field, axis=0)
	# chi2_dof_std = np.nanstd(chi2_dof_field, axis=0)

	# plt.figure(field, figsize=(5,5))
	# plt.imshow(chi2_dof_mean, cmap='inferno', interpolation=None)
	# plt.colorbar()
	# plt.title(f'field {field} mean chi2dof')
	# plt.savefig(basename.rstrip(f'/scan{scan}') + '/chi2dof_mean.png', dpi=500)
	# plt.close()

	# plt.figure(field, figsize=(5,5))
	# plt.imshow(chi2_dof_std, cmap='inferno', interpolation=None)
	# plt.colorbar()
	# plt.title(f'field {field} std chi2dof')
	# plt.savefig(basename.rstrip(f'/scan{scan}') + '/chi2dof_std.png', dpi=500)
	# plt.close()
