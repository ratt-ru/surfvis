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
from surfvis.utils import chisq
from daskms import xds_from_ms, xds_from_table
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr


# COMMAND LINE OPTIONS
def create_parser():
	parser = OptionParser(usage='%prog [options] msname')
	parser.add_option('--rcol', default='RESIDUAL',
					  help='Residual column (default = RESIDUAL)')
	parser.add_option('--wcol', default='WEIGHT_SPECTRUM',
					  help='Weight column (default = WEIGHT_SPECTRUM)')
	parser.add_option('--fcol', default='FLAG',
					  help='Flag column (default = FLAG)')
	parser.add_option('--dataout', default='',
					  help='Output name of zarr dataset')
	parser.add_option('--imagesout', default=None,
					  help='Output folder to place images. '
					  	   'If None (default) no plots are saved')
	parser.add_option('--nthreads', default=4, type=int,
					  help='Number of dask threads to use')
	parser.add_option('--ntimes', default=-1, type=int,
				      help='Number of unique times in each chunk.')
	parser.add_option('--nfreqs', default=128, type=int,
					  help='Number of frequencies in a chunk.')
	return parser

def main():
	(options,args) = create_parser().parse_args()

	if options.dataout == '':
		options.dataout = os.getcwd() + '/chi2'

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
	rbin_idx = []
	rbin_counts = []
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

		nchan = ds.chan.size
		if options.nfreqs in [0, -1]:
			options.nfreqs = nchan

		# list per ds
		chunks.append({'row': tuple(row_chunks), 'chan': options.nfreqs})

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

		fidx = np.arange(0, nchan, options.nfreqs)
		fbin_idx.append(da.from_array(fidx, chunks=1))
		fidx2 = np.append(fidx, nchan)
		fcounts = fidx2[1:] - fidx2[0:-1]
		fbin_counts.append(da.from_array(fcounts, chunks=1))

	xds = xds_from_ms(msname,
					  columns=[options.rcol, options.wcol, options.fcol,
					  		   'ANTENNA1', 'ANTENNA2', 'TIME'],
					  chunks=chunks,
					  group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])

	out_ds = []
	for i, ds in enumerate(xds):
		resid = ds.get(options.rcol).data
		weight = ds.get(options.wcol).data
		flag = ds.get(options.fcol).data
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

		tmp = chisq(resid, weight, flag, ant1, ant2,
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

		out_ds.append(xds_to_zarr(d, options.dataout))


	with ProgressBar():
		dask.compute(out_ds)

	xds = xds_from_zarr(options.dataout)

	if options.imagesout is not None:

		foldername = options.imagesout.rstrip('/')
		if not os.path.isdir(foldername):
				os.system('mkdir '+ foldername)

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
			chi2 = tmp[:, :, :, :, :, 0]
			N = tmp[:, :, :, :, :, 1]
			chi2_dof = np.where(N > 0, chi2/N, np.nan)

			ntime, nfreq, ncorr, _, _ = chi2.shape
			basename = foldername + f'/field{field}' + f'/spw{spw}'+ f'/scan{scan}/'
			for t in range(ntime):
				for f in range(nfreq):
					for c in range(ncorr):
						tmp = chi2_dof[t, f, c]

						fig, ax = plt.subplots(nrows=1, ncols=2)
						im = ax[0].imshow(tmp, cmap='inferno')
						ax[0].axis('off')
						ax[0].set_title('chisq dof')

						divider = make_axes_locatable(ax[0])
						cax = divider.append_axes("bottom", size="10%", pad=0.01)
						cb = fig.colorbar(im, cax=cax, orientation="horizontal")
						cb.outline.set_visible(False)
						# cb.ax.tick_params(length=, width=0.1, labelsize=0.1, pad=0.1)

						ax[1].hist(tmp[tmp != np.nan], bins=27)

						plt.savefig(basename + f't{t}_f{f}_c{c}.png', dpi=250)
						plt.close(fig)

					# reduce over corr
					tmp = np.nanmean(chi2_dof[t, f], axis=0)

					fig, ax = plt.subplots(nrows=1, ncols=2)
					im = ax[0].imshow(tmp, cmap='inferno')
					ax[0].axis('off')
					ax[0].set_title('chisq dof')

					divider = make_axes_locatable(ax[0])
					cax = divider.append_axes("bottom", size="10%", pad=0.01)
					cb = fig.colorbar(im, cax=cax, orientation="horizontal")
					cb.outline.set_visible(False)
					# cb.ax.tick_params(length=, width=0.1, labelsize=0.1, pad=0.1)

					ax[1].hist(tmp[tmp != np.nan], bins=27)

					plt.savefig(basename + f't{t}_f{f}.png', dpi=250)
					plt.close(fig)

				# reduce over freq
				tmp = np.nanmean(chi2_dof[t], axis=(0, 1))

				fig, ax = plt.subplots(nrows=1, ncols=2)
				im = ax[0].imshow(tmp, cmap='inferno')
				ax[0].axis('off')
				ax[0].set_title('chisq dof')

				divider = make_axes_locatable(ax[0])
				cax = divider.append_axes("bottom", size="10%", pad=0.01)
				cb = fig.colorbar(im, cax=cax, orientation="horizontal")
				cb.outline.set_visible(False)
				# cb.ax.tick_params(length=, width=0.1, labelsize=0.1, pad=0.1)

				ax[1].hist(tmp[tmp != np.nan], bins=27)

				plt.savefig(basename + f't{t}.png', dpi=250)
				plt.close(fig)

			# now the entire scan
			tmp = np.nanmean(chi2_dof, axis=(0, 1, 2))

			fig, ax = plt.subplots(nrows=1, ncols=2)
			im = ax[0].imshow(tmp, cmap='inferno')
			ax[0].axis('off')
			ax[0].set_title('chisq dof')

			divider = make_axes_locatable(ax[0])
			cax = divider.append_axes("bottom", size="10%", pad=0.01)
			cb = fig.colorbar(im, cax=cax, orientation="horizontal")
			cb.outline.set_visible(False)
			# cb.ax.tick_params(length=, width=0.1, labelsize=0.1, pad=0.1)

			ax[1].hist(tmp[tmp != np.nan], bins=27)

			plt.savefig(basename + f'scan{scan}.png', dpi=250)
			plt.close(fig)
