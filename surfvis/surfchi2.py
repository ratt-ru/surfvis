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
from surfvis.utils import surfchisq
from daskms import xds_from_ms, xds_from_table
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
# might make for cooler histograms but doesn't work out of the box
from astropy.visualization import hist


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
	parser.add_option("--use-corrs", type=str,
					  help='Comma seprated list of correlations to use (do not use spaces)')
	return parser

def main():
	(options,args) = create_parser().parse_args()

	if options.dataout == '':
		options.dataout = os.getcwd() + '/chi2'

	if os.path.isdir(options.dataout):
		print(f"Removing existing {options.dataout} folder")
		os.system(f"rm -r {options.dataout}")

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
	t0s = []
	tfs = []
	for ds in xds:
		time = ds.TIME.values
		ut, counts = np.unique(time, return_counts=True)
		if options.ntimes in [0, -1]:
			utpc = ut.size
		else:
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

		t0 = ut[tidx]
		t0s.append(da.from_array(t0, chunks=1))
		tf = ut[tidx + tcounts -1]
		tfs.append(da.from_array(tf, chunks=1))

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

	if options.use_corrs is None:
		print('Using only diagonal correlations')
		if len(xds[0].corr) > 1:
			use_corrs = [0, -1]
		else:
			use_corrs = [0]
	else:
		use_corrs = list(map(int, options.use_corrs.split(',')))
		print(f"Using correlations {use_corrs}")
	ncorr = len(use_corrs)

	out_ds = []
	idts = []
	rechunk_chan = False
	for i, ds in enumerate(xds):
		rdims = ds.get(options.rcol).dims
		# LB - I don't think this ever happens?
		# if rdims[0] != 'row':
		# 	ds = ds.swap_dims({rdims[0]: 'row'})
		# 	rechunk_row = True
		if rdims[1] != 'chan':
			ds = ds.swap_dims({rdims[1]: 'chan'})
			rechunk_chan = True
		if rdims[2] != 'corr':
			ds = ds.swap_dims({rdims[2]: 'corr'})

		ds = ds.sel(corr=use_corrs)

		resid = ds.get(options.rcol).data
		if rechunk_chan:
			resid = resid.rechunk({1:options.nfreqs})

		# shape = resid.shape
		# chnks = resid.chunks
		# resid = (da.random.standard_normal(size=shape, chunks=chnks) +
		# 			1.0j * da.random.standard_normal(size=shape, chunks=chnks))
		weight = ds.get(options.wcol).data
		# resid = resid/da.sqrt(2 * weight)
		# weight = da.ones(shape, chunks=chnks)/2.0
		flag = ds.get(options.fcol).data
		# flag = da.zeros(shape, chunks=chnks, dtype=bool)
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
		out_ds.append(xds_to_zarr(d, options.dataout + idt))
		idts.append(idt)


	with ProgressBar():
		dask.compute(out_ds)

	# primitive plotting
	if options.imagesout is not None:
		foldername = options.imagesout.rstrip('/')
		if not os.path.isdir(foldername):
				os.system('mkdir '+ foldername)

		for idt in idts:
			xds = xds_from_zarr(options.dataout + idt)
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
							makeplot(chi2_dof, basename + f't{t}_f{f}_c{c}.png',
									 tbin_idx[t], tbin_idx[t] + tbin_counts[t],
									 fbin_idx[f], fbin_idx[f] + fbin_counts[f])

						# reduce over corr
						chi2 = np.nansum(tmp[t, f, (0, -1), :, :, 0], axis=0)
						N = np.nansum(tmp[t, f, (0, -1), :, :, 1], axis=0)
						chi2_dof = np.zeros_like(chi2)
						chi2_dof[N>0] = chi2[N>0]/N[N>0]
						chi2_dof[N==0] = np.nan
						makeplot(chi2_dof, basename + f't{t}_f{f}.png',
								 tbin_idx[t], tbin_idx[t] + tbin_counts[t],
								 fbin_idx[f], fbin_idx[f] + fbin_counts[f])

					# reduce over freq
					chi2 = np.nansum(tmp[t, :, (0, -1), :, :, 0], axis=(0,1))
					N = np.nansum(tmp[t, :, (0, -1), :, :, 1], axis=(0,1))
					chi2_dof = np.zeros_like(chi2)
					chi2_dof[N>0] = chi2[N>0]/N[N>0]
					chi2_dof[N==0] = np.nan
					makeplot(chi2_dof, basename + f't{t}.png',
					         tbin_idx[t], tbin_idx[t] + tbin_counts[t],
							 0, fbin_idx[-1] + fbin_counts[-1])

				# now the entire scan
				chi2 = np.nansum(tmp[:, :, (0, -1), :, :, 0], axis=(0,1,2))
				N = np.nansum(tmp[:, :, (0, -1), :, :, 1], axis=(0,1,2))
				chi2_dof = np.zeros_like(chi2)
				chi2_dof[N>0] = chi2[N>0]/N[N>0]
				chi2_dof[N==0] = np.nan
				makeplot(chi2_dof, basename + f'scan.png',
						 0, tbin_idx[-1] + tbin_counts[-1],
						 0, fbin_idx[-1] + fbin_counts[-1])

def makeplot(data, name, t0, tf, chan0, chanf):
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

	fig.suptitle(f't {t0}-{tf}, chan {chan0}-{chanf}', fontsize=20)
	plt.savefig(name, dpi=250)
	plt.close(fig)
