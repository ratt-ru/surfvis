#!/usr/bin/env python
import concurrent.futures as cf
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from optparse import OptionParser
from pyrap.tables import table
import os
import sys
import numpy as np
import itertools
from surfvis.utils import surf
from daskms.experimental.zarr import xds_from_zarr

# COMMAND LINE OPTIONS
def create_parser():
	parser = OptionParser(usage='%prog [options] msname')
	parser.add_option('-d','--dcol', help='Data column (default = DATA)', default='DATA')
	parser.add_option('-w','--wcol', help='Weight column (default = WEIGHT_SPECTRUM)', default='WEIGHT_SPECTRUM')
	parser.add_option('-r', '--rcol', help='Residual column (default = RESIDUAL)', default='RESIDUAL')
	parser.add_option('-f', '--fcol', help='Flag column (default = FLAG)', default='FLAG')
	parser.add_option('-o','--outdir', help='Output folder to store plots',
					  default='')
	parser.add_option('-n', '--ncpu', default=4, type=int,
					  help='Number of processes to spawn')
	parser.add_option('-g', '--gains', help='Path to qcal gain table')
	return parser

def main():
	(options,args) = create_parser().parse_args()

	foldername = options.outdir.rstrip('/')
	if not os.path.isdir(foldername):
		os.system('mkdir '+ foldername)

	# Some error trapping
	if len(args) != 1:
		print('Please specify a single Measurement Set to plot.')
		sys.exit(-1)
	else:
		msname = args[0].rstrip('/')

	# gains partitioned by scan
	G = xds_from_zarr(options.gains)

	# MEASUREMENT SET INFO

	ddtab =table(msname+'::DATA_DESCRIPTION')
	# pols = ddtab.getcol('POLARIZATION_ID')
	spw = ddtab.getcol('SPECTRAL_WINDOW_ID')[0]
	ddtab.done()

	fieldtab = table(msname+'::FIELD')
	sourceids = fieldtab.getcol('SOURCE_ID')
	sourcenames = fieldtab.getcol('NAME')
	fieldtab.done()

	spwtab = table(msname+'::SPECTRAL_WINDOW')
	nspw = len(spwtab)
	spwfreqs = spwtab.getcol('REF_FREQUENCY')
	chanwidth = spwtab.getcol('CHAN_WIDTH')[0][0] # probably needs changing if SPWs have different widths
	nchans = spwtab.getcol('NUM_CHAN')
	spwtab.done()

	anttab = table(msname+'::ANTENNA')
	nant = len(anttab)
	antpos = anttab.getcol('POSITION')
	antnames = anttab.getcol('NAME')
	anttab.done()


	ms = table(msname)
	fields = np.unique(ms.getcol('FIELD_ID'))
	scans = np.unique(ms.getcol('SCAN_NUMBER'))
	ants = np.unique(ms.getcol('ANTENNA1'))

	for field in fields:
		if not os.path.isdir(foldername + f'/field{field}'):
			os.system('mkdir '+ foldername + f'/field{field}')

		chi2_dof_field = np.zeros((len(scans), nant, nant))
		for iscan, scan in enumerate(scans):
			basename = foldername + f'/field{field}' + f'/scan{scan}'
			if not os.path.isdir(basename):
				os.system('mkdir '+ basename)

			for g in G:
				if g.SCAN_NUMBER == scan:
					gain = g.gains.values

			wsums = np.zeros((nant, nant))
			chi2_dof = np.zeros((nant, nant))
			futures = []
			with cf.ProcessPoolExecutor(max_workers=options.ncpu) as executor:
				for pq in itertools.combinations_with_replacement(ants, 2):
					p = pq[0]
					q = pq[1]
					subtab = ms.query(query='ANTENNA1=='+str(p)
										+' && ANTENNA2=='+str(q)
										+' && DATA_DESC_ID=='+str(spw)
										+' && FIELD_ID=='+str(field)
										+' && SCAN_NUMBER=='+str(scan))
					data = subtab.getcol(options.dcol)
					resid = subtab.getcol(options.rcol)
					weight = subtab.getcol(options.wcol)
					flag = subtab.getcol(options.fcol)

					gp = gain[:, :, p, 0]
					gq = gain[:, :, q, 0]

					future = executor.submit(surf, p, q, gp, gq, data, resid,
											 weight, flag, basename)
					futures.append(future)

				for f in cf.as_completed(futures):
					p, q, chi2, wsum = f.result()

					chi2_dof_field[iscan, p, q] = chi2
					chi2_dof_field[iscan, q, p] = chi2

					chi2_dof[p, q] = chi2
					chi2_dof[q, p] = chi2
					wsums[p, q] = wsum
					wsums[q, p] = wsum
					print(f'Completed {str(p)} {str(q)}')

			plt.figure(scan, figsize=(5,5))
			plt.imshow(chi2_dof, cmap='inferno', interpolation=None)
			plt.colorbar()
			plt.title('chi2dof')
			plt.savefig(basename + '/chi2dof.png', dpi=500)
			plt.close()

			plt.figure(scan, figsize=(5,5))
			plt.imshow(wsums, cmap='inferno', interpolation=None)
			plt.colorbar()
			plt.title('wsum')
			plt.savefig(basename + '/wsum.png', dpi=500)
			plt.close()

		chi2_dof_mean = np.nanmean(chi2_dof_field, axis=0)
		chi2_dof_std = np.nanstd(chi2_dof_field, axis=0)

		plt.figure(field, figsize=(5,5))
		plt.imshow(chi2_dof_mean, cmap='inferno', interpolation=None)
		plt.colorbar()
		plt.title('mean chi2dof')
		plt.savefig(basename.rstrip(f'/scan{scan}') + '/chi2dof_mean.png', dpi=500)
		plt.close()

		plt.figure(field, figsize=(5,5))
		plt.imshow(chi2_dof_std, cmap='inferno', interpolation=None)
		plt.colorbar()
		plt.title('std chi2dof')
		plt.savefig(basename.rstrip(f'/scan{scan}') + '/chi2dof_std.png', dpi=500)
		plt.close()
