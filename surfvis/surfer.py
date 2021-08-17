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

# COMMAND LINE OPTIONS
def create_parser():
	parser = OptionParser(usage='%prog [options] msname')
	parser.add_option('-d','--dcol', help='Data column to plot (default = DATA)', default='DATA')
	parser.add_option('-w','--wcol', help='Weight column to plot (default = WEIGHT)', default='WEIGHT')
	parser.add_option('-p','--products', help='a=amplitude, p=phase, r=real and i=imag (default = apir)',
					  default='ap')
	parser.add_option('-o','--outdir', help='Output folder to store plots (default = msname___plots)',
					  default='')
	parser.add_option('-n', '--ncpu', default=4, type=int,
					  help='Number of processes to spawn')
	return parser

def main():
	(options,args) = create_parser().parse_args()

	foldername = options.outdir.rstrip('/')

	# Some error trapping
	if len(args) != 1:
		print('Please specify a single Measurement Set to plot.')
		sys.exit(-1)
	else:
		msname = args[0].rstrip('/')


	# MEASUREMENT SET INFO

	ddtab =table(msname+'/DATA_DESCRIPTION')
	# pols = ddtab.getcol('POLARIZATION_ID')
	spw = ddtab.getcol('SPECTRAL_WINDOW_ID')[0]
	ddtab.done()

	fieldtab = table(msname+'/FIELD')
	sourceids = fieldtab.getcol('SOURCE_ID')
	sourcenames = fieldtab.getcol('NAME')
	fieldtab.done()

	spwtab = table(msname+'/SPECTRAL_WINDOW')
	nspw = len(spwtab)
	spwfreqs = spwtab.getcol('REF_FREQUENCY')
	chanwidth = spwtab.getcol('CHAN_WIDTH')[0][0] # probably needs changing if SPWs have different widths
	nchans = spwtab.getcol('NUM_CHAN')
	spwtab.done()

	anttab = table(msname+'/ANTENNA')
	nant = len(anttab)
	antpos = anttab.getcol('POSITION')
	antnames = anttab.getcol('NAME')
	anttab.done()


	ms = table(msname)
	fields = np.unique(ms.getcol('FIELD_ID'))
	scans = np.unique(ms.getcol('SCAN_NUMBER'))
	ants = np.unique(ms.getcol('ANTENNA1'))

	for ifield in fields:
		if not os.path.isdir(options.outdir + f'/field{ifield}'):
			os.system('mkdir '+ foldername + f'/field{ifield}')

		for iscan in scans:
			if not os.path.isdir(options.outdir + f'/field{ifield}' + f'/scan{iscan}'):
				os.system('mkdir '+ foldername + f'/field{ifield}' + f'/scan{iscan}')
			basename = foldername + f'/field{ifield}' + f'/scan{iscan}'


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
										+' && FIELD_ID=='+str(ifield)
										+' && SCAN_NUMBER=='+str(iscan))
					data = subtab.getcol(options.dcol)
					weight = subtab.getcol(options.wcol)
					flag = subtab.getcol('FLAG')

					future = executor.submit(surf, p, q, data, weight, flag,
											 basename, options.products)
					futures.append(future)
					# print(f'Submitted {str(p)} {str(q)}')

				for f in cf.as_completed(futures):
					p, q, chi2, wsum = f.result()

					#chi2, p, q = _surf(p, q, data, weight, flag, basename, products='riap')

					chi2_dof[p, q] = chi2
					chi2_dof[q, p] = chi2
					wsums[p, q] = wsum
					wsums[q, p] = wsum
					print(f'Completed {str(p)} {str(q)}')

			plt.figure(iscan, figsize=(5,5))
			plt.imshow(chi2_dof, cmap='inferno', interpolation=None)
			plt.colorbar()
			plt.title('chi2dof')
			plt.savefig(basename + '/chi2dof.png', dpi=500)
			plt.close()

			plt.figure(iscan, figsize=(5,5))
			plt.imshow(wsums, cmap='inferno', interpolation=None)
			plt.colorbar()
			plt.title('wsum')
			plt.savefig(basename + '/wsum.png', dpi=500)
			plt.close()
