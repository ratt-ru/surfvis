# Read a Measurement Set and produce per-baseline time-frequency surface plots.
# ian.heywood@csiro.au 
# 13.09.16: Re-wrote to make use of TaQL and handle SPWs / DDIDs

import matplotlib
matplotlib.use('Agg')
from optparse import OptionParser
import pylab
import pyrap.tables
import os
import sys
import numpy


def gi(message):
        print '\033[92m'+message+'\033[0m'


def ri(message):
        print '\033[91m'+message+'\033[0m'


# COMMAND LINE OPTIONS


parser = OptionParser(usage='%prog [options] msname')
#parser.add_option('-m','--msname',dest='msname',help='Measurement Set to open',metavar='DIRECTORY')
parser.add_option('-l','--list',dest='dolist',action='store_true',help='List Measurement Set properties and exit',default=False)
parser.add_option('-d','--datacolumn',dest='column',help='Measurement Set column to plot (default = DATA)',default='DATA')
parser.add_option('-f','--field',dest='field',help='Field ID to plot (default = 0)',default=0)
parser.add_option('-s','--spw',dest='myspw',help='Comma separated list of SPWs to plot (default = all)',default='')
parser.add_option('-p','--plot',dest='plot',help='Set to amp, phase, real or imag (default = amp)',default='amp')
parser.add_option('-i','--i',dest='antenna1',help='Antenna 1: plot only this antenna',default=-1)
parser.add_option('-j','--j',dest='antenna2',help='Antenna 2: use with -i to plot a single baseline',default=-1)
parser.add_option('--scale',dest='scale',help='Scale the image peak to this multiple of the per-corr min/max (ignored for phases)',default=1)
parser.add_option('--cmap',dest='mycmap',help='Matplotlib colour map to use (default = jet)',default='jet')
parser.add_option('-o','--opdir',dest='foldername',help='Output folder to store plots (default = msname___plots)',default='')
(options,args) = parser.parse_args()

dolist = options.dolist
column = options.column
fieldid = int(options.field)	
myspw = options.myspw
plot = options.plot
antenna1 = int(options.antenna1)
antenna2 = int(options.antenna2)
scale = float(options.scale)
mycmap = options.mycmap
foldername = options.foldername

# Some error trapping

if len(args) != 1:
	ri('Please specify a single Measurement Set to plot.')
	sys.exit(-1)
else:
	msname = args[0].rstrip('/')

if plot not in ['amp','phase','real','imag']:
	ri('Requested plot not valid, must be one of amp, phase, real or imag.')
	sys.exit(-1)


# MEASUREMENT SET INFO

fieldtab = pyrap.tables.table(msname+'/FIELD')
sourceids = fieldtab.getcol('SOURCE_ID')
sourcenames = fieldtab.getcol('NAME')
fieldtab.done()

spwtab = pyrap.tables.table(msname+'/SPECTRAL_WINDOW')
nspw = len(spwtab)
spwfreqs = spwtab.getcol('REF_FREQUENCY')
nchans = spwtab.getcol('NUM_CHAN')
spwtab.done()

anttab = pyrap.tables.table(msname+'/ANTENNA')
nant = len(anttab)
antpos = anttab.getcol('POSITION')
antnames = anttab.getcol('NAME')
anttab.done()

tt = pyrap.tables.table(msname)
usedants = numpy.unique(tt.getcol('ANTENNA1'))


# PRINT SUMMARY

if dolist:
	print ''
	gi('     '+msname+'/FIELD')
	gi('     ROW   ID            NAME')
	for i in range(0,len(sourceids)):
		print '     %-6s%-14s%-14s' % (i,sourceids[i],sourcenames[i])
	print ''
	gi('     '+msname+'/SPECTRAL_WINDOW')
	gi('     ROW   CHANS         REF_FREQ[MHz]')
	for i in range(0,nspw):
		print '     %-6s%-14s%-14s' % (i,str(nchans[i]),str(spwfreqs[i]/1e6))
	print ''
	gi('     '+msname+'/ANTENNA')
	gi('     ROW   NAME          POSITION')
	for i in range(0,nant):
		if i in usedants:
			print '     %-6s%-14s%-14s' % (i,(antnames[i]),str(antpos[i]))
		else:
			ri('     %-6s%-14s%-14s' % (i,(antnames[i]),str(antpos[i])))
		
	print ''
	tt.done()


# OR ELSE DO THE PLOTS


else:		
	# Create output folder if it doesn't exist
	if foldername == '':
		foldername = msname+'___plots'
	if os.path.isdir(foldername):
		print 'Found',foldername
	else:
		print 'Creating',foldername
		os.system('mkdir '+foldername)

	# Make a complete list of SPWs if one isn't provided
	if myspw == '':
		myspw = numpy.arange(0,nspw)
	else:
		myspw = myspw.split(',')

	fieldname = sourcenames[fieldid]

	# Make a list of baseline pairs based on the antena selections
	baselines = []
	if antenna1 != -1 and antenna2 != -1:
		baselines = [(antenna1,antenna2)]
	elif antenna1 != -1:
		i = antenna1
		for j in usedants:
			if i != j:
				pair = [i,j]
				pair = sorted(pair)
				if pair not in baselines:
					if antenna1 != -1:
						if antenna1 in pair:
							baselines.append(pair)
					else:
						baselines.append(pair)
	else:
		for i in usedants:
			for j in usedants:
				if i != j:
					pair = [i,j]
					pair = sorted(pair)
					if pair not in baselines:
						if antenna1 != -1:
							if antenna1 in pair:
								baselines.append(pair)
						else:
							baselines.append(pair)

	# Loop over baselines
	for baseline in baselines:

		# Determine unprojected baseline length
		ap1 = antpos[baseline[0]]
		ap2 = antpos[baseline[1]]
		blength = (((ap1[0]-ap2[0])**2.0)+((ap1[1]-ap2[1])**2.0)+((ap1[2]-ap1[2])**2.0))**0.5
		blength = str(round(blength/1000.0,2))

		print 'Plotting baseline:',baseline,'     Unprojected length:',blength,'km'
		
		# Get the data
		datacols = []
		flagcols = []
		# Loop over SPWs
		print 'SPW:',
		for spw in myspw:
			print spw,
			subtab = tt.query(query='ANTENNA1=='+str(baseline[0])
				+' && ANTENNA2=='+str(baseline[1])
				+' && DATA_DESC_ID=='+str(spw))
			datacol = subtab.getcol(column)
			flagcol = subtab.getcol('FLAG')
			datacols.append(datacol)
			flagcols.append(flagcol)
		print ''
		# Reshape the data
		baselinedata = datacols[0]
		flagdata = flagcols[0]
		for p in range(1,len(datacols)):
			baselinedata = numpy.concatenate((baselinedata,datacols[p]),axis=1)
			flagdata = numpy.concatenate((flagdata,flagcols[p]),axis=1)

		# Get number of corr products from the data shape
		n_corr = baselinedata.shape[2]

		# Generate png name
		pngname = foldername+'/'+msname.split('/')[-1].rstrip('/')+'_baseline_'+str(baseline[0])+'_'+str(baseline[1])
		pngname+='_field'+str(fieldid)
		pngname+='_'+plot+'.png'

		# Generate figure title
		figtitle = 'MS: '+msname.rstrip('/')
		figtitle += '\nColumn: '+column+', '+plot
		figtitle += '\nBaseline: '+str(baseline[0])+'-'+str(baseline[1])
		figtitle += ' ['+blength+' km]'
		figtitle+='\nField: '+fieldname

		# Create the figure
		fig = pylab.figure(figsize=(20,15))
		t = fig.text(0.5, 0.945, figtitle,horizontalalignment='center',color='blue')

		# A panel for each corr product
		for k in range(0,n_corr):
			if plot == 'phase':
				plotdata = numpy.angle(baselinedata[:,:,k]) # radians
			elif plot == 'real':
				plotdata = baselinedata[:,:,k].real
			elif plot == 'imag':
				plotdata = baselinedata[:,:,k].imag
			else:
				plotdata = numpy.absolute(baselinedata[:,:,k])
		
			flagimage = pylab.cm.gray(plotdata*0.0)
			flagimage[:,:,3] = (flagdata[:,:,k])
		
			ax = fig.add_subplot(1,n_corr,k+1)
			ax.set_xlabel('Channel number')
			if k==0:
				ax.set_ylabel('Time slot')
			elif k==n_corr-1:
				ax.yaxis.tick_right()
				ax.yaxis.set_label_position('right')
				ax.set_ylabel('Time slot')
			else:
				for ytick_i in ax.get_yticklabels():
					ytick_i.set_visible(False)
			if plot != 'phase' and len(plotdata)>0:
				immax = scale*plotdata.max()
				immin = scale*plotdata.min()
				ax.imshow(plotdata,aspect='auto',clim=(immin,immax),cmap=mycmap)
				ax.imshow(flagimage,aspect='auto',interpolation='nearest',cmap='spring')
			elif len(plotdata)>0:
				ax.imshow(plotdata,aspect='auto',cmap=mycmap)
			else:
				ax.imshow(((0,0),(0,0)),aspect='auto')
			
			ax.set_title('Corr product '+str(k))
			
			print '    Corr product:',k,'      Data min,max:',plotdata.min(),plotdata.max()

		for o in fig.findobj(matplotlib.text.Text):	
			o.set_fontsize('11')

		fig.tight_layout(w_pad=0.98,h_pad=0.98,rect=[0.02,0.02,0.95,0.95])

		pylab.savefig(pngname)
		pylab.close()
