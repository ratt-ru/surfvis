# surfvis
Create per-baseline time/frequency plots from a Measurement Set. Uses pyrap.

```
Usage: surfvis.py [options] msname

Options:
  -h, --help            show this help message and exit
  -l, --list            List Measurement Set properties and exit
  -d COLUMN, --datacolumn=COLUMN
                        Measurement Set column to plot (default = DATA)
  -f FIELD, --field=FIELD
                        Field ID to plot (default = 0)
  -s MYSPW, --spw=MYSPW
                        Comma separated list of SPWs to plot (default = all)
  -p PLOT, --plot=PLOT  Set to amp, phase, real or imag (default = amp)
  -i ANTENNA1, --i=ANTENNA1
                        Antenna 1: plot only this antenna
  -j ANTENNA2, --j=ANTENNA2
                        Antenna 2: use with -i to plot a single baseline
  --scale=SCALE         Scale the image peak to this multiple of the per-corr
                        min/max (ignored for phases)
  --cmap=MYCMAP         Matplotlib colour map to use (default = jet)
  -o FOLDERNAME, --opdir=FOLDERNAME
                        Output folder to store plots (default = msname___plots)
```

Plotting can be quite slow for large Measurement Sets. Will probably also fail if you have SPWs with different time/freq shapes. Use of the -s switch can generally get around this. Use at own risk, etc.
