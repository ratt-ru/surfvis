"""List the FIELD, SPECTRAL_WINDOW and ANTENNA tables of a Measurement Set."""

from pathlib import Path

import numpy as np
import pyrap.tables


def _green(message: str) -> None:
    print("\033[92m" + message + "\033[0m")


def _red(message: str) -> None:
    print("\033[91m" + message + "\033[0m")


def summary(ms: Path) -> None:
    """Print a summary of the fields, spectral windows and antennas in ``ms``.

    Antennas that do not appear in the ANTENNA1 column are printed in red.
    """
    msname = str(ms).rstrip("/")

    fieldtab = pyrap.tables.table(msname + "/FIELD")
    sourceids = fieldtab.getcol("SOURCE_ID")
    sourcenames = fieldtab.getcol("NAME")
    fieldtab.done()

    spwtab = pyrap.tables.table(msname + "/SPECTRAL_WINDOW")
    nspw = len(spwtab)
    spwfreqs = spwtab.getcol("REF_FREQUENCY")
    # probably needs changing if SPWs have different widths
    chanwidth = spwtab.getcol("CHAN_WIDTH")[0][0]
    nchans = spwtab.getcol("NUM_CHAN")
    spwtab.done()

    anttab = pyrap.tables.table(msname + "/ANTENNA")
    nant = len(anttab)
    antpos = anttab.getcol("POSITION")
    antnames = anttab.getcol("NAME")
    anttab.done()

    tt = pyrap.tables.table(msname)
    usedants = np.unique(tt.getcol("ANTENNA1"))
    tt.done()

    print("")
    _green("     " + msname + "/FIELD")
    _green("     ROW   ID            NAME")
    for i in range(0, len(sourceids)):
        print("     %-6s%-14s%-14s" % (i, sourceids[i], sourcenames[i]))

    print("")
    _green("     " + msname + "/SPECTRAL_WINDOW")
    _green("     ROW   CHANS         WIDTH[MHz]          REF_FREQ[MHz]")
    for i in range(0, nspw):
        print("     %-6s%-14s%-20s%-14s" % (i, str(nchans[i]), str(chanwidth / 1e6), str(spwfreqs[i] / 1e6)))

    print("")
    _green("     " + msname + "/ANTENNA")
    _green("     ROW   NAME          POSITION")
    for i in range(0, nant):
        if i in usedants:
            print("     %-6s%-14s%-14s" % (i, antnames[i], str(antpos[i])))
        else:
            _red("     %-6s%-14s%-14s" % (i, antnames[i], str(antpos[i])))
    print("")
