"""Per-baseline time/frequency plots from a Measurement Set."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pylab  # noqa: E402
import pyrap.tables  # noqa: E402

VALID_PLOTS = ("amp", "phase", "real", "imag")


def surf(
    ms: Path,
    datacolumn: str = "DATA",
    field: int = 0,
    spw: list[int] | None = None,
    plot: str = "amp",
    i: int | None = None,
    j: int | None = None,
    noflags: bool = False,
    doacorr: bool = False,
    scale: float | None = None,
    cmap: str = "jet",
    opdir: Path | None = None,
) -> None:
    """Write one time/frequency PNG per baseline.

    Args:
        ms: Measurement Set to plot.
        datacolumn: Column to plot.
        field: Field ID to plot.
        spw: Spectral windows to plot. ``None`` plots all of them.
        plot: One of ``amp``, ``phase``, ``real`` or ``imag``.
        i: Antenna 1. ``None`` (or ``-1``) plots every antenna.
        j: Antenna 2, used together with ``i`` to plot a single baseline.
        noflags: Disable the flagged-data overlay.
        doacorr: Also plot auto-correlations.
        scale: Scale the image peak to this multiple of the per-corr min/max.
            ``None`` (or ``-1``) scales the image max to 5 sigma. Ignored for
            phase plots.
        cmap: Matplotlib colour map.
        opdir: Output folder. Defaults to ``<ms>_<datacolumn>__plots``.
    """
    # -1 is the legacy "unset" sentinel for these three; keep honouring it.
    ant1 = -1 if i is None else i
    ant2 = -1 if j is None else j
    imscale = -1.0 if scale is None else scale

    if plot not in VALID_PLOTS:
        raise ValueError(f"Requested plot not valid, must be one of {', '.join(VALID_PLOTS)}.")

    msname = str(ms).rstrip("/")

    fieldtab = pyrap.tables.table(msname + "/FIELD")
    sourcenames = fieldtab.getcol("NAME")
    fieldtab.done()

    spwtab = pyrap.tables.table(msname + "/SPECTRAL_WINDOW")
    nspw = len(spwtab)
    spwtab.done()

    anttab = pyrap.tables.table(msname + "/ANTENNA")
    antpos = anttab.getcol("POSITION")
    anttab.done()

    tt = pyrap.tables.table(msname)
    usedants = np.unique(tt.getcol("ANTENNA1"))

    # Create output folder if it doesn't exist
    if opdir is None:
        foldername = Path(f"{msname}_{datacolumn}__plots")
    else:
        foldername = Path(str(opdir))
    if foldername.is_dir():
        print("Found", foldername)
    else:
        print("Creating", foldername)
        foldername.mkdir(parents=True, exist_ok=True)

    # Make a complete list of SPWs if one isn't provided
    myspw = list(range(0, nspw)) if not spw else list(spw)

    fieldname = sourcenames[field]

    # Make a list of baseline pairs based on the antenna selections
    baselines = []
    if ant1 != -1 and ant2 != -1:
        baselines = [(ant1, ant2)]
    else:
        ants1 = [ant1] if ant1 != -1 else usedants
        for p in ants1:
            for q in usedants:
                if p != q or doacorr:
                    pair = sorted([p, q])
                    if pair not in baselines:
                        if ant1 == -1 or ant1 in pair:
                            baselines.append(pair)

    # Loop over baselines
    for baseline in baselines:
        # Determine unprojected baseline length
        ap1 = antpos[baseline[0]]
        ap2 = antpos[baseline[1]]
        blength = (((ap1[0] - ap2[0]) ** 2.0) + ((ap1[1] - ap2[1]) ** 2.0) + ((ap1[2] - ap1[2]) ** 2.0)) ** 0.5
        blength = str(round(blength / 1000.0, 2))

        print("Plotting baseline:", baseline, "     Deprojected length:", blength, "km")

        # Get the data
        datacols = []
        flagcols = []
        # Loop over SPWs
        print("SPW:")
        for s in myspw:
            print(s)
            subtab = tt.query(
                query="ANTENNA1=="
                + str(baseline[0])
                + " && ANTENNA2=="
                + str(baseline[1])
                + " && DATA_DESC_ID=="
                + str(s)
                + " && FIELD_ID=="
                + str(field)
            )
            datacols.append(subtab.getcol(datacolumn))
            flagcols.append(subtab.getcol("FLAG"))
        print("")

        # Reshape the data
        baselinedata = datacols[0]
        flagdata = flagcols[0]
        for p in range(1, len(datacols)):
            baselinedata = np.concatenate((baselinedata, datacols[p]), axis=1)
            flagdata = np.concatenate((flagdata, flagcols[p]), axis=1)

        # Get number of corr products from the data shape
        n_corr = baselinedata.shape[2]

        # Generate png name
        pngname = str(foldername / f"{msname.split('/')[-1]}_baseline_{baseline[0]}_{baseline[1]}")
        pngname += f"_field{field}"
        pngname += f"_{plot}.png"

        # Generate figure title
        figtitle = "MS: " + msname
        figtitle += "\nColumn: " + datacolumn + ", " + plot
        figtitle += "\nBaseline: " + str(baseline[0]) + "-" + str(baseline[1])
        figtitle += " [" + blength + " km]"
        figtitle += "\nField: " + str(fieldname)

        # Create the figure
        fig = pylab.figure(figsize=(20, 15))
        fig.text(0.5, 0.945, figtitle, horizontalalignment="center", color="blue")

        # A panel for each corr product
        for k in range(0, n_corr):
            if plot == "phase":
                plotdata = np.angle(baselinedata[:, :, k])  # radians
            elif plot == "real":
                plotdata = baselinedata[:, :, k].real
            elif plot == "imag":
                plotdata = baselinedata[:, :, k].imag
            else:
                plotdata = np.absolute(baselinedata[:, :, k])

            flagimage = pylab.cm.gray(plotdata * 0.0)
            flagimage[:, :, 3] = flagdata[:, :, k]

            ax = fig.add_subplot(1, n_corr, k + 1)
            ax.set_xlabel("Channel number")
            if k == 0:
                ax.set_ylabel("Time slot")
            elif k == n_corr - 1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.set_ylabel("Time slot")
            else:
                for ytick_i in ax.get_yticklabels():
                    ytick_i.set_visible(False)

            if plot != "phase" and len(plotdata) > 0:
                if imscale != -1:
                    immax = imscale * plotdata.max()
                    immin = imscale * plotdata.min()
                else:
                    immax = 5.0 * np.std(plotdata)
                    immin = 0.0
                ax.imshow(plotdata, aspect="auto", clim=(immin, immax), cmap=cmap)
                if not noflags:
                    ax.imshow(flagimage, aspect="auto", interpolation="nearest")
            elif len(plotdata) > 0:
                ax.imshow(plotdata, aspect="auto", cmap=cmap)
                if not noflags:
                    ax.imshow(flagimage, aspect="auto", interpolation="nearest")
            else:
                ax.imshow(((0, 0), (0, 0)), aspect="auto")

            ax.set_title("Corr product " + str(k))

            print("    Corr product:", k, "      Data min,max:", plotdata.min(), plotdata.max())

        for o in fig.findobj(matplotlib.text.Text):
            o.set_fontsize("11")

        fig.tight_layout(w_pad=0.98, h_pad=0.98, rect=[0.02, 0.02, 0.95, 0.95])

        pylab.savefig(pngname)
        pylab.close()

    tt.done()
