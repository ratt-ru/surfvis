"""Matplotlib helpers shared by the chi-squared commands."""

import matplotlib as mpl

mpl.rcParams.update({"font.size": 11, "font.family": "serif"})

import dask  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.visualization import hist  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

from surfvis.utils.chisq import _surfchisq_slice  # noqa: E402


def makeplot(data, name, subt):
    """Render an antenna-by-antenna chi-squared image with a marginal histogram."""
    nant, _ = data.shape
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(data, cmap="inferno")
    ax.set_xticks(np.arange(0, nant, 2))
    ax.set_yticks(np.arange(nant))
    ax.tick_params(axis="both", which="major", length=1, width=1, labelsize=4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.2)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=6, pad=0.1)

    rax = divider.append_axes("right", size="50%", pad=0.025)
    x = data[~np.isnan(data)]
    if x.any():
        hist(x, bins="scott", ax=rax, histtype="stepfilled", alpha=0.5, density=False)
        rax.set_yticks([])
        rax.tick_params(axis="y", which="both", bottom=False, top=False, labelbottom=False)
        rax.tick_params(axis="x", which="both", length=1, width=1, labelsize=8)

    fig.suptitle(subt, fontsize=20)
    plt.savefig(name, dpi=250)
    plt.close(fig)


def surfchisq_plot(resid, weight, flag, ant1, ant2, field, spw, scan, figname, subt):
    """Compute chi-squared for one chunk, write its figure, and return the accumulators.

    Runs inside a worker process, so the dask arrays are computed with the
    synchronous scheduler before being handed to the numba kernel.
    """
    resid, weight, flag, ant1, ant2 = dask.compute(resid, weight, flag, ant1, ant2, scheduler="sync")
    chi2, counts = _surfchisq_slice(resid, weight, flag, ant1, ant2)
    nant = np.maximum(ant1.max(), ant2.max()) + 1
    chi2_dof = np.zeros((nant, nant), dtype=float)
    chi2_dof[counts > 0] = chi2[counts > 0] / counts[counts > 0]
    chi2_dof[counts <= 0] = np.nan

    makeplot(chi2_dof, figname, subt)

    return field, spw, scan, chi2, counts
