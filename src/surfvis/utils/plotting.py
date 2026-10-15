"""Matplotlib helpers shared by the chi-squared commands."""

import matplotlib as mpl

# Workers only write PNGs. Without this, a forwarded DISPLAY selects TkAgg, and
# when the X connection dies every worker exits on an XIO error.
mpl.use("Agg")
mpl.rcParams.update({"font.size": 11, "font.family": "serif"})

import dask  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.visualization import hist  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

from surfvis.utils.chisq import _surfchisq_slice  # noqa: E402
from surfvis.utils.products import form  # noqa: E402


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


def surfchisq_products_plot(
    products, labels, resid, weight, flag, ant1, ant2, time, model, gains, grid_times, fignames, subtitles
):
    """Chi-squared for one (time, freq) chunk, per product; write a figure for each.

    Runs inside a worker process, so the dask arrays are computed with the
    synchronous scheduler before anything else touches them. Correction, model
    subtraction and Stokes formation happen here, on the chunk, so nothing
    corrected is ever written back to the MS.

    Returns:
        ``(chi2, counts)``, each ``(nproducts, nant, nant)``.
    """
    resid, weight, flag, ant1, ant2, time, model = dask.compute(
        resid, weight, flag, ant1, ant2, time, model, scheduler="sync"
    )
    tidx = gains.time_index(time, grid_times) if gains is not None else None
    vis, wgt, flg = form(products, labels, resid, weight, flag, ant1, ant2, tidx=tidx, gains=gains, model=model)

    nant = np.maximum(ant1.max(), ant2.max()) + 1
    chi2 = np.zeros((len(products), nant, nant))
    counts = np.zeros((len(products), nant, nant))
    for k in range(len(products)):
        sl = slice(k, k + 1)
        chi2[k], counts[k] = _surfchisq_slice(vis[..., sl], wgt[..., sl], flg[..., sl], ant1, ant2)
        chi2_dof = np.full((nant, nant), np.nan)
        chi2_dof[counts[k] > 0] = chi2[k][counts[k] > 0] / counts[k][counts[k] > 0]
        makeplot(chi2_dof, fignames[k], subtitles[k])

    return chi2, counts
