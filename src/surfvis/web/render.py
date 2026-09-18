"""Colour mapping for the heatmap grid, and matplotlib PNGs for the rest."""

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.colors import LogNorm, Normalize, to_hex  # noqa: E402

CMAP = "inferno"
# Colour for antenna pairs with no unflagged data.
NAN_COLOUR = "#1b1b1b"


SCALES = ("log", "robust", "full")


def make_norm(values: np.ndarray, scale: str = "log"):
    """Build a colour norm for a chi-squared/dof matrix.

    chi-squared/dof routinely spans orders of magnitude -- a handful of bad
    baselines against a sea of ~1 -- so a linear scale renders everything but
    the worst cell black, which defeats the purpose of looking at the grid.
    Log is therefore the default.

    Args:
        values: The matrix, NaN where there was no unflagged data.
        scale: ``log``, ``robust`` (2-98th percentile, linear) or ``full``.

    Returns:
        ``(norm, vmin, vmax)``.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0), 0.0, 1.0

    if scale == "log":
        positive = finite[finite > 0]
        if positive.size == 0:
            return Normalize(vmin=0.0, vmax=1.0), 0.0, 1.0
        vmin, vmax = float(positive.min()), float(positive.max())
        if vmax <= vmin:
            vmax = vmin * 10.0
        return LogNorm(vmin=vmin, vmax=vmax, clip=True), vmin, vmax

    if scale == "robust":
        lo, hi = np.percentile(finite, [2.0, 98.0])
    else:
        lo, hi = finite.min(), finite.max()
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return Normalize(vmin=lo, vmax=hi, clip=True), lo, hi


def cell_colours(matrix: np.ndarray, norm) -> list[list[str]]:
    """Map the chi-squared/dof matrix to per-cell hex colours."""
    cmap = colormaps[CMAP]
    out = []
    for row in matrix:
        colours = []
        for value in row:
            if np.isnan(value) or (isinstance(norm, LogNorm) and value <= 0):
                colours.append(NAN_COLOUR)
            else:
                colours.append(to_hex(cmap(norm(value))))
        out.append(colours)
    return out


def histogram_png(values: np.ndarray, norm, title: str = "") -> bytes:
    """Histogram of chi-squared/dof over all antenna pairs, with a colourbar.

    This is how you judge whether a chunk has outliers at all, rather than a
    colour scale stretched over noise -- so it stays next to the grid. Bins
    follow the colour scale: log bins for a log norm, or the distribution
    collapses into a single bar.
    """
    log = isinstance(norm, LogNorm)
    fig = plt.figure(figsize=(3.4, 3.8))
    ax = fig.add_axes((0.17, 0.30, 0.78, 0.60))
    finite = values[np.isfinite(values)]
    if log:
        finite = finite[finite > 0]

    if finite.size:
        if log:
            bins = np.logspace(np.log10(norm.vmin), np.log10(norm.vmax), 36)
            ax.set_xscale("log")
        else:
            bins = 36
        ax.hist(finite, bins=bins, histtype="stepfilled", alpha=0.65, color="#d95f0e")
        ax.axvline(1.0, color="#4575b4", lw=1.0, ls="--", label=r"$\chi^2/dof=1$")
        ax.legend(fontsize=6, frameon=False, loc="upper right")
    else:
        ax.text(0.5, 0.5, "all flagged", ha="center", va="center", fontsize=9, transform=ax.transAxes)

    ax.set_yticks([])
    ax.tick_params(axis="x", length=2, width=0.8, labelsize=7)
    ax.set_xlabel(r"$\chi^2$/dof", fontsize=8)
    if title:
        ax.set_title(title, fontsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    cax = fig.add_axes((0.17, 0.13, 0.78, 0.045))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cax.tick_params(length=2, width=0.8, labelsize=6)

    return _png(fig)


def waterfall_png(wf, cmap: str = "viridis", show_flags: bool = True) -> bytes:
    """Render a baseline waterfall, outlining the chunk that was clicked."""
    fig = plt.figure(figsize=(9.5, 4.6))
    ax = fig.add_subplot(111)

    values = wf.values
    if values.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return _png(fig)

    unflagged = values[~wf.flags] if wf.flags.shape == values.shape else values
    unflagged = unflagged[np.isfinite(unflagged)]
    if wf.quantity == "phase" or unflagged.size == 0:
        vmin = vmax = None
    else:
        vmin = float(np.nanmin(unflagged))
        vmax = float(np.nanpercentile(unflagged, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1.0

    im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    if show_flags and wf.flags.shape == values.shape and wf.flags.any():
        overlay = np.zeros(values.shape + (4,))
        overlay[..., 3] = wf.flags * 0.55
        ax.imshow(overlay, aspect="auto", origin="lower", interpolation="nearest")

    if wf.rect is not None:
        t0, tf, chan0, chanf = wf.rect
        ax.add_patch(
            plt.Rectangle(
                (chan0 - 0.5, t0 - 0.5),
                max(chanf - chan0, 1),
                max(tf - t0, 1),
                fill=False,
                edgecolor="#39ff14",
                lw=1.8,
            )
        )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Time slot (within scan)")
    ax.set_title(
        f"{wf.antenna1}-{wf.antenna2}   scan {wf.scan}   {wf.column}   {wf.quantity}   pol {wf.polarization}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    return _png(fig)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()
