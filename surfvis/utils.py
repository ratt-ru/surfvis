

import concurrent.futures as cf
import os
import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
import dask
import dask.array as da
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import hist

def surf(p, q, gp, gq, data, resid, weight, flag, basename):
    """
    This here is to get the stack trace when an exception is thrown
    """
    try:
        return _surf(p, q, gp, gq, data, resid, weight, flag, basename)
    except Exception as e:
        traceback_str = traceback.format_exc(e)
        raise StandardError("Error occurred. Original traceback "
                            "is\n%s\n" % traceback_str)

def surfchisq(resid, weight, flag, ant1, ant2,
              rbin_idx, rbin_counts, fbin_idx, fbin_counts):

    nant = da.maximum(ant1.max(), ant2.max()).compute() + 1
    res = da.blockwise(_surfchisq, 'tfcpq2',
                       resid, 'tfc',
                       weight, 'tfc',
                       flag, 'tfc',
                       ant1, 't',
                       ant2, 't',
                       rbin_idx, 't',
                       rbin_counts, 't',
                       fbin_idx, 'f',
                       fbin_counts, 'f',
                       align_arrays=False,
                       dtype=np.float64,
                       adjust_chunks={'t': rbin_idx.chunks[0],
                                      'f': fbin_idx.chunks[0]},
                       new_axes={'p': nant, 'q': nant, '2': 2})
    return res


@njit(nogil=True)
def _surfchisq(resid, weight, flag, ant1, ant2,
               rbin_idx, rbin_counts, fbin_idx, fbin_counts):
    nrow, nchan, ncorr = resid.shape

    nto = rbin_idx.size
    nfo = fbin_idx.size
    uant1 = np.unique(ant1)
    uant2 = np.unique(ant2)
    nant = np.maximum(uant1.max(), uant2.max()) + 1

    # init output array
    out = np.zeros((nto, nfo, ncorr, nant, nant, 2), dtype=np.float64)

    # account for chunk indexing
    rbin_idx2 = rbin_idx - rbin_idx.min()
    fbin_idx2 = fbin_idx - fbin_idx.min()
    for t in range(nto):
        rowi = rbin_idx2[t]
        rowf = rbin_idx2[t] + rbin_counts[t]
        residr = resid[rowi:rowf]
        weightr = weight[rowi:rowf]
        flagr = flag[rowi:rowf]
        ant1r = ant1[rowi:rowf]
        ant2r = ant2[rowi:rowf]
        for f in range(nfo):
            chani = fbin_idx2[f]
            chanf = fbin_idx2[f] + fbin_counts[f]
            residrf = residr[:, chani:chanf]
            weightrf = weightr[:, chani:chanf]
            flagrf = flagr[:, chani:chanf]
            for c in range(ncorr):
                residrfc = residrf[:, :, c]
                weightrfc = weightrf[:, :, c]
                flagrfc = flagrf[:, :, c]
                for p in uant1:
                    Ip = ant1r == p
                    for q in uant2:
                        Iq = ant2r == q
                        Ipq = Ip & Iq
                        R = residrfc[Ipq].ravel()
                        W = weightrfc[Ipq].ravel()
                        F = flagrfc[Ipq].ravel()
                        for i in range(R.size):
                            if not F[i] and p != q:
                                out[t, f, c, p, q, 0] += (np.conj(R[i]) * W[i] * R[i]).real
                                out[t, f, c, p, q, 1] += 1.0
                        out[t, f, c, q, p] = out[t, f, c, p, q]

    return out


def surfchisq_plot(resid, weight, flag, ant1, ant2, field, spw, scan, figname, subt):
    resid, weight, flag, ant1, ant2 = dask.compute(
        resid, weight, flag, ant1, ant2, scheduler='sync'
    )
    chi2, counts = _surfchisq_slice(resid, weight, flag, ant1, ant2)
    nant = np.maximum(ant1.max(), ant2.max()) + 1
    chi2_dof = np.zeros((nant, nant), dtype=float)
    chi2_dof[counts>0] = chi2[counts>0]/counts[counts>0]
    chi2_dof[counts<=0] = np.nan

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(chi2_dof, cmap='inferno')
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
    x = chi2_dof[counts>0]
    if x.any():
        hist(x, bins='scott', ax=rax, histtype='stepfilled',
            alpha=0.5, density=False)
        rax.set_yticks([])
        rax.tick_params(axis='y', which='both',
                        bottom=False, top=False,
                        labelbottom=False)
        rax.tick_params(axis='x', which='both',
                        length=1, width=1, labelsize=8)

    fig.suptitle(subt, fontsize=20)
    plt.savefig(figname, dpi=250)
    plt.close(fig)

    return field, spw, scan, chi2, counts


@njit(nogil=True)
def _surfchisq_slice(resid, weight, flag, ant1, ant2):
    nrow, nchan, ncorr = resid.shape
    uant1 = np.unique(ant1)
    uant2 = np.unique(ant2)
    nant = np.maximum(uant1.max(), uant2.max()) + 1

    # init output array
    chi2 = np.zeros((nant, nant), dtype=np.float64)
    counts = np.zeros((nant, nant), dtype=np.float64)

    for p in uant1:
        Ip = ant1 == p
        for q in uant2:
            Iq = ant2 == q
            Ipq = Ip & Iq
            R = resid[Ipq].ravel()
            W = weight[Ipq].ravel()
            F = flag[Ipq].ravel()
            for i in range(R.size):
                if not F[i] and p != q:
                    chi2[p, q] += (np.conj(R[i]) * W[i] * R[i]).real
                    counts[p, q] += 1.0
            chi2[q, p] = chi2[p, q]
            counts[q, p] = counts[p, q]

    return chi2, counts


def flagchisq(resid, weight, flag, ant1, ant2,
              use_corrs=(), flag_above=5,
              respect_ants=()):

    # import pdb; pdb.set_trace()

    res = da.blockwise(_flagchisq, 'rfc',
                       resid, 'rfc',
                       weight, 'rfc',
                       flag, 'rfc',
                       ant1, 'r',
                       ant2, 'r',
                       use_corrs, None,
                       flag_above, None,
                       respect_ants, None,
                       dtype=bool)
    return res


@njit(fastmath=True, nogil=True)
def _flagchisq(resid, weight, flag, ant1, ant2,
               use_corrs, flag_above, respect_ants):
    nrow, nchan, ncorr = resid.shape
    for r in range(nrow):
        if ant1[r] in respect_ants or ant2[r] in respect_ants:
            continue
        for f in range(nchan):
            for c in use_corrs:
                res = resid[r, f, c]
                w = weight[r, f, c]
                chi2 = (np.conj(res) * w * res).real
                if chi2 > flag_above or chi2 == 0:
                    flag[r, f, :] = True
    return flag
