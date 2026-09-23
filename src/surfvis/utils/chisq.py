"""Numba/dask kernels for per-baseline chi-squared computation and flagging."""

import dask.array as da
import numpy as np
from numba import njit


def surfchisq(resid, weight, flag, ant1, ant2, rbin_idx, rbin_counts, fbin_idx, fbin_counts):
    """Chi-squared per (time bin, freq bin, corr, antenna pair) as a dask array.

    The trailing axis of length 2 holds ``(chi2, count)``.
    """
    nant = da.maximum(ant1.max(), ant2.max()).compute() + 1
    res = da.blockwise(
        _surfchisq,
        "tfcpq2",
        resid,
        "tfc",
        weight,
        "tfc",
        flag,
        "tfc",
        ant1,
        "t",
        ant2,
        "t",
        rbin_idx,
        "t",
        rbin_counts,
        "t",
        fbin_idx,
        "f",
        fbin_counts,
        "f",
        align_arrays=False,
        dtype=np.float64,
        adjust_chunks={"t": rbin_idx.chunks[0], "f": fbin_idx.chunks[0]},
        new_axes={"p": nant, "q": nant, "2": 2},
    )
    return res


@njit(nogil=True)
def _surfchisq(resid, weight, flag, ant1, ant2, rbin_idx, rbin_counts, fbin_idx, fbin_counts):
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


@njit(nogil=True)
def _surfchisq_slice(resid, weight, flag, ant1, ant2):
    """Chi-squared and counts per antenna pair for a single (time, freq, corr) slice."""
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


def flagchisq(resid, weight, flag, ant1, ant2, use_corrs=(), flag_above=5, respect_ants=()):
    """Flag visibilities whose per-visibility chi-squared exceeds ``flag_above``."""
    res = da.blockwise(
        _flagchisq,
        "rfc",
        resid,
        "rfc",
        weight,
        "rfc",
        flag,
        "rfc",
        ant1,
        "r",
        ant2,
        "r",
        use_corrs,
        None,
        flag_above,
        None,
        respect_ants,
        None,
        dtype=bool,
    )
    return res


@njit(fastmath=True, nogil=True)
def _flagchisq(resid, weight, flag, ant1, ant2, use_corrs, flag_above, respect_ants):
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
