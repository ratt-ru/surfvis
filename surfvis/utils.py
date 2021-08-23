

import concurrent.futures as cf
import os
import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
import dask.array as da

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

def chisq(resid, weight, flag, ant1, ant2,
          tbin_idx, tbin_counts, fbin_idx, fbin_counts):

    nant = da.maximum(ant1.max(), ant2.max()).compute() + 1
    res = da.blockwise(_chisq, 'tfcpq2',
                       resid, 'tfc',
                       weight, 'tfc',
                       flag, 'tfc',
                       ant1, 't',
                       ant2, 't',
                       tbin_idx, 't',
                       tbin_counts, 't',
                       fbin_idx, 'f',
                       fbin_counts, 'f',
                       align_arrays=False,
                       dtype=np.float64,
                       adjust_chunks={'t': tbin_idx.chunks[0],
                                      'f': fbin_idx.chunks[0]},
                       new_axes={'p': nant, 'q': nant, '2': 2})
    return res


@njit(fastmath=True, nogil=True)
def _chisq(resid, weight, flag, ant1, ant2,
           tbin_idx, tbin_counts, fbin_idx, fbin_counts):
    nrow, nchan, ncorr = resid.shape

    nto = tbin_idx.size
    nfo = fbin_idx.size
    uant1 = np.unique(ant1)
    uant2 = np.unique(ant2)
    nant = np.maximum(uant1.max(), uant2.max()) + 1

    # init output array
    out = np.zeros((nto, nfo, ncorr, nant, nant, 2), dtype=np.float64)

    # account for chunk indexing
    tbin_idx2 = tbin_idx - tbin_idx.min()
    fbin_idx2 = fbin_idx - fbin_idx.min()
    for t in range(nto):
        rowi = tbin_idx2[t]
        rowf = tbin_idx2[t] + tbin_counts[t]
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


