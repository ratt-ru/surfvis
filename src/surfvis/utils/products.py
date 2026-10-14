"""Corrected residuals and Stokes products, in plain numpy.

Shared by ``core/chi2.py`` and the web layer (which has no numba), so the two
cannot disagree about what "the corrected Stokes V residual" means.

Everything follows one formula per 2x2 visibility::

    r = inv(G_p) R inv(G_q)^H - M

where ``R`` is the input column, ``M`` an optional model and ``G`` optional
gains. ``R`` is assumed to live in the uncorrected frame, with ``weight`` its
inverse variances. Weights are propagated exactly, so for diagonal gains the
corrected residual has the same chi-squared as the uncorrected one -- the gains
change what the waterfall shows, not the statistic.

Stokes products are linear combinations ``s = sum_k a_k v_k`` of correlations,
whose inverse variance is ``1 / sum_k |a_k|^2 / w_k``. radiomesh's generated
``WEIGHT_*`` expressions are imaging weights (``w00 + w11`` for Stokes I), which
agree with that only when the correlation weights are equal, so they are not
used here.
"""

from dataclasses import dataclass

import numpy as np

# casacore Stokes enum -> label, for the correlation types surfvis understands.
CORR_TYPES = {5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
STOKES = ("I", "Q", "U", "V")

# Receptor pair (i, j) behind each correlation, i.e. its place in the 2x2 matrix.
_RECEPTORS = {
    "XX": (0, 0),
    "XY": (0, 1),
    "YX": (1, 0),
    "YY": (1, 1),
    "RR": (0, 0),
    "RL": (0, 1),
    "LR": (1, 0),
    "LL": (1, 1),
}

# Stokes parameter -> {correlation: coefficient}. Same conventions as radiomesh.
_STOKES = {
    "linear": {
        "I": {"XX": 0.5, "YY": 0.5},
        "Q": {"XX": 0.5, "YY": -0.5},
        "U": {"XY": 0.5, "YX": 0.5},
        "V": {"XY": -0.5j, "YX": 0.5j},
    },
    "circular": {
        "I": {"RR": 0.5, "LL": 0.5},
        "Q": {"RL": 0.5, "LR": 0.5},
        "U": {"RL": -0.5j, "LR": 0.5j},
        "V": {"RR": 0.5, "LL": -0.5},
    },
}


@dataclass(frozen=True)
class Gains:
    """Gains for one scan, on the data's (time, channel) grid.

    Attributes:
        jones: ``(ntime, nchan, nant, ngcorr)`` complex, ``ngcorr`` 2 (diagonal)
            or 4 (full 2x2, row-major). ``ntime`` and ``nchan`` may be 1, in
            which case the gains broadcast along that axis.
        flags: ``(ntime, nchan, nant)`` bool.
    """

    jones: np.ndarray
    flags: np.ndarray

    @property
    def full(self) -> bool:
        return self.jones.shape[-1] == 4

    def time_index(self, row_times: np.ndarray, grid_times: np.ndarray) -> np.ndarray:
        """Index into the time axis for each row, given the grid ``jones`` is on."""
        if self.jones.shape[0] == 1:
            return np.zeros(len(row_times), dtype=int)
        return np.searchsorted(grid_times, row_times)

    def subset(self, t0: int, tf: int, chan0: int, chanf: int) -> "Gains":
        """Slice to a chunk, leaving broadcast (length-1) axes alone."""
        tsel = slice(t0, tf) if self.jones.shape[0] > 1 else slice(None)
        fsel = slice(chan0, chanf) if self.jones.shape[1] > 1 else slice(None)
        return Gains(self.jones[tsel, fsel], self.flags[tsel, fsel])


def corr_labels(corr_types) -> list[str]:
    """Translate POLARIZATION::CORR_TYPE to labels such as ``XX``."""
    try:
        return [CORR_TYPES[int(c)] for c in corr_types]
    except KeyError as exc:
        raise ValueError(f"Unsupported correlation type {exc.args[0]} in {list(corr_types)}") from None


def feed_type(labels: list[str]) -> str:
    """``linear`` or ``circular``, from the correlation labels."""
    return "linear" if labels[0][0] in "XY" else "circular"


def constituents(product: str, labels: list[str]) -> list[str]:
    """The correlations a product is formed from."""
    if product in labels:
        return [product]
    if product in STOKES:
        return list(_STOKES[feed_type(labels)][product])
    raise ValueError(f"Unknown product {product!r}. Have correlations {labels} and Stokes {STOKES}")


def available(labels: list[str]) -> list[str]:
    """Products this set of correlations supports: itself plus whichever Stokes it can form."""
    stokes = [s for s in STOKES if set(_STOKES[feed_type(labels)][s]) <= set(labels)]
    return list(labels) + stokes


def needed(products: list[str], labels: list[str], full_jones: bool = False) -> list[str]:
    """Correlations to read to form ``products``, in MS order.

    Full-Jones gains mix all four correlations, so they need all of them.

    Raises:
        ValueError: if a product cannot be formed from ``labels``.
    """
    supported = available(labels)
    missing = [p for p in products if p not in supported]
    if missing:
        raise ValueError(f"Cannot form {missing} from correlations {labels}. Available: {supported}")
    if full_jones:
        if len(labels) != 4:
            raise ValueError(f"Full-Jones gains need all four correlations; this data has {labels}")
        return list(labels)
    want = {c for p in products for c in constituents(p, labels)}
    return [c for c in labels if c in want]


def form(
    products: list[str],
    labels: list[str],
    vis: np.ndarray,
    weight: np.ndarray,
    flag: np.ndarray,
    ant1: np.ndarray | None = None,
    ant2: np.ndarray | None = None,
    tidx: np.ndarray | None = None,
    gains: Gains | None = None,
    model: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correct, subtract the model, and form the requested products.

    Args:
        products: Output labels, correlations (``XX``) and/or Stokes (``I``).
        labels: Correlation labels of the last axis of ``vis``.
        vis: ``(nrow, nchan, ncorr)`` complex, the input column.
        weight: Same shape, inverse variances of ``vis``.
        flag: Same shape, bool.
        ant1: ``(nrow,)`` antenna indices into ``gains``. Needed with gains.
        ant2: ``(nrow,)``.
        tidx: ``(nrow,)`` time indices into ``gains``. Needed with gains.
        gains: Optional :class:`Gains` to correct by.
        model: Optional model, same shape and labels as ``vis``.

    Returns:
        ``(vis, weight, flag)``, each ``(nrow, nchan, nproducts)``. Weights
        of flagged or undetermined values are zero.
    """
    v = {c: vis[..., k].astype(np.complex128) for k, c in enumerate(labels)}
    w = {c: weight[..., k].astype(np.float64) for k, c in enumerate(labels)}
    f = {c: flag[..., k].astype(bool) for k, c in enumerate(labels)}

    if gains is not None:
        if ant1 is None or ant2 is None or tidx is None:
            raise ValueError("Applying gains needs ant1, ant2 and tidx")
        v, w, f = _correct(v, w, f, labels, gains, ant1, ant2, tidx)

    if model is not None:
        for k, c in enumerate(labels):
            v[c] = v[c] - model[..., k]

    out_v, out_w, out_f = [], [], []
    for product in products:
        if product in labels:
            out_v.append(v[product])
            out_w.append(w[product])
            out_f.append(f[product])
            continue
        coeffs = _STOKES[feed_type(labels)][product]
        s = sum(a * v[c] for c, a in coeffs.items())
        with np.errstate(divide="ignore"):
            var = sum(abs(a) ** 2 / w[c] for c, a in coeffs.items())
        bad = np.logical_or.reduce([f[c] for c in coeffs]) | ~np.isfinite(var)
        with np.errstate(divide="ignore"):
            sw = np.where(bad, 0.0, 1.0 / var)
        out_v.append(s)
        out_w.append(sw)
        out_f.append(bad)

    out_f = np.stack(out_f, axis=-1)
    out_w = np.where(out_f, 0.0, np.stack(out_w, axis=-1))
    return np.stack(out_v, axis=-1), out_w, out_f


def _correct(v, w, f, labels, gains, ant1, ant2, tidx):
    """Apply inv(G_p) . inv(G_q)^H to every correlation, propagating weights."""
    # (nrow, nchan_g, ngcorr); a length-1 channel axis broadcasts below.
    gp = gains.jones[tidx, :, ant1]
    gq = gains.jones[tidx, :, ant2]
    gflag = gains.flags[tidx, :, ant1] | gains.flags[tidx, :, ant2]

    if not gains.full:
        out_v, out_w, out_f = {}, {}, {}
        for c in labels:
            i, j = _RECEPTORS[c]
            scale = gp[..., i] * np.conj(gq[..., j])
            dead = scale == 0
            with np.errstate(divide="ignore", invalid="ignore"):
                out_v[c] = np.where(dead, 0, v[c] / np.where(dead, 1, scale))
            out_w[c] = w[c] * np.abs(scale) ** 2
            out_f[c] = f[c] | gflag | dead
        return out_v, out_w, out_f

    if len(labels) != 4:
        raise ValueError(f"Full-Jones gains need all four correlations; this data has {labels}")

    a, det_p = _inv2x2(gp.reshape(gp.shape[:-1] + (2, 2)))
    b, det_q = _inv2x2(gq.reshape(gq.shape[:-1] + (2, 2)))
    shape = np.broadcast_shapes(v[labels[0]].shape, a.shape[:-2])

    vm = np.zeros(shape + (2, 2), dtype=np.complex128)
    var = np.zeros(shape + (2, 2), dtype=np.float64)
    anyflag = np.zeros(shape, dtype=bool)
    for c in labels:
        i, j = _RECEPTORS[c]
        vm[..., i, j] = v[c]
        with np.errstate(divide="ignore"):
            var[..., i, j] = 1.0 / w[c]
        anyflag |= f[c]

    corrected = a @ vm @ np.conj(np.swapaxes(b, -1, -2))
    # var'_ij = sum_kl |A_ik|^2 var_kl |B_jl|^2
    with np.errstate(invalid="ignore"):
        cvar = np.abs(a) ** 2 @ var @ np.swapaxes(np.abs(b) ** 2, -1, -2)
    dead = (det_p == 0) | (det_q == 0)
    bad = anyflag | gflag | dead

    out_v, out_w, out_f = {}, {}, {}
    for c in labels:
        i, j = _RECEPTORS[c]
        cv = cvar[..., i, j]
        undetermined = bad | ~np.isfinite(cv) | (cv == 0)
        out_v[c] = np.where(dead, 0, corrected[..., i, j])
        with np.errstate(divide="ignore"):
            out_w[c] = np.where(undetermined, 0.0, 1.0 / np.where(undetermined, 1.0, cv))
        out_f[c] = undetermined
    return out_v, out_w, out_f


def _inv2x2(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched 2x2 inverse; singular entries come back as zeros with det 0."""
    det = m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]
    safe = np.where(det == 0, 1, det)
    inv = np.empty_like(m)
    inv[..., 0, 0] = m[..., 1, 1]
    inv[..., 0, 1] = -m[..., 0, 1]
    inv[..., 1, 0] = -m[..., 1, 0]
    inv[..., 1, 1] = m[..., 0, 0]
    inv = inv / safe[..., None, None]
    inv[det == 0] = 0
    return inv, det
