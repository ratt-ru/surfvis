"""Read QuartiCal gain tables onto a Measurement Set's (time, channel) grid.

A QuartiCal gain term is a directory of zarr groups, one per (field, spw[,
scan]) partition -- ``gains.qc/NET/NET_0``, ``NET_1``... -- each an xarray
dataset with ``gains`` shaped ``(gain_time, gain_freq, antenna, direction,
correlation)`` and ``gain_flags`` without the last axis. They are opened with
plain xarray (``chunks=None``), so this module is usable from the dask-free web
layer.

Gains are applied on-grid only: each gain axis must either match the data's
times/frequencies or have length one (broadcast). That is the same contract
pfb-imaging's ``hci`` enforces; interpolating off-grid solutions is out of
scope, so write net gains at the data's resolution.
"""

import os
from functools import lru_cache

import numpy as np
import xarray as xr

from surfvis.utils.products import Gains

# Seconds. Data and gain times are both MJD seconds read from TIME columns.
TIME_TOLERANCE = 1e-3
FREQ_RTOL = 1e-6


def term_path(spec: str) -> str:
    """Accept either ``gains.qc/NET`` or pfb's ``gains.qc::NET`` spelling."""
    spec = str(spec).rstrip("/")
    if "::" in spec:
        root, term = spec.rsplit("::", 1)
        return f"{root.rstrip('/')}/{term}"
    return spec


def _datasets(path: str) -> tuple[xr.Dataset, ...]:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No gain term at {path}")
    # Keyed on the directory's mtime: a re-solve replaces the term's groups,
    # and a long-running server must not keep applying the old solutions.
    return _open_term(path, os.stat(path).st_mtime_ns)


@lru_cache(maxsize=8)
def _open_term(path: str, mtime: int) -> tuple[xr.Dataset, ...]:
    term = os.path.basename(path)
    names = sorted(
        (n for n in os.listdir(path) if n.startswith(f"{term}_") and n.removeprefix(f"{term}_").isdigit()),
        key=lambda n: int(n.removeprefix(f"{term}_")),
    )
    if not names:
        raise ValueError(f"{path} holds no {term}_<n> groups -- is this a QuartiCal gain term?")
    return tuple(xr.open_zarr(os.path.join(path, n), chunks=None, consolidated=False) for n in names)


def describe(spec: str) -> str:
    """One line about a gain term, or raise if it cannot be read."""
    dss = _datasets(term_path(spec))
    ncorr = dss[0].sizes["correlation"]
    kind = "full-Jones" if ncorr == 4 else "diagonal"
    return f"{len(dss)} partition(s), {kind}"


def load(
    spec: str,
    field: int,
    spw: int,
    scan: int,
    times: np.ndarray,
    freqs: np.ndarray,
    antenna_names: list[str],
) -> Gains:
    """Gains for one (field, spw, scan) on the given grid.

    Args:
        spec: Gain term, ``gains.qc/NET`` or ``gains.qc::NET``.
        field: FIELD_ID.
        spw: DATA_DESC_ID.
        scan: SCAN_NUMBER.
        times: The scan's unique TIME values (MJD seconds), ascending.
        freqs: Channel frequencies in Hz.
        antenna_names: MS antenna names, indexed by antenna id.

    Returns:
        :class:`Gains` with antennas in MS order. Antennas absent from the
        table get identity gains and are flagged.

    Raises:
        ValueError: if no or several partitions match, or the grids disagree.
    """
    ds = _select(spec, field, spw, scan, times)

    tsel = _align(ds.gain_time.values, np.asarray(times, dtype=float), "time", atol=TIME_TOLERANCE)
    fsel = _align(ds.gain_freq.values, np.asarray(freqs, dtype=float), "frequency", rtol=FREQ_RTOL)

    if ds.sizes.get("direction", 1) > 1:
        raise ValueError(f"{spec} has {ds.sizes['direction']} directions; only direction-independent gains apply")

    jones = np.asarray(ds.gains.values)[..., 0, :]
    flags = np.asarray(ds.gain_flags.values)[..., 0]
    # Stored as int8 by QuartiCal and as float by some writers; NaN is not a solution.
    flags = (flags != 0) | ~np.isfinite(flags)
    jones = jones[tsel][:, fsel]
    flags = flags[tsel][:, fsel]

    ncorr = jones.shape[-1]
    if ncorr == 1:
        jones = np.repeat(jones, 2, axis=-1)
    elif ncorr not in (2, 4):
        raise ValueError(f"{spec} has {ncorr} gain correlations; expected 1, 2 or 4")

    names = [str(n) for n in ds.antenna.values]
    position = {n: k for k, n in enumerate(names)}
    identity = np.array([1, 0, 0, 1] if jones.shape[-1] == 4 else [1, 1], dtype=jones.dtype)
    out_j = np.broadcast_to(identity, jones.shape[:2] + (len(antenna_names), jones.shape[-1])).copy()
    out_f = np.ones(jones.shape[:2] + (len(antenna_names),), dtype=bool)
    for a, name in enumerate(antenna_names):
        k = position.get(str(name))
        if k is not None:
            out_j[:, :, a] = jones[:, :, k]
            out_f[:, :, a] = flags[:, :, k]
    return Gains(jones=out_j, flags=out_f)


def _select(spec: str, field: int, spw: int, scan: int, times: np.ndarray) -> xr.Dataset:
    dss = [
        d
        for d in _datasets(term_path(spec))
        if int(d.attrs.get("FIELD_ID", -1)) == field and int(d.attrs.get("DATA_DESC_ID", -1)) == spw
    ]
    if any("SCAN_NUMBER" in d.attrs for d in dss):
        dss = [d for d in dss if int(d.attrs.get("SCAN_NUMBER", -1)) == scan]
    else:
        # Solved across scans: take the partition whose times cover this scan.
        lo, hi = float(np.min(times)), float(np.max(times))
        dss = [
            d
            for d in dss
            if d.gain_time.values.min() - TIME_TOLERANCE <= hi and d.gain_time.values.max() + TIME_TOLERANCE >= lo
        ]
    if len(dss) != 1:
        raise ValueError(
            f"{'No' if not dss else len(dss)} gain partitions in {spec} for field {field} spw {spw} scan {scan}"
        )
    return dss[0]


def _align(have: np.ndarray, want: np.ndarray, name: str, atol: float = 0.0, rtol: float = 0.0):
    """Indices into ``have`` matching each of ``want``, or a broadcast slice."""
    if have.size == 1:
        return slice(None)
    idx = np.clip(np.searchsorted(have, want), 0, have.size - 1)
    lower = np.clip(idx - 1, 0, have.size - 1)
    idx = np.where(np.abs(have[lower] - want) < np.abs(have[idx] - want), lower, idx)
    if not np.allclose(have[idx], want, atol=atol, rtol=rtol):
        raise ValueError(
            f"Gain {name} axis does not match the data ({have.size} solutions for {want.size} data points). "
            f"Gains apply on-grid only; write net gains at the data's {name} resolution."
        )
    return idx
