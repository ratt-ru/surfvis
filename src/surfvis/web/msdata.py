"""Read baseline waterfalls out of a Measurement Set via xarray-ms.

xarray-ms presents an MSv2 table as ``(time, baseline_id, frequency,
polarization)``, which is already the waterfall shape -- no row-to-grid pivot
is needed. Columns that are not part of the MSv2 standard (``RESIDUAL``, say)
are surfaced automatically as secondary variables, which is how the residual
column surfchi2 works on becomes plottable.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import xarray

QUANTITIES = ("amp", "phase", "real", "imag")


@dataclass(frozen=True)
class Waterfall:
    """A baseline's time/frequency plane, plus where the clicked chunk sits."""

    values: np.ndarray
    flags: np.ndarray
    antenna1: str
    antenna2: str
    scan: int
    column: str
    quantity: str
    polarization: int
    # Chunk rectangle in (time slot, channel) coordinates; None if out of range.
    rect: tuple[int, int, int, int] | None


@lru_cache(maxsize=4)
def _open(ms: str, field: int, spw: int) -> xarray.Dataset:
    """Open one (field, spw) partition. Cached -- opening is not free."""
    # FIELD_ID is not a partition column by default (the default schema is
    # OBSERVATION_ID/PROCESSOR_ID/DATA_DESC_ID/OBS_MODE_ID), so ask for it
    # explicitly -- surfchi2 groups by FIELD_ID and we need to match that.
    return xarray.open_dataset(
        ms,
        engine="xarray-ms:msv2",
        partition_schema=["DATA_DESC_ID", "FIELD_ID"],
        partition_key=(("DATA_DESC_ID", spw), ("FIELD_ID", field)),
    )


def available_columns(ms: str, field: int, spw: int) -> list[str]:
    """Data variables with a (time, baseline, freq, pol) shape, i.e. plottable."""
    ds = _open(ms, field, spw)
    wanted = ("time", "baseline_id", "frequency", "polarization")
    # FLAG has the right shape but is the overlay, not a quantity to plot.
    return sorted(name for name, var in ds.data_vars.items() if var.dims == wanted and name != "FLAG")


def baseline_index(ds: xarray.Dataset, name1: str, name2: str) -> int:
    """Map an antenna-name pair to a baseline_id, in either order."""
    a1 = ds.baseline_antenna1_name.values
    a2 = ds.baseline_antenna2_name.values
    hit = np.where((a1 == name1) & (a2 == name2))[0]
    if hit.size == 0:
        hit = np.where((a1 == name2) & (a2 == name1))[0]
    if hit.size == 0:
        raise KeyError(f"No baseline {name1}-{name2} in this partition")
    return int(hit[0])


def waterfall(
    ms: str,
    field: int,
    spw: int,
    scan: int,
    name1: str,
    name2: str,
    polarization: int,
    column: str,
    quantity: str = "amp",
    rect: tuple[int, int, int, int] | None = None,
) -> Waterfall:
    """Read one baseline's waterfall for a whole scan.

    Args:
        ms: Path to the Measurement Set.
        field: FIELD_ID.
        spw: DATA_DESC_ID.
        scan: SCAN_NUMBER to restrict to.
        name1: Name of the first antenna.
        name2: Name of the second antenna.
        polarization: Index into the polarization axis.
        column: Data variable to plot, e.g. ``RESIDUAL`` or ``VISIBILITY``.
        quantity: One of ``amp``, ``phase``, ``real``, ``imag``.
        rect: Chunk bounds ``(t0, tf, chan0, chanf)`` to outline, in slots
            relative to the start of the scan.

    Returns:
        A :class:`Waterfall`.

    Raises:
        KeyError: if the column or baseline is not present.
        ValueError: if the quantity is not recognised or the scan is empty.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}, got {quantity!r}")

    ds = _open(ms, field, spw)
    if column not in ds.data_vars:
        raise KeyError(f"Column {column!r} not in this Measurement Set. Have: {sorted(ds.data_vars)}")

    mask = ds.scan_name.values.astype(str) == str(scan)
    if not mask.any():
        raise ValueError(f"Scan {scan} has no rows in field {field}, spw {spw}")
    sel = ds.isel(time=np.flatnonzero(mask))

    k = baseline_index(ds, name1, name2)
    plane = sel[column].isel(baseline_id=k, polarization=polarization).values
    flags = sel["FLAG"].isel(baseline_id=k, polarization=polarization).values.astype(bool)

    if quantity == "phase":
        values = np.angle(plane)
    elif quantity == "real":
        values = np.real(plane)
    elif quantity == "imag":
        values = np.imag(plane)
    else:
        values = np.abs(plane)

    return Waterfall(
        values=np.asarray(values, dtype=float),
        flags=flags,
        antenna1=name1,
        antenna2=name2,
        scan=scan,
        column=column,
        quantity=quantity,
        polarization=polarization,
        rect=rect,
    )
