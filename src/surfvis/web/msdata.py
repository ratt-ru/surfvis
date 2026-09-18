"""Read baseline waterfalls out of a Measurement Set via xarray-ms.

xarray-ms presents an MSv2 table as an MSv4 view: variables are shaped
``(time, baseline_id, frequency, polarization)``, which is already the
waterfall shape, so no row-to-grid pivot is needed.

MSv4 renames the columns you are used to -- ``DATA`` is exposed as
``VISIBILITY``, ``WEIGHT_SPECTRUM`` as ``WEIGHT``. Rather than hardcode that,
the names are resolved through the dataset's ``data_groups`` attribute, the
same mechanism pfb-imaging uses (see ``pfb_imaging/core/imager.py``). Columns
that are *not* part of the MSv2 standard -- ``RESIDUAL``, say -- are surfaced
verbatim as secondary variables, which is what makes the residual column
surfchi2 works on plottable.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import xarray

QUANTITIES = ("amp", "phase", "real", "imag")

# Matches pfb-imaging's default. SCAN_NUMBER in the schema means one partition
# per (field, spw, scan) -- exactly how surfchi2 groups, so a partition's time
# axis is the scan and no filtering is needed.
PARTITION_SCHEMA = ("FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER")


@dataclass(frozen=True)
class DataGroup:
    """The MSv4 variable names behind the MSv2 columns people ask for."""

    correlated_data: str
    weight: str
    flag: str

    def resolve(self, column: str) -> str:
        """Translate an MSv2 column name to the MSv4 variable that holds it.

        Non-standard columns (``RESIDUAL``, ``MODEL``, ...) pass through: they
        are not renamed by the MSv4 view, they simply appear as secondary
        variables.
        """
        return {
            "DATA": self.correlated_data,
            "CORRECTED_DATA": self.correlated_data,
            "WEIGHT_SPECTRUM": self.weight,
            "SIGMA_SPECTRUM": self.weight,
            "FLAG": self.flag,
        }.get(column, column)


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
    # Chunk rectangle in (time slot, channel) coordinates; None if not given.
    rect: tuple[int, int, int, int] | None


@lru_cache(maxsize=8)
def data_group(ms: str, group: str = "base") -> DataGroup:
    """Resolve the MSv4 data group for an MS.

    ``data_groups`` is attached by ``open_datatree`` (not ``open_dataset``), so
    this opens the tree once and caches the result -- it is identical across
    partitions.
    """
    from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES

    tree = xarray.open_datatree(ms, engine="xarray-ms:msv2", partition_schema=list(PARTITION_SCHEMA))
    for node in tree.children.values():
        if node.attrs.get("type") not in VISIBILITY_XDS_TYPES:
            continue
        groups = node.ds.attrs.get("data_groups", {})
        if group not in groups:
            raise KeyError(f"No data group {group!r} in {ms}. Have: {sorted(groups)}")
        spec = groups[group]
        return DataGroup(
            correlated_data=spec["correlated_data"],
            weight=spec["weight"],
            flag=spec.get("flag", "FLAG"),
        )
    raise ValueError(f"{ms} contains no visibility partitions")


@lru_cache(maxsize=8)
def _open(ms: str, field: int, spw: int, scan: int) -> xarray.Dataset:
    """Open one (field, spw, scan) partition. Cached -- opening is not free."""
    return xarray.open_dataset(
        ms,
        engine="xarray-ms:msv2",
        partition_schema=list(PARTITION_SCHEMA),
        partition_key=(("FIELD_ID", field), ("DATA_DESC_ID", spw), ("SCAN_NUMBER", scan)),
    )


def available_columns(ms: str, field: int, spw: int, scan: int) -> list[str]:
    """Data variables with a (time, baseline, freq, pol) shape, i.e. plottable."""
    ds = _open(ms, field, spw, scan)
    flag = data_group(ms).flag
    wanted = ("time", "baseline_id", "frequency", "polarization")
    # The flag variable has the right shape but is the overlay, not a quantity.
    return sorted(name for name, var in ds.data_vars.items() if var.dims == wanted and name != flag)


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
        scan: SCAN_NUMBER.
        name1: Name of the first antenna.
        name2: Name of the second antenna.
        polarization: Index into the polarization axis.
        column: Column to plot. MSv2 names are translated through the data
            group, so ``DATA`` finds ``VISIBILITY``; non-standard names such as
            ``RESIDUAL`` are used as given.
        quantity: One of ``amp``, ``phase``, ``real``, ``imag``.
        rect: Chunk bounds ``(t0, tf, chan0, chanf)`` to outline, in slots
            relative to the start of the scan.

    Returns:
        A :class:`Waterfall`.

    Raises:
        KeyError: if the column or baseline is not present.
        ValueError: if the quantity is not recognised.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}, got {quantity!r}")

    ds = _open(ms, field, spw, scan)
    group = data_group(ms)
    resolved = group.resolve(column)
    if resolved not in ds.data_vars:
        raise KeyError(
            f"Column {column!r} (MSv4 {resolved!r}) not in this Measurement Set. Have: {sorted(ds.data_vars)}"
        )

    k = baseline_index(ds, name1, name2)
    plane = ds[resolved].isel(baseline_id=k, polarization=polarization).values
    flags = ds[group.flag].isel(baseline_id=k, polarization=polarization).values.astype(bool)

    if quantity == "phase":
        values = np.angle(plane)
    elif quantity == "real":
        values = np.real(plane)
    elif quantity == "imag":
        values = np.imag(plane)
    else:
        values = np.abs(plane)
    values = np.asarray(values, dtype=float)

    # xarray-ms lays a regular (time, baseline) grid over the MS and fills gaps
    # with NaN. Those slots hold no data, so treat them as flagged rather than
    # letting them punch holes in the colour scale.
    flags |= ~np.isfinite(values)

    return Waterfall(
        values=values,
        flags=flags,
        antenna1=name1,
        antenna2=name2,
        scan=scan,
        column=resolved,
        quantity=quantity,
        polarization=polarization,
        rect=rect,
    )
