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

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
import xarray

from surfvis.utils import gains as gains_mod
from surfvis.utils import products
from surfvis.utils.msv4 import ignore_xarray_ms_warnings

ignore_xarray_ms_warnings()

QUANTITIES = ("amp", "phase", "real", "imag")

# Matches pfb-imaging's default. SCAN_NUMBER in the schema means one partition
# per (field, spw, scan) -- exactly how surfchi2 groups, so a partition's time
# axis is the scan and no filtering is needed.
PARTITION_SCHEMA = ("FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER")

# xarray-ms decodes TIME (MJD seconds) to unix seconds; gain tables and flag
# versions are keyed on the raw MJD values.
MJD_UNIX_OFFSET = 3506716800.0

# Last-seen on-disk signature per MS, so a rewrite invalidates the caches.
_FINGERPRINTS: dict[str, tuple] = {}


@dataclass(frozen=True)
class DataGroup:
    """The MSv4 variable names behind the MSv2 columns people ask for."""

    correlated_data: str
    weight: str
    flag: str

    def resolve(self, column: str) -> str:
        """Translate an MSv2 column name to the MSv4 variable that holds it.

        Only the columns the data group actually maps are translated. Everything
        else passes through: non-standard columns (``RESIDUAL``) and the
        standard ones xarray-ms keeps as secondary variables
        (``CORRECTED_DATA``, ``MODEL_DATA``) appear under their own names.
        Mapping ``CORRECTED_DATA`` to the group's ``VISIBILITY`` would silently
        plot ``DATA`` instead.
        """
        return {
            "DATA": self.correlated_data,
            "WEIGHT_SPECTRUM": self.weight,
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
    product: str
    # Chunk rectangle in (time slot, channel) coordinates; None if not given.
    rect: tuple[int, int, int, int] | None
    # Metres, from ANTENNA::POSITION; None if it could not be read.
    length: float | None = None
    # Unix seconds of each time slot; None if not known.
    times: np.ndarray | None = None
    # The complex plane the values were taken from, and the channel
    # frequencies in Hz; None where not known.
    plane: np.ndarray | None = None
    freqs: np.ndarray | None = None
    # Colour limits per quantity, filled in by the canvas route on a cached plane.
    limits: dict = field(default_factory=dict, compare=False, repr=False)


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
def _open_partition(ms: str, field: int, spw: int, scan: int, fingerprint: tuple) -> xarray.Dataset:
    """Open one (field, spw, scan) partition. Cached -- opening is not free.

    ``fingerprint`` takes no part in the work; it is in the signature so that a
    Measurement Set rewritten on disk gets a different cache key.
    """
    return xarray.open_dataset(
        ms,
        engine="xarray-ms:msv2",
        partition_schema=list(PARTITION_SCHEMA),
        partition_key=(("FIELD_ID", field), ("DATA_DESC_ID", spw), ("SCAN_NUMBER", scan)),
    )


def _fingerprint(ms: str) -> tuple:
    """A cheap signature of the MS's column storage on disk.

    The ``table.f*`` files are the storage managers' data; writing FLAG bumps
    the mtime and possibly the size of the ones holding it. ``table.lock`` is
    deliberately not included -- it is touched by reads as well, so it would
    invalidate the caches on every request.
    """
    entries = []
    try:
        with os.scandir(ms) as listing:
            for entry in listing:
                if entry.name.startswith("table.f") and entry.is_file():
                    stat = entry.stat()
                    entries.append((entry.name, stat.st_mtime_ns, stat.st_size))
    except OSError:
        # Unreadable or gone; let the open itself produce the real error.
        return ()
    return tuple(sorted(entries))


def _release_cached_tables() -> None:
    """Drop the tables xarray-ms holds open, so the next read goes to disk.

    xarray-ms keeps its tables in a process-global multiton cache whose entries
    expire only after a period of *inactivity*, and a real MS stores FLAG in a
    TiledShapeStMan whose tile cache belongs to the open table object. Clearing
    our own caches is therefore not enough: reopening the dataset hands back
    the same underlying table, still serving the flags it read first.

    There is no public API for this, so the private cache is cleared under the
    lock the multiton itself uses. If the internals move, freshness is lost but
    nothing breaks, which is why this is defensive rather than strict.
    """
    try:
        from rarg_python_patterns.multiton import Multiton
    except ImportError:  # pragma: no cover - xarray-ms always brings it
        return
    cache = getattr(Multiton, "_INSTANCE_CACHE", None)
    lock = getattr(Multiton, "_INSTANCE_LOCK", None)
    if cache is None or lock is None:  # pragma: no cover - upstream moved
        return
    with lock:
        cache.clear()
        heap = getattr(Multiton, "_EXPIRY_HEAP", None)
        if heap is not None:
            heap.clear()


def _open(ms: str, field: int, spw: int, scan: int) -> xarray.Dataset:
    """Open one partition, re-reading it if the MS changed on disk.

    A browser is long-running and the Measurement Set under it is not static --
    a flagging run rewrites FLAG while the page is open. Without this check the
    waterfalls keep showing the flags the process read first, which looks like
    a plotting bug and is impossible to tell apart from one.
    """
    fingerprint = _fingerprint(ms)
    if _FINGERPRINTS.get(ms) != fingerprint:
        _open_partition.cache_clear()
        data_group.cache_clear()
        _release_cached_tables()
        _FINGERPRINTS[ms] = fingerprint
    return _open_partition(ms, field, spw, scan, fingerprint)


def available_columns(ms: str, field: int, spw: int, scan: int, complex_only: bool = False) -> list[str]:
    """Data variables with a (time, baseline, freq, pol) shape, i.e. plottable.

    ``complex_only`` restricts to visibility-like columns -- what makes sense
    as a model to subtract, which ``WEIGHT`` does not.
    """
    ds = _open(ms, field, spw, scan)
    flag = data_group(ms).flag
    wanted = ("time", "baseline_id", "frequency", "polarization")
    # The flag variable has the right shape but is the overlay, not a quantity.
    return sorted(
        name
        for name, var in ds.data_vars.items()
        if var.dims == wanted and name != flag and (not complex_only or np.iscomplexobj(var))
    )


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


@lru_cache(maxsize=8)
def field_names(ms: str) -> dict[int, str]:
    """``FIELD_ID`` to ``FIELD::NAME``, or an empty dict if the MS cannot be read."""
    try:
        import arcae

        fields = arcae.table(f"{ms}::FIELD")
        try:
            return {k: str(n) for k, n in enumerate(fields.getcol("NAME"))}
        finally:
            fields.close()
    except Exception:
        return {}


@lru_cache(maxsize=8)
def _antenna_positions(ms: str) -> dict[str, np.ndarray]:
    import arcae

    ant = arcae.table(f"{ms}::ANTENNA")
    try:
        names, positions = ant.getcol("NAME"), np.asarray(ant.getcol("POSITION"), dtype=float)
    finally:
        ant.close()
    return {str(n): positions[k] for k, n in enumerate(names)}


def antenna_positions(ms: str, names: list[str]) -> list[list[float]] | None:
    """ITRF positions for ``names`` in order, or None if the ANTENNA table cannot be read."""
    try:
        positions = _antenna_positions(ms)
        return [positions[n].tolist() for n in names]
    except Exception:
        return None


def baseline_length(ms: str, name1: str, name2: str) -> float:
    """Distance between two antennas in metres, from ``ANTENNA::POSITION`` (ITRF)."""
    positions = _antenna_positions(ms)
    return float(np.linalg.norm(positions[name1] - positions[name2]))


def labels(ms: str, field: int, spw: int, scan: int) -> list[str]:
    """Correlation labels of a partition, such as ``["XX", "XY", "YX", "YY"]``."""
    return [str(p) for p in _open(ms, field, spw, scan).polarization.values]


def available_products(ms: str, field: int, spw: int, scan: int) -> list[str]:
    """Correlations plus whichever Stokes parameters they can form."""
    return products.available(labels(ms, field, spw, scan))


def flag_versions(ms: str) -> list[str]:
    """Names of the CASA flag versions saved next to ``ms``, oldest first.

    CASA keeps them in ``<ms>.flagversions/``: a ``FLAG_VERSION_LIST`` of
    ``name : comment`` lines and one ``flags.<name>`` table per version,
    row-aligned with the main table.
    """
    root = f"{ms.rstrip('/')}.flagversions"
    try:
        with open(os.path.join(root, "FLAG_VERSION_LIST")) as listing:
            names = [line.split(" : ", 1)[0].strip() for line in listing if line.strip()]
    except OSError:
        return []
    return [n for n in names if os.path.isdir(os.path.join(root, f"flags.{n}"))]


def ms_times(ds: xarray.Dataset) -> np.ndarray:
    """The partition's time grid as raw MS TIME values (MJD seconds)."""
    times = np.asarray(ds.time.values, dtype=float)
    if ds.time.attrs.get("format") == "unix":
        times = times + MJD_UNIX_OFFSET
    return times


def baseline_rows(ms: str, field: int, spw: int, scan: int, name1: str, name2: str) -> tuple[np.ndarray, np.ndarray]:
    """Main-table row numbers and TIME values (MJD seconds) of one baseline in one partition.

    Either antenna order matches. Read with an arcae TaQL query, so the web
    layer needs no python-casacore. The row array is a writable copy: arcae
    rejects read-only buffers as row indices.
    """
    import arcae
    from arcae.lib.arrow_tables import Table

    ant = arcae.table(f"{ms}::ANTENNA")
    try:
        ids = {str(n): k for k, n in enumerate(ant.getcol("NAME"))}
    finally:
        ant.close()
    a, b = ids[name1], ids[name2]

    sel = Table.from_taql(
        f"SELECT ROWID() AS ROW, TIME FROM '{ms}' WHERE FIELD_ID=={field} && DATA_DESC_ID=={spw} && "
        f"SCAN_NUMBER=={scan} && ((ANTENNA1=={a} && ANTENNA2=={b}) || (ANTENNA1=={b} && ANTENNA2=={a}))"
    )
    try:
        if not sel.nrow():
            return np.zeros(0, np.int64), np.zeros(0)
        return np.array(sel.getcol("ROW"), dtype=np.int64), np.asarray(sel.getcol("TIME"), dtype=float)
    finally:
        sel.close()


def slot_index(grid: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each TIME's slot on the partition's time grid, and whether it landed on one."""
    if grid.size == 0:
        return np.zeros(times.shape, int), np.zeros(times.shape, bool)
    idx = np.clip(np.searchsorted(grid, times), 0, grid.size - 1)
    # searchsorted gives the insertion point; the nearer neighbour may be below.
    below = np.clip(idx - 1, 0, grid.size - 1)
    idx = np.where(np.abs(grid[below] - times) < np.abs(grid[idx] - times), below, idx)
    return idx, np.abs(grid[idx] - times) < gains_mod.TIME_TOLERANCE


def invalidate(ms: str) -> None:
    """Forget everything cached about ``ms``, so the next read goes to disk.

    For writes made by this process: ``_open`` notices a rewritten MS by its
    storage files' mtimes, but a write and a read inside one mtime tick would
    slip past that.
    """
    _FINGERPRINTS.pop(ms, None)
    _open_partition.cache_clear()
    _cached_plane.cache_clear()
    data_group.cache_clear()
    _release_cached_tables()


def _version_flags(
    ms: str,
    version: str,
    field: int,
    spw: int,
    scan: int,
    name1: str,
    name2: str,
    grid: np.ndarray,
    corr_idx: list[int],
) -> np.ndarray:
    """One baseline's flags from a saved flag version, on the partition's time grid.

    The version tables are row-aligned with the main table, so the baseline's
    rows are found in the main table and the same row numbers read from the
    version. Rows the grid has but the MS does not are flagged, as they are
    for the live FLAG column. Read with arcae, the same library xarray-ms uses,
    so the web layer needs no python-casacore.
    """
    import arcae

    if version not in flag_versions(ms):
        raise KeyError(f"No flag version {version!r} for {ms}. Have: {flag_versions(ms)}")

    rows, times = baseline_rows(ms, field, spw, scan, name1, name2)

    nchan = None
    if rows.size:
        vt = arcae.table(f"{ms.rstrip('/')}.flagversions/flags.{version}")
        try:
            # arcae returns boolean columns as uint8.
            flags = np.array(vt.getcol("FLAG", index=(rows,))[..., corr_idx], dtype=bool)
            flags |= np.asarray(vt.getcol("FLAG_ROW", index=(rows,)), dtype=bool)[:, None, None]
            nchan = flags.shape[1]
        finally:
            vt.close()

    if nchan is None:
        raise KeyError(f"No rows for {name1}-{name2} in scan {scan}")
    out = np.ones((grid.size, nchan, len(corr_idx)), dtype=bool)
    idx, hit = slot_index(grid, times)
    out[idx[hit]] = flags[hit]
    return out


def check(
    ms: str,
    field: int,
    spw: int,
    scan: int,
    product: str,
    column: str,
    model: str = "",
    gtable: str = "",
    flagversion: str = "",
) -> list[str]:
    """Everything that can be wrong with a waterfall request, short of reading it.

    Returns:
        Human-readable problems; empty when the waterfall can be drawn.
    """
    problems = []
    try:
        ds = _open(ms, field, spw, scan)
        group = data_group(ms)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        return [f"Cannot open {ms}: {exc}"]

    for role, name in (("column", column), ("model", model)):
        if name and group.resolve(name) not in ds.data_vars:
            problems.append(f"{role} {name!r} is not in this Measurement Set")

    labels_ = [str(p) for p in ds.polarization.values]
    full = False
    if gtable:
        try:
            full = _gains(ms, gtable, field, spw, scan, ds)[0].full
        except Exception as exc:  # noqa: BLE001
            problems.append(f"gain table: {exc}")
    try:
        products.needed([product], labels_, full_jones=full)
    except ValueError as exc:
        problems.append(str(exc))
    if flagversion and flagversion not in flag_versions(ms):
        problems.append(f"no flag version {flagversion!r}")
    return problems


def _gains(
    ms: str, gtable: str, field: int, spw: int, scan: int, ds: xarray.Dataset
) -> tuple[products.Gains, list[str]]:
    """Gains for every antenna in the partition, on its time and channel grid, and those antennas' names."""
    names = sorted(set(map(str, ds.baseline_antenna1_name.values)) | set(map(str, ds.baseline_antenna2_name.values)))
    gains = gains_mod.load(gtable, field, spw, scan, ms_times(ds), ds.frequency.values, names)
    return gains, names


def waterfall(
    ms: str,
    field: int,
    spw: int,
    scan: int,
    name1: str,
    name2: str,
    product: str,
    column: str,
    quantity: str = "amp",
    rect: tuple[int, int, int, int] | None = None,
    model: str = "",
    gtable: str = "",
    flagversion: str = "",
) -> Waterfall:
    """Read one baseline's waterfall for a whole scan.

    The plotted quantity is ``inv(G_p) column inv(G_q)^H - model``, formed into
    ``product`` -- the same thing ``surfvis chi2 --mcol --gtable --stokes``
    computes, via the same code.

    Args:
        ms: Path to the Measurement Set.
        field: FIELD_ID.
        spw: DATA_DESC_ID.
        scan: SCAN_NUMBER.
        name1: Name of the first antenna.
        name2: Name of the second antenna.
        product: A correlation (``XX``) or Stokes parameter (``V``).
        column: Column to plot. MSv2 names are translated through the data
            group, so ``DATA`` finds ``VISIBILITY``; non-standard names such as
            ``RESIDUAL`` are used as given.
        quantity: One of ``amp``, ``phase``, ``real``, ``imag``.
        rect: Chunk bounds ``(t0, tf, chan0, chanf)`` to outline, in slots
            relative to the start of the scan.
        model: Optional model column, subtracted after correction.
        gtable: Optional QuartiCal gain term to correct by.
        flagversion: Optional CASA flag version to overlay instead of FLAG.

    Returns:
        A :class:`Waterfall`.

    Raises:
        KeyError: if a column, baseline or flag version is not present.
        ValueError: if the quantity, product or gains are not usable.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}, got {quantity!r}")

    ds = _open(ms, field, spw, scan)
    group = data_group(ms)
    resolved = group.resolve(column)
    for name in (resolved, group.resolve(model) if model else None):
        if name is not None and name not in ds.data_vars:
            raise KeyError(f"Column {name!r} not in this Measurement Set. Have: {sorted(ds.data_vars)}")

    labels_ = [str(p) for p in ds.polarization.values]
    gains = None
    ant1 = ant2 = None
    if gtable:
        gains, names = _gains(ms, gtable, field, spw, scan, ds)
        ntime = ds.sizes["time"]
        ant1 = np.full(ntime, names.index(name1))
        ant2 = np.full(ntime, names.index(name2))
    read = products.needed([product], labels_, full_jones=gains is not None and gains.full)
    corr_idx = [labels_.index(c) for c in read]

    k = baseline_index(ds, name1, name2)
    sel = {"baseline_id": k, "polarization": corr_idx}
    vis = ds[resolved].isel(sel).values
    mvis = ds[group.resolve(model)].isel(sel).values if model else None
    if flagversion:
        flags = _version_flags(ms, flagversion, field, spw, scan, name1, name2, ms_times(ds), corr_idx)
    else:
        flags = ds[group.flag].isel(sel).values.astype(bool)
    # xarray-ms lays a regular (time, baseline) grid over the MS and fills gaps
    # with NaN. Those slots hold no data, so treat them as flagged rather than
    # letting them punch holes in the colour scale.
    flags |= ~np.isfinite(vis)
    vis = np.where(np.isfinite(vis), vis, 0)
    if mvis is not None:
        mvis = np.where(np.isfinite(mvis), mvis, 0)

    tidx = None
    if gains is not None:
        grid = ms_times(ds)
        tidx = gains.time_index(grid, grid)
    plane, _, pflags = products.form(
        [product],
        read,
        vis,
        np.ones(vis.shape),
        flags,
        ant1,
        ant2,
        tidx=tidx,
        gains=gains,
        model=mvis,
    )
    plane, pflags = plane[..., 0], pflags[..., 0]

    if quantity == "phase":
        values = np.angle(plane)
    elif quantity == "real":
        values = np.real(plane)
    elif quantity == "imag":
        values = np.imag(plane)
    else:
        values = np.abs(plane)
    values = np.asarray(values, dtype=float)

    return Waterfall(
        values=values,
        flags=pflags,
        antenna1=name1,
        antenna2=name2,
        scan=scan,
        column=_describe(resolved, group.resolve(model) if model else "", gtable),
        quantity=quantity,
        product=product,
        rect=rect,
        length=baseline_length(ms, name1, name2),
        times=ms_times(ds) - MJD_UNIX_OFFSET,
        plane=plane,
        freqs=np.asarray(ds.frequency.values, dtype=float),
    )


def cached_plane(
    ms: str,
    field: int,
    spw: int,
    scan: int,
    name1: str,
    name2: str,
    product: str,
    column: str,
    model: str = "",
    gtable: str = "",
    flagversion: str = "",
) -> Waterfall:
    """One baseline's formed complex plane, cached so zooming does not re-read the MS.

    The key includes the MS's on-disk fingerprint and the gain term's mtime,
    so a rewritten MS or a re-solved gain table is read afresh. ``quantity``
    and the chunk rectangle are not part of the key. The entry is slimmed to
    save memory: ``plane`` is complex64 and ``values`` is None, so use
    ``plane``; ``rect`` is None.
    """
    return _cached_plane(
        ms,
        field,
        spw,
        scan,
        name1,
        name2,
        product,
        column,
        model,
        gtable,
        flagversion,
        _fingerprint(ms),
        _gtable_stamp(gtable),
    )


@lru_cache(maxsize=4)
def _cached_plane(
    ms, field, spw, scan, name1, name2, product, column, model, gtable, flagversion, fingerprint, gstamp
) -> Waterfall:
    # fingerprint and gstamp only key the cache.
    wf = waterfall(
        ms,
        field,
        spw,
        scan,
        name1,
        name2,
        product,
        column,
        quantity="amp",
        rect=None,
        model=model,
        gtable=gtable,
        flagversion=flagversion,
    )
    return replace(wf, plane=wf.plane.astype(np.complex64), values=None)


def _gtable_stamp(gtable: str) -> int | None:
    """mtime of a gain term's directory, or None."""
    if not gtable:
        return None
    try:
        return os.stat(gains_mod.term_path(gtable)).st_mtime_ns
    except OSError:
        return None


def _describe(column: str, model: str, gtable: str) -> str:
    """How the plotted quantity was formed, for the plot title."""
    text = f"corrected {column}" if gtable else column
    return f"{text} - {model}" if model else text
