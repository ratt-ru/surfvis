"""Read the chi-squared zarr written by ``surfvis chi2 --dataout``."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


@dataclass(frozen=True)
class Selection:
    """One chunk of the chi-squared cube: a (field, spw, scan) at one bin."""

    field: int
    spw: int
    scan: int
    time_bin: int
    freq_bin: int
    corr: int


class Chi2Store:
    """Lazy reader over the chi-squared zarr hierarchy."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No chi-squared dataset at {self.path}")
        self._scans = _discover_scans(self.path)
        if not self._scans:
            raise ValueError(f"{self.path} contains no field*/spw*/scan* groups")

    @property
    def scans(self) -> list[tuple[int, int, int]]:
        """Every (field, spw, scan) in the store, in sorted order."""
        return self._scans

    @property
    def fields(self) -> list[int]:
        return sorted({f for f, _, _ in self._scans})

    def spws(self, field: int) -> list[int]:
        return sorted({s for f, s, _ in self._scans if f == field})

    def scan_numbers(self, field: int, spw: int) -> list[int]:
        return sorted({n for f, s, n in self._scans if f == field and s == spw})

    def group(self, field: int, spw: int, scan: int) -> xr.Dataset:
        """Open one (field, spw, scan) group."""
        return _open_group(str(self.path), field, spw, scan)

    def chunk(self, sel: Selection) -> "Chunk":
        """Return the antenna-by-antenna chi-squared/dof matrix for one chunk."""
        ds = self.group(sel.field, sel.spw, sel.scan)
        corr_index = int(np.where(ds.corr.values == sel.corr)[0][0])
        chi2 = ds.chi2.values[sel.time_bin, sel.freq_bin, corr_index]
        counts = ds.counts.values[sel.time_bin, sel.freq_bin, corr_index]

        with np.errstate(invalid="ignore", divide="ignore"):
            chi2_dof = np.where(counts > 0, chi2 / np.where(counts > 0, counts, 1), np.nan)

        return Chunk(
            selection=sel,
            chi2_dof=chi2_dof,
            antenna_names=list(ds.attrs["antenna_names"]),
            t0=int(ds.t0.values[sel.time_bin]),
            tf=int(ds.tf.values[sel.time_bin]),
            chan0=int(ds.chan0.values[sel.freq_bin]),
            chanf=int(ds.chanf.values[sel.freq_bin]),
            ms=str(ds.attrs["ms"]),
            rcol=str(ds.attrs["rcol"]),
        )

    def bins(self, field: int, spw: int, scan: int) -> tuple[list[int], list[int], list[int]]:
        """Available (time bins, freq bins, polarization indices) for a scan."""
        ds = self.group(field, spw, scan)
        return (
            [int(v) for v in ds.time_bin.values],
            [int(v) for v in ds.freq_bin.values],
            [int(v) for v in ds.corr.values],
        )


@dataclass(frozen=True)
class Chunk:
    """A single chunk's chi-squared/dof matrix plus the context to plot it."""

    selection: Selection
    chi2_dof: np.ndarray
    antenna_names: list[str]
    t0: int
    tf: int
    chan0: int
    chanf: int
    ms: str
    rcol: str

    @property
    def finite(self) -> np.ndarray:
        """The non-NaN values, i.e. the antenna pairs that had unflagged data."""
        return self.chi2_dof[~np.isnan(self.chi2_dof)]


@lru_cache(maxsize=32)
def _open_group(path: str, field: int, spw: int, scan: int) -> xr.Dataset:
    # chunks=None reads eagerly into numpy. Without it xarray returns
    # dask-backed arrays, which would make dask a hard requirement of the web
    # extra for no benefit -- a single chunk's matrix is a few hundred KB and
    # every consumer here calls .values immediately.
    return xr.open_zarr(path, group=f"field{field}/spw{spw}/scan{scan}", chunks=None)


def _discover_scans(path: Path) -> list[tuple[int, int, int]]:
    """Walk the zarr hierarchy for field*/spw*/scan* groups."""
    root = zarr.open_group(str(path), mode="r")
    found = []
    for fname in root:
        if not fname.startswith("field"):
            continue
        for sname in root[fname]:
            if not sname.startswith("spw"):
                continue
            for nname in root[fname][sname]:
                if not nname.startswith("scan"):
                    continue
                found.append(
                    (
                        int(fname.removeprefix("field")),
                        int(sname.removeprefix("spw")),
                        int(nname.removeprefix("scan")),
                    )
                )
    return sorted(found)
