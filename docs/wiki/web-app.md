---
type: reference
title: Chi-squared browser
description: How surfvis serve links surfchi2 output to on-demand baseline waterfalls, and the xarray-ms facts it depends on.
tags: [web, fastapi, htmx, xarray-ms, zarr]
timestamp: 2026-09-18
last_verified_commit: PENDING
---

# Chi-squared browser

Source: `src/surfvis/web/` (store, msdata, render, app), `src/surfvis/cli/serve.py`,
and the zarr writer at the end of `src/surfvis/core/chi2.py`.

**Not a Stimela cab, and never will be.** A long-running GUI is not a batch
task. `serve` carries no `@stimela_cab` decorator, and `parse_module` skips
undecorated functions, so `generate-cabs` emits nothing for it.

## Data flow

```
surfvis chi2 --dataout chi2.zarr   →  chi2.zarr        (numbers, not pictures)
                                          ↓ store.py
surfvis serve --data chi2.zarr     →  heatmap grid + histogram
                                          ↓ click  → msdata.py
                                      waterfall PNG   (read from the MS)
```

The MS is opened only when a waterfall is requested. Browsing the grid touches
the zarr alone.

## The zarr contract

One group per `(field, spw, scan)`, because the number of time and frequency
bins differs between them:

```
chi2.zarr/field{F}/spw{S}/scan{N}
  chi2, counts        (time_bin, freq_bin, corr, antenna1, antenna2)
  t0/tf               time-slot bounds of each time bin, within the scan
  chan0/chanf         channel bounds of each frequency bin
  attrs               ms, rcol, wcol, fcol, antenna_names
```

Two things are easy to get wrong here:

- **`corr` holds resolved polarization indices, not offsets into `--use-corrs`.**
  The default `[0, -1]` is stored as `[0, 3]` for a 4-correlation MS. Store the
  offset instead and every waterfall shows the wrong polarization.
- **Bin bounds must be stored.** Without `t0`/`chan0` the app cannot outline the
  clicked chunk on the scan's waterfall, and would have to re-derive the
  chunking from `--ntimes`/`--nfreqs`.

`surfchisq_plot` returns `(field, spw, scan, chi2, counts)` and does not say
which chunk it was. `core/chi2.py` therefore keys futures by `(ds, t, f, c)` in
the parent — `futures[fut] = (i, t, f, c)` — rather than changing the worker.

## xarray-ms facts this depends on

Verified against 0.5.9 (see `tests/fixtures/ms.py`):

- Variables come out as `(time, baseline_id, frequency, polarization)`, which
  *is* the waterfall shape. No row-to-grid pivot is needed, unlike dask-ms.
- **`FIELD_ID` is not a partition column by default.** The default schema is
  `OBSERVATION_ID/PROCESSOR_ID/DATA_DESC_ID/OBS_MODE_ID`, so `msdata._open`
  passes `partition_schema=["DATA_DESC_ID", "FIELD_ID"]` explicitly to match how
  surfchi2 groups.
- **Scan is a coordinate, not a partition.** Select with a boolean index over
  `scan_name`, whose values are *strings*.
- `VISIBILITY` is hard-wired to the `DATA` column and cannot be repointed. But
  columns that are *not* part of the MSv2 standard are surfaced automatically as
  secondary variables — which is exactly why `RESIDUAL` is plottable. Standard
  columns other than `DATA`/`WEIGHT_SPECTRUM` are dropped, except
  `CORRECTED_DATA`, `MODEL_DATA`, `FLOAT_DATA` and `CORRECTED_WEIGHT_SPECTRUM`.
- Baselines are addressed by antenna *name*, via the `baseline_antenna1_name` /
  `baseline_antenna2_name` coordinates, hence the names stored in the zarr.

## Colour scaling

Default is **log**. chi-squared/dof spans orders of magnitude, and on a linear
scale one bad baseline renders every other cell black — which defeats the point
of looking at the grid. The same applies to the histogram: linear bins collapse
the bulk into a single bar, so log bins follow the log norm. `robust` (2nd-98th
percentile) and `full` remain available; note that `robust` is degenerate when
there are few antenna pairs, since the 98th percentile lands inside the
outliers.

## The fork deadlock

`core/chi2.py` builds its `ProcessPoolExecutor` with an explicit **spawn**
context. `dask.config.set(pool=ThreadPool(...))` has already started threads by
that point, and forking a process with live threads deadlocks the children —
the parent sits in `as_completed` forever. This was reproducible on every run
before the fix.

## Testing

`tests/fixtures/ms.py` writes a real Measurement Set with python-casacore, small
enough to build in well under a second. Three subtables have to be populated or
xarray-ms refuses to construct the MSv4 view: `FEED` (validated against
`ANTENNA1`/`ANTENNA2`), `STATE`, and the `FIELD` direction columns
(`PHASE_DIR`/`REFERENCE_DIR`/`DELAY_DIR`). The fixture plants an inflated
residual on one baseline, and the tests follow that outlier all the way to the
rendered grid.

Tests skip cleanly without the `[full,web]` extras; `tests/conftest.py` carries
the `needs_ms` / `needs_web` markers.
