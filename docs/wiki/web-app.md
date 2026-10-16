---
type: reference
title: Chi-squared browser
description: How surfvis serve links surfchi2 output to on-demand baseline waterfalls, the waterfall's correction / Stokes / flag-version controls, and the xarray-ms facts it depends on.
tags: [web, fastapi, htmx, xarray-ms, zarr]
timestamp: 2026-10-16
last_verified_commit: 1ab4f7c
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
  chi2, counts        (time_bin, freq_bin, product, antenna1, antenna2)
  t0/tf               time-slot bounds of each time bin, within the scan
  chan0/chanf         channel bounds of each frequency bin
  attrs               ms, rcol, wcol, fcol, mcol, gtable, corr_labels, antenna_names
```

Two things are easy to get wrong here:

- **`product` holds labels** -- `XX`, `YY` for the default `--use-corrs`, then
  any `--stokes` (`I`, `V`). They are resolved from `POLARIZATION::CORR_TYPE`,
  not offsets into `--use-corrs`, and the waterfall forms the same label from
  the MS. Stores written before products existed have a `corr` axis instead;
  `Chi2Store.bins` refuses them with a message to rerun `chi2`.
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
- **The default partition schema is not what you want.** It is
  `OBSERVATION_ID/PROCESSOR_ID/DATA_DESC_ID/OBS_MODE_ID` — no `FIELD_ID`, no
  `SCAN_NUMBER`. `msdata.PARTITION_SCHEMA` is
  `("FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER")`, matching `pfb-imaging`'s
  default and surfchi2's `group_cols` exactly. One partition is then one
  (field, spw, scan), so a partition's time axis *is* the scan and the chunk
  can be addressed by integer ids via `partition_key` — no name lookup, no
  boolean filtering over `scan_name`.
- **Column names are MSv4, resolved via `data_groups`.** `DATA` is exposed as
  `VISIBILITY` and `WEIGHT_SPECTRUM` as `WEIGHT`. Do not hardcode that: read
  `node.ds.attrs["data_groups"][group]["correlated_data"]` / `["weight"]` /
  `["flag"]`, as `pfb-imaging`'s `core/imager.py` does. `msdata.DataGroup`
  wraps this and `DataGroup.resolve()` translates the MSv2 names the data group
  maps (`DATA`, `WEIGHT_SPECTRUM`, `FLAG`), so both spellings work in the UI.
  It must translate **nothing else**: it once mapped `CORRECTED_DATA` to
  `VISIBILITY` too, so choosing `CORRECTED_DATA` plotted `DATA`.
- **`data_groups` is attached by `open_datatree`, not `open_dataset`.** Hence
  `msdata.data_group()` opens the tree once and caches it, while the per-chunk
  reads use `open_dataset(partition_key=...)`.
- Columns that are *not* part of the MSv2 standard are surfaced automatically as
  secondary variables under their own names — which is exactly why `RESIDUAL` is
  plottable. Standard columns other than `DATA`/`WEIGHT_SPECTRUM` are dropped,
  except `CORRECTED_DATA`, `MODEL_DATA`, `FLOAT_DATA` and
  `CORRECTED_WEIGHT_SPECTRUM`. Those four appear under their own names --
  only once their cells are defined, so an empty `MODEL_DATA` is invisible.
- **`time` is decoded to unix seconds** (`attrs["format"] == "unix"`). Gain
  tables and flag versions are keyed on the MS's MJD-second `TIME`, so
  `msdata.ms_times()` adds `MJD_UNIX_OFFSET` back before matching.
- The partition's `polarization` coordinate already holds labels (`XX` ...),
  which is what `msdata.labels()` returns.
- Baselines are addressed by antenna *name*, via the `baseline_antenna1_name` /
  `baseline_antenna2_name` coordinates, hence the names stored in the zarr.
- **xarray-ms lays a regular (time, baseline) grid and fills gaps with NaN.**
  Those slots hold no data, so `waterfall()` folds `~isfinite` into the flag
  mask rather than letting them punch holes in the colour scale.

- **xarray-ms warns on every open** (imputed metadata, column shapes, UVW
  frame, irregular grids). `msdata` calls `utils.msv4.ignore_xarray_ms_warnings()`
  at module scope, the same filter set as pfb-imaging.

## Waterfall controls: product, model, gains, flag version

The waterfall panel carries six controls: product, column, model, gain table,
flag version, quantity. They recompute **the waterfall only**. The heatmap is
what `chi2` measured, so the controls default to how it was formed (the zarr's
`rcol`/`mcol`/`gtable` attrs and the selected product), and the panel shows a
note once they no longer describe the same quantity. Recomputing the heatmap
on the fly was ruled out: one chunk is every baseline across the chunk, which
at MeerKAT scale is hundreds of MB per click.

The plotted quantity is `inv(G_p) column inv(G_q)^H - model` formed into the
product, via `utils/products.form` -- the same function `chi2` uses, so the two
cannot drift apart. `chi2`'s recorded `DATA` is resolved to `VISIBILITY` before
it reaches the menus, or the column would be listed twice.

- **Gain table** is free text: a QuartiCal term (`gains.qc/NET`, or pfb's
  `gains.qc::NET`). Gains are read only for the antennas in the partition and
  apply on-grid only (see [chi2-pipeline.md](chi2-pipeline.md)). The term is
  cached on the directory's mtime, so a re-solve is picked up.
- **Flag version** lists CASA's `<ms>.flagversions/FLAG_VERSION_LIST`
  (`name : comment` lines) where `flags.<name>` exists. Version tables are
  row-aligned with the main table, so the baseline's rows are found with an
  arcae TaQL query (`Table.from_taql`, `ROWID()`) on the main table and the
  same row numbers read from the version; `FLAG_ROW` is ORed in and grid slots
  with no row are flagged. arcae returns these boolean columns as `uint8`, and
  wants a writable row-index array. The web layer needs no python-casacore.
  Only listed names are accepted, which also keeps the name from being a path.
- **The column menu lists complex columns only.** The column is corrected and
  has a model subtracted, which means nothing for `WEIGHT`.
- **Problems are reported in the panel**, not as a broken image: the
  `/waterfall` route runs `msdata.check()` (columns exist, product formable,
  gains load and align, version exists) before emitting the `<img>`. If the
  PNG still fails, the page fetches it again and shows the route's `detail`.
- **Choices survive a change of baseline.** The route marks each control whose
  value differs from the heatmap's default with `class="keep"`, and
  `static/app.js` adds those to a cell click's parameters in
  `htmx:configRequest` (an `hx-include` would warn on every click while
  nothing is marked). Only *changed* controls are
  carried, so a product left at its default follows the heatmap's product
  when that changes, while an explicit choice sticks.

## The grid

The matrix holds each baseline twice, as `(p, q)` and `(q, p)`, so the header
counts the upper triangle. Diagonal cells are autocorrelations, which `chi2`
does not measure: they carry no `hx-get`, since the waterfall route would
answer them with a 404 that renders as a broken image.

Fields are listed by `FIELD::NAME` (`msdata.field_names`, arcae), with the
`FIELD_ID` as the option value; the zarr stays keyed by ID. If the MS cannot be
read the menu falls back to IDs.

The grid and the histogram sit in two equal columns of one CSS grid, and the
histogram PNG is square, so the two render at the same size.

The waterfall title carries the baseline length: the distance between the two
`ANTENNA::POSITION` rows (ITRF metres), read once per MS with arcae
(`msdata.baseline_length`). Its time axis is labelled in UTC; pixels stay in
slot coordinates, since the chunk outline (and later, a flagging selection)
uses them.

The heatmap ships one JSON blob (`#grid-data`: antenna names, the χ²/dof
matrix, antenna positions) rather than attributes on every cell. `app.js`
reads it for:

- **the hover tooltip** at the cursor, mirrored in the readout under the grid:
  names with antenna numbers (the grid index is the `ANTENNA` row, since
  `chi2` records every antenna), χ²/dof, baseline length. A custom element,
  not `title=`, which shows only after a delay;
- **arrow-key navigation**: up/down change antenna1, left/right antenna2,
  skipping the diagonal;
- **following the selection.** The selected baseline is held by antenna
  *names*, since another scan can hold a different antenna set. When the
  heatmap re-renders (field, scan, bin or product) and a waterfall is open, the
  same baseline is re-fetched there, with the carried settings. This hooks
  `htmx:afterSettle`, not `afterSwap`: htmx has not wired the new cells' `hx-get`
  at `afterSwap`, so a click there does nothing.

Antenna names label the grid: slanted at 60° along the top, horizontal down the
side. They sit in a gutter and band around the square (`.grid-frame`), sized to
the cell pitch with container query units (`100cqw / --n`); the histogram pads
by the same amounts, on its right, so the two squares stay equal.

While an htmx request is in flight, `#heatmap` and `#waterfall` show a spinner
over their dimmed content (the children are dimmed, not the region, so the
spinner stays opaque). Without it, the first click showed nothing in the
waterfall area: the per-PNG spinner only exists once a panel has rendered.

**Save as PNG** renders `<body>` in the browser with `static/html-to-image.js`
(html-to-image 1.11.11, MIT, vendored like htmx so a node without internet
still works). It goes through an SVG `foreignObject`, so the page comes out as
shown, scrolled-off parts included, at `pixelRatio: 2`. The library re-fetches
every `<img>` it embeds; for the waterfall that would mean reading the MS
again, so `savePng` first swaps each loaded image's `src` for a data URL of its
decoded pixels (the original is kept in `data-src`). Printing is no substitute:
browsers drop background colours by default, and every grid cell is one.

The histogram's axis covers the data; the χ²/dof = 1 reference is drawn only
when it falls inside, since forcing it in left most decades empty on
uncalibrated data.

## Loading feedback

Three layers, because the slow part is not always an htmx request:

- `#busy`, a bar across the top, is the `hx-indicator` for every htmx request
  (set once on `<body>`, inherited). It also stays lit while any `.plot.loading`
  exists (`body:has(...)`), since the waterfall PNG is the slow part and its
  panel is usually below the fold when a cell is clicked.
- The region a request will replace (`#heatmap`, `#waterfall`, `#selectors`)
  is dimmed between `htmx:beforeRequest` and `htmx:afterRequest`.
- The PNGs are fetched by the browser, not htmx, and the waterfall PNG is where
  the MS is actually read. Each sits in a `.plot.loading` wrapper showing a
  spinner until the `<img>`'s `onload`.

Static URLs carry `?v=<digest of the static files>`. `StaticFiles` sends no
`Cache-Control`, so browsers cached the stylesheet heuristically and served
the old one with new HTML after an upgrade: no indicators, an unsized
histogram.

The selected cell keeps an outline (`.selected`) across heatmap re-renders.

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

Spawned workers re-import matplotlib, so `utils/plotting.py` forces `Agg`.
Otherwise a forwarded `DISPLAY` selects TkAgg, and when the X connection drops
(a closed SSH session under `screen`) every worker dies on an XIO error and the
parent raises `BrokenProcessPool`.

## Testing

`tests/fixtures/ms.py` writes a real Measurement Set with python-casacore, small
enough to build in well under a second. Three subtables have to be populated or
xarray-ms refuses to construct the MSv4 view: `FEED` (validated against
`ANTENNA1`/`ANTENNA2`), `STATE`, and the `FIELD` direction columns
(`PHASE_DIR`/`REFERENCE_DIR`/`DELAY_DIR`). The fixture plants an inflated
residual on one baseline, and the tests follow that outlier all the way to the
rendered grid. Its columns are mutually consistent --
`DATA = G_p MODEL_DATA G_q^H + RESIDUAL` and
`CORRECTED_DATA = inv(G_p) DATA inv(G_q)^H`, with diagonal gains varying in
time, channel and antenna -- so the corrections are checked exactly.
`write_gain_table` writes those gains as a QuartiCal term and
`add_flag_version` writes a CASA flag version.

Tests skip cleanly without the `[full,web]` extras; `tests/conftest.py` carries
the `needs_ms` / `needs_web` markers.
