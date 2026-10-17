---
type: log
title: Wiki changelog
description: Chronological record of what changed in the wiki and why.
timestamp: 2026-10-16
last_verified_commit: d4a40a0
---

# Wiki changelog

## 2026-10-16 — plane cache and widget fixes (verified at `d4a40a0`)

- **Updated** `web-app.md`: "Plane cache" section (key, memory, colour limits),
  `wheelFactor`, single first fetch, fetch abort.

## 2026-10-16 — canvas waterfall widget (verified at `dfee997`)

- **Updated** `web-app.md`: new "Canvas waterfall" section (route, wire format,
  modules, interactions, `no-cache` static header); loading-feedback bullet.

## 2026-10-16 — flagging write path (verified at `3eb66e8`)

- **Updated** `web-app.md`: new "Manual flagging (write path)" section --
  polygon rasterisation, `/flag`, the backup flag version (and why `_c_order`
  stays in the descriptor), undo, cache invalidation.

## 2026-10-16 — Save as PNG (verified at `1ab4f7c`)

- **Updated** `web-app.md`: in-browser page capture with vendored
  html-to-image, and why loaded images are passed as data URLs.

## 2026-10-16 — antenna labels and tooltip (verified at `366256a`)

- **Updated** `web-app.md`: slanted antenna labels around the grid, a cursor
  tooltip with antenna numbers, and a region spinner for the first click.

## 2026-10-16 — README and agent files for PR #17 (verified at `a811509`)

- **Updated** `index.md`: web-app.md now covers navigation and loading.
- README gains "Corrected residuals on the fly"; CLAUDE.md and
  `.claude/rules/architecture.md` describe `web/`, its JS sharp edges, and
  point at the flagging design (#16).

## 2026-10-16 — browser navigation (verified at `3cd34f5`)

- **Updated** `web-app.md`: hover readout and arrow keys from a per-heatmap
  JSON blob; the waterfall follows the selected baseline (by name) into a new
  field/scan/bin, hooked on `afterSettle`; UTC time axis; histogram axis
  limited to the data; complex-only column menu.

## 2026-10-16 — static cache busting (verified at `1950c3f`)

- **Updated** `web-app.md`: static URLs are versioned; the busy bar also
  covers PNG loads.

## 2026-10-16 — browser usability (verified at `461673f`)

- **Updated** `web-app.md`: fields listed by name; loading feedback (request
  bar, dimmed targets, PNG spinners); waterfall controls changed from the
  heatmap's defaults persist across baselines; histogram sized to the grid.

## 2026-10-15 — first run on real data (verified at `fb5c6ae`)

- **Updated** `web-app.md`: `chi2` workers force matplotlib's `Agg` backend
  (a dropped X forward killed the pool); the grid's diagonal is not
  clickable and the header counts each baseline once.
- **Updated** `web-app.md`: the waterfall title shows the baseline length.
- **Updated** `web-app.md`: xarray-ms's per-open warnings are filtered.

## 2026-10-15 — write-line pins, 3.10 CLI-only (verified at `1dbfc61`)

- **Updated** `chi2-pipeline.md`: dropped the claim that arcae is what makes
  `chi2` need 3.11 -- all native execution needs 3.11 now; 3.10 is CLI plus
  container only. The xarray-ms/arcae write-line pins are recorded in
  CLAUDE.md's sharp edges.

## 2026-10-14 — arcae for new table reads (verified at `85c6bcd`)

- **Updated** `web-app.md` and `chi2-pipeline.md`: flag versions and the
  spectral setup are read with arcae, not python-casacore; arcae's `uint8`
  booleans and its read-only-index refusal; `chi2` now needs Python 3.11+.

## 2026-10-14 — corrections, Stokes and flag versions (verified at `81f37f1`)

Issues #12, #13 and #14, on the browser branch.

- **Updated** `chi2-pipeline.md`: the `inv(G_p) R inv(G_q)^H - M` formula behind
  `--mcol`/`--gtable`, exact Stokes weights (and why radiomesh's imaging
  weights are not used), on-grid QuartiCal gains, one task per (time, freq)
  forming every product, new image names, and the inherited bug that made
  every per-correlation image identical.
- **Updated** `web-app.md`: the zarr's `product` axis and new attrs, the
  waterfall controls and why they leave the heatmap alone, CASA flag versions,
  the `CORRECTED_DATA` resolve bug, and xarray-ms decoding TIME to unix seconds.
- **Updated** `migration.md` (new chi2 options, image names) and `index.md`.

## 2026-09-19 — chi-squared browser (verified at `0070303`)

- **Added** `web-app.md` for `surfvis serve`: the zarr contract written by
  `chi2 --dataout`, the xarray-ms behaviour it relies on (partition schema,
  MSv4 column resolution via `data_groups`), why the colour scale is log, and
  the spawn-vs-fork deadlock.
- **Updated** `chi2-pipeline.md`: `--dataout` is no longer vestigial.

## 2026-09-19 — conversion review pass (verified at `c3bd176`)

- **Revised** `cli-contract.md`: the typer dialect is closed and enforced at
  parse time by hip-cargo 0.4.0, and writing the cab is the recommended
  direction — `--ms` and the absent short flags are properties of the format,
  not compromises. Replaced the "two quirks" section with the help-string cases
  that still emit silently-wrong YAML (a help ending in a colon becomes a dict;
  non-ASCII survives as a literal escape), verified end to end rather than
  assumed. The `-1` quirk is fixed upstream; `| None = None` stays on its own
  merits.
- **Added** `chi2-pipeline.md` §The fork deadlock: the process pool needs a
  spawn context, and why the failure is intermittent.

## 2026-09-18 — wiki created alongside the hip-cargo conversion

Initial bundle, written while converting surfvis from three `optparse` console
scripts to a single hip-cargo/Typer app.

- **Added** `index.md`, `cli-contract.md`, `chi2-pipeline.md`, `migration.md`.
