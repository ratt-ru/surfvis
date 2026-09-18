---
type: log
title: Wiki changelog
description: Chronological record of what changed in the wiki and why.
timestamp: 2026-09-18
last_verified_commit: 91fd17f
---

# Wiki changelog

## 2026-09-19 — review pass after hip-cargo #112/#113

- **Revised** `cli-contract.md`: the typer dialect is closed and enforced at
  parse time upstream, and writing the cab is the recommended direction —
  `--ms` and the absent short flags are properties of the format, not
  compromises. Replaced the "two quirks" section with the help-string cases
  that still emit silently-wrong YAML on the pinned branch (help ending in a
  colon becomes a dict; non-ASCII survives as a literal escape), verified end
  to end rather than assumed. The `-1` quirk is fixed upstream; `| None = None`
  stays on its own merits.
- **Re-stamped** `chi2-pipeline.md`, `index.md` and `web-app.md`, which had
  drifted from the commits that changed them.

## 2026-09-18 — chi-squared browser (verified at `6b8f4f7`)

- **Added** `web-app.md` for `surfvis serve`: the zarr contract written by
  `chi2 --dataout`, the xarray-ms behaviour it relies on (partition schema,
  scan-as-coordinate, secondary columns), why the colour scale is log, and the
  spawn-vs-fork deadlock in `core/chi2.py`.
- **Updated** `chi2-pipeline.md`: `--dataout` is no longer vestigial.

## 2026-09-18 — wiki created alongside the hip-cargo conversion

Initial bundle, written while converting surfvis from three `optparse` console
scripts to a single hip-cargo/Typer app.

- **Added** `index.md`, `cli-contract.md`, `chi2-pipeline.md`, `migration.md`.
- All pages stamped against `91fd17f`, the conversion commit.
