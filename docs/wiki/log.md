---
type: log
title: Wiki changelog
description: Chronological record of what changed in the wiki and why.
timestamp: 2026-09-18
last_verified_commit: 91fd17f
---

# Wiki changelog

## 2026-09-18 — chi-squared browser

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
