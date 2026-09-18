---
type: log
title: Wiki changelog
description: Chronological record of what changed in the wiki and why.
timestamp: 2026-09-18
last_verified_commit: 9276bd3
---

# Wiki changelog

## 2026-09-18 — wiki created alongside the hip-cargo conversion

Initial bundle, written while converting surfvis from three `optparse` console
scripts to a single hip-cargo/Typer app.

- **Added** `index.md`, `cli-contract.md`, `chi2-pipeline.md`, `migration.md`.
- Pages describe the working tree of the conversion, which is the commit
  *after* the `last_verified_commit` stamps they currently carry. Re-stamp all
  four with `git rev-parse --short HEAD` once the conversion commit lands.
