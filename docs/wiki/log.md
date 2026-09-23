---
type: log
title: Wiki changelog
description: Chronological record of what changed in the wiki and why.
timestamp: 2026-09-18
last_verified_commit: 91fd17f
---

# Wiki changelog

## 2026-09-19 — conversion review pass

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
