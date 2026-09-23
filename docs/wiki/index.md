---
type: index
title: surfvis LLM wiki
description: Progressive-disclosure listing of the in-repo knowledge bundle.
timestamp: 2026-09-18
last_verified_commit: PENDING
---

# surfvis LLM wiki

In-repo knowledge bundle in the [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
style: plain markdown + YAML frontmatter, readable by humans without tools and
by agents without SDKs. The primary reader is an LLM agent; humans are a close
second.

**This is the canonical reference for what is implemented.**
Specs and plans are ephemeral process artifacts and are not retained in the
repo — do not cite them.

**Verification contract:** every page's frontmatter carries
`last_verified_commit` — the commit its claims were last checked against.
To assess staleness: `git diff <stamp>..HEAD -- <files the page covers>`.
Maintenance rule (also in `CLAUDE.md`): if your change invalidates or extends
a page, update the page and refresh its stamp **in the same session**.

## Pages

| Page | Covers | Read when |
|------|--------|-----------|
| [cli-contract.md](cli-contract.md) | CLI ↔ cab ↔ core contract, round-trip constraints, hip-cargo quirks to route around | Touching anything under `src/surfvis/cli/` or `src/surfvis/cabs/` |
| [web-app.md](web-app.md) | The chi-squared browser: zarr contract, xarray-ms behaviour, colour scaling, the spawn fix | Touching `src/surfvis/web/`, `cli/serve.py` or the zarr writer |
| [chi2-pipeline.md](chi2-pipeline.md) | MS chunking, numba kernels, the parallel reduce, output layout, in-place flagging | Touching `core/chi2.py`, `core/flag_chi2.py` or `utils/` |
| [migration.md](migration.md) | What the pre-hip-cargo entry points mapped to, option-by-option | Answering "where did `surfchi2` go?" or porting an old command line |
| [log.md](log.md) | Chronological wiki changelog | Checking what changed and when |

## Not covered here

- **How to edit this codebase** (linting, commit format, Typer patterns):
  `.claude/rules/*.md` — harness instructions, kept separately.
- **hip-cargo's own machinery** (container fallback internals, GPU
  passthrough, remote URIs, the `_container_image.py` contract):
  [hip-cargo's wiki](https://github.com/landmanbester/hip-cargo/tree/main/docs/wiki).
  This repo only *uses* those mechanisms.
