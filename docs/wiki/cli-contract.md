---
type: reference
title: CLI, cab and core contract
description: How src/surfvis/cli, src/surfvis/cabs and src/surfvis/core stay in agreement, and the hip-cargo quirks this repo routes around.
tags: [cli, cabs, typer, stimela, round-trip]
timestamp: 2026-09-18
last_verified_commit: 9276bd3
---

# CLI, cab and core contract

Source: `src/surfvis/cli/*.py` (wrappers), `src/surfvis/cabs/*.yml`
(generated), `src/surfvis/core/*.py` (implementations),
`tests/test_roundtrip.py` (the enforcement).

## The three layers

| Layer | Imports | Written by |
|---|---|---|
| `cli/<cmd>.py` | `typer`, `hip_cargo` only | Hand-edited, then **regenerated to canonical form** (see below) |
| `cabs/<cmd>.yml` | — | `hip-cargo generate-cabs`, never by hand |
| `core/<cmd>.py` | numpy, dask, daskms, matplotlib, pyrap, numba | Hand-written |

`cli/` must stay importable with only the lightweight install
(`pip install surfvis`). It lazy-imports `core/` inside
`try/except ImportError`; on failure `run_in_container()` re-runs the command
inside `ghcr.io/ratt-ru/surfvis:<tag>`. Never add a heavy import to `cli/` or
to `cli/__init__.py`.

`utils/chisq.py` and `utils/plotting.py` are heavy too. They are imported from
`core/` only.

## Round-trip is the contract

`tests/test_roundtrip.py` runs `generate-cabs` then `generate-function` on each
`cli/*.py` and asserts the regenerated source is **byte-identical**. Practical
consequence: the committed `cli/*.py` files are exactly what
`hip-cargo generate-function` emits. When you change a CLI signature, the
reliable procedure is:

```bash
# 1. edit src/surfvis/cli/<cmd>.py by hand
uv run hip-cargo generate-cabs --module 'src/surfvis/cli/*.py' --output-dir src/surfvis/cabs
# 2. regenerate the wrapper from the cab and adopt it verbatim
uv run hip-cargo generate-function --cab-file src/surfvis/cabs/<cmd>.yml \
    --output-file src/surfvis/cli/<cmd>.py --config-file pyproject.toml
# 3. regenerate cabs again (the adopted file may reflow help strings)
uv run hip-cargo generate-cabs --module 'src/surfvis/cli/*.py' --output-dir src/surfvis/cabs
uv run pytest tests/test_roundtrip.py
```

Add a `test_roundtrip_<cmd>` case for every new command.

Because the reverse generator only ever emits `typer.Option`, **required
parameters are options, not positional CLI arguments** — hence `--ms` rather
than a bare `MS` argument. The cab still records `policies: {positional: true}`,
which is correct: `flavour: python` means Stimela calls the *core* function,
where `ms` genuinely is the first positional argument.

## Two hip-cargo quirks this repo works around

1. **Negative defaults serialise as YAML strings.** `= -1` in a CLI signature
   emits `default: '-1'` under an `dtype: int` — a string where Stimela wants an
   int. surfvis therefore expresses every "unset" sentinel as `| None = None`
   (`--i`, `--j`, `--scale`, `--ntimes`). The core functions still honour an
   explicit `-1` for backwards compatibility, e.g. `ant1 = -1 if i is None else i`.
2. **A colon in a multi-sentence `help=` produces invalid YAML.** hip-cargo
   splits `help` into one YAML line per sentence, and single-quotes only the
   sentence containing the colon, which breaks the block. Keep colons out of
   any `help=` string that has more than one sentence — `"Antenna 1: plot only
   this antenna. Defaults to all of them."` had to become `"Index of antenna 1.
   Plot only this antenna. Defaults to all of them."`.

## Per-command specifics

- **`flag-chi2` rewrites the MS in place.** Its `ms` input carries
  `StimelaMeta(writable=True)`, which becomes `writable: true` in the cab and
  makes the container-fallback runner bind-mount the MS read-write. Drop it and
  flagging silently fails inside a container.
- **`surf` and `chi2` declare `@stimela_output` decorators** for `opdir`, and
  `dataout`/`imagesout` respectively. An output name must be the kebab-case form
  of the matching parameter name, otherwise the parameter is emitted as *both*
  an input and an output.
- **List-valued options use `ListInt` + `parse_list_int`**, not `list[int]`:
  `--spw 0,1,2`, `--use-corrs 0,3`, `--respect-ants 12,41`. See hip-cargo's
  README §Quirks for why Click cannot type these as `list[int]`.
- **`--plot` is a `Literal`**, which becomes `choices:` in the cab. `core/surf.py`
  re-validates against `VALID_PLOTS` because Stimela calls the core function
  directly and does not go through Typer.

## Image tag lifecycle

`src/surfvis/_container_image.py` is the single source of truth. On a feature
branch, set the tag to the branch name by hand; the `update-cabs` workflow
resets it to `latest` on merge to `master`; `tbump` writes the semantic version
during a release. The tag appears in every cab's `image:` field, so cabs must be
regenerated whenever it changes — both the pre-commit hook and `tbump.toml` do
this for you.
