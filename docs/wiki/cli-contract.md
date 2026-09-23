---
type: reference
title: CLI, cab and core contract
description: How src/surfvis/cli, src/surfvis/cabs and src/surfvis/core stay in agreement, and the hip-cargo quirks this repo routes around.
tags: [cli, cabs, typer, stimela, round-trip]
timestamp: 2026-09-18
last_verified_commit: c3bd176
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

### Which side you write is free; the dialect is not

A cab and a CLI module are two renderings of one definition, and hip-cargo
generates either from the other. **Writing the YAML cab and generating the CLI
is the simpler direction and the recommended one** — the procedure above is for
when you would rather edit Python. What you cannot do is hand-edit the
generated side and stop there.

That bijection is why the writable language is closed: a typer construct with no
cab representation cannot survive the round trip. `typer.Argument` and
`typer.Option("-x", "--ex", ...)` are both rejected at parse time, with a message
naming the rule. So **required parameters are options, not positional CLI
arguments** — `--ms`, not a bare `MS` — and flag names derive from the parameter
name, so surfvis has no short flags. Those are properties of the format, not
compromises this project made.

The cab still records `policies: {positional: true}` for required inputs, which
is correct and unrelated: `flavour: python` means Stimela calls the *core*
function, where `ms` genuinely is the first positional argument.

Upstream reference: hip-cargo's
[`docs/wiki/cli-dialect.md`](https://github.com/landmanbester/hip-cargo/blob/main/docs/wiki/cli-dialect.md).

## Help strings: what still breaks

`format_info_fields` rewrites each `info:` value as text after `safe_dump` has
already quoted it, splitting it into one line per sentence. It re-derives the
quoting from a heuristic, so some help strings emerge wrong. Verified against
hip-cargo 0.4.0, one parameter per case, `generate-cabs` run end to end:

| `help=` | Result |
|---|---|
| `"Options are as follows:"` | **silent** — `info` loads as a *dict*, `{'Options are as follows': None}` |
| `"Angle in degrees (°). Second sentence."` | **silent** — `info` is `'Angle in degrees (\xB0). Second sentence.'` |
| `"Weights column. Options are as follows:"` | raises `ValueError` |
| `"Use channel #3. Second sentence."` | raises |
| `"- leading dash. Second sentence."` | raises |

The two silent rows are the hazard; the rest fail loudly at generation, which is
the `yaml.safe_load` guard doing its job. **Non-ASCII is the one most likely to
bite here** — `safe_dump` defaults to `allow_unicode=False`, so `°`, `λ` and `μ`
come back as a double-quoted scalar and the escape survives into the cab. Radio
astronomy help text reaches for those characters.

A colon is safe in a single-sentence help, and an apostrophe alongside it now
round-trips correctly (it used to emerge as `don''t`) -- `--i` and `--j` carry
their original `"Antenna 1: ..."` phrasing again. Ending a help string with a
colon is not safe.

## A fixed quirk worth keeping the shape of

`= -1` in a CLI signature used to emit `default: '-1'` — a string under
`dtype: int`. That was hip-cargo #109 and is fixed in 0.4.0. surfvis
keeps `| None = None` for every "unset" sentinel (`--i`, `--j`, `--scale`,
`--ntimes`) regardless, because it is the honest expression of "not set" and it
makes the cab nullable rather than sentinel-valued. The core functions still
honour an explicit `-1`, e.g. `ant1 = -1 if i is None else i`, so old command
lines keep working.

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
