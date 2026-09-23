# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**surfvis** produces per-baseline diagnostics from a Measurement Set. One Typer
app, several commands:

| Command | Core module | What it does |
|---|---|---|
| `surfvis summary` | `core/summary.py` | Print the FIELD / SPECTRAL_WINDOW / ANTENNA tables. |
| `surfvis surf` | `core/surf.py` | One time/frequency PNG per baseline. |
| `surfvis chi2` | `core/chi2.py` | Per-(time, freq, corr) chi-squared images plus a per-scan combination. |
| `surfvis flag-chi2` | `core/flag_chi2.py` | Flag visibilities whose chi-squared exceeds a threshold, in place. |
| `surfvis onboard` | `core/onboard.py` | Prints remaining CI/CD setup steps. Delete once GitHub is configured. |

This is a [hip-cargo](https://github.com/landmanbester/hip-cargo) package: CLI
commands are decorated so Stimela cab definitions are generated from the CLI
source, and the same commands run from a recipe or the shell interchangeably.

## Commands

```bash
uv sync --group dev --group test          # dev environment (lightweight: no MS deps)
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

uv run ruff format . && uv run ruff check . --fix   # mandatory after any code change

uv run pytest                              # round-trip + install tests; MS tests skip
uv run pytest tests/test_roundtrip.py::test_roundtrip_chi2 -v   # one test

uv run hip-cargo generate-cabs --module 'src/surfvis/cli/*.py' --output-dir src/surfvis/cabs
```

The heavy stack (python-casacore, dask-ms, numba) is **not** in the dev
environment, so anything touching a Measurement Set skips locally. Run those in
the container:

```bash
docker build -t surfvis:local .
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/src -w /src \
  surfvis:local bash -c \
  "pip install --quiet --target /tmp/t pytest; \
   PYTHONPATH=/tmp/t:/src/src:/src /tmp/t/bin/pytest tests/ -q -W ignore"
```

That is the only way to run the full suite (15 tests). **CI does not run them** —
`.github/workflows/ci.yml` installs the lightweight package only, deliberately,
since python-casacore and numba make CI slow and brittle. Verify locally.

`tests/fixtures/ms.py::make_ms` writes a real Measurement Set in under a second,
with a deliberately bad baseline for the flagging tests to find.

## Architecture

Three layers, and the boundary between them is load-bearing:

- **`cli/`** imports only `typer` and `hip_cargo`. It lazy-imports `core/`
  inside `try/except ImportError`; on failure `run_in_container()` re-runs the
  command inside `ghcr.io/ratt-ru/surfvis`. Adding a heavy import here silently
  breaks the lightweight install and the container fallback.
- **`core/`** holds implementations and imports freely (numpy, dask, daskms,
  matplotlib, pyrap, numba).
- **`cabs/`** is generated. Never edit by hand.
- **`utils/`** is heavy too (`chisq.py` numba kernels, `plotting.py` matplotlib)
  and is imported only from `core/`.

`pip install surfvis` gets the CLI and cabs only; `[full]` adds the science
stack. The Dockerfile installs `[full]`.

## Sharp edges

Things that have already cost time. Most are documented in depth in
`docs/wiki/`.

**The committed `cli/*.py` are generated artifacts.** `tests/test_roundtrip.py`
asserts `cli → cab → cli` is byte-identical. After changing a CLI signature:
regenerate cabs, run `generate-function` back over the cab, adopt its output
verbatim, regenerate cabs again, then test. Editing by hand and stopping there
will fail the round-trip. See `docs/wiki/cli-contract.md` for the exact loop.

**The dialect is closed, and writing the cab is the easier direction.** A cab
and a CLI module are two renderings of one definition; hip-cargo generates
either from the other, and the round-trip test enforces the bijection. So a
typer construct with no cab representation is not part of the language:
`typer.Argument` and `typer.Option("-x", "--ex", ...)` both raise at parse time
with a message naming the rule. Required params are therefore options (`--ms`,
not a bare argument) and there are no short flags — properties of the format,
not compromises. If editing generated Python feels like fighting the tool,
write the YAML cab and run `generate-function` instead. Upstream:
hip-cargo's `docs/wiki/cli-dialect.md`.

**Prefer `| None = None` to a sentinel default.** surfvis uses it for `--i`,
`--j`, `--scale`, `--ntimes`: it makes the cab nullable rather than
sentinel-valued, and the core functions still honour an explicit `-1` so old
command lines keep working. (A negative default also used to serialise as the
*string* `'-1'` — hip-cargo #109, fixed in 0.4.0. The idiom is worth keeping on
its own merits.)

**Two help strings still emit silently-wrong cab YAML.** Verified against
hip-cargo 0.4.0:

| `help=` | Result |
|---|---|
| ends in a colon, as the only sentence | `info` becomes a **dict** |
| contains non-ASCII (`°`, `λ`, `μ`) | escape survives as literal `\xB0` |

Multi-sentence help containing `": "`, `" #"` or a leading `"- "` raises at
generation instead, which is fine. Non-ASCII is the one to watch — radio
astronomy help text reaches for those characters.

**`core/chi2.py` must use a spawn context** for its `ProcessPoolExecutor`.
`dask.config.set(pool=ThreadPool(...))` has started threads by then, and forking
with live threads deadlocks the children — the parent hangs in `as_completed`
forever, with no output. The failure is load-dependent, so it presents as an
intermittent hang.

**`--dataout` and `--imagesout` are deleted and recreated** on every `chi2` run.
`--dataout` is currently vestigial: the directory is removed and nothing is
written to it. Implementing it is the subject of the `serve-ui` branch.

**The fixture MS populates more subtables than casacore needs.** `FEED`, `STATE`
and the `FIELD` direction columns are there because an MSv4 reader refuses to
build its view without them. Keep them when editing the fixture.

**A green test run proves less than it looks.** The heavy module skips at
*import*, so pytest reports one skip for the whole file: a lightweight run says
"10 passed, 1 skipped" while 5 tests did not run.
`tests/test_suite_integrity.py` pins the per-module test counts statically so a
deletion fails loudly; update those counts deliberately when adding or removing
a test.

**Expect a regeneration when hip-cargo #114 lands.** `generate-function` emits
`MS = NewType("MS", Path)` while hip-cargo documents those types as UPath-backed
— false for a remote URI, since `S3Path` is not a `pathlib.Path` subclass. The
fix changes generated CLI source, so every `cli/*.py` here will need
regenerating with it.

## Direction of travel

The project is **moving away from Dask and distributed**. `core/chi2.py` and
`core/flag_chi2.py` still use dask-ms + dask arrays. Before writing new parallel
or kernel code, consult `rarg-ray-patterns` (Ray actor autoscaling,
`wrap_future` bridging `ObjectRef` to asyncio) and `rarg-numba-patterns` (atomic
spinlocks, typed pointer intrinsics). The spawn workaround above is a band-aid
over the fork+threads architecture that this migration removes: numba atomics
would let the chi-squared kernel accumulate into one shared matrix, deleting the
process pool, the pickling and the parent reduction together.

## LLM wiki

`docs/wiki/` is the canonical reference for what is implemented; start at
`docs/wiki/index.md`. Each page's frontmatter carries a `last_verified_commit`.

**Update-as-you-touch:** if a change invalidates or extends a page, update it,
refresh its `last_verified_commit` (`git rev-parse --short HEAD`) and
`timestamp` in the same session, and add a line to `docs/wiki/log.md`.

| Rule file | Read it when editing |
|---|---|
| `.claude/rules/architecture.md` | `src/surfvis/**` — package layout, install modes, container fallback, cab generation. |
| `.claude/rules/python-standards.md` | any `**/*.py` — type hints, lazy imports, Typer syntax, hip-cargo types. |
| `.claude/rules/testing-and-ci.md` | `tests/**` or `.github/workflows/**` — round-trip tests, dev workflow, commits. |

Commits follow Conventional Commits; the changelog is generated from them by
git-cliff, and a `commit-msg` hook enforces the prefixes.
