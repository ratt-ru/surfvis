# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**surfvis** produces per-baseline diagnostics from a Measurement Set. One Typer
app, several commands:

| Command | Core module | What it does |
|---|---|---|
| `surfvis summary` | `core/summary.py` | Print the FIELD / SPECTRAL_WINDOW / ANTENNA tables. |
| `surfvis surf` | `core/surf.py` | One time/frequency PNG per baseline. |
| `surfvis chi2` | `core/chi2.py` | Per-(time, freq, corr) chi-squared images, a per-scan combination, and the zarr the browser reads. |
| `surfvis flag-chi2` | `core/flag_chi2.py` | Flag visibilities whose chi-squared exceeds a threshold, in place. |
| `surfvis serve` | `web/` | FastAPI + htmx browser over `chi2 --dataout`. Not a cab, deliberately. |
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

The heavy stack (python-casacore, dask-ms, numba, xarray-ms) is **not** in the
dev environment, so anything touching a Measurement Set skips locally. Run those
in the container:

```bash
docker build -t surfvis-web:local .
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/src -w /src \
  surfvis-web:local bash -c \
  "pip install --quiet --target /tmp/t pytest httpx2; \
   PYTHONPATH=/tmp/t:/src/src:/src /tmp/t/bin/pytest tests/ -q -W ignore"
```

That is the only way to run the full suite (22 tests). **CI does not run them** —
`.github/workflows/ci.yml` installs the lightweight package only, deliberately,
since python-casacore and numba make CI slow and brittle. Verify locally.

To exercise the web app end to end, `tests/fixtures/ms.py::make_ms` writes a
real Measurement Set in under a second; feed it to `chi2 --dataout` and point
`serve` at the result.

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
- **`web/`** is the browser: `store.py` (zarr), `msdata.py` (MS via xarray-ms),
  `render.py` (colours + PNGs), `app.py` (FastAPI). Dask-free.

`pip install surfvis` gets the CLI and cabs only; `[full]` adds the science
stack, `[web]` the browser. The Dockerfile installs `[full,web]`.

Data flow for the browser: `chi2 --dataout` writes χ² and counts per
`(time bin, freq bin, corr, antenna, antenna)`, one zarr group per
`(field, spw, scan)`. `serve` reads that for the heatmap and only opens the MS
when a waterfall is requested.

## Sharp edges

Things that have already cost time. Most are documented in depth in
`docs/wiki/`.

**The committed `cli/*.py` are generated artifacts.** `tests/test_roundtrip.py`
asserts `cli → cab → cli` is byte-identical. After changing a CLI signature:
regenerate cabs, run `generate-function` back over the cab, adopt its output
verbatim, regenerate cabs again, then test. Editing by hand and stopping there
will fail the round-trip. See `docs/wiki/cli-contract.md` for the exact loop.

**Required params are options, not positional args.** hip-cargo's reverse
generator only emits `typer.Option`, so the MS is `--ms`, not a bare argument.
The cab still says `positional: true`, which is correct — Stimela calls the core
function, where it genuinely is positional.

**Never use a negative default in a CLI signature.** hip-cargo serialises `-1`
as the *string* `'-1'` under a numeric dtype. Express "unset" as `| None = None`
and honour a legacy `-1` in the core function. (Fixed in the pinned hip-cargo
branch; the idiom stays because it is the better expression anyway.)

**Never put a colon in a multi-sentence `help=` string.** It produces
unparseable cab YAML. A single-sentence help with a colon is fine.

**hip-cargo is pinned to a git branch** (`fix-cab-yaml-emission`,
landmanbester/hip-cargo#111). Consequences: **PyPI rejects direct URL
dependencies, so `publish.yml` cannot ship a release** while the pin stands, and
the Dockerfile carries an `apt-get install git` layer purely for it. Both come
out when that lands in a release.

**`core/chi2.py` must use a spawn context** for its `ProcessPoolExecutor`.
`dask.config.set(pool=ThreadPool(...))` has started threads by then, and forking
with live threads deadlocks the children — the parent hangs in `as_completed`
forever, with no output.

**MSv4 renames columns; resolve, do not hardcode.** `DATA` is exposed as
`VISIBILITY`, `WEIGHT_SPECTRUM` as `WEIGHT`. Read them from
`attrs["data_groups"][group]`, as `pfb-imaging`'s `core/imager.py` does.
`data_groups` is attached by `open_datatree`, **not** `open_dataset`. Columns
outside the MSv2 standard (`RESIDUAL`) are not renamed.

**xarray-ms's default partition schema is wrong for us.** It omits `FIELD_ID`
and `SCAN_NUMBER`. Use `("FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER")` — it
matches both pfb-imaging and surfchi2's grouping, so one partition is one
(field, spw, scan). xarray-ms also fills gaps in its regular (time, baseline)
grid with NaN; treat those as flagged.

**`corr` in the zarr is a resolved polarization index**, not an offset into
`--use-corrs`. The default `[0, -1]` is stored as `[0, 3]`. Get this wrong and
every waterfall shows the wrong polarization.

**Colour scales default to log.** χ²/dof spans orders of magnitude; on a linear
scale one bad baseline renders every other cell black. The histogram bins follow
the same scale for the same reason.

**A synthetic MS needs more than the main table.** xarray-ms refuses to build
the MSv4 view unless `FEED` (validated against ANTENNA1/2), `STATE`, and the
`FIELD` direction columns are populated. `tests/fixtures/ms.py` does this.

**`--dataout` and `--imagesout` are deleted and recreated** on every `chi2` run.

**`serve` carries no `@stimela_cab`** and must not. A long-running GUI is not a
batch task. `generate-cabs` skips undecorated functions, which is what keeps
`serve.yml` from existing.

**The `[web]` extra needs Python 3.11+** (xarray-ms). The batch commands still
support 3.10, hence the environment marker in `pyproject.toml`.

## Direction of travel

The project is **moving away from Dask and distributed**. `core/chi2.py` and
`core/flag_chi2.py` still use dask-ms + dask arrays; the web layer already does
not. Before writing new parallel or kernel code, consult `rarg-ray-patterns`
(Ray actor autoscaling, `wrap_future` bridging `ObjectRef` to asyncio) and
`rarg-numba-patterns` (atomic spinlocks, typed pointer intrinsics). The spawn
workaround above is a band-aid over the fork+threads architecture that this
migration removes: numba atomics would let the chi-squared kernel accumulate
into one shared matrix, deleting the process pool, the pickling and the parent
reduction together.

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
