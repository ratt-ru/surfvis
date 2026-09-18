# CLAUDE.md — Project Context

## Project Overview

**surfvis** produces per-baseline diagnostics from a Measurement Set. Four
commands, one Typer app:

| Command | Core module | What it does |
|---|---|---|
| `surfvis summary` | `core/summary.py` | Print the FIELD / SPECTRAL_WINDOW / ANTENNA tables. |
| `surfvis surf` | `core/surf.py` | One time/frequency PNG per baseline. |
| `surfvis chi2` | `core/chi2.py` | Per-(time, freq, corr) chi-squared images plus a per-scan combination. |
| `surfvis flag-chi2` | `core/flag_chi2.py` | Flag visibilities whose chi-squared exceeds a threshold, in place. |
| `surfvis serve` | `web/` | FastAPI + htmx browser over `chi2 --dataout`. Not a cab, deliberately. |

`onboard` is the scaffolding command that prints the remaining CI/CD setup
steps. Delete it once GitHub is fully configured.

The numerical work lives in `src/surfvis/utils/`: `chisq.py` holds the numba
kernels and the dask `blockwise` wrappers, `plotting.py` the matplotlib
helpers. Both import heavy third-party packages, so they are only ever
imported from `core/`, never from `cli/`.

## LLM Wiki (canonical implementation reference)

Deep reference documentation lives in `docs/wiki/` (start at
`docs/wiki/index.md`). Each page's frontmatter carries a
`last_verified_commit` stamp — the commit its claims were last checked
against.

* **Read the relevant wiki page before working in a subsystem.**
* **Update-as-you-touch rule:** if a change you make invalidates or extends a
  wiki page, update that page and refresh its `last_verified_commit`
  (`git rev-parse --short HEAD`) and `timestamp` in the same session, and add
  a line to `docs/wiki/log.md`.

## Mandatory Development Workflow

**Always run linting after adding or modifying any code:**

```bash
uv run ruff format . && uv run ruff check . --fix
```

**Always regenerate cabs after touching anything under `src/surfvis/cli/`:**

```bash
uv run hip-cargo generate-cabs --module 'src/surfvis/cli/*.py' --output-dir src/surfvis/cabs
uv run pytest tests/test_roundtrip.py
```

The pre-commit hook does the first of these for you; the round-trip test is
what catches a CLI module that cab generation cannot reproduce.

---

This project was bootstrapped with [`hip-cargo init`](https://github.com/landmanbester/hip-cargo).
It is a **hip-cargo package**: a Python CLI whose commands are decorated so that
Stimela cab definitions are generated automatically from the CLI source. Cabs
let the same commands be invoked from Stimela recipes and from `surfvis`
on the command line interchangeably.

When working in this repo, treat the patterns in the rules below as load-bearing
— they are what makes the round-trip between CLI source, generated cabs, and
container fallback work. If you find yourself wanting to deviate, stop and check
[hip-cargo's own docs](https://github.com/landmanbester/hip-cargo) first.

*Note: Detailed architecture/domain logic, Python standards, and testing/CI
rules have been modularized into the `.claude/rules/` directory for progressive
disclosure. Read the relevant file before editing the matching files.*

| Rule file | Read it when editing |
|---|---|
| `.claude/rules/architecture.md` | `src/surfvis/**` — package layout, install modes, container fallback, cab generation. |
| `.claude/rules/python-standards.md` | any `**/*.py` — type hints, lazy imports, Typer syntax, hip-cargo types. |
| `.claude/rules/testing-and-ci.md` | `tests/**` or `.github/workflows/**` — round-trip tests, dev workflow, commits. |

---

## Where to Go Deeper

- hip-cargo source & docs: <https://github.com/landmanbester/hip-cargo>
- Stimela: <https://github.com/caracal-pipeline/stimela>
- Twelve-factor principles guide most architectural decisions in this repo:
  <https://12factor.net/>
