---
type: reference
title: Migration from the optparse entry points
description: Option-by-option mapping from the pre-hip-cargo surfvis/surfchi2/flagchi2 scripts to the single surfvis Typer app.
tags: [migration, cli, breaking-changes]
timestamp: 2026-09-18
last_verified_commit: c3bd176
---

# Migration from the optparse entry points

Before the hip-cargo conversion, surfvis installed three console scripts built
on `optparse`: `surfvis`, `surfchi2` and `flagchi2`. It now installs **one**:
`surfvis`, a Typer app with subcommands. The old executables are gone.

## Command mapping

| Was | Is now |
|---|---|
| `surfvis --list MS` | `surfvis summary --ms MS` |
| `surfvis MS ...` | `surfvis surf --ms MS ...` |
| `surfchi2 MS ...` | `surfvis chi2 --ms MS ...` |
| `flagchi2 MS ...` | `surfvis flag-chi2 --ms MS ...` |

The Measurement Set moved from a positional argument to a required `--ms`
option. That is forced by hip-cargo's reverse generator, which only emits
`typer.Option` — see [cli-contract.md](cli-contract.md). Stimela still receives
it positionally, because cabs use `flavour: python` and call the core function.

## `surfvis` → `surfvis surf`

Long option names are unchanged. The single-dash short forms (`-d`, `-f`, `-s`,
`-p`, `-i`, `-j`, `-o`) are **gone**: hip-cargo derives flag names from the
Python parameter name and does not support explicit `param_decls`.

| Old | New | Change |
|---|---|---|
| `-l`, `--list` | — | Split out into `surfvis summary`. |
| `-d`, `--datacolumn` | `--datacolumn` | — |
| `-f`, `--field` | `--field` | — |
| `-s`, `--spw` | `--spw` | Now `List[int]`. `--spw 0,1` as before; omit the flag for "all" instead of passing `--spw ''`. |
| `-p`, `--plot` | `--plot` | Now a `Literal`, validated by Typer against `amp|phase|real|imag`. |
| `-i`, `--i` | `--i` | Defaults to `None` (all antennas) rather than `-1`. `-1` is still accepted. |
| `-j`, `--j` | `--j` | As above. |
| `--noflags` | `--noflags` / `--no-noflags` | Typer renders booleans as a flag pair. |
| `--doacorr` | `--doacorr` / `--no-doacorr` | As above. |
| `--scale` | `--scale` | Defaults to `None` (scale to 5 sigma) rather than `-1`. `-1` is still accepted. |
| `--cmap` | `--cmap` | — |
| `-o`, `--opdir` | `--opdir` | Defaults to `None`, which still resolves to `<ms>_<datacolumn>__plots`. |

## `surfchi2` → `surfvis chi2`

All long names unchanged: `--rcol`, `--wcol`, `--fcol`, `--dataout`,
`--imagesout`, `--nthreads`, `--ntimes`, `--nfreqs`, `--use-corrs`.

- `--ntimes` defaults to `None` rather than `-1`; both mean "all unique times".
- `--use-corrs` is typed `List[int]` rather than a raw string, so the cab
  advertises the right dtype. The command line form (`--use-corrs 0,3`) is
  identical.
- `--dataout`/`--imagesout` default to `None` rather than `''`; both still
  resolve to `$CWD/chi2`.

## `flagchi2` → `surfvis flag-chi2`

All long names unchanged: `--rcol`, `--wcol`, `--fcol`, `--flag-above`,
`--nthreads`, `--nrows`, `--nfreqs`, `--use-corrs`, `--respect-ants`.
`--use-corrs` and `--respect-ants` are now `List[int]`, same command-line form.

## Behavioural notes

- Output directories are created with `Path.mkdir(parents=True)` and removed
  with `shutil.rmtree` instead of `os.system("mkdir ...")` /
  `os.system("rm -r ...")`.
- `surfvis surf` raises `ValueError` on an invalid `--plot` instead of calling
  `sys.exit(-1)`, so Stimela sees a real exception.
- `surfvis.utils.surf()`, a wrapper that referenced the Python-2-only
  `StandardError` and could never have run, was dropped. Everything else in the
  old `surfvis/utils.py` survives, split between `utils/chisq.py` and
  `utils/plotting.py`.
- `chi2` no longer re-raises inside its `as_completed` loop via an explicit
  `try/except`; a failing future propagates directly, which is the same
  observable behaviour with a shorter traceback.

## Packaging

`setup.py` is gone. The package is PEP 621 (`pyproject.toml`, `uv_build`) and
installs lightweight by default. The heavy stack — `python-casacore`,
`dask-ms`, `numba`, `matplotlib`, `astropy`, `xarray`, `dask` — lives in the
`full` extra:

```bash
pip install surfvis          # CLI + cabs only; commands run in a container
pip install 'surfvis[full]'  # run natively
```
