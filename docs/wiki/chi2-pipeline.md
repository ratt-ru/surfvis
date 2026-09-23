---
type: reference
title: Chi-squared pipeline
description: How chi2 and flag-chi2 chunk a Measurement Set, what the numba kernels compute, how results are reduced, and where images land.
tags: [chi2, flagging, dask, numba, daskms]
timestamp: 2026-09-18
last_verified_commit: PENDING
---

# Chi-squared pipeline

Source: `src/surfvis/core/chi2.py`, `src/surfvis/core/flag_chi2.py`,
`src/surfvis/utils/chisq.py`, `src/surfvis/utils/plotting.py`.

## The quantity

Both commands compute, per visibility, `chi2 = real(conj(r) * w * r)` from a
residual column `r` and a weight column `w`. Passing
`--wcol SIGMA_SPECTRUM` switches the weight to `1 / sigma**2`. Flagged
visibilities and auto-correlations (`p == q`) are excluded from the sums.

## `chi2` — imaging the chi-squared surface

1. **Pass one** reads only `TIME` and the flag column, grouped by
   `(FIELD_ID, DATA_DESC_ID, SCAN_NUMBER)`, to derive the chunking. `--ntimes`
   unique times per chunk (default: all of them) and `--nfreqs` channels per
   chunk (default 128) give the row/time/frequency bin indices and counts.
2. **Pass two** re-opens the MS with those chunks and the residual, weight,
   flag, `ANTENNA1`, `ANTENNA2` and `TIME` columns.
3. Every `(time bin, freq bin, corr)` triple is submitted to a
   `concurrent.futures.ProcessPoolExecutor` (`--nthreads` workers) running
   `utils.plotting.surfchisq_plot`. That function computes its dask slices with
   the **synchronous** scheduler (it is already inside a worker process), calls
   the numba kernel `_surfchisq_slice`, writes one PNG, and returns
   `(field, spw, scan, chi2, counts)`.
4. The parent accumulates those returns into per-`field/spw/scan` arrays and
   finally writes a `combined.png` per scan from `chi2 / counts`, with `NaN`
   where `counts == 0`.

Output layout, rooted at `--imagesout` (default `$CWD/chi2`):

```
<imagesout>/field<F>/spw<S>/scan<N>/t<t>_f<f>_c<c>.png   # one per chunk
<imagesout>/field<F>/spw<S>/scan<N>/combined.png         # per-scan reduction
```

`--imagesout` and `--dataout` are both **deleted and recreated** on every run.
`--dataout` now writes the zarr its name always implied — see
[web-app.md](web-app.md) for the layout. It is what `surfvis serve` reads.

The `--nthreads` value does double duty: it sizes the process pool *and* the
dask `ThreadPool` set via `dask.config`.

**That pool must use a spawn context.** The `ThreadPool` above is already
running by the time the executor is built, and forking a process with live
threads deadlocks the children — the parent waits in `as_completed` forever,
printing nothing. See [web-app.md](web-app.md) §The fork deadlock.

## `flag-chi2` — flagging in place

Simpler: one pass over the MS chunked by `--nrows` rows and `--nfreqs`
channels. `utils.chisq.flagchisq` wraps the numba kernel `_flagchisq` in a
`da.blockwise`. A visibility whose chi-squared exceeds `--flag-above`, **or is
exactly zero**, flags the whole `(row, chan)` cell across every correlation.
Baselines touching any antenna in `--respect-ants` are skipped entirely.

`FLAG_ROW` is recomputed as `all(flag)` over `(chan, corr)` and both columns are
written back with `xds_to_storage_table`. This **modifies the input MS**, which
is why the cab marks `ms` as `writable: true`.

## Kernels

| Symbol | Kind | Notes |
|---|---|---|
| `_surfchisq_slice` | `@njit(nogil=True)` | `(nant, nant)` chi2 and counts for one already-sliced chunk. The one the pipeline actually uses. |
| `_surfchisq` / `surfchisq` | `@njit` + `da.blockwise` | Whole-array variant producing `(t, f, corr, p, q, 2)`. Retained from the original codebase but **not called** by any command. |
| `_flagchisq` / `flagchisq` | `@njit(fastmath=True)` + `da.blockwise` | Mutates and returns the flag array. |

All of them fill both triangles (`out[q, p] = out[p, q]`), so the resulting
matrices are symmetric.

Ruff's `N806` is disabled for `utils/chisq.py` in `pyproject.toml`: the kernels
keep the upper-case names of the maths they implement (`R`, `W`, `F` for
residual/weight/flag; `Ip`, `Iq`, `Ipq` for the antenna index masks).

## Correlation selection

`--use-corrs` takes a comma-separated list of correlation indices. When it is
omitted both commands fall back to the diagonal correlations: `[0, -1]` for a
multi-correlation MS, `[0]` otherwise.
