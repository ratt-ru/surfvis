---
type: reference
title: Chi-squared pipeline
description: How chi2 and flag-chi2 chunk a Measurement Set, what the numba kernels compute, how results are reduced, and where images land.
tags: [chi2, flagging, dask, numba, daskms]
timestamp: 2026-10-14
last_verified_commit: 85c6bcd
---

# Chi-squared pipeline

Source: `src/surfvis/core/chi2.py`, `src/surfvis/core/flag_chi2.py`,
`src/surfvis/utils/chisq.py`, `src/surfvis/utils/plotting.py`.

## The quantity

Both commands compute, per visibility, `chi2 = real(conj(r) * w * r)` from a
residual `r` and a weight `w`. Passing `--wcol SIGMA_SPECTRUM` switches the
weight to `1 / sigma**2`. Flagged visibilities and auto-correlations (`p == q`)
are excluded from the sums.

For `chi2`, `r` need not be a column. Per 2x2 visibility it is

```
r = inv(G_p) R inv(G_q)^H - M
```

with `R` the `--rcol` column, `M` the optional `--mcol` and `G` the optional
`--gtable`. So `--rcol DATA --mcol MODEL_DATA --gtable gains.qc/NET` forms the
residual on the fly, and `--rcol RESIDUAL --gtable ...` gives the same thing.
`utils/products.py` does the algebra in plain numpy (shared with the browser):
weights are propagated exactly (`w |g_p|^2 |g_q|^2` for diagonal gains,
`1 / sum |A_ik|^2 |B_jl|^2 / w_kl` for full Jones), gain flags OR into the data
flags, and a singular gain flags its visibilities. For diagonal gains the
chi-squared is therefore *unchanged* by correction; a test pins that.

`--stokes I,V` adds Stokes products formed from the corrected correlations,
with weight `1 / sum_k |a_k|^2 / w_k` -- the exact inverse variance. radiomesh's
generated `WEIGHT_*` expressions are imaging weights (`w00 + w11` for I) and
agree with that only for equal correlation weights. A product the MS cannot
form (V from linear feeds without XY/YX) raises before any work starts.

Gains come from a QuartiCal gain term (`utils/gains.py`): one zarr group per
partition under `gains.qc/NET/NET_<n>`, read with plain xarray. They apply
**on-grid only** -- each gain axis must match the scan's unique `TIME` values
and the channel frequencies, or have length one. Antennas missing from the
table are flagged.

## `chi2` — imaging the chi-squared surface

1. **Pass one** reads only `TIME` and the flag column, grouped by
   `(FIELD_ID, DATA_DESC_ID, SCAN_NUMBER)`, to derive the chunking. `--ntimes`
   unique times per chunk (default: all of them) and `--nfreqs` channels per
   chunk (default 128) give the row/time/frequency bin indices and counts.
2. **Pass two** re-opens the MS with those chunks and the residual, weight,
   flag, optional model, `ANTENNA1`, `ANTENNA2` and `TIME` columns, keeping
   only the correlations the requested products need (all four for full-Jones
   gains).
3. Every `(time bin, freq bin)` pair is submitted to a
   `concurrent.futures.ProcessPoolExecutor` (`--nthreads` workers) running
   `utils.plotting.surfchisq_products_plot`, along with that chunk's slice of
   the gains. The worker computes its dask slices with the **synchronous**
   scheduler (it is already inside a worker process), forms every product with
   `utils.products.form`, calls the numba kernel `_surfchisq_slice` once per
   product, writes one PNG per product, and returns `(chi2, counts)` shaped
   `(nproducts, nant, nant)`. The parent keys futures by `(ds, t, f)`.
4. The parent accumulates those into per-scan cubes and writes `combined.png`
   per scan from the correlation products, plus `combined_<S>.png` for each
   Stokes product, as `chi2 / counts` with `NaN` where `counts == 0`.

Before this, one task per `(time, freq, corr)` was submitted but each was given
*all* of `--use-corrs` -- a bug inherited from surfchi2 -- so every
per-correlation image was the same sum, computed `ncorr` times.

Output layout, rooted at `--imagesout` (default `$CWD/chi2`):

```
<imagesout>/field<F>/spw<S>/scan<N>/t<t>_f<f>_<product>.png   # one per chunk and product
<imagesout>/field<F>/spw<S>/scan<N>/combined.png               # correlations, per scan
<imagesout>/field<F>/spw<S>/scan<N>/combined_<S>.png           # each Stokes product, per scan
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
| `_surfchisq_slice` | `@njit(nogil=True)` | `(nant, nant)` chi2 and counts for one already-sliced chunk. The one the pipeline actually uses, once per product. |
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
multi-correlation MS, `[0]` otherwise. In `chi2` these resolve to labels
(`XX`, `YY`) via `POLARIZATION::CORR_TYPE`, and `--stokes` appends Stokes
products; `--use-corrs` does not restrict which Stokes can be formed. Every
spectral window must have the same correlations. `CORR_TYPE` and `CHAN_FREQ`
are read with arcae (`_spectral_setup`), not python-casacore, which makes
`chi2` need Python 3.11+.
