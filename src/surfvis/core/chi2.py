"""Per-baseline chi-squared plots from a Measurement Set."""

import concurrent.futures as cf
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import dask
import numpy as np
import pyrap.tables
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_from_storage_table as xds_from_table

from surfvis.utils import products as products_mod
from surfvis.utils.plotting import makeplot, surfchisq_products_plot


def chi2(
    ms: Path,
    rcol: str = "RESIDUAL",
    wcol: str = "WEIGHT_SPECTRUM",
    fcol: str = "FLAG",
    mcol: str | None = None,
    gtable: Path | None = None,
    dataout: Path | None = None,
    imagesout: Path | None = None,
    nthreads: int = 4,
    ntimes: int | None = None,
    nfreqs: int = 128,
    use_corrs: list[int] | None = None,
    stokes: list[str] | None = None,
) -> None:
    """Write per-(time, freq, product) chi-squared images plus a per-scan combination.

    The quantity is ``inv(G_p) R inv(G_q)^H - M`` per visibility, ``R`` being
    ``rcol``, ``M`` the optional ``mcol`` and ``G`` the optional ``gtable``,
    with weights propagated through the correction (see
    :mod:`surfvis.utils.products`).

    Args:
        ms: Measurement Set to inspect.
        rcol: Residual column, or the data column when ``mcol`` is given.
        wcol: Weight column. ``SIGMA_SPECTRUM`` initialises weights as 1/sigma**2.
        fcol: Flag column.
        mcol: Model column, subtracted after correction.
        gtable: QuartiCal gain term (``gains.qc/NET``) to correct by.
        dataout: Output name of the zarr dataset holding chi-squared and counts
            per (time bin, freq bin, product, antenna, antenna). Defaults to
            ``$CWD/chi2.zarr``. This is what ``surfvis serve`` reads.
        imagesout: Output folder for images. Defaults to ``$CWD/chi2``.
        nthreads: Number of worker processes (also the dask thread-pool size).
        ntimes: Number of unique times in each chunk. ``None`` means all of them.
        nfreqs: Number of frequencies in a chunk. ``-1`` means all of them.
        use_corrs: Correlations to use. Defaults to the diagonal correlations.
        stokes: Stokes parameters to compute as well, e.g. ``["I", "V"]``.
    """
    from multiprocessing.pool import ThreadPool

    msname = str(ms).rstrip("/")
    gtable = str(gtable) if gtable is not None else None

    dataout = Path(str(dataout)) if dataout is not None else Path(os.getcwd()) / "chi2.zarr"
    if dataout.is_dir():
        print(f"Removing existing {dataout} folder")
        shutil.rmtree(dataout)

    imagesout = Path(str(imagesout)) if imagesout is not None else Path(os.getcwd()) / "chi2"
    if imagesout.is_dir():
        print(f"Removing existing {imagesout} folder")
        shutil.rmtree(imagesout)

    dask.config.set(pool=ThreadPool(nthreads))

    # chunking info
    schema = {fcol: {"dims": ("chan", "corr")}}
    xds = xds_from_ms(
        msname,
        chunks={"row": -1},
        columns=["TIME", fcol],
        group_cols=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"],
        table_schema=schema,
    )

    chunks = []
    utimes = []
    rbin_idx = []
    rbin_counts = []
    tbin_idx = []
    tbin_counts = []
    fbin_idx = []
    fbin_counts = []
    for ds in xds:
        time = ds.TIME.values
        ut, counts = np.unique(time, return_counts=True)
        utimes.append(ut)
        utpc = ut.size if ntimes in [None, 0, -1] else ntimes
        row_chunks = [np.sum(counts[i : i + utpc]) for i in range(0, ut.size, utpc)]

        nchan = ds.chan.size
        if nfreqs in [0, -1]:
            nfreqs = nchan

        # list per ds
        chunks.append({"row": tuple(row_chunks), "chan": nfreqs})

        ridx = np.zeros(len(row_chunks))
        ridx[1:] = np.cumsum(row_chunks)[0:-1]
        rbin_idx.append(ridx.astype(int))
        rbin_counts.append(row_chunks)

        ntime = ut.size
        tidx = np.arange(0, ntime, utpc)
        tbin_idx.append(tidx.astype(int))
        tidx2 = np.append(tidx, ntime)
        tbin_counts.append(tidx2[1:] - tidx2[0:-1])

        fidx = np.arange(0, nchan, nfreqs)
        fbin_idx.append(fidx)
        fidx2 = np.append(fidx, nchan)
        fbin_counts.append(fidx2[1:] - fidx2[0:-1])

    columns = [rcol, wcol, fcol] + ([mcol] if mcol else [])
    schema = {col: {"dims": ("chan", "corr")} for col in columns}

    xds = xds_from_ms(
        msname,
        columns=columns + ["ANTENNA1", "ANTENNA2", "TIME"],
        chunks=chunks,
        group_cols=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"],
        table_schema=schema,
    )

    spw_freqs, corr_types = _spectral_setup(msname)
    labels = products_mod.corr_labels(corr_types[int(xds[0].DATA_DESC_ID)])
    for ds in xds:
        if products_mod.corr_labels(corr_types[int(ds.DATA_DESC_ID)]) != labels:
            raise ValueError("surfvis chi2 needs the same correlations in every spectral window")

    ncorr_total = len(labels)
    if not use_corrs:
        print("Using only diagonal correlations")
        corrs = [0, -1] if ncorr_total > 1 else [0]
    else:
        corrs = list(use_corrs)
        print(f"Using correlations {corrs}")
    # -1 and friends are positional; resolve them to labels so the zarr records
    # real polarizations, not offsets into use_corrs.
    products = list(dict.fromkeys(labels[c % ncorr_total] for c in corrs))
    products += [s for s in dict.fromkeys(s.upper() for s in (stokes or [])) if s not in products]
    print(f"Products {products}")

    ant_ds = xds_from_table(f"{msname}::ANTENNA")[0]
    antenna_names = [str(n) for n in ant_ds.NAME.values]
    nant = len(antenna_names)

    gains = {}
    if gtable is not None:
        from surfvis.utils import gains as gains_mod

        print(f"Applying gains from {gtable}: {gains_mod.describe(gtable)}")
        for i, ds in enumerate(xds):
            gains[i] = gains_mod.load(
                gtable,
                int(ds.FIELD_ID),
                int(ds.DATA_DESC_ID),
                int(ds.SCAN_NUMBER),
                utimes[i],
                spw_freqs[int(ds.DATA_DESC_ID)],
                antenna_names,
            )
    full_jones = any(g.full for g in gains.values())
    read = products_mod.needed(products, labels, full_jones=full_jones)
    read_idx = [labels.index(c) for c in read]
    nprod = len(products)

    # Per-chunk cubes, so the zarr keeps what the PNGs throw away.
    cubes = {}
    keys = {}
    futures = {}
    foldername = str(imagesout).rstrip("/")
    # Spawn, not fork: dask's ThreadPool is already running by this point, and
    # forking a process with live threads deadlocks the children.
    with cf.ProcessPoolExecutor(max_workers=nthreads, mp_context=mp.get_context("spawn")) as executor:
        for i, ds in enumerate(xds):
            field = ds.FIELD_ID
            spw = ds.DATA_DESC_ID
            scan = ds.SCAN_NUMBER

            basename = foldername + f"/field{field}" + f"/spw{spw}" + f"/scan{scan}/"
            odir = Path(basename).resolve()
            odir.mkdir(parents=True, exist_ok=True)

            ntime = tbin_idx[i].size
            nfreq = fbin_idx[i].size
            for t in range(ntime):
                for f in range(nfreq):
                    t0 = tbin_idx[i][t]
                    tf = t0 + tbin_counts[i][t]
                    chan0 = fbin_idx[i][f]
                    chanf = chan0 + fbin_counts[i][f]
                    row0 = rbin_idx[i][t]
                    rowf = rbin_idx[i][t] + rbin_counts[i][t]
                    dso = ds[{"row": slice(row0, rowf), "chan": slice(chan0, chanf)}]
                    dso = dso.isel(corr=read_idx)
                    resid = dso.get(rcol).data
                    if wcol == "SIGMA_SPECTRUM":
                        weight = 1.0 / dso.get(wcol).data ** 2
                    else:
                        weight = dso.get(wcol).data
                    fut = executor.submit(
                        surfchisq_products_plot,
                        products,
                        read,
                        resid,
                        weight,
                        dso.get(fcol).data,
                        dso.ANTENNA1.data,
                        dso.ANTENNA2.data,
                        dso.TIME.data,
                        dso.get(mcol).data if mcol else None,
                        gains[i].subset(t0, tf, chan0, chanf) if gains else None,
                        utimes[i][t0:tf],
                        [basename + f"t{t}_f{f}_{p}.png" for p in products],
                        [f"t {t0}-{tf}, chan {chan0}-{chanf}, {p}" for p in products],
                    )
                    futures[fut] = (i, t, f)

            keys[i] = (int(field), int(spw), int(scan))
            cubes[i] = (
                np.zeros((ntime, nfreq, nprod, nant, nant), dtype=float),
                np.zeros((ntime, nfreq, nprod, nant, nant), dtype=float),
            )
            print(f"Submitted field{field}_spw{spw}_scan{scan}")

        num_completed = 0
        num_futures = len(futures)
        for fut in cf.as_completed(futures):
            num_completed += 1
            print(f"\rProcessing: {num_completed}/{num_futures}", end="", flush=True)
            chi2_chunk, count = fut.result()
            i, t, f = futures[fut]
            chi2_cube, count_cube = cubes[i]
            # A chunk only spans the antennas present in it; pad into the full grid.
            na = chi2_chunk.shape[-1]
            chi2_cube[t, f, :, :na, :na] = chi2_chunk
            count_cube[t, f, :, :na, :na] = count
        print()

    _write_zarr_dataset(
        dataout,
        cubes,
        keys,
        tbin_idx,
        tbin_counts,
        fbin_idx,
        fbin_counts,
        products,
        antenna_names,
        msname,
        {
            "rcol": rcol,
            "wcol": wcol,
            "fcol": fcol,
            "mcol": mcol or "",
            "gtable": gtable or "",
            "corr_labels": list(labels),
        },
    )

    # Correlations combine into one image per scan, as before; Stokes products
    # are not independent of them, so each gets its own.
    print("Plotting per scan")
    is_corr = np.array([p in labels for p in products])
    for i, (chi2_cube, count_cube) in cubes.items():
        field, spw, scan = keys[i]
        basename = foldername + f"/field{field}" + f"/spw{spw}" + f"/scan{scan}/"
        groups = [("combined.png", is_corr, f"scan {scan}")]
        groups += [
            (f"combined_{p}.png", np.array([q == p for q in products]), f"scan {scan} {p}")
            for p in products
            if p not in labels
        ]
        for name, sel, title in groups:
            if not sel.any():
                continue
            val = chi2_cube[:, :, sel].sum(axis=(0, 1, 2))
            count = count_cube[:, :, sel].sum(axis=(0, 1, 2))
            chi2_dof = np.full_like(val, np.nan)
            chi2_dof[count > 0] = val[count > 0] / count[count > 0]
            makeplot(chi2_dof, basename + name, title)


def _spectral_setup(msname: str) -> tuple[dict[int, np.ndarray], dict[int, list[int]]]:
    """Channel frequencies and correlation types per DATA_DESC_ID."""
    ddt = pyrap.tables.table(f"{msname}::DATA_DESCRIPTION", ack=False)
    spwt = pyrap.tables.table(f"{msname}::SPECTRAL_WINDOW", ack=False)
    polt = pyrap.tables.table(f"{msname}::POLARIZATION", ack=False)
    try:
        freqs, corrs = {}, {}
        for ddid in range(ddt.nrows()):
            freqs[ddid] = np.asarray(spwt.getcell("CHAN_FREQ", int(ddt.getcell("SPECTRAL_WINDOW_ID", ddid))))
            corrs[ddid] = [int(c) for c in polt.getcell("CORR_TYPE", int(ddt.getcell("POLARIZATION_ID", ddid)))]
    finally:
        for t in (ddt, spwt, polt):
            t.close()
    return freqs, corrs


def _write_zarr_dataset(
    dataout,
    cubes,
    keys,
    tbin_idx,
    tbin_counts,
    fbin_idx,
    fbin_counts,
    products,
    antenna_names,
    msname,
    columns,
):
    """Write the per-chunk chi-squared cubes to a zarr store.

    One group per (field, spw, scan), because the number of time and frequency
    bins differs between them. Bin bounds ride along as coordinates so a reader
    can locate a chunk inside the scan's waterfall without re-deriving the
    chunking. ``columns`` records how the quantity was formed, so a reader can
    reproduce it.
    """
    import xarray as xr

    print(f"Writing chi-squared dataset to {dataout}")
    nant = len(antenna_names)
    attrs = {
        "ms": str(msname),
        **columns,
        "antenna_names": list(antenna_names),
    }

    for i, (chi2_cube, count_cube) in cubes.items():
        field, spw, scan = keys[i]
        ntime, nfreq, nprod = chi2_cube.shape[:3]
        t0 = np.asarray(tbin_idx[i][:ntime], dtype=int)
        tf = t0 + np.asarray(tbin_counts[i][:ntime], dtype=int)
        chan0 = np.asarray(fbin_idx[i][:nfreq], dtype=int)
        chanf = chan0 + np.asarray(fbin_counts[i][:nfreq], dtype=int)

        dims = ("time_bin", "freq_bin", "product", "antenna1", "antenna2")
        ds = xr.Dataset(
            data_vars={
                "chi2": (dims, chi2_cube),
                "counts": (dims, count_cube),
                "t0": ("time_bin", t0),
                "tf": ("time_bin", tf),
                "chan0": ("freq_bin", chan0),
                "chanf": ("freq_bin", chanf),
            },
            coords={
                "time_bin": np.arange(ntime),
                "freq_bin": np.arange(nfreq),
                "product": np.asarray(products[:nprod], dtype=str),
                "antenna1": np.arange(nant),
                "antenna2": np.arange(nant),
            },
            attrs={**attrs, "field": field, "spw": spw, "scan": scan},
        )
        ds.to_zarr(dataout, group=f"field{field}/spw{spw}/scan{scan}", mode="a")
