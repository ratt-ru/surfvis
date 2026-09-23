"""Per-baseline chi-squared plots from a Measurement Set."""

import concurrent.futures as cf
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import dask
import numpy as np
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_from_storage_table as xds_from_table

from surfvis.utils.plotting import makeplot, surfchisq_plot


def chi2(
    ms: Path,
    rcol: str = "RESIDUAL",
    wcol: str = "WEIGHT_SPECTRUM",
    fcol: str = "FLAG",
    dataout: Path | None = None,
    imagesout: Path | None = None,
    nthreads: int = 4,
    ntimes: int | None = None,
    nfreqs: int = 128,
    use_corrs: list[int] | None = None,
) -> None:
    """Write per-(time, freq, corr) chi-squared images plus a per-scan combination.

    Args:
        ms: Measurement Set to inspect.
        rcol: Residual column.
        wcol: Weight column. ``SIGMA_SPECTRUM`` initialises weights as 1/sigma**2.
        fcol: Flag column.
        dataout: Output name of the zarr dataset holding chi-squared and counts
            per (time bin, freq bin, corr, antenna, antenna). Defaults to
            ``$CWD/chi2.zarr``. This is what ``surfvis serve`` reads.
        imagesout: Output folder for images. Defaults to ``$CWD/chi2``.
        nthreads: Number of worker processes (also the dask thread-pool size).
        ntimes: Number of unique times in each chunk. ``None`` means all of them.
        nfreqs: Number of frequencies in a chunk. ``-1`` means all of them.
        use_corrs: Correlations to use. Defaults to the diagonal correlations.
    """
    from multiprocessing.pool import ThreadPool

    msname = str(ms).rstrip("/")

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
    rbin_idx = []
    rbin_counts = []
    tbin_idx = []
    tbin_counts = []
    fbin_idx = []
    fbin_counts = []
    for ds in xds:
        time = ds.TIME.values
        ut, counts = np.unique(time, return_counts=True)
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

    schema = {
        rcol: {"dims": ("chan", "corr")},
        wcol: {"dims": ("chan", "corr")},
        fcol: {"dims": ("chan", "corr")},
    }

    xds = xds_from_ms(
        msname,
        columns=[rcol, wcol, fcol, "ANTENNA1", "ANTENNA2", "TIME"],
        chunks=chunks,
        group_cols=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"],
        table_schema=schema,
    )

    ncorr_total = xds[0].corr.size
    if not use_corrs:
        print("Using only diagonal correlations")
        corrs = [0, -1] if ncorr_total > 1 else [0]
    else:
        corrs = list(use_corrs)
        print(f"Using correlations {corrs}")
    ncorr = len(corrs)
    # -1 and friends are positional; resolve them so the zarr records real
    # polarization indices, not offsets into use_corrs.
    pol_idx = [c % ncorr_total for c in corrs]

    ant_ds = xds_from_table(f"{msname}::ANTENNA")[0]
    antenna_names = [str(n) for n in ant_ds.NAME.values]
    nant = len(antenna_names)

    chi2s = {}
    counts = {}
    # Per-chunk cubes, keyed the same way, so the zarr keeps what the PNGs throw away.
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
                    for c in range(ncorr):
                        t0 = tbin_idx[i][t]
                        tf = t0 + tbin_counts[i][t]
                        chan0 = fbin_idx[i][f]
                        chanf = chan0 + fbin_counts[i][f]
                        row0 = rbin_idx[i][t]
                        rowf = rbin_idx[i][t] + rbin_counts[i][t]
                        dso = ds[{"row": slice(row0, rowf), "chan": slice(chan0, chanf)}]
                        dso = dso.sel(corr=corrs)
                        resid = dso.get(rcol).data
                        if wcol == "SIGMA_SPECTRUM":
                            weight = 1.0 / dso.get(wcol).data ** 2
                        else:
                            weight = dso.get(wcol).data
                        flag = dso.get(fcol).data
                        ant1 = dso.ANTENNA1.data
                        ant2 = dso.ANTENNA2.data
                        fut = executor.submit(
                            surfchisq_plot,
                            resid,
                            weight,
                            flag,
                            ant1,
                            ant2,
                            field,
                            spw,
                            scan,
                            basename + f"t{t}_f{f}_c{c}.png",
                            f"t {t0}-{tf}, chan {chan0}-{chanf}, corr {c}",
                        )
                        futures[fut] = (i, t, f, c)

            # to reduce over time, freq and corr at the end
            key = f"field{field}_spw{spw}_scan{scan}"
            keys[i] = (key, int(field), int(spw), int(scan))
            chi2s[key] = np.zeros((nant, nant), dtype=float)
            counts[key] = np.zeros((nant, nant), dtype=float)
            cubes[i] = (
                np.zeros((ntime, nfreq, ncorr, nant, nant), dtype=float),
                np.zeros((ntime, nfreq, ncorr, nant, nant), dtype=float),
            )
            print(f"Submitted field{field}_spw{spw}_scan{scan}")

        # reduce per scan
        num_completed = 0
        num_futures = len(futures)
        for fut in cf.as_completed(futures):
            num_completed += 1
            print(f"\rProcessing: {num_completed}/{num_futures}", end="", flush=True)
            field, spw, scan, chi2_chunk, count = fut.result()
            chi2s[f"field{field}_spw{spw}_scan{scan}"] += chi2_chunk
            counts[f"field{field}_spw{spw}_scan{scan}"] += count
            i, t, f, c = futures[fut]
            chi2_cube, count_cube = cubes[i]
            # A chunk only spans the antennas present in it; pad into the full grid.
            na = chi2_chunk.shape[0]
            chi2_cube[t, f, c, :na, :na] = chi2_chunk
            count_cube[t, f, c, :na, :na] = count
        print()

    _write_zarr_dataset(
        dataout,
        cubes,
        keys,
        tbin_idx,
        tbin_counts,
        fbin_idx,
        fbin_counts,
        pol_idx,
        antenna_names,
        msname,
        rcol,
        wcol,
        fcol,
    )

    # LB - is it worth doing this in parallel?
    print("Plotting per scan")
    for key, val in chi2s.items():
        field, spw, scan = key.split("_")
        field = field.strip("field")
        spw = spw.strip("spw")
        scan = scan.strip("scan")
        count = counts[key]
        chi2_dof = np.zeros_like(val)
        chi2_dof[count > 0] = val[count > 0] / count[count > 0]
        chi2_dof[count <= 0] = np.nan

        basename = foldername + f"/field{field}" + f"/spw{spw}" + f"/scan{scan}/"
        makeplot(chi2_dof, basename + "combined.png", f"scan {scan}.png")


def _write_zarr_dataset(
    dataout,
    cubes,
    keys,
    tbin_idx,
    tbin_counts,
    fbin_idx,
    fbin_counts,
    pol_idx,
    antenna_names,
    msname,
    rcol,
    wcol,
    fcol,
):
    """Write the per-chunk chi-squared cubes to a zarr store.

    One group per (field, spw, scan), because the number of time and frequency
    bins differs between them. Bin bounds ride along as coordinates so a reader
    can locate a chunk inside the scan's waterfall without re-deriving the
    chunking.
    """
    import xarray as xr

    print(f"Writing chi-squared dataset to {dataout}")
    nant = len(antenna_names)
    attrs = {
        "ms": str(msname),
        "rcol": rcol,
        "wcol": wcol,
        "fcol": fcol,
        "antenna_names": list(antenna_names),
    }

    for i, (chi2_cube, count_cube) in cubes.items():
        _, field, spw, scan = keys[i]
        ntime, nfreq, ncorr = chi2_cube.shape[:3]
        t0 = np.asarray(tbin_idx[i][:ntime], dtype=int)
        tf = t0 + np.asarray(tbin_counts[i][:ntime], dtype=int)
        chan0 = np.asarray(fbin_idx[i][:nfreq], dtype=int)
        chanf = chan0 + np.asarray(fbin_counts[i][:nfreq], dtype=int)

        dims = ("time_bin", "freq_bin", "corr", "antenna1", "antenna2")
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
                "corr": np.asarray(pol_idx[:ncorr], dtype=int),
                "antenna1": np.arange(nant),
                "antenna2": np.arange(nant),
            },
            attrs={**attrs, "field": field, "spw": spw, "scan": scan},
        )
        ds.to_zarr(dataout, group=f"field{field}/spw{spw}/scan{scan}", mode="a")
