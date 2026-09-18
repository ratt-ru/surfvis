"""Per-baseline chi-squared plots from a Measurement Set."""

import concurrent.futures as cf
import os
import shutil
from pathlib import Path

import dask
import numpy as np
from daskms import xds_from_storage_ms as xds_from_ms

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
        dataout: Output name of the zarr dataset. Defaults to ``$CWD/chi2``.
        imagesout: Output folder for images. Defaults to ``$CWD/chi2``.
        nthreads: Number of worker processes (also the dask thread-pool size).
        ntimes: Number of unique times in each chunk. ``None`` means all of them.
        nfreqs: Number of frequencies in a chunk. ``-1`` means all of them.
        use_corrs: Correlations to use. Defaults to the diagonal correlations.
    """
    from multiprocessing.pool import ThreadPool

    msname = str(ms).rstrip("/")

    dataout = Path(str(dataout)) if dataout is not None else Path(os.getcwd()) / "chi2"
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

    if not use_corrs:
        print("Using only diagonal correlations")
        corrs = [0, -1] if len(xds[0].corr) > 1 else [0]
    else:
        corrs = list(use_corrs)
        print(f"Using correlations {corrs}")
    ncorr = len(corrs)

    chi2s = {}
    counts = {}
    futures = []
    foldername = str(imagesout).rstrip("/")
    with cf.ProcessPoolExecutor(max_workers=nthreads) as executor:
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
                        futures.append(fut)

            # to reduce over time, freq and corr at the end
            nant = np.maximum(ant1.compute().max(), ant2.compute().max()) + 1
            chi2s[f"field{field}_spw{spw}_scan{scan}"] = np.zeros((nant, nant), dtype=float)
            counts[f"field{field}_spw{spw}_scan{scan}"] = np.zeros((nant, nant), dtype=float)
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
