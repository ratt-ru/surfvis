"""Flag visibilities whose per-visibility chi-squared exceeds a threshold."""

from pathlib import Path

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_to_storage_table as xds_to_table

from surfvis.utils.chisq import flagchisq


def flag_chi2(
    ms: Path,
    rcol: str = "RESIDUAL",
    wcol: str = "WEIGHT_SPECTRUM",
    fcol: str = "FLAG",
    flag_above: float = 3.0,
    nthreads: int = 4,
    nrows: int = 250000,
    nfreqs: int = 512,
    use_corrs: list[int] | None = None,
    respect_ants: list[int] | None = None,
) -> None:
    """Update the flag column of ``ms`` in place.

    Args:
        ms: Measurement Set to flag. Modified in place.
        rcol: Residual column.
        wcol: Weight column. ``SIGMA_SPECTRUM`` initialises weights as 1/sigma**2.
        fcol: Flag column, written back to the Measurement Set.
        flag_above: Flag data with chi-squared above this value.
        nthreads: Number of dask threads to use.
        nrows: Number of rows in each chunk.
        nfreqs: Number of frequencies in a chunk.
        use_corrs: Correlations to use. Defaults to the diagonal correlations.
        respect_ants: Antennas whose baselines are left untouched.
    """
    from multiprocessing.pool import ThreadPool

    msname = str(ms).rstrip("/")

    dask.config.set(pool=ThreadPool(nthreads))

    schema = {
        rcol: {"dims": ("chan", "corr")},
        wcol: {"dims": ("chan", "corr")},
        fcol: {"dims": ("chan", "corr")},
    }

    xds = xds_from_ms(
        msname,
        columns=[rcol, wcol, fcol, "ANTENNA1", "ANTENNA2"],
        chunks={"row": nrows, "chan": nfreqs},
        group_cols=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"],
        table_schema=schema,
    )

    if not use_corrs:
        print("Using only diagonal correlations")
        corrs = (0, -1) if len(xds[0].corr) > 1 else (0,)
    else:
        corrs = tuple(use_corrs)
        print(f"Using correlations {corrs}")

    rants = tuple(respect_ants) if respect_ants else ()

    out_data = []
    for ds in xds:
        resid = ds.get(rcol).data
        if wcol == "SIGMA_SPECTRUM":
            weight = 1.0 / ds.get(wcol).data ** 2
        else:
            weight = ds.get(wcol).data
        flag = ds.get(fcol).data
        ant1 = ds.ANTENNA1.data
        ant2 = ds.ANTENNA2.data

        uflag = flagchisq(
            resid,
            weight,
            flag,
            ant1,
            ant2,
            use_corrs=corrs,
            flag_above=flag_above,
            respect_ants=rants,
        )

        out_ds = ds.assign(**{fcol: (("row", "chan", "corr"), uflag)})

        # update FLAG_ROW
        flag_row = da.all(uflag.rechunk({1: -1, 2: -1}), axis=(1, 2))

        out_ds = out_ds.assign(**{"FLAG_ROW": (("row",), flag_row)})

        out_data.append(out_ds)

    writes = xds_to_table(out_data, msname, columns=[fcol, "FLAG_ROW"], rechunk=True)

    with ProgressBar():
        dask.compute(writes)
