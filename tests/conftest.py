"""Shared fixtures.

Everything that touches a Measurement Set needs the heavy extras, so those
tests skip cleanly on a lightweight install rather than erroring.
"""

import pytest


def _have(*modules: str) -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in modules)


needs_ms = pytest.mark.skipif(
    not _have("casacore", "daskms", "numba"),
    reason="needs the [full] extras (python-casacore, dask-ms, numba)",
)
needs_web = pytest.mark.skipif(
    not _have("casacore", "daskms", "numba", "fastapi", "xarray_ms"),
    reason="needs the [full,web] extras",
)


@pytest.fixture(scope="session")
def tiny_ms(tmp_path_factory):
    """A small Measurement Set with a known-bad baseline."""
    from tests.fixtures.ms import make_ms

    path = tmp_path_factory.mktemp("ms") / "tiny.ms"
    return make_ms(str(path))


@pytest.fixture(scope="session")
def chi2_zarr(tiny_ms, tmp_path_factory):
    """The chi-squared dataset produced from :func:`tiny_ms`."""
    from surfvis.core.chi2 import chi2

    out = tmp_path_factory.mktemp("chi2")
    chi2(
        tiny_ms["path"],
        dataout=out / "chi2.zarr",
        imagesout=out / "imgs",
        nthreads=2,
        nfreqs=16,
        ntimes=6,
    )
    return {"zarr": out / "chi2.zarr", "ms": tiny_ms}


@pytest.fixture(scope="session")
def gain_table(tiny_ms, tmp_path_factory):
    """The fixture MS's gains, written as a QuartiCal gain term."""
    from tests.fixtures.ms import write_gain_table

    return write_gain_table(tiny_ms, str(tmp_path_factory.mktemp("gains") / "gains.qc"))


@pytest.fixture(scope="session")
def chi2_corrected(tiny_ms, gain_table, tmp_path_factory):
    """chi2 formed on the fly from DATA, MODEL_DATA and the gains, with Stokes I and V."""
    from surfvis.core.chi2 import chi2

    out = tmp_path_factory.mktemp("chi2c")
    chi2(
        tiny_ms["path"],
        rcol="DATA",
        mcol="MODEL_DATA",
        gtable=gain_table,
        dataout=out / "chi2.zarr",
        imagesout=out / "imgs",
        nthreads=2,
        nfreqs=16,
        ntimes=6,
        stokes=["I", "v"],
    )
    return {"zarr": out / "chi2.zarr", "imgs": out / "imgs", "ms": tiny_ms, "gtable": gain_table}
