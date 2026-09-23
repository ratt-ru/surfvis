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
