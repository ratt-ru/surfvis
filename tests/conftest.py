"""Shared fixtures.

Everything that touches a Measurement Set needs the heavy extras, so those
tests skip cleanly on a lightweight install rather than erroring.
"""

import importlib.util

import pytest


def _have(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


needs_ms = pytest.mark.skipif(
    not _have("casacore", "daskms", "numba"),
    reason="needs the [full] extras (python-casacore, dask-ms, numba)",
)


@pytest.fixture(scope="session")
def tiny_ms(tmp_path_factory):
    """A small Measurement Set with a known-bad baseline."""
    from tests.fixtures.ms import make_ms

    path = tmp_path_factory.mktemp("ms") / "tiny.ms"
    return make_ms(str(path))
