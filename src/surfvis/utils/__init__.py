"""Shared numerical kernels and plotting helpers for surfvis.

Everything in here imports heavy third-party packages (numba, dask, matplotlib)
and is therefore only ever imported from ``surfvis.core``, never from
``surfvis.cli``. The exceptions are ``products`` and ``gains``: numpy and
xarray only, so the dask-free web layer can share them.
"""
