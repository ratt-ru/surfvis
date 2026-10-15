"""Helpers for reading Measurement Sets through xarray-ms."""

import warnings

from xarray_ms.errors import (
    ColumnShapeImputationWarning,
    FrameConversionWarning,
    IrregularGridWarning,
    MissingMetadataWarning,
)


def ignore_xarray_ms_warnings() -> None:
    """Silence the xarray-ms warnings every MSv4 open repeats.

    They fire per partition and per open, so a server that reads a waterfall
    per click buries its log under them. Call at module scope in every module
    that opens MSv4 data. Same filters as pfb-imaging's ``utils/msv4.py``.
    """
    for category in (
        IrregularGridWarning,
        MissingMetadataWarning,
        FrameConversionWarning,
        ColumnShapeImputationWarning,
    ):
        warnings.filterwarnings("ignore", category=category)
