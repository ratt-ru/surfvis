"""Serve the chi-squared browser.

Deliberately NOT decorated with ``@stimela_cab``: a long-running GUI is not a
batch task and will never be part of a Stimela recipe. ``generate-cabs`` skips
undecorated commands, so no cab is produced for this module.
"""

from pathlib import Path
from typing import Annotated

import typer


def serve(
    data: Annotated[
        Path,
        typer.Option(
            ...,
            help="Chi-squared zarr written by 'surfvis chi2 --dataout'.",
            rich_help_panel="Inputs",
        ),
    ],
    ms: Annotated[
        Path | None,
        typer.Option(
            help="Measurement Set to read waterfalls from. Defaults to the path recorded in the dataset.",
            rich_help_panel="Inputs",
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option(help="Interface to bind to.", rich_help_panel="Server"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(help="Port to listen on.", rich_help_panel="Server"),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option(help="Reload on source changes (development only).", rich_help_panel="Server"),
    ] = False,
):
    """
    Browse surfchi2 output and plot baseline waterfalls on demand.
    """
    try:
        import msv4_utils  # noqa: F401
        import uvicorn
        import xarray_ms  # noqa: F401

        from surfvis.web.app import create_app
    except ImportError as exc:  # pragma: no cover - depends on install mode
        import sys

        hint = "pip install 'surfvis[web]'"
        if sys.version_info < (3, 11):
            hint += " -- note xarray-ms needs Python 3.11 or newer"
        raise typer.BadParameter(f"The web interface needs the optional extras: {hint} ({exc})") from exc

    app = create_app(data, ms)
    typer.echo(f"surfvis serving {data} on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=reload)
