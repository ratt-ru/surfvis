from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import (
    ListInt,
    StimelaMeta,
    parse_list_int,
    parse_upath,
    stimela_cab,
    stimela_output,
)

Directory = NewType("Directory", Path)
MS = NewType("MS", Path)


@stimela_cab(
    name="surf",
    info="Create per-baseline time/frequency plots from a Measurement Set.",
)
@stimela_output(
    dtype="Directory",
    name="opdir",
    info="Output folder to store plots. Defaults to <ms>_<datacolumn>__plots.",
    metadata={"rich_help_panel": "Outputs"},
)
def surf(
    ms: Annotated[
        MS,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Measurement Set to plot.",
            rich_help_panel="Inputs",
        ),
    ],
    datacolumn: Annotated[
        str,
        typer.Option(
            help="Measurement Set column to plot.",
            rich_help_panel="Inputs",
        ),
    ] = "DATA",
    field: Annotated[
        int,
        typer.Option(
            help="Field ID to plot.",
            rich_help_panel="Inputs",
        ),
    ] = 0,
    spw: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="Comma separated list of SPWs to plot. Defaults to all of them.",
            rich_help_panel="Inputs",
        ),
    ] = None,
    plot: Annotated[
        Literal["amp", "phase", "real", "imag"],
        typer.Option(
            help="Quantity to plot.",
            rich_help_panel="Inputs",
        ),
    ] = "amp",
    i: Annotated[
        int | None,
        typer.Option(
            help="Antenna 1: plot only this antenna. Defaults to all of them.",
            rich_help_panel="Inputs",
        ),
    ] = None,
    j: Annotated[
        int | None,
        typer.Option(
            help="Antenna 2: use with i to plot a single baseline.",
            rich_help_panel="Inputs",
        ),
    ] = None,
    noflags: Annotated[
        bool,
        typer.Option(
            help="Disable flagged data overlay.",
            rich_help_panel="Inputs",
        ),
    ] = False,
    doacorr: Annotated[
        bool,
        typer.Option(
            help="Plot auto-correlations.",
            rich_help_panel="Inputs",
        ),
    ] = False,
    scale: Annotated[
        float | None,
        typer.Option(
            help="Scale the image peak to this multiple of the per-corr min/max. "
            "The default scales the image max to 5 sigma. "
            "Ignored for phase plots.",
            rich_help_panel="Inputs",
        ),
    ] = None,
    cmap: Annotated[
        str,
        typer.Option(
            help="Matplotlib colour map to use.",
            rich_help_panel="Inputs",
        ),
    ] = "jet",
    opdir: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Output folder to store plots. Defaults to <ms>_<datacolumn>__plots.",
            rich_help_panel="Outputs",
        ),
    ] = None,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = False,
):
    """
    Create per-baseline time/frequency plots from a Measurement Set.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                surf,
                dict(
                    ms=ms,
                    datacolumn=datacolumn,
                    field=field,
                    spw=spw,
                    plot=plot,
                    i=i,
                    j=j,
                    noflags=noflags,
                    doacorr=doacorr,
                    scale=scale,
                    cmap=cmap,
                    opdir=opdir,
                ),
            )

            # Lazy import the core implementation
            from surfvis.core.surf import surf as surf_core  # noqa: E402

            # Call the core function with all parameters
            surf_core(
                ms,
                datacolumn=datacolumn,
                field=field,
                spw=spw,
                plot=plot,
                i=i,
                j=j,
                noflags=noflags,
                doacorr=doacorr,
                scale=scale,
                cmap=cmap,
                opdir=opdir,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("surfvis")
    if image is None:
        raise RuntimeError("No Container URL in surfvis metadata.")

    run_in_container(
        surf,
        dict(
            ms=ms,
            datacolumn=datacolumn,
            field=field,
            spw=spw,
            plot=plot,
            i=i,
            j=j,
            noflags=noflags,
            doacorr=doacorr,
            scale=scale,
            cmap=cmap,
            opdir=opdir,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
