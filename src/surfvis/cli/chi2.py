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
    name="chi2",
    info="Create per-baseline chi-squared plots from a Measurement Set.",
)
@stimela_output(
    dtype="Directory",
    name="dataout",
    info="Output name of zarr dataset. Saved in CWD/chi2 by default.",
    metadata={"rich_help_panel": "Outputs"},
)
@stimela_output(
    dtype="Directory",
    name="imagesout",
    info="Output folder to place images. Saved in CWD/chi2 by default.",
    metadata={"rich_help_panel": "Outputs"},
)
def chi2(
    ms: Annotated[
        MS,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Measurement Set to inspect.",
            rich_help_panel="Inputs",
        ),
    ],
    rcol: Annotated[
        str,
        typer.Option(
            help="Residual column.",
            rich_help_panel="Inputs",
        ),
    ] = "RESIDUAL",
    wcol: Annotated[
        str,
        typer.Option(
            help="Weight column. "
            "The special value SIGMA_SPECTRUM can be passed to initialise the weights as 1/sigma**2.",
            rich_help_panel="Inputs",
        ),
    ] = "WEIGHT_SPECTRUM",
    fcol: Annotated[
        str,
        typer.Option(
            help="Flag column.",
            rich_help_panel="Inputs",
        ),
    ] = "FLAG",
    nthreads: Annotated[
        int,
        typer.Option(
            help="Number of dask threads to use.",
            rich_help_panel="Inputs",
        ),
    ] = 4,
    ntimes: Annotated[
        int | None,
        typer.Option(
            help="Number of unique times in each chunk. Defaults to all of them.",
            rich_help_panel="Inputs",
        ),
    ] = None,
    nfreqs: Annotated[
        int,
        typer.Option(
            help="Number of frequencies in a chunk. Set to -1 to use all of them.",
            rich_help_panel="Inputs",
        ),
    ] = 128,
    use_corrs: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="Comma separated list of correlations to use. Defaults to the diagonal correlations.",
            rich_help_panel="Inputs",
        ),
    ] = None,
    dataout: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Output name of zarr dataset. Saved in CWD/chi2 by default.",
            rich_help_panel="Outputs",
        ),
    ] = None,
    imagesout: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Output folder to place images. Saved in CWD/chi2 by default.",
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
    Create per-baseline chi-squared plots from a Measurement Set.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                chi2,
                dict(
                    ms=ms,
                    rcol=rcol,
                    wcol=wcol,
                    fcol=fcol,
                    nthreads=nthreads,
                    ntimes=ntimes,
                    nfreqs=nfreqs,
                    use_corrs=use_corrs,
                    dataout=dataout,
                    imagesout=imagesout,
                ),
            )

            # Lazy import the core implementation
            from surfvis.core.chi2 import chi2 as chi2_core  # noqa: E402

            # Call the core function with all parameters
            chi2_core(
                ms,
                rcol=rcol,
                wcol=wcol,
                fcol=fcol,
                nthreads=nthreads,
                ntimes=ntimes,
                nfreqs=nfreqs,
                use_corrs=use_corrs,
                dataout=dataout,
                imagesout=imagesout,
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
        chi2,
        dict(
            ms=ms,
            rcol=rcol,
            wcol=wcol,
            fcol=fcol,
            nthreads=nthreads,
            ntimes=ntimes,
            nfreqs=nfreqs,
            use_corrs=use_corrs,
            dataout=dataout,
            imagesout=imagesout,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
