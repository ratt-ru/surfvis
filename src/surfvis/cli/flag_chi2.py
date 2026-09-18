from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import (
    ListInt,
    StimelaMeta,
    parse_list_int,
    parse_upath,
    stimela_cab,
)

MS = NewType("MS", Path)


@stimela_cab(
    name="flag_chi2",
    info="Flag data with a per-visibility chi-squared above a threshold.",
)
def flag_chi2(
    ms: Annotated[
        MS,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Measurement Set to flag. The flag column is updated in place.",
            rich_help_panel="Inputs",
        ),
        StimelaMeta(
            writable=True,
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
            help="Flag column. This is the column that gets updated.",
            rich_help_panel="Inputs",
        ),
    ] = "FLAG",
    flag_above: Annotated[
        float,
        typer.Option(
            help="Flag data with chi-squared above this value.",
            rich_help_panel="Inputs",
        ),
    ] = 3.0,
    nthreads: Annotated[
        int,
        typer.Option(
            help="Number of dask threads to use.",
            rich_help_panel="Inputs",
        ),
    ] = 4,
    nrows: Annotated[
        int,
        typer.Option(
            help="Number of rows in each chunk.",
            rich_help_panel="Inputs",
        ),
    ] = 250000,
    nfreqs: Annotated[
        int,
        typer.Option(
            help="Number of frequencies in a chunk.",
            rich_help_panel="Inputs",
        ),
    ] = 512,
    use_corrs: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="Comma separated list of correlations to use. Defaults to the diagonal correlations.",
            rich_help_panel="Inputs",
        ),
    ] = None,
    respect_ants: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="Comma separated list of antennas to respect. Baselines to these antennas are left untouched.",
            rich_help_panel="Inputs",
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
    Flag data with a per-visibility chi-squared above a threshold.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                flag_chi2,
                dict(
                    ms=ms,
                    rcol=rcol,
                    wcol=wcol,
                    fcol=fcol,
                    flag_above=flag_above,
                    nthreads=nthreads,
                    nrows=nrows,
                    nfreqs=nfreqs,
                    use_corrs=use_corrs,
                    respect_ants=respect_ants,
                ),
            )

            # Lazy import the core implementation
            from surfvis.core.flag_chi2 import flag_chi2 as flag_chi2_core  # noqa: E402

            # Call the core function with all parameters
            flag_chi2_core(
                ms,
                rcol=rcol,
                wcol=wcol,
                fcol=fcol,
                flag_above=flag_above,
                nthreads=nthreads,
                nrows=nrows,
                nfreqs=nfreqs,
                use_corrs=use_corrs,
                respect_ants=respect_ants,
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
        flag_chi2,
        dict(
            ms=ms,
            rcol=rcol,
            wcol=wcol,
            fcol=fcol,
            flag_above=flag_above,
            nthreads=nthreads,
            nrows=nrows,
            nfreqs=nfreqs,
            use_corrs=use_corrs,
            respect_ants=respect_ants,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
