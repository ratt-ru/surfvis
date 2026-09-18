from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab

MS = NewType("MS", Path)


@stimela_cab(
    name="summary",
    info="List Measurement Set properties.",
)
def summary(
    ms: Annotated[
        MS,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Measurement Set to summarise.",
            rich_help_panel="Inputs",
        ),
    ],
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
    List Measurement Set properties.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                summary,
                dict(
                    ms=ms,
                ),
            )

            # Lazy import the core implementation
            from surfvis.core.summary import summary as summary_core  # noqa: E402

            # Call the core function with all parameters
            summary_core(
                ms,
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
        summary,
        dict(
            ms=ms,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
