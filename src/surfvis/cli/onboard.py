from typing import Annotated, Literal

import typer
from hip_cargo import StimelaMeta, stimela_cab


@stimela_cab(
    name="onboard",
    info="Print setup instructions for CI/CD, PyPI publishing, and GitHub configuration.",
)
def onboard(
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
    Print setup instructions for CI/CD, PyPI publishing, and GitHub configuration.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                onboard,
                dict(),
            )

            # Lazy import the core implementation
            from surfvis.core.onboard import onboard as onboard_core  # noqa: E402

            # Call the core function with all parameters
            onboard_core()
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
        onboard,
        dict(),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
