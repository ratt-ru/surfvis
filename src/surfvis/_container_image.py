CONTAINER_IMAGE = "ghcr.io/ratt-ru/surfvis:latest"

# Optional GPU passthrough for the container-fallback path.
# Set GPU = True for a CUDA/GPU image, or "auto" to request a GPU only
# when one is detected (and, for docker/podman, the NVIDIA Container
# Toolkit is present). Absent => no GPU flags (the default).
# GPU = True

# Optional per-backend extra arguments, passed verbatim to the container
# runtime during fallback. Example: RUN_ARGS_APPTAINER = ["--ipc=host"].
# RUN_ARGS_DOCKER = []
# RUN_ARGS_PODMAN = []
# RUN_ARGS_APPTAINER = []
# RUN_ARGS_SINGULARITY = []
