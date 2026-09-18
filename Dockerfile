FROM python:3.11-slim

WORKDIR /app

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /usr/local/bin/uv

# TEMPORARY: git is only needed because pyproject.toml pins hip-cargo to a
# branch (landmanbester/hip-cargo#111). python:3.11-slim ships without it, so
# the build fails at `uv pip install` without this. Drop both this layer and
# the pin once the fix is in a hip-cargo release.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files (LICENSE is required by pyproject's license-files glob)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install package with full dependencies.
# python-casacore, numba and dask-ms all ship manylinux wheels, so no apt-get
# of casacore/llvm is needed on this base image.
RUN uv pip install --system --no-cache ".[full]"

# Make CLI available
CMD ["surfvis", "--help"]
