FROM python:3.11-slim

WORKDIR /app

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /usr/local/bin/uv

# Copy package files (LICENSE is required by pyproject's license-files glob)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install package with full dependencies.
# python-casacore, numba and dask-ms all ship manylinux wheels, so no apt-get
# of casacore/llvm is needed on this base image.
RUN uv pip install --system --no-cache ".[full]"

# Make CLI available
CMD ["surfvis", "--help"]
