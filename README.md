# surfvis

Per-baseline time/frequency and chi-squared diagnostics from a Measurement Set.

`surfvis` is a [`hip-cargo`](https://github.com/landmanbester/hip-cargo)
package: one Typer CLI whose commands double as
[`stimela`](https://github.com/caracal-pipeline/stimela) cabs, backed by a
container image on the GitHub Container Registry.

![](http://i.imgur.com/wHeA9JT.jpg)

## Installation

```bash
pip install surfvis
```

That is the **lightweight** install — the CLI and the cab definitions, nothing
else. It is safe to install alongside `stimela` and `cult-cargo`. Commands run
inside `ghcr.io/ratt-ru/surfvis` via container fallback, so you need a
container runtime (apptainer, singularity, docker or podman) but not the
scientific Python stack.

To run natively instead, install the heavy dependencies
(`python-casacore`, `dask-ms`, `numba`, `matplotlib`, `astropy`, `xarray`,
`dask`):

```bash
pip install 'surfvis[full]'
```

Every command takes `--backend` (`auto`, `native`, `apptainer`, `singularity`,
`docker`, `podman`) and `--always-pull-images`. `auto` tries native execution
and falls back to a container when the heavy imports are missing.

## Usage

```bash
surfvis --help
```

| Command | What it does |
|---|---|
| `surfvis summary` | Print the FIELD, SPECTRAL_WINDOW and ANTENNA tables of an MS. |
| `surfvis surf` | One time/frequency PNG per baseline. |
| `surfvis chi2` | Per-(time, freq, corr) chi-squared images plus a per-scan combination. |
| `surfvis flag-chi2` | Flag visibilities whose chi-squared exceeds a threshold. Modifies the MS in place. |

### Examples

```bash
# What is in this MS?
surfvis summary --ms /data/my.ms

# Amplitude waterfalls for every baseline in field 0, SPWs 0 and 1
surfvis surf --ms /data/my.ms --datacolumn CORRECTED_DATA --plot amp --field 0 --spw 0,1

# A single baseline, phases, no flag overlay
surfvis surf --ms /data/my.ms --i 4 --j 17 --plot phase --noflags

# Chi-squared images, 8 workers, 64 channels per chunk
surfvis chi2 --ms /data/my.ms --rcol RESIDUAL --wcol WEIGHT_SPECTRUM \
    --nthreads 8 --nfreqs 64 --imagesout /scratch/chi2

# Flag anything above chi2 = 5, leaving baselines to antennas 0 and 1 alone
surfvis flag-chi2 --ms /data/my.ms --flag-above 5 --respect-ants 0,1
```

Comma-separated list options (`--spw`, `--use-corrs`, `--respect-ants`) take a
single argument with no spaces: `--use-corrs 0,3`.

> **Coming from the old scripts?** `surfvis`, `surfchi2` and `flagchi2` were
> three separate `optparse` executables. See
> [docs/wiki/migration.md](docs/wiki/migration.md) for the option-by-option
> mapping.

## Using the cabs from a stimela recipe

The lightweight install ships generated cab definitions, so a recipe can pull
them in directly:

```yaml
_include:
  - (surfvis.cabs)chi2.yml
  - (surfvis.cabs)flag_chi2.yml
```

`stimela` matches the cab's `image:` field against the published container, so
the version you install and the version that runs are the same.

## Caveats

Plotting can be slow for large Measurement Sets. `surfvis surf` will probably
fail on MSs whose SPWs have different time/frequency shapes — use `--spw` to
select a compatible subset.

`surfvis flag-chi2` writes to the flag column of the MS you give it. There is
no dry-run.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ratt-ru/surfvis.git
cd surfvis
uv sync --group dev --group test
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

The CLI wrappers under `src/surfvis/cli/` and the cab YAML under
`src/surfvis/cabs/` must stay in agreement — `tests/test_roundtrip.py` enforces
that they round-trip byte-identically. After changing a CLI signature:

```bash
uv run ruff format . && uv run ruff check . --fix
uv run hip-cargo generate-cabs --module 'src/surfvis/cli/*.py' --output-dir src/surfvis/cabs
uv run pytest
```

The pre-commit hook regenerates cabs for you. `docs/wiki/` is the canonical
reference for how this repo is put together — start at
[docs/wiki/index.md](docs/wiki/index.md).

Remaining CI/CD setup (PyPI trusted publishing, the GitHub App used by the
`update-cabs` workflow, branch protection) is described by:

```bash
uv run surfvis onboard
```

## Credits

Originally written by Ian Heywood (`ianh@astro.ox.ac.uk`).

## License

MIT — see [LICENSE](LICENSE).
