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
| `surfvis serve` | Browse chi-squared output in a browser and plot baseline waterfalls on demand. |

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

## Browsing chi-squared interactively

`surfvis serve` turns `surfchi2` output into a browsable page: an
antenna-by-antenna chi-squared grid for one chunk, the histogram over all
antenna pairs beside it, and a click on any cell plotting that baseline's
waterfall straight from the Measurement Set.

This is a GUI, so it is deliberately **not** a Stimela cab and never will be.

### 1. Produce the dataset

The browser reads the zarr written by `chi2 --dataout`, not the PNGs. If you
have only ever run `surfchi2` for its images, you need one more run:

```bash
surfvis chi2 --ms /data/my.ms \
    --dataout /data/chi2.zarr \
    --imagesout /data/chi2 \
    --rcol RESIDUAL --wcol WEIGHT_SPECTRUM \
    --nthreads 16 --nfreqs 128
```

`--dataout` and `--imagesout` are **deleted and recreated** on every run, so
point them somewhere you do not mind losing. The zarr costs roughly 4 MB per
scan for 64 antennas at `--nfreqs 128`; a long observation is a few hundred MB.

The Measurement Set path is recorded inside the dataset, so `serve` finds it on
its own. Pass `--ms` only if the MS has moved since.

### 2. Start the server

```bash
pip install 'surfvis[full,web]'
surfvis serve --data /data/chi2.zarr
```

Then open <http://127.0.0.1:8000>.

Useful flags:

| Flag | Default | Notes |
|---|---|---|
| `--data` | *required* | The zarr from `chi2 --dataout`. |
| `--ms` | from the dataset | Override if the MS moved. |
| `--host` | `127.0.0.1` | Use `0.0.0.0` in a container, or to accept remote connections. |
| `--port` | `8000` | Change if the port is taken. |
| `--reload` | off | Development only. |

`--host 127.0.0.1` means nothing outside the machine can reach it. That is the
right default; to view it from your laptop, forward the port rather than
binding to `0.0.0.0` on a public interface.

### Running where the data lives

The Measurement Set is usually on a compute node, not your laptop, and moving a
few hundred GB to look at it is not an option. Run the server next to the data
and forward the port.

**On a remote machine you can ssh to directly:**

```bash
# on the remote machine
surfvis serve --data /data/chi2.zarr --port 8000

# on your laptop, in another terminal
ssh -N -L 8000:localhost:8000 you@remote
```

Then open <http://localhost:8000> on your laptop. `-N` means "no remote
command", so the tunnel just sits there; `Ctrl-C` closes it.

**On a compute node behind a login node** (the usual Slurm case), forward
through the login node in one hop:

```bash
# on the compute node, e.g. inside your job
surfvis serve --data /data/chi2.zarr --host 0.0.0.0 --port 8000

# on your laptop -- note the node name, not localhost, on the right of the colon
ssh -N -L 8000:compute-node-042:8000 you@login.cluster
```

`--host 0.0.0.0` is needed here because the connection arrives from the login
node, not from the compute node itself.

If the cluster only allows one hop, use `-J`:

```bash
ssh -N -J you@login.cluster -L 8000:localhost:8000 you@compute-node-042
```

**Inside a container**, publish the port and bind to all interfaces:

```bash
docker run --rm -p 8000:8000 -v /data:/data ghcr.io/ratt-ru/surfvis \
    surfvis serve --data /data/chi2.zarr --host 0.0.0.0 --port 8000
```

This needs no local dependencies at all. `-v /data:/data` mounts the data with
the *same path inside the container*, which matters because the MS path
recorded in the dataset is resolved inside the container. If you mount
somewhere else, pass `--ms` to match.

Combine the two when the container runs remotely: publish the port on the
remote host, then forward it to your laptop exactly as above.

If port 8000 is already taken, pick another and keep both sides consistent:
`--port 8042` with `-L 8042:localhost:8042`.

### Using the interface

**The controls along the top** choose which chunk you are looking at: field,
spw, scan, then the time bin and frequency bin within that scan, then the
polarization. These are the same chunks `surfchi2` wrote PNGs for -- the bins
follow the `--ntimes` and `--nfreqs` you ran with.

**The grid** is chi-squared per degree of freedom for every antenna pair, the
same quantity as the `surfchi2` images. Rows are antenna1, columns antenna2,
and it is symmetric. Dark grey cells along the diagonal are auto-correlations,
which carry no chi-squared; dark grey anywhere else means every visibility for
that pair was flagged. Hover a cell for the antenna names. A single bad antenna
shows up as a bright row *and* column -- that cross is the pattern worth
looking for, and it is why the grid is worth keeping rather than just ranking
the worst pairs.

**The histogram** beside the grid is the whole distribution over antenna pairs,
with a dashed line at chi-squared/dof = 1. This is how you judge whether a
chunk has outliers at all: a single clean peak means the colour scale is just
stretched over noise, whereas a long tail or a detached bar means something is
genuinely wrong. The colourbar underneath is the scale the grid is using.

**The scale selector** matters more than it looks. chi-squared/dof routinely
spans orders of magnitude, so the default is **log** -- on a linear scale one
bad baseline renders every other cell black and you learn nothing beyond "there
is one bad baseline". `robust` clips to the 2nd-98th percentile, which spreads
the bulk across the colourmap and saturates the outliers; it is usually the
better view once you know where the problem is, though it degenerates when
there are few antenna pairs. `full` is plain min-to-max.

**Clicking any cell** plots that baseline's waterfall below: the whole scan by
all channels, for the polarization you are viewing. The chunk you clicked is
outlined in green, so you can see whether the bad chunk is isolated or part of
something larger. Flagged data is greyed out, as are slots where the MS has no
data at all.

**The column and quantity selectors** under the waterfall re-plot without
changing your place in the grid. Quantity is `amp`, `phase`, `real` or `imag`.
Column defaults to whatever `--rcol` the chi-squared was computed from, which
is usually what you want -- you are asking "does this residual look like
noise?" -- but switching to the data column tells you whether the problem is in
the data or in the model subtracted from it.

### A note on column names (MSv2 vs MSv4)

Waterfalls are read through
[xarray-ms](https://github.com/ratt-ru/xarray-ms), which presents an MSv2 table
as an **MSv4** view. MSv4 renames the columns you are used to:

| You know it as | MSv4 calls it |
|---|---|
| `DATA` | `VISIBILITY` |
| `CORRECTED_DATA` | `VISIBILITY` (of the corrected data group) |
| `WEIGHT_SPECTRUM` | `WEIGHT` |
| `FLAG` | `FLAG` |

surfvis resolves this through the dataset's `data_groups` attribute rather than
hardcoding the mapping, the same way `pfb-imaging` does, so **either name
works** in the column selector and on the URL. Columns that are not part of the
MSv2 standard -- `RESIDUAL`, `MODEL`, anything your calibration pipeline wrote
-- are not renamed and appear under their own names. That is precisely why the
residual column the chi-squared came from is plottable.

### Troubleshooting

**`The web interface needs the optional extras`** -- install
`surfvis[full,web]`. Note xarray-ms needs Python 3.11 or newer; the batch
commands still run on 3.10.

**`No chi-squared dataset at ...`** -- `--data` wants the zarr from
`chi2 --dataout`, not the `--imagesout` directory of PNGs.

**`Column 'X' (MSv4 'Y') not in this Measurement Set`** -- the error lists what
*is* available in that partition. If you expected a column your pipeline wrote,
check it exists in the MS and is not empty.

**Nothing at `localhost:8000` through a tunnel** -- check the server says
`Uvicorn running on ...`, that both sides of `-L` use the same port, and that
you used `--host 0.0.0.0` if the tunnel terminates anywhere other than the
machine running the server.

**The grid is all one colour** -- switch the scale selector. On `log` with a
very tight distribution everything genuinely is the same; on `robust` with few
antennas the percentiles collapse.

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
