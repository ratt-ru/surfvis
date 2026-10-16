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
(`python-casacore`, `arcae`, `dask-ms`, `numba`, `matplotlib`, `astropy`,
`xarray`, `dask`); native execution needs Python 3.11 or newer:

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
| `surfvis chi2` | Per-(time, freq, product) chi-squared images plus a per-scan combination. Can form the corrected residual on the fly and add Stokes products. |
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

# The same, without a RESIDUAL column (see below)
surfvis chi2 --ms /data/my.ms --rcol DATA --mcol MODEL_DATA \
    --gtable /data/gains.qc/NET --imagesout /scratch/chi2

# Flag anything above chi2 = 5, leaving baselines to antennas 0 and 1 alone
surfvis flag-chi2 --ms /data/my.ms --flag-above 5 --respect-ants 0,1
```

Comma-separated list options (`--spw`, `--use-corrs`, `--respect-ants`) take a
single argument with no spaces: `--use-corrs 0,3`.

## Corrected residuals on the fly

`surfvis chi2` no longer needs a `RESIDUAL` column. Give it the data, the model
and the gains, and it forms the corrected residual itself, per 2x2 visibility:

```
r = inv(G_p) D inv(G_q)^H - M
```

where `D` is `--rcol`, `M` is `--mcol` and `G` is the QuartiCal gain term in
`--gtable`. Nothing is written back to the MS.

```bash
surfvis chi2 --ms /data/my.ms \
    --rcol DATA --mcol MODEL_DATA --gtable /data/gains.qc/NET \
    --wcol WEIGHT_SPECTRUM \
    --dataout /data/chi2.zarr --imagesout /data/chi2
```

Which options to pass depends on what the MS already holds:

| The MS has | Pass |
|---|---|
| a residual column your pipeline wrote | `--rcol RESIDUAL` (the default) |
| `DATA`, `MODEL_DATA` and a QuartiCal gain solution | `--rcol DATA --mcol MODEL_DATA --gtable gains.qc/NET` |
| gains already applied to `CORRECTED_DATA` | `--rcol CORRECTED_DATA --mcol MODEL_DATA` |

Things to know:

- **`MODEL_DATA` is the uncorrupted model.** It is subtracted *after* the
  gains are removed, so do not pass a model that already has the gains applied.
- **The gain term** is a QuartiCal output directory such as `gains.qc/NET`
  (pfb-imaging's `gains.qc::NET` spelling also works). Use the net gains if you
  solved for several terms.
- **Gains must be on the data's grid.** Each gain axis has to match the scan's
  times and the channel frequencies exactly, or have length one. Solutions on
  coarser intervals are rejected, not interpolated, so write net gains at the
  data's resolution. Antennas missing from the table are flagged.
- **Weights are propagated through the correction exactly**, and gain flags are
  ORed into the data flags. For diagonal gains the chi-squared is therefore the
  same as for the uncorrected residual `D - G_p M G_q^H`; the gains matter for
  what you *see* in the waterfall and for Stokes products.
- **`--stokes I,V`** adds Stokes products, formed from the corrected
  correlations, alongside `--use-corrs`. Each needs the correlations it is
  formed from (V from linear feeds needs `XY` and `YX`).
- The zarr from `--dataout` records `--rcol`, `--mcol` and `--gtable`, so
  `surfvis serve` plots waterfalls of the same quantity by default.

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

To browse corrected residuals without a `RESIDUAL` column, form them on the
fly as described in [Corrected residuals on the fly](#corrected-residuals-on-the-fly):
add `--rcol DATA --mcol MODEL_DATA --gtable gains.qc/NET` to the command above.

The waterfall panel can switch product (correlation or Stokes), subtract a
model column, apply a QuartiCal gain term (`gains.qc/NET`) and overlay a saved
CASA flag version, all on the fly. Those controls change the waterfall only;
the heatmap is what `chi2` computed, and the panel says when the two differ.

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
| `--allow-flagging` | off | Let the browser write FLAG (see below). |
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

**Save as PNG** (top right) saves the whole page, including whatever is
scrolled out of view, as a PNG at twice screen resolution. The file is named
after the current selection, e.g.
`surfvis_J1331+3030_scan1_t0_f0_XX_m009-m051.png`.

A bar across the top of the page runs while anything is loading. The part being
replaced dims, and plots show a spinner until they arrive; reading a waterfall
from the MS is the slow part.

**The controls along the top** choose which chunk you are looking at: field
(listed by name), spw, scan, then the time bin and frequency bin within that scan, then the
polarization. These are the same chunks `surfchi2` wrote PNGs for -- the bins
follow the `--ntimes` and `--nfreqs` you ran with.

**The grid** is chi-squared per degree of freedom for every antenna pair, the
same quantity as the `surfchi2` images. Rows are antenna1, columns antenna2,
and it is symmetric. Dark grey cells along the diagonal are auto-correlations,
which carry no chi-squared; dark grey anywhere else means every visibility for
that pair was flagged. Antenna names label both axes. Hovering a cell shows
both antennas' names and numbers, the chi-squared/dof and the baseline length. A single bad antenna
shows up as a bright row *and* column -- that cross is the pattern worth
looking for, and it is why the grid is worth keeping rather than just ranking
the worst pairs.

**The histogram** beside the grid is the whole distribution over antenna pairs,
with a dashed line at chi-squared/dof = 1 when that falls within the data. This is how you judge whether a
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
all channels against UTC, for the product you are viewing. The chunk you
clicked is outlined in green, so you can see whether the bad chunk is isolated
or part of something larger. Flagged data is greyed out, as are slots where the
MS has no data at all.

The selected baseline stays selected. **Arrow keys** move it (up/down change
antenna1, left/right antenna2), and changing the field, scan, bin or product
re-plots the same baseline there.

**The controls under the waterfall** re-plot it without changing your place in
the grid: product, column, model, gain table, flag version and quantity (`amp`,
`phase`, `real` or `imag`). They default to how `chi2` formed the chi-squared,
which is usually what you want -- you are asking "does this residual look like
noise?" -- but dropping the model tells you whether the problem is in the data
or in the model subtracted from it. Anything you change carries over as you
click from baseline to baseline. Once the settings no longer match the heatmap,
the panel says so.

### A note on column names (MSv2 vs MSv4)

Waterfalls are read through
[xarray-ms](https://github.com/ratt-ru/xarray-ms), which presents an MSv2 table
as an **MSv4** view. MSv4 renames the columns you are used to:

| You know it as | MSv4 calls it |
|---|---|
| `DATA` | `VISIBILITY` |
| `CORRECTED_DATA` | `CORRECTED_DATA` (kept under its own name) |
| `WEIGHT_SPECTRUM` | `WEIGHT` |
| `FLAG` | `FLAG` |

surfvis resolves this through the dataset's `data_groups` attribute rather than
hardcoding the mapping, the same way `pfb-imaging` does, so **either name
works** in the column selector and on the URL. `CORRECTED_DATA`, `MODEL_DATA`
and columns that are not part of the MSv2 standard -- `RESIDUAL`, anything your
calibration pipeline wrote -- are not renamed and appear under their own names. That is precisely why the
residual column the chi-squared came from is plottable.

### Troubleshooting

**`The web interface needs the optional extras`** -- install
`surfvis[full,web]`. Native execution needs Python 3.11 or newer; on
3.10 only the lightweight CLI is supported, and it runs every batch command in
the container.

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

`surfvis serve` is read-only unless started with `--allow-flagging`. Manual
flagging from the waterfall is in progress
([#16](https://github.com/ratt-ru/surfvis/issues/16)): the server side exists,
the selection tool in the page does not yet. Before its first write, a
flagging session saves FLAG as a CASA flag version named
`surfvis_before_<UTC timestamp>`, so `flagmanager` (or the flags menu) can
restore it.

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

## Credits

Originally written by Ian Heywood (`ianh@astro.ox.ac.uk`).

## License

MIT — see [LICENSE](LICENSE).
