# Architectural Rules & Domain Logic

Read this when editing `src/surfvis/**` files.

## 1. Package Layout

```
surfvis/
├── src/surfvis/
│   ├── __init__.py
│   ├── _container_image.py    # CONTAINER_IMAGE — single source of truth for the image tag
│   ├── cli/                   # Lightweight Typer wrappers. THIS is what generate-cabs parses.
│   │   ├── __init__.py        # Builds the Typer `app` and registers subcommands
│   │   └── onboard.py         # One file per subcommand (delete onboard once setup is done)
│   ├── core/                  # Real implementations. Heavy deps live here.
│   │   ├── __init__.py
│   │   └── onboard.py         # Mirrors cli/onboard.py — same function name, no decorators
│   └── cabs/                  # AUTO-GENERATED Stimela YAMLs. Do NOT hand-edit.
│       ├── __init__.py
│       └── onboard.yml
├── tests/
│   ├── test_install.py
│   └── test_roundtrip.py      # Guards the CLI → cab → CLI round-trip
├── Dockerfile                 # Builds the image referenced by _container_image.py
├── pyproject.toml
├── tbump.toml                 # Release tooling — updates _container_image.py + cabs
├── .pre-commit-config.yaml    # Runs generate-cabs on every commit
└── .github/workflows/
    ├── ci.yml
    ├── publish.yml             # PyPI on tag push
    ├── publish-container.yml   # ghcr.io on tag + every push to master
    └── update-cabs.yml         # Regenerates cabs on merge to master
```

### Role of each directory

| Directory | What lives there | What does NOT live there |
|---|---|---|
| `cli/` | Thin Typer wrappers with `@stimela_cab` (and optional `@stimela_output`). One file per command. **Imports from `core/` must be lazy** (inside the function body). | Heavy imports at module top. Business logic. NumPy / pandas / domain libs. |
| `core/` | The actual implementation. Type-hinted function with the same name as the CLI wrapper, but **no Typer / hip-cargo decorators**. Free to import anything. | Typer. `@stimela_cab`. UI concerns. `typer.Exit(...)`. |
| `cabs/` | Generated `<command>.yml` files. Committed to source control. Loaded by Stimela. | Anything you wrote by hand. Drift from `cli/*.py`. |

### Adding a new command

1. Create `src/surfvis/cli/<name>.py` with a `@stimela_cab`-decorated
   Typer function. Lazily import the core implementation inside the function.
2. Create `src/surfvis/core/<name>.py` with the real implementation —
   same function name, no decorators, free to import heavy deps.
3. Register the new command in `src/surfvis/cli/__init__.py` (next to
   the existing `onboard` registration; mirror its pattern).
4. Commit. The pre-commit hook regenerates `src/surfvis/cabs/<name>.yml`
   automatically.

**Never** create files under `cabs/` by hand. They are derived artefacts.

## 2. Lightweight vs Full Installation

This package supports two install modes. The split is what makes the
container-fallback pattern below work.

| Mode | Command | What it pulls | When to use |
|---|---|---|---|
| **Lightweight** | `pip install surfvis` | `hip-cargo` + `typer` only | Cab consumers (Stimela), CI machines that only need to dispatch commands into containers, anyone who already has the project's container image available. |
| **Full** | `pip install surfvis[full]` | Lightweight + everything listed under `[project.optional-dependencies].full` in `pyproject.toml` | Local development; native (non-container) execution. |

The lightweight install is **always sufficient to invoke any command** because
the generated CLI wrappers fall back to running the same command inside the
project's container when native imports fail (see §3).

### When you add a heavy dep

- Add it to `[project.optional-dependencies].full` in `pyproject.toml`, **not**
  to the top-level `dependencies`. The top-level list must stay tiny so the
  lightweight install remains lightweight.
- Import it **only from inside `core/`**. Never import it from `cli/` at module
  scope.

## 3. Container Fallback & Backends

Every generated CLI wrapper in `cli/*.py` follows the same shape (this is
emitted by `hip-cargo generate-function`, but the pattern matters when you
write a new command by hand too):

```python
def my_command(...):
    if backend == "native" or backend == "auto":
        try:
            from hip_cargo.utils.runner import preflight_remote_must_exist
            preflight_remote_must_exist(my_command, dict(...))
            from surfvis.core.my_command import my_command as my_command_core
            my_command_core(...)
            return
        except ImportError:
            if backend == "native":
                raise
    # Heavy deps missing OR backend explicitly chose a container → run in container.
    from hip_cargo.utils.config import get_container_image
    from hip_cargo.utils.runner import run_in_container
    image = get_container_image("surfvis")
    run_in_container(my_command, dict(...), image=image, backend=backend, ...)
```

### How `--backend` flows

Every command auto-grows two parameters via `hip-cargo generate-function`:

| Flag | Values | Effect |
|---|---|---|
| `--backend` | `auto` (default), `native`, `apptainer`, `singularity`, `docker`, `podman` | `auto` tries native then falls back to a detected container runtime. `native` forces in-process execution and surfaces the `ImportError` if `[full]` is not installed. The explicit backends skip the native attempt entirely and dispatch into the matching runtime. |
| `--always-pull-images` | bool | Forces a fresh `pull` before each container run. |

Both flags are decorated with `StimelaMeta(skip=True)` so they appear in the
Python CLI but **not** in the generated cab YAML — Stimela manages container
execution on its own side and doesn't need them.

### Image resolution

The image tag is owned by `src/surfvis/_container_image.py`:

```python
CONTAINER_IMAGE = "ghcr.io/ratt-ru/surfvis:latest"
```

Three things keep this in sync — do not bypass them:

1. **Feature branches:** Edit `_container_image.py` by hand to point at your
   branch tag (e.g. `:my-feature`). The `publish-container.yml` workflow builds
   and pushes that tag on every push of the PR.
2. **Merge to `master`:** The `update-cabs.yml` workflow resets the
   tag to `latest` and regenerates cabs in a `[skip checks]` commit.
3. **Releases:** `tbump <version>` rewrites the tag to the semantic version and
   regenerates cabs as a `before_commit` hook.

### GPU passthrough & extra run-args

For a GPU image, declare it in `src/surfvis/_container_image.py` alongside
`CONTAINER_IMAGE`:

```python
GPU = True                  # True | False | "auto" | a device spec ("device=0,1")
RUN_ARGS_APPTAINER = []     # per-backend extras: _DOCKER / _PODMAN / _APPTAINER / _SINGULARITY
```

On the container-fallback path the runner translates `GPU` per runtime (`--gpus`
for docker, CDI `--device nvidia.com/gpu=...` for podman, `--nv` for
apptainer/singularity) and appends the matching `RUN_ARGS_*` verbatim. `"auto"`
only requests a GPU when one is detected (and, for docker/podman, the NVIDIA
Container Toolkit is present). Override per-invocation with `HIP_CARGO_GPUS`
(e.g. `HIP_CARGO_GPUS=none`) and `HIP_CARGO_RUN_ARGS`. These constants live only
in `_container_image.py` — they are deliberately kept out of the cab YAML, since
Stimela manages its own container execution. Explicit device specs (e.g. `device=0,1`) target docker and apptainer/singularity; on podman prefer `True`/`"auto"`/`"all"` since CDI uses a different device grammar.

### Remote URIs (S3 / GCS / Azure)

Path-typed parameters (`File`, `Directory`, `MS`, `URI`) accept both local
paths and remote URIs (`s3://...`, `gs://...`, `az://...`). When the path is
remote:

- `_resolve_mounts` skips it (nothing to bind-mount).
- `preflight_remote_must_exist` checks existence via fsspec.
- `run_in_container` forwards the matching credentials (`AWS_*`, `~/.aws`,
  `GOOGLE_APPLICATION_CREDENTIALS`, `~/.config/gcloud`, `AZURE_*`, `~/.azure`).

Users who want native remote access install the right extra: `pip install
hip-cargo[s3]`, `[gcs]`, or `[azure]`. Without it, the wrapper's existing
`try/except ImportError` routes them into the container, which already has the
backends.

## 4. Cab Generation is Automatic

**The `src/surfvis/cabs/*.yml` files are generated artefacts. Never edit
them by hand and never run `hip-cargo generate-cabs` manually.**

Three automated paths keep them in sync with `cli/*.py`:

1. **Pre-commit hook** (`.pre-commit-config.yaml`): on every commit, runs
   `hip-cargo generate-cabs --module src/surfvis/cli/*.py --output-dir
   src/surfvis/cabs`. If it modifies files, pre-commit will "fail" the
   commit — re-run `git add -u && git commit` to include the updates.
2. **`update-cabs.yml` workflow**: on merge to `master`, resets the
   container tag to `latest` and regenerates cabs in a `[skip checks]` commit.
3. **`tbump`**: on release, rewrites the container tag to the version and
   regenerates cabs.

If you ever see a cab YAML in a diff that wasn't generated by one of these
three paths, that's a bug — revert it and edit the corresponding `cli/*.py`
instead.

> **Heads-up:** `generate-cabs` resolves the `image:` field from the *installed*
> package metadata. Activate the project venv (so `surfvis` is importable)
> before committing — otherwise the regenerated cab is written without an
> `image:` field.

### How CLI source maps to cab YAML

- `@stimela_cab(name=..., info=...)` → the cab's name and top-level info.
- `@stimela_output(...)` → entries under `outputs:` in the cab.
- Each Typer parameter → an entry under `inputs:` (dtype inferred from the type
  hint, `info` from `help=`, defaults from `= ...`).
- `Annotated[..., StimelaMeta(skip=True)]` → omitted from the cab (used for
  `--backend`, `--always-pull-images`, etc.).
- `Annotated[..., StimelaMeta(metadata={"rich_help_panel": "Inputs", "tunable":
  True})]` → flows into the cab's `metadata:` dict.
- Inline comments after `Annotated[...]` rows are preserved through the round
  trip — they show up as `# noqa: ...` or similar on the matching cab field.
