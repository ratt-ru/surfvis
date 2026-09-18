# Python Standards & CLI Implementation Guidelines

Read this when editing or creating any `**/*.py` files.

## 1. Type Hints and Modern Python

- **Python 3.10+.** Use modern syntax (`X | Y`, `list[int]`, etc.).
- **Type hints on every function signature.**
- Use `from typing import Annotated` for Typer parameter annotations.

## 2. Lazy Imports in `cli/`

Heavy imports live in `core/` only. CLI wrappers under `cli/` must import from
`core/` **inside the function body**, never at module scope. This keeps the
lightweight install fast and lets the container-fallback pattern work (see
`architecture.md` §3).

- **fsspec backends stay lazy.** Never import `s3fs`, `gcsfs`, or `adlfs`
  directly. fsspec loads the matching backend on demand when a remote UPath is
  first accessed.

## 3. Typer Option / Argument Syntax (CRITICAL)

**Never pass `None` as the positional default to `typer.Option()`** — it raises
`AttributeError`. Follow these exact patterns:

- **Required:** `Annotated[T, typer.Option(..., help="...")]` (no `= default`).
- **Optional w/ default:** `Annotated[T, typer.Option(help="...")] = default`.
- **Optional None:** `Annotated[T | None, typer.Option(help="...")] = None`.

## 4. hip-cargo Types

- **Comma-separated lists:** use `ListInt`, `ListFloat`, `ListStr` from
  `hip_cargo`, with their matching `parse_list_*` parsers. Typer cannot natively
  handle variable-length lists as a single option; these `NewType` wrappers wrap
  `str` for Typer but parse into `list[int]` etc. at runtime.
- **UPath-backed path types:** `File`, `Directory`, `MS`, `URI` are
  `NewType(..., UPath)`. Generated CLIs use `parser=parse_upath` so the same
  signature accepts local paths and remote URIs. User functions receive a
  `universal_pathlib.UPath` and call `.open()` / `.exists()` directly.
- **The `stimela` metadata dict:** `Annotated[..., StimelaMeta(...)]` overrides
  inferred cab metadata. Use `StimelaMeta(skip=True)` to exclude a parameter from
  the generated cab YAML entirely.

## 5. Architectural Style

- Prefer functional, explicit-over-implicit code. Use classes when state or
  polymorphism genuinely helps.
- Keep `core/` implementations straightforward; let exceptions propagate. Use
  `typer.Exit(code=1)` for CLI errors in `cli/` (never in `core/`).
- Use Google-style docstrings (document Args, Returns, Raises). Keep them
  concise. Add short inline comments only when intent isn't obvious.
