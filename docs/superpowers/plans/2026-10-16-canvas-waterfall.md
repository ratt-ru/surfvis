# Canvas Waterfall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the waterfall PNG in `surfvis serve` with a canvas drawn in the browser, fed by a binary data route that returns a screen-sized reduction of a cached plane, with wheel zoom, shift-drag pan and a hover readout.

**Architecture:** The server forms one baseline's complex plane once (`msdata.cached_plane`, LRU of 4, invalidated by flag writes), reduces any window of it to at most `max_t × max_c` bins (`web/planes.py`, numpy only) and sends it as a small binary block (`GET /waterfall.bin`). The page decodes the block, colours it with a 256-entry LUT and draws it on a `<canvas>` (`static/wf/*.mjs`, plain ES modules, no build step), refetching the visible window after each zoom or pan.

**Tech Stack:** Python 3.11, FastAPI, numpy, xarray-ms/arcae (existing); vanilla JS ES modules; node ≥ 18 `node --test` for the pure JS modules.

**Spec:** `docs/superpowers/specs/2026-10-16-canvas-waterfall-design.md` — read it first; this plan argues from it.

## Global Constraints

- Branch `flagging`. Read `CLAUDE.md` and `.claude/rules/*.md` before starting.
- `web/` stays dask-free. No new Python dependencies. No new vendored JS library, no CDN, no build step.
- After any Python change: `uv run ruff format . && uv run ruff check . --fix`.
- Full suite (the only complete run): `docker build -q -t surfvis-web:local . && docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/src -w /src surfvis-web:local bash -c "pip install --quiet --target /tmp/t pytest httpx2; PYTHONPATH=/tmp/t:/src/src:/src /tmp/t/bin/pytest tests/ -q -W ignore -p no:cacheprovider"`. The local venv also has the extras: `uv run pytest -q -p no:cacheprovider` runs everything, plus the node tests (node is not in the container).
- Every new or removed test function: update `EXPECTED` in `tests/test_suite_integrity.py` (module → (total, MS-marked)). A module is either all `@needs_web`/`@needs_ms` or none.
- Header JSON must never contain NaN or Infinity (`json.dumps(..., allow_nan=False)`).
- Data coordinates: slot `i` spans `[i-0.5, i+0.5]`, channel `j` spans `[j-0.5, j+0.5]` (same as `web/flagging.rasterise`).
- Bin caps: `MAX_BINS = 4096` per axis; route default `max_t = max_c = 1024`.
- Commits: Conventional Commits, imperative, first line < 72 chars, ending with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Do not push.
- Do not edit `src/surfvis/cabs/` or `src/surfvis/cli/*.py` other than `serve.py` (none needed here).

## Review Focus

- **All-flagged or constant plane** → colour limits finite (`[0, 1]` or `vmax = vmin + 1`), header encodes; tested in Task 2 and Task 3.
- **Plane sizes not divisible by the bin size** (ragged last bin) → the last bin reduces only the cells that exist, and its flag fraction counts only those; tested in Task 2.
- **A flag write or a settings change while a plane is cached** → the next block reflects it, never a stale plane; tested in Task 1.
- **Fast zooming, responses arriving out of order** → only the newest response is drawn; tested in Task 4 (`Latest`).
- **Windows partly or wholly outside the plane, inverted, or narrower than a cell; planes one slot long** → clamped, a 400 only when nothing remains, zoom never collapses or escapes the extent; tested in Tasks 3 and 4.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/surfvis/web/msdata.py` (modify) | `Waterfall.plane`, `Waterfall.freqs`; `cached_plane`; `invalidate` clears it | 1 |
| `src/surfvis/web/planes.py` (create) | `reduce_block`, `colour_limits`, `encode`, `decode` — numpy only | 2 |
| `src/surfvis/web/app.py` (modify) | `GET /waterfall.bin`; static files with `Cache-Control: no-cache` | 3, 5 |
| `src/surfvis/web/static/wf/format.mjs` | decode the wire format | 4 |
| `src/surfvis/web/static/wf/luts.mjs` | generated colormap tables | 4 |
| `tools/make_luts.py` | generates `luts.mjs` from matplotlib | 4 |
| `src/surfvis/web/static/wf/colormap.mjs` | `paint` values+flags into RGBA | 4 |
| `src/surfvis/web/static/wf/view.mjs` | `View`, `Latest`, `niceTicks` | 4 |
| `src/surfvis/web/static/wf/widget.mjs` | canvas widget | 5 |
| `src/surfvis/web/templates/_waterfall.html`, `static/app.js`, `static/style.css` (modify) | mount the widget | 5 |
| `tests/test_planes.py` (create, light) | Task 2 tests | 2 |
| `tests/test_js.py` (create, light) + `tests/js/*.test.mjs` | node runner and JS tests | 4 |
| `tests/test_web.py`, `tests/test_flagging.py` (modify) | Tasks 1, 3, 5 tests | 1, 3, 5 |
| `docs/wiki/web-app.md`, `docs/wiki/log.md`, `README.md`, `CLAUDE.md` | docs | 5 |

---

### Task 1: Cached complex plane

**Files:**
- Modify: `src/surfvis/web/msdata.py` (the `Waterfall` dataclass; end of `waterfall()`; `invalidate()`; new `cached_plane`, `_cached_plane`, `_gtable_stamp`)
- Test: `tests/test_web.py`, `tests/test_flagging.py`, `tests/test_suite_integrity.py`

**Interfaces:**
- Produces: `msdata.cached_plane(ms: str, field: int, spw: int, scan: int, name1: str, name2: str, product: str, column: str, model: str = "", gtable: str = "", flagversion: str = "", rect: tuple[int, int, int, int] | None = None) -> Waterfall`. The returned `Waterfall` has `plane: np.ndarray` (complex, `(ntime, nchan)`), `flags: np.ndarray` (bool, same shape), `times: np.ndarray` (unix s, `(ntime,)`), `freqs: np.ndarray` (Hz, `(nchan,)`), `rect`, `length`, `antenna1`, `antenna2`, `scan`, `column`, `product`. Its `quantity`/`values` are `"amp"` and must not be relied on.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_web.py`:

```python
@needs_web
def test_plane_cache_reuses_and_invalidates(chi2_zarr, monkeypatch):
    """Zooming re-slices a cached plane; new settings or an invalidate re-read the MS."""
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    msdata.invalidate(ms)
    calls = []
    real = msdata.waterfall
    monkeypatch.setattr(msdata, "waterfall", lambda *a, **k: calls.append(1) or real(*a, **k))

    a = msdata.cached_plane(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    b = msdata.cached_plane(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert a is b and len(calls) == 1
    assert np.iscomplexobj(a.plane) and a.plane.shape == a.flags.shape == (12, 32)
    assert a.freqs.shape == (32,) and a.times.shape == (12,)

    msdata.cached_plane(ms, 0, 0, 1, names[0], names[2], "YY", "RESIDUAL")
    assert len(calls) == 2
    msdata.invalidate(ms)
    msdata.cached_plane(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert len(calls) == 3
```

Append to `tests/test_flagging.py`:

```python
@needs_web
def test_cached_plane_sees_a_flag_write(chi2_zarr, writable_ms):
    """A plane cached before a flag write must not be served after it."""
    from surfvis.web import msdata
    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    before = msdata.cached_plane(writable_ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert not before.flags[2:4, 5:9].all()
    Flagger(writable_ms).apply(0, 0, 1, names[0], names[2], np.array(_rect(2, 3, 5, 8)), corr_idx=[0])
    after = msdata.cached_plane(writable_ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert after.flags[2:4, 5:9].all()
```

In `tests/test_suite_integrity.py` change `"test_web.py": (24, 24)` to `(25, 25)` and `"test_flagging.py": (6, 6)` to `(7, 7)`.

- [ ] **Step 2: Run them to see them fail.**
Run: `uv run pytest tests/test_web.py::test_plane_cache_reuses_and_invalidates tests/test_flagging.py::test_cached_plane_sees_a_flag_write -q -p no:cacheprovider`
Expected: FAIL with `AttributeError: module 'surfvis.web.msdata' has no attribute 'cached_plane'`.

- [ ] **Step 3: Implement.** In `msdata.py`, add two fields at the end of `Waterfall`:

```python
    # The complex plane the values were taken from, and the channel
    # frequencies in Hz; None where not known.
    plane: np.ndarray | None = None
    freqs: np.ndarray | None = None
```

In `waterfall()`, add to the `Waterfall(...)` it returns:

```python
        plane=plane,
        freqs=np.asarray(ds.frequency.values, dtype=float),
```

Add after `waterfall()`:

```python
def cached_plane(
    ms: str,
    field: int,
    spw: int,
    scan: int,
    name1: str,
    name2: str,
    product: str,
    column: str,
    model: str = "",
    gtable: str = "",
    flagversion: str = "",
    rect: tuple[int, int, int, int] | None = None,
) -> Waterfall:
    """One baseline's formed complex plane, cached so zooming does not re-read the MS.

    The key includes the MS's on-disk fingerprint and the gain term's mtime,
    so a rewritten MS or a re-solved gain table is read afresh. ``quantity``
    is not part of the key: use ``plane``, not ``values``.
    """
    return _cached_plane(
        ms, field, spw, scan, name1, name2, product, column, model, gtable, flagversion, rect,
        _fingerprint(ms), _gtable_stamp(gtable),
    )


@lru_cache(maxsize=4)
def _cached_plane(ms, field, spw, scan, name1, name2, product, column, model, gtable, flagversion, rect, fingerprint, gstamp):
    # fingerprint and gstamp only key the cache.
    return waterfall(
        ms, field, spw, scan, name1, name2, product, column,
        quantity="amp", rect=rect, model=model, gtable=gtable, flagversion=flagversion,
    )


def _gtable_stamp(gtable: str) -> int | None:
    """mtime of a gain term's directory, or None."""
    if not gtable:
        return None
    try:
        return os.stat(gains_mod.term_path(gtable)).st_mtime_ns
    except OSError:
        return None
```

In `invalidate()`, add `_cached_plane.cache_clear()` after `_open_partition.cache_clear()`. Run ruff (it will reflow the long calls).

- [ ] **Step 4: Run the tests to see them pass.** Same command as Step 2. Expected: 2 passed. Then `uv run pytest tests/test_web.py tests/test_flagging.py tests/test_suite_integrity.py -q -p no:cacheprovider` — all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/surfvis/web/msdata.py tests/test_web.py tests/test_flagging.py tests/test_suite_integrity.py
git commit -m "feat(web): cache a baseline's complex plane for zooming

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Reduction and wire format (`planes.py`)

**Files:**
- Create: `src/surfvis/web/planes.py`
- Test: `tests/test_planes.py` (numpy only, no marker), `tests/test_suite_integrity.py`

**Interfaces:**
- Produces:
  - `planes.MAX_BINS: int = 4096`
  - `planes.Block` (frozen dataclass): `values: np.ndarray` (float32, `(nt, nc)`), `flags: np.ndarray` (uint8, `(nt, nc)`), `t0, t1, c0, c1, tstep, cstep: int`
  - `planes.reduce_block(plane, flags, quantity, t0, t1, c0, c1, max_t, max_c) -> Block`
  - `planes.colour_limits(plane, flags, quantity) -> tuple[float, float]`
  - `planes.encode(header: dict, block: Block) -> bytes` (sets `header["nt"]`, `["nc"]`, `["version"]` itself)
  - `planes.decode(buf: bytes) -> tuple[dict, np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing tests** in `tests/test_planes.py`:

```python
"""Reducing a waterfall plane to a screen-sized block, and its wire format.

numpy only, so these run wherever numpy is installed -- no Measurement Set.
"""

import math

import pytest

np = pytest.importorskip("numpy")

from surfvis.web import planes  # noqa: E402


def _plane(nt, nc, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(nt, nc)) + 1j * rng.normal(size=(nt, nc))


def test_full_resolution_is_the_quantity_itself():
    z = _plane(5, 7)
    f = np.zeros(z.shape, bool)
    for quantity, expect in (("amp", np.abs(z)), ("real", z.real), ("imag", z.imag), ("phase", np.angle(z))):
        b = planes.reduce_block(z, f, quantity, 0, 5, 0, 7, 100, 100)
        assert (b.tstep, b.cstep) == (1, 1) and b.values.dtype == np.float32
        np.testing.assert_allclose(b.values, expect, rtol=1e-6, atol=1e-6)
        assert not b.flags.any()


def test_amp_is_the_peak_of_the_unflagged_cells():
    z = np.ones((2, 2), complex)
    z[0, 1] = 9  # unflagged spike: must survive the reduction
    f = np.zeros(z.shape, bool)
    assert planes.reduce_block(z, f, "amp", 0, 2, 0, 2, 1, 1).values[0, 0] == 9
    f[0, 1] = True  # now flagged: ignored while unflagged cells remain
    b = planes.reduce_block(z, f, "amp", 0, 2, 0, 2, 1, 1)
    assert b.values[0, 0] == 1 and b.flags[0, 0] == round(255 * 1 / 4)


def test_fully_flagged_bin_falls_back_to_all_cells():
    z = np.array([[1.0, 5.0]], complex)
    b = planes.reduce_block(z, np.ones(z.shape, bool), "amp", 0, 1, 0, 2, 1, 1)
    assert b.values[0, 0] == 5 and b.flags[0, 0] == 255


def test_ragged_last_bins_use_only_cells_that_exist():
    # 5 x 7 into at most 2 x 3 bins: steps 3 x 3, bins 2 x 3, last bins ragged.
    z = np.arange(35, dtype=float).reshape(5, 7).astype(complex)
    f = np.zeros(z.shape, bool)
    f[4, 6] = True  # the only cell in the corner bin that is flagged...
    b = planes.reduce_block(z, f, "amp", 0, 5, 0, 7, 2, 3)
    assert (b.tstep, b.cstep) == (3, 3) and b.values.shape == (2, 3)
    # Corner bin holds rows 3-4, channel 6 only: cells 27 and 34(flagged).
    assert b.values[1, 2] == 27
    assert b.flags[1, 2] == round(255 * 1 / 2)  # ...out of 2 cells, not 9
    assert b.values[0, 0] == np.abs(z[0:3, 0:3]).max()


def test_signed_peak_and_circular_mean():
    z = np.array([[1 - 4j, -6 + 1j]])
    f = np.zeros(z.shape, bool)
    assert planes.reduce_block(z, f, "real", 0, 1, 0, 2, 1, 1).values[0, 0] == -6
    assert planes.reduce_block(z, f, "imag", 0, 1, 0, 2, 1, 1).values[0, 0] == -4
    # Phases either side of +-pi average to pi, not 0.
    w = np.exp(1j * np.array([[3.1, -3.1]]))
    ph = planes.reduce_block(w, f, "phase", 0, 1, 0, 2, 1, 1).values[0, 0]
    assert abs(abs(ph) - math.pi) < 1e-6


def test_window_matches_reducing_the_slice():
    z = _plane(10, 20, seed=3)
    f = np.random.default_rng(4).random(z.shape) < 0.2
    win = planes.reduce_block(z, f, "amp", 2, 7, 5, 17, 2, 4)
    sub = planes.reduce_block(z[2:7, 5:17], f[2:7, 5:17], "amp", 0, 5, 0, 12, 2, 4)
    np.testing.assert_array_equal(win.values, sub.values)
    np.testing.assert_array_equal(win.flags, sub.flags)
    assert (win.t0, win.t1, win.c0, win.c1) == (2, 7, 5, 17)


def test_colour_limits_are_finite_and_ordered():
    z = _plane(20, 30)
    f = np.zeros(z.shape, bool)
    lo, hi = planes.colour_limits(z, f, "amp")
    assert lo == pytest.approx(np.abs(z).min()) and hi == pytest.approx(np.percentile(np.abs(z), 99))
    assert planes.colour_limits(z, f, "phase") == (-math.pi, math.pi)
    assert planes.colour_limits(z, np.ones(z.shape, bool), "amp") == (0.0, 1.0)
    assert planes.colour_limits(np.full((3, 3), 2 + 0j), np.zeros((3, 3), bool), "amp") == (2.0, 3.0)


def test_encode_round_trips_and_rejects_nan_headers():
    z = _plane(3, 5)
    b = planes.reduce_block(z, np.zeros(z.shape, bool), "amp", 0, 3, 0, 5, 10, 10)
    header = {"title": "m000-m001 χ", "vmin": 0.0, "vmax": 1.0}
    buf = planes.encode(header, b)
    assert buf[:4] == b"SVWF"
    got, values, flags = planes.decode(buf)
    assert got["nt"] == 3 and got["nc"] == 5 and got["version"] == 1 and got["title"] == header["title"]
    np.testing.assert_array_equal(values, b.values)
    np.testing.assert_array_equal(flags, b.flags)
    with pytest.raises(ValueError):
        planes.encode({"vmin": float("nan")}, b)
```

In `tests/test_suite_integrity.py` add `"test_planes.py": (8, 0),` to `EXPECTED`.

- [ ] **Step 2: Run them to see them fail.**
Run: `uv run pytest tests/test_planes.py -q -p no:cacheprovider`
Expected: FAIL (`ImportError: cannot import name 'planes'`).

- [ ] **Step 3: Implement** `src/surfvis/web/planes.py`:

```python
"""Reduce a baseline's complex plane to a screen-sized block, and encode it for the browser.

numpy only. The browser asks for a window of the plane at no more bins than it
has pixels; this module does the reduction and the binary packing. The wire
format is documented in docs/superpowers/specs/2026-10-16-canvas-waterfall-design.md.
"""

import json
import math
import struct
from dataclasses import dataclass

import numpy as np

MAX_BINS = 4096
MAGIC = b"SVWF"
VERSION = 1


@dataclass(frozen=True)
class Block:
    """A window of a plane, reduced to bins. Bins are aligned at ``t0``/``c0``; the last may be ragged."""

    values: np.ndarray
    flags: np.ndarray
    t0: int
    t1: int
    c0: int
    c1: int
    tstep: int
    cstep: int


def _binned(a: np.ndarray, tstep: int, cstep: int, fill) -> np.ndarray:
    """``(T, C)`` -> ``(nt, nc, tstep * cstep)``, padding the ragged edges with ``fill``."""
    T, C = a.shape
    nt, nc = -(-T // tstep), -(-C // cstep)
    padded = np.full((nt * tstep, nc * cstep), fill, dtype=a.dtype)
    padded[:T, :C] = a
    return padded.reshape(nt, tstep, nc, cstep).transpose(0, 2, 1, 3).reshape(nt, nc, tstep * cstep)


def reduce_block(
    plane: np.ndarray,
    flags: np.ndarray,
    quantity: str,
    t0: int,
    t1: int,
    c0: int,
    c1: int,
    max_t: int,
    max_c: int,
) -> Block:
    """Reduce ``plane[t0:t1, c0:c1]`` to at most ``max_t x max_c`` bins.

    Per bin, the cells used are the unflagged ones, or all of them if every
    cell is flagged. ``amp`` takes the peak (RFI must not be averaged away),
    ``real``/``imag`` the signed value of largest magnitude, ``phase`` the
    circular mean. The flag byte is the flagged fraction of the bin's cells,
    scaled to 0-255.
    """
    if max_t < 1 or max_c < 1:
        raise ValueError("max_t and max_c must be at least 1")
    z = plane[t0:t1, c0:c1]
    f = flags[t0:t1, c0:c1]
    tstep = -(-z.shape[0] // max_t)
    cstep = -(-z.shape[1] // max_c)
    zb = _binned(z, tstep, cstep, 0j)
    fb = _binned(f, tstep, cstep, False)
    present = _binned(np.ones(z.shape, bool), tstep, cstep, False)

    flagged = present & fb
    used = present & ~fb
    none_left = ~used.any(axis=-1, keepdims=True)
    used = np.where(none_left, present, used)

    if quantity == "amp":
        values = np.where(used, np.abs(zb), -np.inf).max(axis=-1)
    elif quantity in ("real", "imag"):
        part = zb.real if quantity == "real" else zb.imag
        k = np.where(used, np.abs(part), -1.0).argmax(axis=-1)
        values = np.take_along_axis(part, k[..., None], axis=-1)[..., 0]
    elif quantity == "phase":
        mag = np.abs(zb)
        unit = np.divide(zb, mag, out=np.zeros_like(zb), where=mag > 0)
        values = np.angle(np.where(used, unit, 0).sum(axis=-1))
    else:
        raise ValueError(f"Unknown quantity {quantity!r}")

    frac = flagged.sum(axis=-1) / present.sum(axis=-1)
    return Block(
        values=values.astype(np.float32),
        flags=np.round(frac * 255).astype(np.uint8),
        t0=t0,
        t1=t1,
        c0=c0,
        c1=c1,
        tstep=tstep,
        cstep=cstep,
    )


def colour_limits(plane: np.ndarray, flags: np.ndarray, quantity: str) -> tuple[float, float]:
    """Colour range from the whole plane's unflagged, finite values; always finite and ordered."""
    if quantity == "phase":
        return (-math.pi, math.pi)
    part = {"amp": np.abs, "real": np.real, "imag": np.imag}[quantity](plane)
    v = part[~flags & np.isfinite(part)]
    if v.size == 0:
        return (0.0, 1.0)
    lo, hi = (float(v.min()), float(np.percentile(v, 99))) if quantity == "amp" else tuple(map(float, np.percentile(v, [1, 99])))
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def encode(header: dict, block: Block) -> bytes:
    """Pack a block: magic, version, header length, JSON header, padding, float32 values, uint8 flags."""
    header = {**header, "version": VERSION, "nt": int(block.values.shape[0]), "nc": int(block.values.shape[1])}
    meta = json.dumps(header, allow_nan=False).encode()
    head = MAGIC + struct.pack("<II", VERSION, len(meta)) + meta
    head += b"\0" * (-len(head) % 4)
    return head + block.values.astype("<f4").tobytes() + block.flags.astype(np.uint8).tobytes()


def decode(buf: bytes) -> tuple[dict, np.ndarray, np.ndarray]:
    """Inverse of :func:`encode`."""
    if buf[:4] != MAGIC:
        raise ValueError("Not a surfvis waterfall block")
    _, n = struct.unpack("<II", buf[4:12])
    header = json.loads(buf[12 : 12 + n])
    offset = 12 + n + (-(12 + n) % 4)
    count = header["nt"] * header["nc"]
    values = np.frombuffer(buf, "<f4", count, offset).reshape(header["nt"], header["nc"])
    flags = np.frombuffer(buf, np.uint8, count, offset + 4 * count).reshape(header["nt"], header["nc"])
    return header, values, flags
```

Run ruff.

- [ ] **Step 4: Run the tests to see them pass.**
Run: `uv run pytest tests/test_planes.py tests/test_suite_integrity.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/surfvis/web/planes.py tests/test_planes.py tests/test_suite_integrity.py
git commit -m "feat(web): reduce waterfall planes to screen-sized binary blocks

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `GET /waterfall.bin`

**Files:**
- Modify: `src/surfvis/web/app.py` (new route next to `waterfall_png`; import `planes`)
- Test: `tests/test_web.py`, `tests/test_suite_integrity.py`

**Interfaces:**
- Consumes: `msdata.cached_plane` (Task 1); `planes.reduce_block`, `planes.colour_limits`, `planes.encode`, `planes.MAX_BINS` (Task 2); the existing `_chunk`, `_waterfall_settings` helpers inside `create_app`.
- Produces: `GET /waterfall.bin` with the query parameters of `/waterfall.png` plus `t0: int = 0`, `t1: int | None = None`, `c0: int = 0`, `c1: int | None = None`, `max_t: int = 1024`, `max_c: int = 1024`, `meta: bool = False`. Response `application/octet-stream`, `Cache-Control: no-store`, body per `planes.encode`. Header keys: `t0, t1, c0, c1, tstep, cstep, ntime, nchan, vmin, vmax, quantity, colormap, title, rect` (+ `times`, `freqs` with `meta`), plus `version, nt, nc` from `encode`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_web.py`:

```python
@needs_web
def test_waterfall_bin(chi2_zarr):
    """Overview, window, clamping, caps and errors of the canvas data route."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web import msdata
    from surfvis.web.app import create_app
    from surfvis.web.planes import decode

    client = TestClient(create_app(chi2_zarr["zarr"]))
    names = chi2_zarr["ms"]["antenna_names"]
    base = "/waterfall.bin?field=0&spw=0&scan=1&t=0&f=0&c=XX&p=1&q=5&column=RESIDUAL"

    r = client.get(base + "&meta=true")
    assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
    assert r.headers["cache-control"] == "no-store"
    head, values, flags = decode(r.content)
    assert (head["nt"], head["nc"], head["tstep"], head["cstep"]) == (12, 32, 1, 1)
    assert (head["ntime"], head["nchan"]) == (12, 32) and head["rect"] == [0, 6, 0, 16]
    assert len(head["times"]) == 12 and len(head["freqs"]) == 32
    assert head["colormap"] == "viridis" and head["vmin"] < head["vmax"]
    assert head["title"].startswith(f"{names[1]}-{names[5]}")
    wf = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[1], names[5], "XX", "RESIDUAL")
    ok = ~wf.flags
    np.testing.assert_allclose(values[ok], wf.values[ok], rtol=1e-5)

    small, _, _ = decode(client.get(base + "&max_t=4&max_c=5").content)
    assert (small["tstep"], small["cstep"], small["nt"], small["nc"]) == (3, 7, 4, 5)
    assert "times" not in small
    assert (small["vmin"], small["vmax"]) == (head["vmin"], head["vmax"])  # colours do not move on zoom

    win, wvalues, _ = decode(client.get(base + "&t0=2&t1=5&c0=10&c1=20").content)
    assert (win["nt"], win["nc"], win["t0"], win["c1"]) == (3, 10, 2, 20)
    np.testing.assert_array_equal(wvalues, values[2:5, 10:20])

    clamped, _, _ = decode(client.get(base + "&t0=-5&t1=999&max_t=100000").content)
    assert (clamped["t0"], clamped["t1"], clamped["nt"]) == (0, 12, 12)

    phase, _, _ = decode(client.get(base + "&quantity=phase").content)
    assert phase["colormap"] == "twilight" and phase["quantity"] == "phase"

    assert client.get(base + "&t0=5&t1=5").status_code == 400
    assert client.get(base + "&c0=40").status_code == 400
    assert client.get(base + "&quantity=nope").status_code == 400
    assert client.get(base.replace("RESIDUAL", "NOPE")).status_code == 404
```

In `tests/test_suite_integrity.py` bump `"test_web.py"` by one, to `(26, 26)`.

- [ ] **Step 2: Run it to see it fail.**
Run: `uv run pytest tests/test_web.py::test_waterfall_bin -q -p no:cacheprovider`
Expected: FAIL, `assert 404 == 200` (no such route).

- [ ] **Step 3: Implement.** In `app.py` add `planes` to the import line (`from surfvis.web import msdata, planes, render`). Add, directly after the `waterfall_png` route inside `create_app`:

```python
    @app.get("/waterfall.bin")
    def waterfall_bin(
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        t: int = Query(...),
        f: int = Query(...),
        c: str = Query(...),
        p: int = Query(...),
        q: int = Query(...),
        product: str = Query(""),
        column: str = Query(""),
        model: str | None = Query(None),
        gtable: str | None = Query(None),
        flagversion: str = Query(""),
        quantity: str = Query("amp"),
        t0: int = Query(0),
        t1: int | None = Query(None),
        c0: int = Query(0),
        c1: int | None = Query(None),
        max_t: int = Query(1024),
        max_c: int = Query(1024),
        meta: bool = Query(False),
    ):
        """A window of one baseline's waterfall, reduced to at most max_t x max_c bins, for the canvas."""
        if quantity not in msdata.QUANTITIES:
            raise HTTPException(status_code=400, detail=f"quantity must be one of {msdata.QUANTITIES}")
        chunk = _chunk(field, spw, scan, t, f, c)
        names = chunk.antenna_names
        settings = _waterfall_settings(chunk, product, column, model, gtable, flagversion, quantity)
        settings.pop("quantity")
        try:
            wf = msdata.cached_plane(
                ms_path,
                field,
                spw,
                scan,
                names[p],
                names[q],
                rect=(chunk.t0, chunk.tf, chunk.chan0, chunk.chanf),
                **settings,
            )
        except (KeyError, IndexError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        ntime, nchan = wf.plane.shape
        t0, t1 = max(t0, 0), min(ntime if t1 is None else t1, ntime)
        c0, c1 = max(c0, 0), min(nchan if c1 is None else c1, nchan)
        if t0 >= t1 or c0 >= c1:
            raise HTTPException(status_code=400, detail=f"Empty window: slots {t0}-{t1}, channels {c0}-{c1}")
        cap = planes.MAX_BINS
        block = planes.reduce_block(
            wf.plane, wf.flags, quantity, t0, t1, c0, c1, min(max(max_t, 1), cap), min(max(max_c, 1), cap)
        )
        vmin, vmax = planes.colour_limits(wf.plane, wf.flags, quantity)
        length = f" ({wf.length:.0f} m)" if wf.length is not None else ""
        header = {
            "t0": t0,
            "t1": t1,
            "c0": c0,
            "c1": c1,
            "tstep": block.tstep,
            "cstep": block.cstep,
            "ntime": ntime,
            "nchan": nchan,
            "vmin": vmin,
            "vmax": vmax,
            "quantity": quantity,
            "colormap": "twilight" if quantity == "phase" else "viridis",
            "title": f"{wf.antenna1}-{wf.antenna2}{length}   scan {wf.scan}   {wf.column}   {quantity}   {wf.product}",
            "rect": list(wf.rect) if wf.rect is not None else None,
        }
        if meta:
            header["times"] = [float(x) for x in wf.times] if wf.times is not None else None
            header["freqs"] = [float(x) for x in wf.freqs] if wf.freqs is not None else None
        return Response(
            planes.encode(header, block),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )
```

Run ruff.

- [ ] **Step 4: Run the tests to see them pass.**
Run: `uv run pytest tests/test_web.py tests/test_suite_integrity.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/surfvis/web/app.py tests/test_web.py tests/test_suite_integrity.py
git commit -m "feat(web): serve waterfall windows as binary blocks

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Pure JS modules — decoding, colours, view maths

**Files:**
- Create: `src/surfvis/web/static/wf/format.mjs`, `src/surfvis/web/static/wf/colormap.mjs`, `src/surfvis/web/static/wf/view.mjs`
- Create: `tools/make_luts.py`; generated `src/surfvis/web/static/wf/luts.mjs`
- Create: `tests/js/format.test.mjs`, `tests/js/colormap.test.mjs`, `tests/js/view.test.mjs`, `tests/test_js.py`
- Modify: `tests/test_suite_integrity.py`

**Interfaces:**
- Consumes: the wire format from Task 2 (`planes.encode`).
- Produces (all ES module named exports):
  - `format.mjs`: `decode(buffer: ArrayBuffer) -> {header: object, values: Float32Array, flags: Uint8Array}`; throws `Error` on a bad magic or version.
  - `luts.mjs`: `LUTS: {viridis: Uint8Array(768), twilight: Uint8Array(768)}` (RGB triplets).
  - `colormap.mjs`: `NAN_RGB = [128, 128, 128]`, `FLAG_ALPHA = 0.55`, `paint(values, flags, lut, vmin, vmax, out: Uint8ClampedArray, {nt, nc, showFlags = true})` — writes RGBA with **block row 0 (earliest time) as the bottom image row**.
  - `view.mjs`: `class View(ntime, nchan)` with `win {t0, t1, c0, c1}` (data coordinates), `extent`, `reset()`, `zoomAt(factor, t, c)`, `pan(dt, dc)`, `toScreen(t, c, rect) -> {x, y}`, `toData(x, y, rect) -> {t, c}`, `request(rect, dpr, cap = 4096) -> {t0, t1, c0, c1, max_t, max_c}`; `class Latest` with `next() -> int` and `accept(id) -> boolean`; `niceTicks(lo, hi, target = 6, integer = false) -> number[]`. `rect` is `{x, y, w, h}` in CSS pixels; time increases **upward**.

- [ ] **Step 1: Generate the LUTs.** Create `tools/make_luts.py`:

```python
"""Write src/surfvis/web/static/wf/luts.mjs from matplotlib's colormaps.

Run from the repo root: uv run python tools/make_luts.py
"""

from pathlib import Path

import numpy as np
from matplotlib import colormaps

OUT = Path(__file__).resolve().parents[1] / "src/surfvis/web/static/wf/luts.mjs"
NAMES = ("viridis", "twilight")


def table(name: str) -> str:
    rgb = np.round(colormaps[name](np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(int)
    return ",".join(map(str, rgb.ravel()))


lines = [
    "// Generated by tools/make_luts.py from matplotlib -- do not edit.",
    "// 256 RGB triplets per colormap.",
    "export const LUTS = {",
    *(f"  {name}: new Uint8Array([{table(name)}])," for name in NAMES),
    "};",
    "",
]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines))
print(f"wrote {OUT}")
```

Run: `uv run python tools/make_luts.py`. Expected: `wrote .../luts.mjs`.

- [ ] **Step 2: Write the failing JS tests.** `tests/js/format.test.mjs`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { decode } from "../../src/surfvis/web/static/wf/format.mjs";

// Build a block the way planes.encode does: magic, version, length, JSON, pad to 4, f32, u8.
function encode(header, values, flags) {
  const meta = new TextEncoder().encode(JSON.stringify(header));
  const headLen = 12 + meta.length;
  const pad = (4 - (headLen % 4)) % 4;
  const buf = new ArrayBuffer(headLen + pad + values.length * 5);
  const u8 = new Uint8Array(buf);
  u8.set([83, 86, 87, 70]); // "SVWF"
  const dv = new DataView(buf);
  dv.setUint32(4, 1, true);
  dv.setUint32(8, meta.length, true);
  u8.set(meta, 12);
  new Float32Array(buf, headLen + pad, values.length).set(values);
  u8.set(flags, headLen + pad + values.length * 4);
  return buf;
}

test("decodes header, values and flags", () => {
  const buf = encode({ nt: 2, nc: 3, title: "m000 χ" }, [1, 2, 3, 4, 5, NaN], [0, 255, 0, 0, 128, 0]);
  const { header, values, flags } = decode(buf);
  assert.equal(header.title, "m000 χ");
  assert.deepEqual(Array.from(values.slice(0, 5)), [1, 2, 3, 4, 5]);
  assert.ok(Number.isNaN(values[5]));
  assert.deepEqual(Array.from(flags), [0, 255, 0, 0, 128, 0]);
});

test("rejects a buffer that is not a block", () => {
  assert.throws(() => decode(new ArrayBuffer(16)), /not a surfvis waterfall block/i);
});
```

`tests/js/colormap.test.mjs`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { paint, NAN_RGB, FLAG_ALPHA } from "../../src/surfvis/web/static/wf/colormap.mjs";
import { LUTS } from "../../src/surfvis/web/static/wf/luts.mjs";

const lut = LUTS.viridis;
const rgb = (out, px) => Array.from(out.slice(px * 4, px * 4 + 3));

test("endpoints map to the ends of the table, and values clamp", () => {
  const out = new Uint8ClampedArray(4 * 4);
  paint(new Float32Array([0, 10, -5, 99]), new Uint8Array(4), lut, 0, 10, out, { nt: 1, nc: 4 });
  assert.deepEqual(rgb(out, 0), Array.from(lut.slice(0, 3)));
  assert.deepEqual(rgb(out, 1), Array.from(lut.slice(765, 768)));
  assert.deepEqual(rgb(out, 2), rgb(out, 0));
  assert.deepEqual(rgb(out, 3), rgb(out, 1));
  assert.equal(out[3], 255);
});

test("NaN is grey, and flags darken by their fraction", () => {
  const out = new Uint8ClampedArray(3 * 4);
  paint(new Float32Array([NaN, 10, 10]), new Uint8Array([0, 255, 0]), lut, 0, 10, out, { nt: 1, nc: 3 });
  assert.deepEqual(rgb(out, 0), NAN_RGB);
  const top = Array.from(lut.slice(765, 768));
  assert.deepEqual(rgb(out, 1), top.map((v) => Math.round(v * (1 - FLAG_ALPHA))));
  const off = new Uint8ClampedArray(3 * 4);
  paint(new Float32Array([NaN, 10, 10]), new Uint8Array([0, 255, 0]), lut, 0, 10, off, { nt: 1, nc: 3, showFlags: false });
  assert.deepEqual(rgb(off, 1), top);
});

test("the earliest time bin is the bottom image row", () => {
  const out = new Uint8ClampedArray(2 * 4);
  paint(new Float32Array([0, 10]), new Uint8Array(2), lut, 0, 10, out, { nt: 2, nc: 1 });
  assert.deepEqual(rgb(out, 1), Array.from(lut.slice(0, 3))); // row 0 of the block -> image row 1
  assert.deepEqual(rgb(out, 0), Array.from(lut.slice(765, 768)));
});
```

`tests/js/view.test.mjs`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { View, Latest, niceTicks } from "../../src/surfvis/web/static/wf/view.mjs";

const rect = { x: 10, y: 20, w: 400, h: 200 };
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test("starts at the full extent in cell-edge coordinates", () => {
  const v = new View(12, 32);
  assert.deepEqual(v.win, { t0: -0.5, t1: 11.5, c0: -0.5, c1: 31.5 });
});

test("toScreen and toData are inverse, with time upward", () => {
  const v = new View(12, 32);
  const bottomLeft = v.toScreen(-0.5, -0.5, rect);
  assert.deepEqual(bottomLeft, { x: 10, y: 220 });
  const p = v.toData(123, 77, rect);
  const s = v.toScreen(p.t, p.c, rect);
  close(s.x, 123);
  close(s.y, 77);
});

test("zoom keeps the point under the cursor fixed and stays inside the extent", () => {
  const v = new View(100, 100);
  v.zoomAt(0.5, 49.5, 49.5);
  close(v.win.t1 - v.win.t0, 50);
  close((49.5 - v.win.t0) / (v.win.t1 - v.win.t0), 0.5);
  v.reset();
  v.zoomAt(0.5, -0.5, -0.5); // corner: the window shifts back inside
  assert.ok(v.win.t0 >= -0.5 && v.win.c0 >= -0.5);
  v.zoomAt(100, 0, 0); // zooming out past everything gives the full extent
  assert.deepEqual(v.win, v.extent);
});

test("zoom never collapses below two cells, even on a one-slot plane", () => {
  const v = new View(1, 50);
  for (let i = 0; i < 50; i++) v.zoomAt(0.5, 0, 25);
  close(v.win.t1 - v.win.t0, 1); // the whole (one-cell) extent
  assert.ok(v.win.c1 - v.win.c0 >= 2 - 1e-9);
});

test("pan is clamped to the extent", () => {
  const v = new View(100, 100);
  v.zoomAt(0.1, 50, 50);
  const span = v.win.t1 - v.win.t0;
  v.pan(1000, -1000);
  close(v.win.t1, 99.5);
  close(v.win.c0, -0.5);
  close(v.win.t1 - v.win.t0, span);
});

test("request covers every visible cell at screen resolution", () => {
  const v = new View(12, 32);
  assert.deepEqual(v.request(rect, 2), { t0: 0, t1: 12, c0: 0, c1: 32, max_t: 400, max_c: 800 });
  v.win = { t0: 2.3, t1: 5.1, c0: 2.5, c1: 9.5 };
  const r = v.request(rect, 1, 64);
  assert.deepEqual([r.t0, r.t1, r.c0, r.c1], [2, 6, 3, 10]);
  assert.deepEqual([r.max_t, r.max_c], [64, 64]); // capped
});

test("Latest accepts only the newest response", () => {
  const l = new Latest();
  const a = l.next();
  const b = l.next();
  assert.equal(l.accept(b), true);
  assert.equal(l.accept(a), false); // arrived late
  assert.equal(l.accept(b), true);
});

test("niceTicks steps by 1, 2 or 5 times a power of ten", () => {
  assert.deepEqual(niceTicks(0, 100, 6), [0, 20, 40, 60, 80, 100]);
  assert.deepEqual(niceTicks(-0.5, 3.5, 6, true), [0, 1, 2, 3]);
  assert.deepEqual(niceTicks(0.3, 0.9, 3), [0.5]);
});
```

`tests/test_js.py`:

```python
"""Run the browser code's unit tests (tests/js/*.test.mjs) with node, if it is installed.

Only the pure modules are tested this way: decoding, colours and view maths.
node is not in the container image, so this skips there.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

JS_TESTS = Path(__file__).parent / "js"


def test_javascript_unit_tests():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    major = int(subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v").split(".")[0])
    if major < 18:
        pytest.skip(f"node {major} has no node:test")
    files = sorted(str(p) for p in JS_TESTS.glob("*.test.mjs"))
    result = subprocess.run([node, "--test", *files], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
```

Add `"test_js.py": (1, 0),` to `EXPECTED` in `tests/test_suite_integrity.py`.

- [ ] **Step 3: Run them to see them fail.**
Run: `node --test tests/js/`
Expected: FAIL, `Cannot find module .../format.mjs`.

- [ ] **Step 4: Implement.** `src/surfvis/web/static/wf/format.mjs`:

```js
// Decode a waterfall block from GET /waterfall.bin. Layout (little-endian):
// "SVWF", uint32 version, uint32 header length, JSON header, zero padding to a
// multiple of 4, float32 values, uint8 flag fractions. Row = time bin.

export function decode(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 12 || String.fromCharCode(...bytes.subarray(0, 4)) !== "SVWF") {
    throw new Error("Not a surfvis waterfall block");
  }
  const dv = new DataView(buffer);
  const version = dv.getUint32(4, true);
  if (version !== 1) throw new Error(`Unsupported waterfall block version ${version}`);
  const n = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + n)));
  const offset = 12 + n + ((4 - ((12 + n) % 4)) % 4);
  const count = header.nt * header.nc;
  // Float32Array uses platform byte order; every browser surfvis runs in is little-endian.
  const values = new Float32Array(buffer, offset, count);
  const flags = new Uint8Array(buffer, offset + 4 * count, count);
  return { header, values, flags };
}
```

`src/surfvis/web/static/wf/colormap.mjs`:

```js
// Colour a block's values with a 256-entry LUT and darken flagged bins.

export const NAN_RGB = [128, 128, 128];
// Matches the PNG renderer's flag overlay (black at alpha 0.55).
export const FLAG_ALPHA = 0.55;

export function paint(values, flags, lut, vmin, vmax, out, { nt, nc, showFlags = true }) {
  const scale = 255 / (vmax - vmin);
  for (let k = 0; k < nt; k++) {
    // Time runs upward: block row 0 (the earliest bin) is the bottom image row.
    const dst = (nt - 1 - k) * nc;
    for (let j = 0; j < nc; j++) {
      const i = k * nc + j;
      const v = values[i];
      let r, g, b;
      if (Number.isNaN(v)) {
        [r, g, b] = NAN_RGB;
      } else {
        const x = Math.min(255, Math.max(0, Math.round((v - vmin) * scale)));
        r = lut[3 * x];
        g = lut[3 * x + 1];
        b = lut[3 * x + 2];
      }
      if (showFlags && flags[i]) {
        const keep = 1 - (FLAG_ALPHA * flags[i]) / 255;
        r *= keep;
        g *= keep;
        b *= keep;
      }
      const o = (dst + j) * 4;
      out[o] = r;
      out[o + 1] = g;
      out[o + 2] = b;
      out[o + 3] = 255;
    }
  }
}
```

`src/surfvis/web/static/wf/view.mjs`:

```js
// The waterfall's view: which part of the plane is on screen, in data
// coordinates (slot i spans [i - 0.5, i + 0.5], channel j likewise), and the
// maths between that, the screen and the block to request.

// A zoom never makes the window narrower than this many cells (or the whole axis, if shorter).
const MIN_SPAN = 2;

function fit(lo, hi, min, max) {
  // Shift [lo, hi] inside [min, max] without changing its width (or clamp it if wider).
  const w = hi - lo;
  if (w >= max - min) return [min, max];
  if (lo < min) return [min, min + w];
  if (hi > max) return [max - w, max];
  return [lo, hi];
}

export class View {
  constructor(ntime, nchan) {
    this.ntime = ntime;
    this.nchan = nchan;
    this.reset();
  }

  get extent() {
    return { t0: -0.5, t1: this.ntime - 0.5, c0: -0.5, c1: this.nchan - 0.5 };
  }

  reset() {
    this.win = { ...this.extent };
  }

  zoomAt(factor, t, c) {
    const e = this.extent;
    const axis = (lo, hi, at, min, max) => {
      const full = max - min;
      const w = Math.min(Math.max((hi - lo) * factor, Math.min(MIN_SPAN, full)), full);
      const nlo = at - ((at - lo) * w) / (hi - lo);
      return fit(nlo, nlo + w, min, max);
    };
    [this.win.t0, this.win.t1] = axis(this.win.t0, this.win.t1, t, e.t0, e.t1);
    [this.win.c0, this.win.c1] = axis(this.win.c0, this.win.c1, c, e.c0, e.c1);
  }

  pan(dt, dc) {
    const e = this.extent;
    [this.win.t0, this.win.t1] = fit(this.win.t0 + dt, this.win.t1 + dt, e.t0, e.t1);
    [this.win.c0, this.win.c1] = fit(this.win.c0 + dc, this.win.c1 + dc, e.c0, e.c1);
  }

  toScreen(t, c, rect) {
    const w = this.win;
    return {
      x: rect.x + ((c - w.c0) / (w.c1 - w.c0)) * rect.w,
      y: rect.y + rect.h - ((t - w.t0) / (w.t1 - w.t0)) * rect.h,
    };
  }

  toData(x, y, rect) {
    const w = this.win;
    return {
      t: w.t0 + ((rect.y + rect.h - y) / rect.h) * (w.t1 - w.t0),
      c: w.c0 + ((x - rect.x) / rect.w) * (w.c1 - w.c0),
    };
  }

  request(rect, dpr, cap = 4096) {
    // Cells overlapping the window: i + 0.5 > t0 and i - 0.5 < t1.
    const w = this.win;
    const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
    const t0 = clamp(Math.floor(w.t0 + 0.5), 0, this.ntime - 1);
    const c0 = clamp(Math.floor(w.c0 + 0.5), 0, this.nchan - 1);
    return {
      t0,
      t1: clamp(Math.ceil(w.t1 + 0.5), t0 + 1, this.ntime),
      c0,
      c1: clamp(Math.ceil(w.c1 + 0.5), c0 + 1, this.nchan),
      max_t: clamp(Math.round(rect.h * dpr), 1, cap),
      max_c: clamp(Math.round(rect.w * dpr), 1, cap),
    };
  }
}

// Responses can arrive out of order while zooming: draw only the newest.
export class Latest {
  constructor() {
    this.issued = 0;
    this.shown = 0;
  }

  next() {
    return ++this.issued;
  }

  accept(id) {
    if (id < this.shown) return false;
    this.shown = id;
    return true;
  }
}

// Tick positions in [lo, hi], about `target` of them, stepping by 1, 2 or 5 x 10^k.
export function niceTicks(lo, hi, target = 6, integer = false) {
  const raw = (hi - lo) / Math.max(target, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  let step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  if (integer) step = Math.max(1, Math.round(step));
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    ticks.push(Number(v.toPrecision(12)));
  }
  return ticks;
}
```

- [ ] **Step 5: Run the tests to see them pass.**
Run: `node --test tests/js/` — Expected: all pass. Then `uv run pytest tests/test_js.py tests/test_suite_integrity.py -q -p no:cacheprovider` — Expected: pass.

- [ ] **Step 6: Commit.**

```bash
git add tools/make_luts.py src/surfvis/web/static/wf tests/js tests/test_js.py tests/test_suite_integrity.py
git commit -m "feat(web): decoding, colour and view modules for the canvas waterfall

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Canvas widget in the page

**Files:**
- Create: `src/surfvis/web/static/wf/widget.mjs`
- Modify: `src/surfvis/web/templates/_waterfall.html` (the `{% else %}` branch with the `<img class="waterfall">`), `src/surfvis/web/static/app.js` (mount on settle), `src/surfvis/web/static/style.css`, `src/surfvis/web/app.py` (static `Cache-Control`)
- Test: `tests/test_web.py` (`test_waterfall_controls` asserts), `tests/test_suite_integrity.py` (unchanged count)
- Docs: `docs/wiki/web-app.md`, `docs/wiki/log.md`, `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `GET /waterfall.bin` (Task 3); `decode` (`format.mjs`), `LUTS` (`luts.mjs`), `paint` (`colormap.mjs`), `View`, `Latest`, `niceTicks` (`view.mjs`) (Task 4).
- Produces: `widget.mjs` export `mount(root: HTMLElement) -> Promise<api>`, where `api = {canvas, toData(x, y) -> {t, c}, toScreen(t, c) -> {x, y}, addLayer(fn), redraw(), view}`; `fn(ctx, api)` is called on every redraw, clipped to the plot area, coordinates in CSS pixels. Also sets `root.waterfall = api` and dispatches a bubbling `wf:ready` event with `detail = api`. Step 3 builds on exactly this.

- [ ] **Step 1: Change the panel test first.** In `tests/test_web.py::test_waterfall_controls` replace `assert "waterfall.png?" in panel.text` with:

```python
    assert 'data-src="/waterfall.bin?' in panel.text and "<canvas" in panel.text
```

and replace `assert "waterfall.png?" not in broken.text` with `assert "waterfall.bin?" not in broken.text`. Add to `test_routes`, after the `index` asserts:

```python
    # Static files revalidate: ES module imports cannot carry the ?v= version.
    assert client.get("/static/wf/widget.mjs").headers["cache-control"] == "no-cache"
```

Run: `uv run pytest tests/test_web.py::test_waterfall_controls tests/test_web.py::test_routes -q -p no:cacheprovider` — Expected: FAIL.

- [ ] **Step 2: Serve static files with `Cache-Control: no-cache`.** In `app.py`, above `create_app`:

```python
class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate (by ETag) on every use.

    The ``?v=`` digest busts caches for files the page names, but imports
    between ES modules under ``static/wf/`` cannot carry it.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response
```

and mount with it: `app.mount("/static", RevalidatedStaticFiles(directory=HERE / "static"), name="static")`.

- [ ] **Step 3: Replace the image in the panel.** In `_waterfall.html`, replace the `<div class="plot loading" ...> <img class="waterfall" ...> </div>` block with:

```html
  <div class="wf" data-src="/waterfall.bin?{{ query }}"
       data-module="/static/wf/widget.mjs?v={{ static_version }}">
    <div class="wf-bar">
      <span class="wf-title"></span>
      <label class="wf-check"><input type="checkbox" class="wf-flags" checked> show flags</label>
      <button type="button" class="wf-reset">Reset view</button>
    </div>
    <div class="plot loading" data-label="Reading visibilities&hellip;">
      <canvas class="wf-canvas" aria-label="Waterfall for baseline {{ name1 }} to {{ name2 }}"></canvas>
    </div>
    <p class="axis wf-readout">wheel: zoom &middot; shift-drag or middle-drag: pan &middot; 0: reset</p>
  </div>
```

- [ ] **Step 4: Mount it.** In `static/app.js`, add after the existing `htmx:afterSettle` listener:

```js
// The waterfall panel is drawn by static/wf/widget.mjs; mount it whenever the panel is (re)rendered.
document.body.addEventListener("htmx:afterSettle", (e) => {
  if (e.detail.target.id !== "waterfall") return;
  const root = e.detail.target.querySelector(".wf[data-src]");
  if (!root) return;
  import(root.dataset.module).then((m) => m.mount(root));
});
```

- [ ] **Step 5: Styles.** In `static/style.css`, replace the `.waterfall { ... }` rule with:

```css
.wf-bar { display: flex; align-items: center; gap: 0.9rem; margin: 0 0 0.4rem; }
.wf-title { flex: 1; font-size: 0.85rem; color: var(--text); font-family: ui-monospace, monospace; white-space: pre; overflow: hidden; text-overflow: ellipsis; }
.wf-check { display: flex; align-items: center; gap: 0.3rem; color: var(--muted); font-size: 0.78rem; }
.wf-reset { background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 4px; padding: 0.2rem 0.6rem; font: inherit; font-size: 0.78rem; cursor: pointer; }
.wf-reset:hover { border-color: var(--accent); }
.wf-canvas { display: block; width: 100%; background: #fff; border-radius: 6px; touch-action: none; }
.wf .plot.refining::after { content: "refining\2026"; top: 0.4rem; right: 0.6rem; left: auto; text-align: right; }
```

- [ ] **Step 6: The widget.** Create `src/surfvis/web/static/wf/widget.mjs`:

```js
// The waterfall drawn in the browser. Fetches blocks from /waterfall.bin,
// paints them on a canvas, zooms (wheel) and pans (shift- or middle-drag),
// and reports what is under the cursor. Plain left-drag, click and
// double-click are deliberately unbound: step 3 of #16 uses them to draw
// flagging polygons, through toData/toScreen/addLayer.

import { decode } from "./format.mjs";
import { LUTS } from "./luts.mjs";
import { paint } from "./colormap.mjs";
import { View, Latest, niceTicks } from "./view.mjs";

// CSS pixels around the image: time labels, channel labels, colourbar.
const MARGIN = { left: 76, right: 78, top: 8, bottom: 40 };
const ASPECT = 0.48; // canvas height / width, as the PNG's 9.5 x 4.6
const REFINE_MS = 150;
const OUTLINE = "#39ff14";
const FONT = "12px ui-sans-serif, system-ui, sans-serif";

function utc(seconds) {
  return new Date(seconds * 1000).toISOString().slice(11, 19);
}

export async function mount(root) {
  const canvas = root.querySelector(".wf-canvas");
  const plot = canvas.parentElement;
  const readout = root.querySelector(".wf-readout");
  const help = readout.textContent;
  const ctx = canvas.getContext("2d");
  const latest = new Latest();
  const layers = [];
  let view = null;
  let meta = null; // overview header: times, freqs
  let block = null; // {header, values, flags}
  let image = null; // offscreen canvas of block
  let refineTimer = null;
  let drag = null;

  const rect = () => ({
    x: MARGIN.left,
    y: MARGIN.top,
    w: canvas.clientWidth - MARGIN.left - MARGIN.right,
    h: canvas.clientHeight - MARGIN.top - MARGIN.bottom,
  });
  const showFlags = () => root.querySelector(".wf-flags").checked;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = plot.clientWidth;
    const h = Math.round(w * ASPECT);
    canvas.style.height = `${h}px`;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  async function fetchBlock(params) {
    const id = latest.next();
    const res = await fetch(`${root.dataset.src}&${new URLSearchParams(params)}`);
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        detail = (await res.json()).detail ?? detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const decoded = decode(await res.arrayBuffer());
    return latest.accept(id) ? decoded : null;
  }

  function setBlock(decoded) {
    block = decoded;
    const { header, values, flags } = decoded;
    const off = document.createElement("canvas");
    off.width = header.nc;
    off.height = header.nt;
    const octx = off.getContext("2d");
    const img = octx.createImageData(header.nc, header.nt);
    paint(values, flags, LUTS[header.colormap], header.vmin, header.vmax, img.data, {
      nt: header.nt,
      nc: header.nc,
      showFlags: showFlags(),
    });
    octx.putImageData(img, 0, 0);
    image = off;
  }

  function draw() {
    const r = rect();
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    if (!view || !block) return;
    const h = block.header;
    const e = view.extent;
    const lo = view.toScreen(e.t0, e.c0, r);
    const hi = view.toScreen(e.t1, e.c1, r);

    // The image spans whole bins from (t0, c0); a ragged last bin is drawn
    // full-width and clipped at the plane's edge.
    const a = view.toScreen(h.t0 - 0.5, h.c0 - 0.5, r);
    const b = view.toScreen(h.t0 + h.nt * h.tstep - 0.5, h.c0 + h.nc * h.cstep - 0.5, r);
    ctx.save();
    ctx.beginPath();
    const x0 = Math.max(r.x, lo.x);
    const y0 = Math.max(r.y, hi.y);
    ctx.rect(x0, y0, Math.min(r.x + r.w, hi.x) - x0, Math.min(r.y + r.h, lo.y) - y0);
    ctx.clip();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(image, a.x, b.y, b.x - a.x, a.y - b.y);
    if (h.rect) {
      const [t0, tf, c0, cf] = h.rect;
      const p = view.toScreen(t0 - 0.5, c0 - 0.5, r);
      const q = view.toScreen(tf - 0.5, cf - 0.5, r);
      ctx.strokeStyle = OUTLINE;
      ctx.lineWidth = 2;
      ctx.strokeRect(p.x, q.y, q.x - p.x, p.y - q.y);
    }
    for (const layer of layers) layer(ctx, api);
    ctx.restore();

    drawAxes(r);
    drawColourbar(r, h);
  }

  function drawAxes(r) {
    ctx.strokeStyle = "#222";
    ctx.lineWidth = 1;
    ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w, r.h);
    ctx.fillStyle = "#222";
    ctx.font = FONT;
    const w = view.win;

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const c of niceTicks(Math.max(w.c0, 0), Math.min(w.c1, view.nchan - 1), 8, true)) {
      const { x } = view.toScreen(0, c, r);
      ctx.fillRect(x, r.y + r.h, 1, 4);
      ctx.fillText(String(c), x, r.y + r.h + 6);
    }
    ctx.fillText("Channel", r.x + r.w / 2, r.y + r.h + 22);

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of niceTicks(Math.max(w.t0, 0), Math.min(w.t1, view.ntime - 1), 6, true)) {
      const { y } = view.toScreen(t, 0, r);
      ctx.fillRect(r.x - 4, y, 4, 1);
      ctx.fillText(meta.times ? utc(meta.times[t]) : String(t), r.x - 6, y);
    }
    ctx.save();
    ctx.translate(14, r.y + r.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText(meta.times ? `UTC on ${new Date(meta.times[0] * 1000).toISOString().slice(0, 10)}` : "Time slot", 0, 0);
    ctx.restore();
  }

  function drawColourbar(r, h) {
    const lut = LUTS[h.colormap];
    const x = r.x + r.w + 12;
    const bw = 14;
    for (let i = 0; i < r.h; i++) {
      const k = Math.min(255, Math.floor(((r.h - 1 - i) / (r.h - 1)) * 256));
      ctx.fillStyle = `rgb(${lut[3 * k]},${lut[3 * k + 1]},${lut[3 * k + 2]})`;
      ctx.fillRect(x, r.y + i, bw, 1);
    }
    ctx.strokeStyle = "#222";
    ctx.strokeRect(x + 0.5, r.y + 0.5, bw, r.h);
    ctx.fillStyle = "#222";
    ctx.font = FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const v of niceTicks(h.vmin, h.vmax, 5)) {
      const y = r.y + r.h - ((v - h.vmin) / (h.vmax - h.vmin)) * r.h;
      ctx.fillRect(x + bw, y, 4, 1);
      ctx.fillText(Number(v.toPrecision(4)).toString(), x + bw + 6, y);
    }
  }

  function scheduleRefine() {
    clearTimeout(refineTimer);
    refineTimer = setTimeout(refine, REFINE_MS);
  }

  async function refine() {
    plot.classList.add("refining");
    try {
      const decoded = await fetchBlock(view.request(rect(), window.devicePixelRatio || 1));
      if (decoded) {
        setBlock(decoded);
        draw();
      }
    } catch (err) {
      readout.textContent = `Could not refine: ${err.message}`;
    } finally {
      plot.classList.remove("refining");
    }
  }

  function describe(x, y) {
    const r = rect();
    if (x < r.x || x > r.x + r.w || y < r.y || y > r.y + r.h) return help;
    const { t, c } = view.toData(x, y, r);
    const slot = Math.round(t);
    const chan = Math.round(c);
    if (slot < 0 || slot >= view.ntime || chan < 0 || chan >= view.nchan) return help;
    const h = block.header;
    const k = Math.floor((slot - h.t0) / h.tstep);
    const j = Math.floor((chan - h.c0) / h.cstep);
    const parts = [
      meta.times ? `${utc(meta.times[slot])} UTC (slot ${slot})` : `slot ${slot}`,
      meta.freqs ? `channel ${chan} (${(meta.freqs[chan] / 1e6).toFixed(3)} MHz)` : `channel ${chan}`,
    ];
    if (k >= 0 && k < h.nt && j >= 0 && j < h.nc) {
      const i = k * h.nc + j;
      const binned = h.tstep * h.cstep > 1 ? ` over ${h.tstep}×${h.cstep} cells` : "";
      parts.push(`${h.quantity} ${Number(block.values[i].toPrecision(5))}${binned}`);
      parts.push(`${Math.round((100 * block.flags[i]) / 255)}% flagged`);
    }
    return parts.join(" · ");
  }

  const api = {
    canvas,
    toData: (x, y) => view.toData(x, y, rect()),
    toScreen: (t, c) => view.toScreen(t, c, rect()),
    addLayer(fn) {
      layers.push(fn);
      draw();
    },
    redraw: () => draw(),
    get view() {
      return view;
    },
  };

  // Input. Listeners on window/document stop once the panel is replaced.
  canvas.addEventListener(
    "wheel",
    (e) => {
      if (!view) return;
      e.preventDefault();
      const { t, c } = view.toData(e.offsetX, e.offsetY, rect());
      view.zoomAt(Math.exp(e.deltaY * 0.0015), t, c);
      draw();
      scheduleRefine();
    },
    { passive: false },
  );
  canvas.addEventListener("pointerdown", (e) => {
    if (!view || !((e.button === 0 && e.shiftKey) || e.button === 1)) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    drag = { x: e.offsetX, y: e.offsetY };
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!view) return;
    if (drag) {
      const r = rect();
      const a = view.toData(drag.x, drag.y, r);
      const b = view.toData(e.offsetX, e.offsetY, r);
      view.pan(a.t - b.t, a.c - b.c);
      drag = { x: e.offsetX, y: e.offsetY };
      draw();
    }
    readout.textContent = describe(e.offsetX, e.offsetY);
  });
  canvas.addEventListener("pointerup", () => {
    if (drag) scheduleRefine();
    drag = null;
  });
  canvas.addEventListener("pointerleave", () => {
    readout.textContent = help;
  });
  root.querySelector(".wf-reset").addEventListener("click", () => {
    view.reset();
    draw();
    scheduleRefine();
  });
  root.querySelector(".wf-flags").addEventListener("change", () => {
    if (block) setBlock(block);
    draw();
  });
  const onKey = (e) => {
    if (!root.isConnected) return document.removeEventListener("keydown", onKey);
    if (e.key === "0" && view && !e.target.closest("input, select, textarea")) {
      view.reset();
      draw();
      scheduleRefine();
    }
  };
  document.addEventListener("keydown", onKey);
  const observer = new ResizeObserver(() => {
    if (!root.isConnected) return observer.disconnect();
    resize();
    draw();
    if (view) scheduleRefine();
  });

  // First load: the whole plane at screen resolution, with the slot times and frequencies.
  resize();
  try {
    const dpr = window.devicePixelRatio || 1;
    const r = rect();
    const decoded = await fetchBlock({ meta: "true", max_t: Math.round(r.h * dpr), max_c: Math.round(r.w * dpr) });
    if (!decoded) return api;
    meta = decoded.header;
    view = new View(meta.ntime, meta.nchan);
    root.querySelector(".wf-title").textContent = meta.title;
    setBlock(decoded);
    plot.classList.remove("loading");
    draw();
    observer.observe(plot);
  } catch (err) {
    const p = document.createElement("p");
    p.className = "problems";
    p.textContent = err.message;
    plot.replaceWith(p);
    return api;
  }
  root.waterfall = api;
  root.dispatchEvent(new CustomEvent("wf:ready", { bubbles: true, detail: api }));
  return api;
}
```

- [ ] **Step 7: Run the Python tests.**
Run: `uv run ruff format . && uv run ruff check . --fix && uv run pytest -q -p no:cacheprovider`
Expected: all pass (the node test included).

- [ ] **Step 8: Verify in a browser.** Build a demo and drive it with headless Chrome (`/usr/bin/google-chrome`; playwright runs via `uv run --with playwright`, using `channel="chrome"`). In a scratch directory (never the repo):

```python
# mkdemo.py -- run with: uv run python mkdemo.py <outdir>   (from the repo root, with PYTHONPATH=.)
import sys
from tests.fixtures.ms import make_ms
from surfvis.core.chi2 import chi2
if __name__ == "__main__":  # chi2 uses a spawn pool: this guard is required
    d = sys.argv[1]
    info = make_ms(f"{d}/demo.ms", nant=16, nchan=256, ntime=40)
    chi2(info["path"], dataout=f"{d}/chi2.zarr", imagesout=f"{d}/imgs", nthreads=2, nfreqs=64, ntimes=10)
```

Then `uv run surfvis serve --data <outdir>/chi2.zarr --port 8765 &` and a playwright script that: loads `/`, waits for `.hist .plot:not(.loading)`, clicks `.grid i[data-pq="1,5"]`, waits for `.wf .plot:not(.loading)`, and checks with `page.evaluate` that `document.querySelector('.wf').waterfall` exists; reads `.wf-title`; hovers the canvas centre and prints `.wf-readout`; dispatches `page.mouse.wheel(0, -600)` over the canvas centre, waits 600 ms, and confirms (via `page.on("request")`) that a `/waterfall.bin` request with `t0=`/`c0=` other than the full range was made and the readout shows `over` no longer (full resolution when zoomed in far enough); shift-drags; presses `0`; toggles `.wf-flags`; changes the quantity select to `phase` and confirms the colourbar is redrawn; clicks `#save-png` and saves the download. Take screenshots before and after zooming and look at them. There must be no console errors (`page.on("console")`, `page.on("pageerror")`). Stop the server afterwards (`kill` its PID; do not `pkill -f` a pattern that matches your own shell).

Fix anything that fails, then re-run Step 7.

- [ ] **Step 9: Docs.**
  - `docs/wiki/web-app.md`: add a section "Canvas waterfall" summarising the spec (route, reduction table, wire format, cache, modules, interactions, the `no-cache` static header and why), and note `/waterfall.png` is kept but unused by the page. Update the "Loading feedback" bullet about the waterfall PNG.
  - `README.md` ("Using the interface"): the waterfall is drawn in the browser; wheel zooms, shift-drag pans, `0` resets, hover shows time, frequency and value, "show flags" toggles the overlay.
  - `CLAUDE.md`: in the `web/` bullet mention `planes.py` and `static/wf/` (ES modules, node-tested via `tests/test_js.py`, node not in the container); update the full-suite test count to the number the container run reports.
  - Then commit the code, stamp `last_verified_commit` in `docs/wiki/web-app.md` and `docs/wiki/log.md` with `git rev-parse --short HEAD`, add a `log.md` entry, and commit the docs separately.

- [ ] **Step 10: Commit.**

```bash
git add src/surfvis/web tests README.md CLAUDE.md
git commit -m "feat(web): draw the waterfall on a canvas with zoom and pan

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
# then the docs commit per Step 9
```
