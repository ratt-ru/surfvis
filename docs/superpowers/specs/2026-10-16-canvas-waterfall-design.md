# Canvas waterfall — design

Step 2 of [#16](https://github.com/ratt-ru/surfvis/issues/16) (manual flagging
from the waterfall). Branch: `flagging`, stacked on `serve-ui` (PR #17).

## Goal

Replace the waterfall PNG in `surfvis serve` with a waterfall drawn in the
browser on a `<canvas>`, so that the user can zoom to single-channel precision,
see exact values under the cursor, and — in step 3 — draw flagging polygons on
it without a server round trip per redraw.

Step 3 (polygon and drag-rectangle selection wired to `POST /flag`) is **out of
scope**, but this design must leave it a clean place to plug in: coordinate
transforms, an overlay hook, and plain left-drag / click / double-click left
unbound.

## What the user sees

- Clicking a grid cell shows the baseline's waterfall as today: the whole
  scan by all channels, time upward, the clicked chunk outlined in green,
  flagged cells darkened, the title line (`m009-m051 (968 m)  scan 1
  VISIBILITY  amp  XX`), UTC on the time axis, channel on the x axis, a
  colourbar.
- **Mouse wheel** zooms about the cursor. **Shift + drag** or **middle-button
  drag** pans. **Reset view** (button, or the `0` key) returns to the full
  plane.
- While zoomed, the current image is rescaled immediately, then replaced by a
  sharper one once the server returns the visible window at screen
  resolution. Zooming in far enough shows individual cells.
- Hovering shows, under the plot: UTC time, slot, channel, frequency (MHz),
  the value, and the flagged fraction of the hovered bin.
- A **Show flags** checkbox toggles the flag overlay.
- The waterfall controls (product, column, model, gains, flags, quantity),
  the "no longer matches the heatmap" note, carried-over settings, arrow-key
  baseline navigation and Save as PNG all keep working.
- Phase uses a cyclic colormap (`twilight`); amp/real/imag use `viridis`.

## Data route

`GET /waterfall.bin` takes the same query parameters as `/waterfall.png`
(`field, spw, scan, t, f, c, p, q, product, column, model, gtable,
flagversion, quantity`) plus:

| Param | Meaning | Default |
|---|---|---|
| `t0`, `t1` | half-open slot range to return | `0`, `ntime` |
| `c0`, `c1` | half-open channel range to return | `0`, `nchan` |
| `max_t`, `max_c` | most bins to return along each axis | `1024`; capped at `4096` |
| `meta` | include per-slot `times` and per-channel `freqs` | `false` |

Ranges are clamped to the plane. After clamping, `t0 >= t1` or `c0 >= c1` is a
400. Unknown column/baseline etc. is a 404 with a JSON `detail`, as for the
PNG route.

### Reduction

Bin sizes are `tstep = ceil((t1 - t0) / max_t)` and `cstep = ceil((c1 - c0) /
max_c)`. Bins are aligned at `t0`/`c0`; the last bin along an axis may be
ragged (fewer cells). Per bin, the cells used are the unflagged ones if any,
otherwise all of them (so fully flagged data stays visible under the overlay,
as in the PNG today). From the complex plane `z` of the used cells:

| quantity | bin value |
|---|---|
| `amp` | `max |z|` — RFI must not be averaged away |
| `real` / `imag` | the signed value of largest magnitude |
| `phase` | `angle(sum(z / |z|))`, the circular mean; `0` if every `z` is 0 |

The flag byte is `round(255 * flagged fraction of the bin's cells)`.

### Colour limits

From the **whole** plane's unflagged, finite values, independent of the window,
so zooming does not change the colours: `amp` → `[min, p99]`; `real`/`imag` →
`[p1, p99]`; `phase` → `[-π, π]`. If there are no such values, `[0, 1]`. If
`vmax <= vmin`, `vmax = vmin + 1`.

### Wire format

Little-endian:

```
bytes 0-3   b"SVWF"
bytes 4-7   uint32 version = 1
bytes 8-11  uint32 N = length of the header
N bytes     UTF-8 JSON header
pad         zero bytes up to a multiple of 4 from the start of the buffer
nt*nc*4     float32 values, row-major, row = time bin (bin 0 = earliest)
nt*nc       uint8 flag bytes, same layout
```

Header keys: `version, nt, nc, t0, t1, c0, c1, tstep, cstep, ntime, nchan,
vmin, vmax, quantity, colormap ("viridis" | "twilight"), title, rect ([t0, tf,
chan0, chanf] of the clicked chunk, or null)`, and with `meta`: `times` (unix
seconds per slot) and `freqs` (Hz per channel). The header must be valid JSON:
**no NaN or Infinity** anywhere in it.

### Caching

The formed complex plane is cached per `(ms, field, spw, scan, baseline,
product, column, model, gtable, flagversion)` plus the MS fingerprint and the
gain term's mtime, at most 4 planes. `msdata.invalidate(ms)` (called after a
flag write) clears it. Zooming and changing quantity therefore re-slice a
cached plane instead of re-reading the MS. `quantity` is not part of the key.

## Client

Plain ES modules under `static/wf/`, no build step, no new vendored library:

| Module | Responsibility | Tested with |
|---|---|---|
| `format.mjs` | decode the wire format | node |
| `colormap.mjs` | 256-entry `viridis` and `twilight` LUTs; value → RGBA | node |
| `view.mjs` | view state in data coordinates, zoom/pan/clamp, data ↔ screen transforms, which block to request, discarding stale responses | node |
| `widget.mjs` | canvas drawing, axes, colourbar, input, fetching, hover | browser |

Data coordinates: slot `i` spans `[i - 0.5, i + 0.5]`, channel `j` spans
`[j - 0.5, j + 0.5]` — the same convention as `flagging.rasterise`, so step 3
can send the vertices it draws straight to `/flag`.

The widget exposes, for step 3: `toData(x, y)`, `toScreen(t, c)`,
`addLayer(draw)` (called with the 2D context on every redraw), `redraw()`, and
the `canvas` element.

The canvas backing store is sized by `devicePixelRatio`, so it is sharp on
HiDPI screens, and `max_t`/`max_c` are requested in device pixels.

Static files are served with `Cache-Control: no-cache` (revalidate by ETag),
because ES module imports between `static/wf/*.mjs` cannot carry the `?v=`
version query.

## Kept

`/waterfall.png` stays (route, tests and matplotlib renderer), unused by the
page. The heatmap, histogram and χ² images stay matplotlib.

## Not in this step

Selection tools, `/flag` calls from the page, a colour-limit control,
frequency on the x axis.
