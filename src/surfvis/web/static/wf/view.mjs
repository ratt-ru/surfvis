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

// Zoom factor for one wheel event. Browsers report deltaY in pixels (0), lines
// (1, Firefox) or pages (2); normalise to pixels and clamp so one event cannot
// jump the zoom. Negative deltaY zooms in (factor < 1).
export function wheelFactor(deltaY, deltaMode = 0, perPixel = 0.0015, maxPixels = 300) {
  const px = deltaY * (deltaMode === 1 ? 40 : deltaMode === 2 ? 800 : 1);
  return Math.exp(Math.min(Math.max(px, -maxPixels), maxPixels) * perPixel);
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
