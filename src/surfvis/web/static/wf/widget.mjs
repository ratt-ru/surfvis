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
