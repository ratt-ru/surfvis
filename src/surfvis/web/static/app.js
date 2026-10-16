// Behaviour htmx does not cover: loading feedback, the selected baseline,
// the hover readout, keyboard navigation, and carrying waterfall settings
// from one baseline to the next.

// The selected baseline, by antenna names rather than grid indices: a
// different scan can hold a different set of antennas.
let selected = null;

function gridData() {
  const node = document.getElementById("grid-data");
  return node ? JSON.parse(node.textContent) : null;
}

function cellFor(names) {
  const data = gridData();
  if (!data || !names) return null;
  const p = data.names.indexOf(names[0]);
  const q = data.names.indexOf(names[1]);
  if (p < 0 || q < 0 || p === q) return null;
  return document.querySelector(`.grid i[data-pq="${p},${q}"]`);
}

function pq(cell) {
  return cell.dataset.pq.split(",").map(Number);
}

function markSelected() {
  document.querySelectorAll(".grid i.selected").forEach((c) => c.classList.remove("selected"));
  cellFor(selected)?.classList.add("selected");
}

// Dim whatever a request is about to replace, so a click visibly did something.
document.body.addEventListener("htmx:beforeRequest", (e) => e.detail.target.classList.add("loading"));
document.body.addEventListener("htmx:afterRequest", (e) => e.detail.target.classList.remove("loading"));

// A cell click carries the waterfall controls the user moved off the
// heatmap's defaults (the server marks them .keep), so a choice survives a
// change of baseline.
document.body.addEventListener("htmx:configRequest", (e) => {
  if (!e.detail.elt.matches(".grid i[data-pq]")) return;
  document.querySelectorAll("#waterfall .keep").forEach((el) => {
    e.detail.parameters[el.name] = el.value;
  });
});

document.body.addEventListener("click", (e) => {
  const cell = e.target.closest(".grid i[data-pq]");
  if (!cell) return;
  const data = gridData();
  const [p, q] = pq(cell);
  selected = [data.names[p], data.names[q]];
  markSelected();
});

document.body.addEventListener("htmx:afterSettle", (e) => {
  if (e.detail.target.id !== "heatmap") return;
  markSelected();
  // A new field, scan, bin or product: follow the selected baseline into it,
  // rather than leave a waterfall from somewhere else on screen.
  if (!selected || !document.querySelector("#waterfall .panel")) return;
  const cell = cellFor(selected);
  if (cell) {
    cell.click();
  } else {
    document.getElementById("waterfall").innerHTML =
      `<p class="hint">${selected[0]} &times; ${selected[1]} has no data here. Click a cell to plot another baseline.</p>`;
  }
});

// Hover: a tooltip at the cursor and the readout under the grid, both naming
// the antennas with their numbers (the grid index is the ANTENNA row).
function describe(cell) {
  const data = gridData();
  const [p, q] = pq(cell);
  const value = data.values[p][q];
  const names = `${data.names[p]} (${p}) \u00d7 ${data.names[q]} (${q})`;
  let stats = `\u03c7\u00b2/dof ${value === null ? "flagged" : value.toFixed(2)}`;
  if (data.positions) {
    const [a, b] = [data.positions[p], data.positions[q]];
    stats += ` \u00b7 ${Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]).toFixed(0)} m`;
  }
  return [names, stats];
}

let readoutDefault = null;
function showReadout(cell) {
  const readout = document.getElementById("readout");
  if (!readout) return;
  readoutDefault ??= readout.textContent;
  readout.textContent = cell ? describe(cell).join(" \u00b7 ") : readoutDefault;
}

document.body.addEventListener("mousemove", (e) => {
  const tip = document.getElementById("tip");
  const cell = e.target.closest(".grid i[data-pq]");
  showReadout(cell);
  if (!tip) return;
  if (!cell) {
    tip.hidden = true;
    return;
  }
  tip.textContent = describe(cell).join("\n");
  tip.hidden = false;
  // Below-right of the cursor, flipped when it would leave the viewport.
  const pad = 14;
  const { width, height } = tip.getBoundingClientRect();
  const x = e.clientX + pad + width > window.innerWidth ? e.clientX - pad - width : e.clientX + pad;
  const y = e.clientY + pad + height > window.innerHeight ? e.clientY - pad - height : e.clientY + pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
});

// Arrow keys move the selection: up/down change antenna1 (the row),
// left/right antenna2 (the column), skipping the diagonal.
document.addEventListener("keydown", (e) => {
  const steps = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  if (!(e.key in steps) || e.target.closest("input, select, textarea")) return;
  const current = cellFor(selected);
  const data = gridData();
  if (!current || !data) return;
  e.preventDefault();
  const n = data.names.length;
  let [p, q] = pq(current);
  const [dp, dq] = steps[e.key];
  do {
    p = (p + dp + n) % n;
    q = (q + dq + n) % n;
  } while (p === q);
  const next = document.querySelector(`.grid i[data-pq="${p},${q}"]`);
  if (!next) return;
  next.click();
  showReadout(next);
});

// PNGs are fetched by the browser, not htmx, so they report their own progress.
function plotLoaded(img) {
  img.parentElement.classList.remove("loading");
}

async function plotFailed(img) {
  const wrap = img.parentElement;
  wrap.classList.remove("loading");
  let detail = "Could not render this plot.";
  try {
    const body = await (await fetch(img.src)).json();
    if (body.detail) detail = body.detail;
  } catch (_) {}
  const p = document.createElement("p");
  p.className = "problems";
  p.textContent = detail;
  wrap.replaceWith(p);
}

// Save the whole page as a PNG, including what is scrolled out of view, which
// a screenshot cannot do. html-to-image (vendored) renders the DOM through an
// SVG foreignObject, so the grid, labels and controls come out as shown.
function pageName() {
  const pick = (name) => {
    const el = document.querySelector(`#selectors select[name="${name}"]`);
    return el ? el.selectedOptions[0]?.textContent.trim() : "";
  };
  const parts = ["surfvis", pick("field"), `scan${pick("scan")}`, `t${pick("t")}`, `f${pick("f")}`, pick("c")];
  if (selected && document.querySelector("#waterfall .panel")) parts.push(`${selected[0]}-${selected[1]}`);
  return parts.filter(Boolean).join("_").replace(/[^A-Za-z0-9_.+-]+/g, "") + ".png";
}

async function savePng(button) {
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Saving\u2026";
  try {
    // The library re-downloads every <img> it embeds; for the waterfall that
    // would mean reading the MS again. Hand it the already-decoded pixels.
    for (const img of document.querySelectorAll("img")) {
      if (!img.complete || !img.naturalWidth || img.src.startsWith("data:")) continue;
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
      img.dataset.src = img.src;
      img.src = canvas.toDataURL("image/png");
    }
    const body = document.body;
    const url = await htmlToImage.toPng(body, {
      backgroundColor: getComputedStyle(body).backgroundColor,
      width: body.scrollWidth,
      height: body.scrollHeight,
      pixelRatio: 2,
      skipFonts: true,
      filter: (node) => !["busy", "tip", "save-png"].includes(node.id),
    });
    const a = document.createElement("a");
    a.href = url;
    a.download = pageName();
    a.click();
  } catch (err) {
    alert(`Could not save the page: ${err}`);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}
