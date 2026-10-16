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
