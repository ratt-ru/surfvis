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
