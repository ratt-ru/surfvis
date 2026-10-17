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
