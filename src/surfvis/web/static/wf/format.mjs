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
