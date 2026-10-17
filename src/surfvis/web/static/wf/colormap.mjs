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
