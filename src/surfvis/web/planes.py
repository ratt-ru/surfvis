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
    n_rows, n_cols = a.shape
    nt, nc = -(-n_rows // tstep), -(-n_cols // cstep)
    padded = np.full((nt * tstep, nc * cstep), fill, dtype=a.dtype)
    padded[:n_rows, :n_cols] = a
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
    lo, hi = (
        (float(v.min()), float(np.percentile(v, 99)))
        if quantity == "amp"
        else tuple(map(float, np.percentile(v, [1, 99])))
    )
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
