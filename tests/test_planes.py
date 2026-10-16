"""Reducing a waterfall plane to a screen-sized block, and its wire format.

numpy only, so these run wherever numpy is installed -- no Measurement Set.
"""

import math

import pytest

np = pytest.importorskip("numpy")

from surfvis.web import planes  # noqa: E402


def _plane(nt, nc, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(nt, nc)) + 1j * rng.normal(size=(nt, nc))


def test_full_resolution_is_the_quantity_itself():
    z = _plane(5, 7)
    f = np.zeros(z.shape, bool)
    for quantity, expect in (("amp", np.abs(z)), ("real", z.real), ("imag", z.imag), ("phase", np.angle(z))):
        b = planes.reduce_block(z, f, quantity, 0, 5, 0, 7, 100, 100)
        assert (b.tstep, b.cstep) == (1, 1) and b.values.dtype == np.float32
        np.testing.assert_allclose(b.values, expect, rtol=1e-6, atol=1e-6)
        assert not b.flags.any()


def test_amp_is_the_peak_of_the_unflagged_cells():
    z = np.ones((2, 2), complex)
    z[0, 1] = 9  # unflagged spike: must survive the reduction
    f = np.zeros(z.shape, bool)
    assert planes.reduce_block(z, f, "amp", 0, 2, 0, 2, 1, 1).values[0, 0] == 9
    f[0, 1] = True  # now flagged: ignored while unflagged cells remain
    b = planes.reduce_block(z, f, "amp", 0, 2, 0, 2, 1, 1)
    assert b.values[0, 0] == 1 and b.flags[0, 0] == round(255 * 1 / 4)


def test_fully_flagged_bin_falls_back_to_all_cells():
    z = np.array([[1.0, 5.0]], complex)
    b = planes.reduce_block(z, np.ones(z.shape, bool), "amp", 0, 1, 0, 2, 1, 1)
    assert b.values[0, 0] == 5 and b.flags[0, 0] == 255


def test_ragged_last_bins_use_only_cells_that_exist():
    # 5 x 7 into at most 2 x 3 bins: steps 3 x 3, bins 2 x 3, last bins ragged.
    z = np.arange(35, dtype=float).reshape(5, 7).astype(complex)
    f = np.zeros(z.shape, bool)
    f[4, 6] = True  # the only cell in the corner bin that is flagged...
    b = planes.reduce_block(z, f, "amp", 0, 5, 0, 7, 2, 3)
    assert (b.tstep, b.cstep) == (3, 3) and b.values.shape == (2, 3)
    # Corner bin holds rows 3-4, channel 6 only: cells 27 and 34(flagged).
    assert b.values[1, 2] == 27
    assert b.flags[1, 2] == round(255 * 1 / 2)  # ...out of 2 cells, not 9
    assert b.values[0, 0] == np.abs(z[0:3, 0:3]).max()


def test_signed_peak_and_circular_mean():
    z = np.array([[1 - 4j, -6 + 1j]])
    f = np.zeros(z.shape, bool)
    assert planes.reduce_block(z, f, "real", 0, 1, 0, 2, 1, 1).values[0, 0] == -6
    assert planes.reduce_block(z, f, "imag", 0, 1, 0, 2, 1, 1).values[0, 0] == -4
    # Phases either side of +-pi average to pi, not 0.
    w = np.exp(1j * np.array([[3.1, -3.1]]))
    ph = planes.reduce_block(w, f, "phase", 0, 1, 0, 2, 1, 1).values[0, 0]
    assert abs(abs(ph) - math.pi) < 1e-6


def test_window_matches_reducing_the_slice():
    z = _plane(10, 20, seed=3)
    f = np.random.default_rng(4).random(z.shape) < 0.2
    win = planes.reduce_block(z, f, "amp", 2, 7, 5, 17, 2, 4)
    sub = planes.reduce_block(z[2:7, 5:17], f[2:7, 5:17], "amp", 0, 5, 0, 12, 2, 4)
    np.testing.assert_array_equal(win.values, sub.values)
    np.testing.assert_array_equal(win.flags, sub.flags)
    assert (win.t0, win.t1, win.c0, win.c1) == (2, 7, 5, 17)


def test_colour_limits_are_finite_and_ordered():
    z = _plane(20, 30)
    f = np.zeros(z.shape, bool)
    lo, hi = planes.colour_limits(z, f, "amp")
    assert lo == pytest.approx(np.abs(z).min()) and hi == pytest.approx(np.percentile(np.abs(z), 99))
    assert planes.colour_limits(z, f, "phase") == (-math.pi, math.pi)
    assert planes.colour_limits(z, np.ones(z.shape, bool), "amp") == (0.0, 1.0)
    assert planes.colour_limits(np.full((3, 3), 2 + 0j), np.zeros((3, 3), bool), "amp") == (2.0, 3.0)


def test_encode_round_trips_and_rejects_nan_headers():
    z = _plane(3, 5)
    b = planes.reduce_block(z, np.zeros(z.shape, bool), "amp", 0, 3, 0, 5, 10, 10)
    header = {"title": "m000-m001 χ", "vmin": 0.0, "vmax": 1.0}
    buf = planes.encode(header, b)
    assert buf[:4] == b"SVWF"
    got, values, flags = planes.decode(buf)
    assert got["nt"] == 3 and got["nc"] == 5 and got["version"] == 1 and got["title"] == header["title"]
    np.testing.assert_array_equal(values, b.values)
    np.testing.assert_array_equal(flags, b.flags)
    with pytest.raises(ValueError):
        planes.encode({"vmin": float("nan")}, b)
