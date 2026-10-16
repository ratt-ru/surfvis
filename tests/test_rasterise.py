"""Polygon rasterisation for manual flagging: which cells a selection covers.

numpy only, so these run wherever numpy is installed -- no Measurement Set.
"""

import pytest

np = pytest.importorskip("numpy")

from surfvis.web.flagging import rasterise  # noqa: E402


def _rect(t0, t1, c0, c1):
    """The four-vertex polygon around slots t0..t1 and channels c0..c1, inclusive."""
    return [(t0 - 0.5, c0 - 0.5), (t0 - 0.5, c1 + 0.5), (t1 + 0.5, c1 + 0.5), (t1 + 0.5, c0 - 0.5)]


def test_rectangle_selects_exactly_its_cells():
    mask = rasterise(np.array(_rect(2, 4, 3, 7)), 10, 12)
    expected = np.zeros((10, 12), bool)
    expected[2:5, 3:8] = True
    np.testing.assert_array_equal(mask, expected)


def test_vertex_order_and_direction_do_not_matter():
    rect = np.array(_rect(1, 3, 0, 5))
    np.testing.assert_array_equal(rasterise(rect, 8, 8), rasterise(rect[::-1], 8, 8))
    np.testing.assert_array_equal(rasterise(rect, 8, 8), rasterise(np.roll(rect, 2, axis=0), 8, 8))


def test_triangle_uses_cell_centres():
    # Right triangle with legs just outside slot 0 and channel 0 and its
    # hypotenuse on c = 3.5 - t: centre (i, j) is inside when i + j <= 3.
    mask = rasterise(np.array([(-0.5, -0.5), (-0.5, 4.0), (4.0, -0.5)]), 6, 6)
    i, j = np.indices((6, 6))
    np.testing.assert_array_equal(mask, (i + j) < 3.5)


def test_self_intersecting_polygon_is_even_odd():
    # A bow tie: two triangles meeting at (5, 5); the crossing point is not
    # double-counted and the two lobes are both inside.
    bow = np.array([(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)])
    mask = rasterise(bow, 11, 11)
    assert mask[1, 5] and mask[9, 5]  # inside the upper and lower lobes
    assert not mask[5, 1] and not mask[5, 9]  # the side notches are outside


def test_polygon_beyond_the_grid_is_clipped():
    mask = rasterise(np.array(_rect(-5, 2, 30, 40)), 6, 34)
    expected = np.zeros((6, 34), bool)
    expected[0:3, 30:34] = True
    np.testing.assert_array_equal(mask, expected)


def test_rasterise_rejects_degenerate_input():
    with pytest.raises(ValueError):
        rasterise(np.array([(0.0, 0.0), (1.0, 1.0)]), 4, 4)
    with pytest.raises(ValueError):
        rasterise(np.array([(0.0, 0.0), (1.0, np.nan), (2.0, 0.0)]), 4, 4)
