"""Exact checks on the correction and Stokes algebra.

numpy only, so these run wherever numpy is installed -- no Measurement Set.
"""

import pytest

np = pytest.importorskip("numpy")

from surfvis.utils import products  # noqa: E402


def _rand(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_stokes_weights_are_inverse_variances():
    """Unequal correlation weights: Stokes I's weight is 4ab/(a+b), not radiomesh's imaging a+b."""
    vis = np.array([[[2.0, 0, 0, 4.0]]], dtype=complex)
    weight = np.array([[[1.0, 1.0, 1.0, 3.0]]])
    flag = np.zeros(vis.shape, bool)

    out, w, f = products.form(["I", "Q"], ["XX", "XY", "YX", "YY"], vis, weight, flag)
    np.testing.assert_allclose(out[0, 0], [3.0, -1.0])
    np.testing.assert_allclose(w[0, 0], [4 * 1 * 3 / (1 + 3)] * 2)
    assert not f.any()


def test_stokes_inherit_any_constituent_flag():
    vis = np.ones((1, 2, 4), complex)
    flag = np.zeros(vis.shape, bool)
    flag[0, 0, 3] = True
    _, w, f = products.form(["XX", "I", "U"], ["XX", "XY", "YX", "YY"], vis, np.ones(vis.shape), flag)
    assert f[0, 0].tolist() == [False, True, False]
    assert w[0, 0, 1] == 0


def test_circular_feeds():
    vis = np.array([[[3.0, 1.0, 1.0, 1.0]]], dtype=complex)
    out, _, _ = products.form(["I", "V"], ["RR", "RL", "LR", "LL"], vis, np.ones(vis.shape), np.zeros(vis.shape, bool))
    np.testing.assert_allclose(out[0, 0], [2.0, 1.0])


def test_availability():
    assert products.available(["XX", "YY"]) == ["XX", "YY", "I", "Q"]
    assert products.needed(["I"], ["XX", "XY", "YX", "YY"]) == ["XX", "YY"]
    assert products.needed(["XX"], ["XX", "XY", "YX", "YY"], full_jones=True) == ["XX", "XY", "YX", "YY"]
    with pytest.raises(ValueError, match="Cannot form"):
        products.needed(["V"], ["XX", "YY"])
    assert products.corr_labels([5, 8]) == ["RR", "LL"]


def _setup(rng, full):
    nrow, nchan, nant = 6, 3, 3
    ant1 = np.array([0, 0, 1, 0, 0, 1])
    ant2 = np.array([1, 2, 2, 1, 2, 2])
    tidx = np.array([0, 0, 0, 1, 1, 1])
    jones = _rand(rng, (2, nchan, nant, 4 if full else 2))
    gains = products.Gains(jones, np.zeros((2, nchan, nant), bool))
    model = _rand(rng, (nrow, nchan, 4))
    return nrow, nchan, ant1, ant2, tidx, gains, model


def _matrix(x):
    return x.reshape(x.shape[:-1] + (2, 2))


def test_full_jones_correction_recovers_the_model():
    rng = np.random.default_rng(1)
    nrow, nchan, ant1, ant2, tidx, gains, model = _setup(rng, full=True)
    gp = _matrix(gains.jones[tidx, :, ant1])
    gq = _matrix(gains.jones[tidx, :, ant2])
    data = (gp @ _matrix(model) @ np.conj(np.swapaxes(gq, -1, -2))).reshape(nrow, nchan, 4)

    labels = ["XX", "XY", "YX", "YY"]
    out, w, f = products.form(
        labels, labels, data, np.ones(data.shape), np.zeros(data.shape, bool), ant1, ant2, tidx, gains, model
    )
    np.testing.assert_allclose(out, 0, atol=1e-10)
    assert (w > 0).all() and not f.any()


def test_diagonal_correction_leaves_chi_squared_unchanged():
    """With exactly propagated weights, correcting does not change |r|^2 w."""
    rng = np.random.default_rng(2)
    nrow, nchan, ant1, ant2, tidx, gains, _ = _setup(rng, full=False)
    labels = ["XX", "YY"]
    resid = _rand(rng, (nrow, nchan, 2))
    weight = rng.uniform(0.5, 2, size=resid.shape)
    out, w, _ = products.form(labels, labels, resid, weight, np.zeros(resid.shape, bool), ant1, ant2, tidx, gains)
    np.testing.assert_allclose(np.abs(out) ** 2 * w, np.abs(resid) ** 2 * weight)


def test_flagged_or_singular_gains_flag_the_data():
    rng = np.random.default_rng(3)
    nrow, nchan, ant1, ant2, tidx, gains, _ = _setup(rng, full=False)
    gains.flags[0, :, 2] = True
    gains.jones[1, 0, 1, 0] = 0
    vis = _rand(rng, (nrow, nchan, 2))
    _, w, f = products.form(
        ["XX", "YY"], ["XX", "YY"], vis, np.ones(vis.shape), np.zeros(vis.shape, bool), ant1, ant2, tidx, gains
    )
    # Rows 1 and 2 touch antenna 2 at time 0; row 5 has antenna 1 at time 1, channel 0, receptor X.
    assert f[1].all() and f[2].all()
    assert f[5, 0, 0] and not f[5, 0, 1]
    assert (w[f] == 0).all()
