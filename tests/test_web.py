"""Tests for the chi-squared browser.

The fixture MS plants an inflated residual on one baseline; every test here
checks that the pipeline carries that outlier through to something a human
would see.
"""

import pytest

from tests.conftest import needs_ms, needs_web

# Skip the whole module rather than fail collection on a lightweight install.
np = pytest.importorskip("numpy")


@needs_ms
def test_chi2_writes_a_readable_zarr(chi2_zarr):
    """chi2 --dataout produces a store the web layer can open."""
    from surfvis.web.store import Chi2Store

    store = Chi2Store(chi2_zarr["zarr"])
    assert store.scans == [(0, 0, 1), (0, 0, 2)]
    assert store.fields == [0]
    assert store.spws(0) == [0]

    times, freqs, corrs = store.bins(0, 0, 1)
    assert times == [0, 1]
    assert freqs == [0, 1]
    # Diagonal correlations of a 4-correlation MS, resolved from [0, -1].
    assert corrs == [0, 3]


@needs_ms
def test_hot_baseline_dominates_its_chunk(chi2_zarr):
    """The planted bad baseline is the worst pair, by a wide margin."""
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    chunk = store.chunk(Selection(0, 0, 1, 0, 0, 0))
    matrix = chunk.chi2_dof

    assert matrix.shape == (8, 8)
    assert np.isnan(np.diag(matrix)).all(), "auto-correlations carry no chi-squared"

    p, q = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    assert {int(p), int(q)} == set(chi2_zarr["ms"]["hot"])
    assert matrix[p, q] > 100 * np.nanmedian(matrix)
    assert matrix[p, q] == matrix[q, p], "the matrix is symmetric"


@needs_ms
def test_chunk_bounds_describe_the_scan(chi2_zarr):
    """Bin bounds are what let a reader outline the chunk on a waterfall."""
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    first = store.chunk(Selection(0, 0, 1, 0, 0, 0))
    second = store.chunk(Selection(0, 0, 1, 1, 1, 0))

    assert (first.t0, first.tf) == (0, 6)
    assert (first.chan0, first.chanf) == (0, 16)
    assert second.t0 == 6
    assert second.chan0 == 16
    assert first.rcol == "RESIDUAL"


@needs_web
def test_waterfall_reads_the_right_baseline(chi2_zarr):
    """The hot baseline reads back brighter than a quiet one."""
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    p, q = chi2_zarr["ms"]["hot"]
    names = chi2_zarr["ms"]["antenna_names"]

    hot = msdata.waterfall(ms, 0, 0, 1, names[p], names[q], 0, "RESIDUAL")
    cold = msdata.waterfall(ms, 0, 0, 1, names[0], names[2], 0, "RESIDUAL")

    assert hot.values.shape == (12, 32), "one scan of times by all channels"
    assert np.median(hot.values) > 10 * np.median(cold.values)


@needs_web
def test_waterfall_accepts_either_antenna_order(chi2_zarr):
    """Clicking (p, q) or (q, p) in the grid finds the same baseline."""
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    forward = msdata.waterfall(ms, 0, 0, 1, names[1], names[5], 0, "RESIDUAL")
    reverse = msdata.waterfall(ms, 0, 0, 1, names[5], names[1], 0, "RESIDUAL")
    assert np.array_equal(forward.values, reverse.values)


@needs_web
def test_residual_is_plottable_but_flag_is_not(chi2_zarr):
    """RESIDUAL is non-standard, so xarray-ms surfaces it; FLAG is the overlay."""
    from surfvis.web import msdata

    columns = msdata.available_columns(chi2_zarr["ms"]["path"], 0, 0, 1)
    assert "RESIDUAL" in columns
    assert "VISIBILITY" in columns
    assert "FLAG" not in columns


@needs_web
def test_msv2_column_names_resolve_to_msv4(chi2_zarr):
    """DATA is VISIBILITY and WEIGHT_SPECTRUM is WEIGHT in the MSv4 view."""
    from surfvis.web import msdata

    group = msdata.data_group(chi2_zarr["ms"]["path"])
    assert group.correlated_data == "VISIBILITY"
    assert group.weight == "WEIGHT"

    assert group.resolve("DATA") == "VISIBILITY"
    assert group.resolve("WEIGHT_SPECTRUM") == "WEIGHT"
    assert group.resolve("FLAG") == "FLAG"
    # Non-standard columns are not renamed by the MSv4 view.
    assert group.resolve("RESIDUAL") == "RESIDUAL"


@needs_web
def test_data_column_is_plottable_by_its_msv2_name(chi2_zarr):
    """Asking for DATA must work even though the variable is called VISIBILITY."""
    from surfvis.web import msdata

    names = chi2_zarr["ms"]["antenna_names"]
    wf = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], 0, "DATA")
    assert wf.column == "VISIBILITY"


@needs_web
def test_each_partition_is_one_scan(chi2_zarr):
    """SCAN_NUMBER is in the partition schema, so a partition's time axis is the scan."""
    from surfvis.web import msdata

    names = chi2_zarr["ms"]["antenna_names"]
    one = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], 0, "RESIDUAL")
    two = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 2, names[0], names[1], 0, "RESIDUAL")
    assert one.values.shape == two.values.shape == (12, 32)
    assert not np.array_equal(one.values, two.values)


@needs_web
@pytest.mark.parametrize("quantity", ["amp", "phase", "real", "imag"])
def test_every_quantity_renders(chi2_zarr, quantity):
    from surfvis.web import msdata

    names = chi2_zarr["ms"]["antenna_names"]
    wf = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], 0, "RESIDUAL", quantity=quantity)
    assert np.isfinite(wf.values).all()


@needs_web
def test_log_scale_separates_the_outlier(chi2_zarr):
    """A linear scale buries the bulk; the log default must not."""
    from surfvis.web import render
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    matrix = store.chunk(Selection(0, 0, 1, 0, 0, 0)).chi2_dof

    norm, vmin, vmax = render.make_norm(matrix, "log")
    colours = render.cell_colours(matrix, norm)
    assert vmin > 0 and vmax > vmin
    # The bulk must not all collapse onto a single colour.
    flat = [c for row in colours for c in row if c != render.NAN_COLOUR]
    assert len(set(flat)) > 5

    with pytest.raises(ValueError):
        render.make_norm(matrix, "nonsense")


@needs_web
def test_routes(chi2_zarr):
    """Every endpoint answers, and a missing chunk 404s rather than 500s."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    client = TestClient(create_app(chi2_zarr["zarr"]))

    assert client.get("/healthz").json()["scans"] == 2
    assert client.get("/").status_code == 200

    heatmap = client.get("/heatmap?field=0&spw=0&scan=1&t=0&f=0&c=0")
    assert heatmap.status_code == 200
    # One clickable cell per antenna pair.
    assert heatmap.text.count('hx-get="/waterfall') == 64

    png = client.get("/histogram.png?field=0&spw=0&scan=1&t=0&f=0&c=0")
    assert png.headers["content-type"] == "image/png"
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"

    wf = client.get("/waterfall.png?field=0&spw=0&scan=1&t=0&f=0&c=0&p=1&q=5&column=RESIDUAL&quantity=amp")
    assert wf.headers["content-type"] == "image/png"

    assert client.get("/heatmap?field=0&spw=0&scan=99&t=0&f=0&c=0").status_code == 404
    assert client.get("/heatmap?field=0&spw=0&scan=1&t=0&f=0&c=0&scale=nope").status_code == 400
