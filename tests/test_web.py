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
    assert corrs == ["XX", "YY"]


@needs_ms
def test_hot_baseline_dominates_its_chunk(chi2_zarr):
    """The planted bad baseline is the worst pair, by a wide margin."""
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    chunk = store.chunk(Selection(0, 0, 1, 0, 0, "XX"))
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
    first = store.chunk(Selection(0, 0, 1, 0, 0, "XX"))
    second = store.chunk(Selection(0, 0, 1, 1, 1, "XX"))

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

    hot = msdata.waterfall(ms, 0, 0, 1, names[p], names[q], "XX", "RESIDUAL")
    cold = msdata.waterfall(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")

    assert hot.values.shape == (12, 32), "one scan of times by all channels"
    assert np.median(hot.values) > 10 * np.median(cold.values)


@needs_web
def test_waterfall_accepts_either_antenna_order(chi2_zarr):
    """Clicking (p, q) or (q, p) in the grid finds the same baseline."""
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    forward = msdata.waterfall(ms, 0, 0, 1, names[1], names[5], "XX", "RESIDUAL")
    reverse = msdata.waterfall(ms, 0, 0, 1, names[5], names[1], "XX", "RESIDUAL")
    assert np.array_equal(forward.values, reverse.values)


@needs_web
def test_waterfall_carries_the_baseline_length(chi2_zarr):
    """The title's baseline length is the distance between ANTENNA::POSITION rows."""
    from casacore.tables import table

    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    with table(f"{ms}::ANTENNA", ack=False) as ant:
        positions = ant.getcol("POSITION")
    wf = msdata.waterfall(ms, 0, 0, 1, names[1], names[5], "XX", "RESIDUAL")
    assert wf.length == pytest.approx(np.linalg.norm(positions[1] - positions[5]))
    assert wf.length > 0


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
    # Nor are the standard columns xarray-ms keeps as secondary variables.
    assert group.resolve("CORRECTED_DATA") == "CORRECTED_DATA"
    assert group.resolve("MODEL_DATA") == "MODEL_DATA"


@needs_web
def test_data_column_is_plottable_by_its_msv2_name(chi2_zarr):
    """Asking for DATA must work even though the variable is called VISIBILITY."""
    from surfvis.web import msdata

    names = chi2_zarr["ms"]["antenna_names"]
    wf = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], "XX", "DATA")
    assert wf.column == "VISIBILITY"


@needs_web
def test_each_partition_is_one_scan(chi2_zarr):
    """SCAN_NUMBER is in the partition schema, so a partition's time axis is the scan."""
    from surfvis.web import msdata

    names = chi2_zarr["ms"]["antenna_names"]
    one = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], "XX", "RESIDUAL")
    two = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 2, names[0], names[1], "XX", "RESIDUAL")
    assert one.values.shape == two.values.shape == (12, 32)
    assert not np.array_equal(one.values, two.values)


@needs_web
@pytest.mark.parametrize("quantity", ["amp", "phase", "real", "imag"])
def test_every_quantity_renders(chi2_zarr, quantity):
    from surfvis.web import msdata

    names = chi2_zarr["ms"]["antenna_names"]
    wf = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], "XX", "RESIDUAL", quantity=quantity)
    assert np.isfinite(wf.values).all()


@needs_web
def test_log_scale_separates_the_outlier(chi2_zarr):
    """A linear scale buries the bulk; the log default must not."""
    from surfvis.web import render
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    matrix = store.chunk(Selection(0, 0, 1, 0, 0, "XX")).chi2_dof

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

    heatmap = client.get("/heatmap?field=0&spw=0&scan=1&t=0&f=0&c=XX")
    assert heatmap.status_code == 200
    # One clickable cell per ordered antenna pair; the diagonal is not a baseline.
    assert heatmap.text.count('hx-get="/waterfall') == 64 - 8
    assert heatmap.text.count('class="auto"') == 8
    assert "28 baselines" in heatmap.text

    png = client.get("/histogram.png?field=0&spw=0&scan=1&t=0&f=0&c=XX")
    assert png.headers["content-type"] == "image/png"
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"

    wf = client.get("/waterfall.png?field=0&spw=0&scan=1&t=0&f=0&c=XX&p=1&q=5&column=RESIDUAL&quantity=amp")
    assert wf.headers["content-type"] == "image/png"

    assert client.get("/heatmap?field=0&spw=0&scan=99&t=0&f=0&c=XX").status_code == 404
    assert client.get("/heatmap?field=0&spw=0&scan=1&t=0&f=0&c=XX&scale=nope").status_code == 400


@needs_web
def test_waterfall_sees_flags_written_after_the_first_read(chi2_zarr, tmp_path):
    """A long-running server must not pin the flags it read first.

    xarray-ms caches its tables in a process-global multiton whose TTL resets
    on every access, and a real MS keeps FLAG in a TiledShapeStMan whose tile
    cache belongs to that table object. Together they mean ``surfvis serve``
    renders the flags as they were when it opened the partition, for the life
    of the process, however often the MS is re-flagged underneath it.
    """
    import subprocess
    import sys

    from surfvis.web import msdata
    from tests.fixtures.ms import copy_with_tiled_flags

    ms = copy_with_tiled_flags(chi2_zarr["ms"]["path"], str(tmp_path / "reflagged.ms"))
    names = chi2_zarr["ms"]["antenna_names"]

    before = msdata.waterfall(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert not before.flags[:, 7].any(), "fixture channel is unflagged to start with"

    # The flagger is another program -- skarabina, CASA -- so the write has to
    # come from another process. Writing through this process's own casacore
    # handle would invalidate its table cache and hide the bug.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from casacore.tables import table; "
            "t = table(sys.argv[1], readonly=False, ack=False); "
            "f = t.getcol('FLAG'); f[:, 7, :] = True; t.putcol('FLAG', f); t.close()",
            ms,
        ],
        check=True,
    )

    after = msdata.waterfall(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert after.flags[:, 7].all(), "waterfall is still showing the flags from the first read"


@needs_ms
def test_each_correlation_gets_its_own_chi2(chi2_zarr):
    """Each product's slice is computed from that product alone.

    surfchi2 used to hand every correlation's task all of --use-corrs, so the
    per-correlation images were identical and the pol selector did nothing.
    """
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    xx = store.chunk(Selection(0, 0, 1, 0, 0, "XX")).chi2_dof
    yy = store.chunk(Selection(0, 0, 1, 0, 0, "YY")).chi2_dof
    assert not np.allclose(xx, yy, equal_nan=True)


@needs_ms
def test_residual_formed_on_the_fly_matches_the_residual_column(chi2_zarr, chi2_corrected):
    """inv(G) DATA inv(G)^H - MODEL, with propagated weights, is the RESIDUAL chi-squared."""
    from surfvis.web.store import Chi2Store, Selection

    plain = Chi2Store(chi2_zarr["zarr"])
    formed = Chi2Store(chi2_corrected["zarr"])
    for product in ("XX", "YY"):
        for t, f in ((0, 0), (1, 1)):
            want = plain.chunk(Selection(0, 0, 2, t, f, product)).chi2_dof
            got = formed.chunk(Selection(0, 0, 2, t, f, product)).chi2_dof
            np.testing.assert_allclose(got, want, rtol=1e-3, equal_nan=True)

    attrs = formed.group(0, 0, 1).attrs
    assert (attrs["rcol"], attrs["mcol"], attrs["gtable"]) == ("DATA", "MODEL_DATA", chi2_corrected["gtable"])


@needs_ms
def test_stokes_chi2_is_normalised_and_finds_the_bad_baseline(chi2_corrected):
    """Stokes weights are exact inverse variances, so chi2/dof matches the correlations'."""
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_corrected["zarr"])
    assert store.bins(0, 0, 1)[2] == ["XX", "YY", "I", "V"]

    xx = store.chunk(Selection(0, 0, 1, 0, 0, "XX")).chi2_dof
    for product in ("I", "V"):
        matrix = store.chunk(Selection(0, 0, 1, 0, 0, product)).chi2_dof
        p, q = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        assert {int(p), int(q)} == set(chi2_corrected["ms"]["hot"])
        # Complex unit-variance noise gives chi2/dof ~ 2 for XX and for Stokes alike.
        assert 0.75 < np.nanmedian(matrix) / np.nanmedian(xx) < 1.33
    assert (chi2_corrected["imgs"] / "field0" / "spw0" / "scan1" / "combined_V.png").exists()


@needs_ms
def test_chi2_refuses_stokes_it_cannot_form(tmp_path):
    """Stokes V of linear feeds needs XY and YX; an XX,YY-only MS must fail before any work."""
    from surfvis.core.chi2 import chi2
    from tests.fixtures.ms import make_ms

    info = make_ms(str(tmp_path / "dual.ms"), nant=4, nchan=4, ncorr=2, ntime=2, nscan=1)
    with pytest.raises(ValueError, match="Cannot form"):
        chi2(info["path"], dataout=tmp_path / "c.zarr", imagesout=tmp_path / "i", nthreads=1, stokes=["V"])


@needs_web
def test_corrected_data_is_its_own_column(chi2_zarr):
    """CORRECTED_DATA used to resolve to VISIBILITY, silently plotting DATA."""
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    data = msdata.waterfall(ms, 0, 0, 1, names[0], names[1], "XX", "DATA")
    corrected = msdata.waterfall(ms, 0, 0, 1, names[0], names[1], "XX", "CORRECTED_DATA")
    assert corrected.column == "CORRECTED_DATA"
    assert not np.allclose(data.values, corrected.values)
    assert "CORRECTED_DATA" in msdata.available_columns(ms, 0, 0, 1)


@needs_web
@pytest.mark.parametrize("quantity", ["real", "imag"])
def test_waterfall_forms_the_corrected_residual(chi2_zarr, gain_table, quantity):
    """DATA corrected by the gains, minus MODEL_DATA, is RESIDUAL in the corrected frame."""
    from surfvis.web import msdata

    info = chi2_zarr["ms"]
    names = info["antenna_names"]
    p, q = 2, 6
    formed = msdata.waterfall(
        info["path"], 0, 0, 2, names[p], names[q], "XX", "DATA", quantity, model="MODEL_DATA", gtable=gain_table
    )
    raw = msdata.waterfall(info["path"], 0, 0, 2, names[p], names[q], "XX", "RESIDUAL", "real")
    raw_i = msdata.waterfall(info["path"], 0, 0, 2, names[p], names[q], "XX", "RESIDUAL", "imag")
    g = info["gains"][info["ntime"] :]
    want = (raw.values + 1j * raw_i.values) / (g[:, :, p, 0] * np.conj(g[:, :, q, 0]))
    want = want.real if quantity == "real" else want.imag
    np.testing.assert_allclose(formed.values, want, atol=1e-4)
    assert formed.column == "corrected VISIBILITY - MODEL_DATA"


@needs_web
def test_stokes_waterfall_combines_correlations(chi2_zarr):
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    assert msdata.available_products(ms, 0, 0, 1) == ["XX", "XY", "YX", "YY", "I", "Q", "U", "V"]

    def wf(product):
        return msdata.waterfall(ms, 0, 0, 1, names[0], names[1], product, "RESIDUAL", "real").values

    np.testing.assert_allclose(wf("I"), 0.5 * (wf("XX") + wf("YY")), atol=1e-5)
    np.testing.assert_allclose(wf("Q"), 0.5 * (wf("XX") - wf("YY")), atol=1e-5)


@needs_web
def test_gains_off_the_data_grid_are_refused(chi2_zarr, gain_table, tmp_path):
    """On-grid only: a solution interval coarser than the data is an error, not a guess."""
    import xarray as xr

    from surfvis.web import msdata

    coarse = tmp_path / "coarse.qc" / "NET"
    for n in range(2):
        ds = xr.open_zarr(f"{gain_table}/NET_{n}", chunks=None).load()
        ds.isel(gain_time=slice(0, None, 2)).to_zarr(coarse / f"NET_{n}", mode="w")

    names = chi2_zarr["ms"]["antenna_names"]
    with pytest.raises(ValueError, match="on-grid"):
        msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], "XX", "RESIDUAL", gtable=str(coarse))
    problems = msdata.check(chi2_zarr["ms"]["path"], 0, 0, 1, "XX", "RESIDUAL", gtable=str(coarse))
    assert any("on-grid" in p for p in problems)


@needs_web
def test_flag_versions_are_listed_and_overlaid(chi2_zarr, tmp_path):
    """A saved CASA flag version replaces FLAG in the overlay, and the live column is untouched."""
    import shutil

    from casacore.tables import table

    from surfvis.web import msdata
    from tests.fixtures.ms import add_flag_version

    ms = str(tmp_path / "versioned.ms")
    shutil.copytree(chi2_zarr["ms"]["path"], ms)
    assert msdata.flag_versions(ms) == []

    t = table(ms, ack=False)
    flags = t.getcol("FLAG")
    t.close()
    flags[:, 3, :] = True
    add_flag_version(ms, "before_tricolour", flags, "flags autosave")
    assert msdata.flag_versions(ms) == ["before_tricolour"]

    names = chi2_zarr["ms"]["antenna_names"]
    live = msdata.waterfall(ms, 0, 0, 1, names[0], names[1], "XX", "RESIDUAL")
    saved = msdata.waterfall(ms, 0, 0, 1, names[0], names[1], "XX", "RESIDUAL", flagversion="before_tricolour")
    assert not live.flags[:, 3].any()
    assert saved.flags[:, 3].all()
    assert not saved.flags[:, 2].any()

    with pytest.raises(KeyError):
        msdata.waterfall(ms, 0, 0, 1, names[0], names[1], "XX", "RESIDUAL", flagversion="../../etc")


@needs_web
def test_waterfall_controls(chi2_corrected):
    """The panel defaults to how the heatmap was formed, and reports problems instead of a broken image."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    client = TestClient(create_app(chi2_corrected["zarr"]))
    base = "/waterfall?field=0&spw=0&scan=1&t=0&f=0&c=I&p=1&q=5"

    panel = client.get(base)
    assert panel.status_code == 200
    assert f'value="{chi2_corrected["gtable"]}"' in panel.text
    assert '<option value="MODEL_DATA" selected>' in panel.text
    # chi2 recorded DATA; the menu knows it as VISIBILITY and must not list both.
    assert '<option value="VISIBILITY" selected>' in panel.text
    assert 'value="DATA"' not in panel.text
    assert (
        'name="model"' in panel.text
        and '<option value="WEIGHT"' not in panel.text.split('name="model"')[1].split("</select>")[0]
    )
    assert "no longer shows" not in panel.text
    assert "waterfall.png?" in panel.text

    png = client.get(base.replace("/waterfall?", "/waterfall.png?"))
    assert png.headers["content-type"] == "image/png"

    changed = client.get(base + "&product=V&column=DATA&model=MODEL_DATA&gtable=" + chi2_corrected["gtable"])
    assert "no longer shows" in changed.text

    broken = client.get(base + "&gtable=/nowhere/NET")
    assert "gain table" in broken.text
    assert "waterfall.png?" not in broken.text
