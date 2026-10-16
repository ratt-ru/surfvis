"""Manual flagging writes: exact cells, the backup flag version, undo and the routes.

Needs a Measurement Set, and works on a copy so the session fixtures stay
pristine. Which cells a polygon covers is tested in ``test_rasterise.py``.
"""

import shutil

import pytest

from tests.conftest import needs_web

np = pytest.importorskip("numpy")


def _rect(t0, t1, c0, c1):
    """The four-vertex polygon around slots t0..t1 and channels c0..c1, inclusive."""
    return [(t0 - 0.5, c0 - 0.5), (t0 - 0.5, c1 + 0.5), (t1 + 0.5, c1 + 0.5), (t1 + 0.5, c0 - 0.5)]


@pytest.fixture
def writable_ms(chi2_zarr, tmp_path):
    """A private copy of the fixture MS, free to be flagged."""
    path = str(tmp_path / "flagme.ms")
    shutil.copytree(chi2_zarr["ms"]["path"], path)
    return path


def _flags(ms: str) -> tuple[np.ndarray, np.ndarray]:
    from casacore.tables import table

    with table(ms, ack=False) as t:
        return t.getcol("FLAG"), t.getcol("FLAG_ROW")


def _baseline_rows(ms: str, a: int, b: int, scan: int) -> np.ndarray:
    from casacore.tables import table

    with table(ms, ack=False) as t:
        a1, a2, sc = t.getcol("ANTENNA1"), t.getcol("ANTENNA2"), t.getcol("SCAN_NUMBER")
    return np.flatnonzero((a1 == a) & (a2 == b) & (sc == scan))


@needs_web
def test_flag_writes_exactly_the_selected_cells(chi2_zarr, writable_ms):
    """Rows, channels and correlations outside the polygon are untouched."""
    from surfvis.web import msdata
    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    before, before_row = _flags(writable_ms)
    rows = _baseline_rows(writable_ms, 0, 2, scan=1)

    flagger = Flagger(writable_ms)
    out = flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(3, 5, 10, 14)), corr_idx=[0, 3])

    after, after_row = _flags(writable_ms)
    expected = before.copy()
    # The baseline's rows in scan 1 are its slots, in time order.
    sel = np.ix_(rows[3:6], np.arange(10, 15), [0, 3])
    expected[sel] = True
    np.testing.assert_array_equal(after, expected)
    np.testing.assert_array_equal(after_row, before_row | expected.all(axis=(1, 2)))
    assert out["cells"] == 3 * 5 * 2
    assert out["newly_flagged"] == int((~before[sel] & ~before_row[rows[3:6], None, None]).sum())

    # The browser sees the write at once.
    wf = msdata.waterfall(writable_ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert wf.flags[3:6, 10:15].all()


@needs_web
def test_first_write_saves_a_backup_version(chi2_zarr, writable_ms):
    """Before the first write, FLAG is saved as a CASA flag version; later writes reuse it."""
    from surfvis.web import msdata
    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    before, before_row = _flags(writable_ms)
    flagger = Flagger(writable_ms)
    first = flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(0, 1, 0, 3)), corr_idx=[0])
    second = flagger.apply(0, 0, 1, names[1], names[3], np.array(_rect(0, 1, 0, 3)), corr_idx=[0])

    assert first["backup"] == second["backup"] and first["backup"].startswith("surfvis_before_")
    assert first["backup"] in msdata.flag_versions(writable_ms)

    from casacore.tables import table

    with table(f"{writable_ms}.flagversions/flags.{first['backup']}", ack=False) as t:
        np.testing.assert_array_equal(t.getcol("FLAG"), before)
        np.testing.assert_array_equal(t.getcol("FLAG_ROW"), before_row)


@needs_web
def test_undo_restores_flags_in_reverse_order(chi2_zarr, writable_ms):
    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    original, original_row = _flags(writable_ms)
    flagger = Flagger(writable_ms)
    flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(0, 4, 0, 31)), corr_idx=[0, 1, 2, 3])
    middle, _ = _flags(writable_ms)
    flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(5, 8, 2, 6)), corr_idx=[0])

    assert flagger.undo().id == 2
    np.testing.assert_array_equal(_flags(writable_ms)[0], middle)
    assert flagger.undo().id == 1
    after, after_row = _flags(writable_ms)
    np.testing.assert_array_equal(after, original)
    np.testing.assert_array_equal(after_row, original_row)
    assert flagger.undo() is None


@needs_web
def test_dry_run_counts_without_writing(chi2_zarr, writable_ms):
    import os

    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    before, _ = _flags(writable_ms)
    flagger = Flagger(writable_ms)
    out = flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(0, 2, 0, 3)), corr_idx=[0], dry_run=True)
    assert out["id"] is None and out["cells"] == 3 * 4 and out["backup"] is None
    np.testing.assert_array_equal(_flags(writable_ms)[0], before)
    assert not os.path.exists(f"{writable_ms}.flagversions") or not any(
        name.startswith("flags.surfvis_") for name in os.listdir(f"{writable_ms}.flagversions")
    )
    assert flagger.edits == []


@needs_web
def test_flag_routes(chi2_zarr, writable_ms):
    """Read-only by default; with --allow-flagging the routes flag, report and undo."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    body = {"field": 0, "spw": 0, "scan": 1, "p": 0, "q": 2, "product": "I", "vertices": _rect(1, 2, 4, 9)}

    readonly = TestClient(create_app(chi2_zarr["zarr"], ms=writable_ms))
    assert readonly.post("/flag", json=body).status_code == 403
    assert readonly.get("/flag/state").json() == {"enabled": False}

    client = TestClient(create_app(chi2_zarr["zarr"], ms=writable_ms, allow_flagging=True))
    assert client.get("/healthz").json()["flagging"] is True
    before, _ = _flags(writable_ms)

    # Stokes I is formed from XX and YY, so those are the correlations flagged.
    out = client.post("/flag", json=body).json()
    assert out["cells"] == 2 * 6 * 2
    after, _ = _flags(writable_ms)
    rows = _baseline_rows(writable_ms, 0, 2, scan=1)
    changed = np.argwhere(after != before)
    assert set(changed[:, 0]) <= set(rows[1:3]) and set(changed[:, 2]) <= {0, 3}

    allcorr = client.post("/flag", json={**body, "correlations": "all", "dry_run": True}).json()
    assert allcorr["cells"] == 2 * 6 * 4

    state = client.get("/flag/state").json()
    assert state["enabled"] and len(state["edits"]) == 1 and state["backup"]

    assert client.post("/flag/undo").json()["undone"] == out["id"]
    np.testing.assert_array_equal(_flags(writable_ms)[0], before)
    assert client.post("/flag/undo").status_code == 409

    assert client.post("/flag", json={**body, "vertices": [(0, 0), (1, 1)]}).status_code == 422
    assert client.post("/flag", json={**body, "vertices": _rect(50, 60, 0, 3)}).status_code == 400
    # Linear feeds cannot form RR.
    assert client.post("/flag", json={**body, "product": "RR"}).status_code == 400


@needs_web
def test_backup_of_a_tiled_flag_column(chi2_zarr, tmp_path):
    """A real MS keeps FLAG in a TiledShapeStMan; the backup must copy it, shape intact."""
    from casacore.tables import table

    from surfvis.web.flagging import save_flag_version
    from tests.fixtures.ms import copy_with_tiled_flags

    ms = copy_with_tiled_flags(chi2_zarr["ms"]["path"], str(tmp_path / "tiled.ms"))
    with table(ms, ack=False) as t:
        assert t.getdminfo("FLAG")["TYPE"] == "TiledShapeStMan"
        before = t.getcol("FLAG")
    save_flag_version(ms, "probe", "a probe")
    with table(f"{ms}.flagversions/flags.probe", ack=False) as v:
        np.testing.assert_array_equal(v.getcol("FLAG"), before)
    with pytest.raises(FileExistsError):
        save_flag_version(ms, "probe", "again")
