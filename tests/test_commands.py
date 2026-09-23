"""Smoke tests for the plotting and flagging commands.

Neither had any coverage: `test_roundtrip.py` only proves the CLI and the cab
agree, which says nothing about whether the command runs. These exercise the
core functions against the fixture MS.
"""

import shutil

import pytest

from tests.conftest import needs_ms

np = pytest.importorskip("numpy")


@needs_ms
def test_summary_reports_every_antenna(tiny_ms, capsys):
    """summary prints one row per antenna, per field and per spectral window."""
    from surfvis.core.summary import summary

    summary(tiny_ms["path"])
    out = capsys.readouterr().out

    for name in tiny_ms["antenna_names"]:
        assert name in out
    assert "testfield" in out
    assert "/ANTENNA" in out and "/FIELD" in out and "/SPECTRAL_WINDOW" in out


@needs_ms
def test_surf_writes_one_png_per_baseline(tiny_ms, tmp_path):
    """surf renders the baseline it was asked for, and names it after the pair."""
    from surfvis.core.surf import surf

    opdir = tmp_path / "plots"
    surf(tiny_ms["path"], datacolumn="DATA", field=0, i=0, j=1, opdir=opdir, plot="amp")

    pngs = sorted(opdir.glob("*.png"))
    assert len(pngs) == 1
    assert "baseline_0_1" in pngs[0].name
    assert "_amp" in pngs[0].name
    assert pngs[0].stat().st_size > 0


@needs_ms
def test_surf_rejects_an_unknown_quantity(tiny_ms, tmp_path):
    """Stimela calls the core function directly, so it validates for itself."""
    from surfvis.core.surf import surf

    with pytest.raises(ValueError, match="must be one of"):
        surf(tiny_ms["path"], opdir=tmp_path / "plots", plot="nonsense")


@needs_ms
def test_flag_chi2_flags_the_bad_baseline_in_place(tiny_ms, tmp_path):
    """The planted residual is flagged, and quiet baselines are left alone."""
    from casacore.tables import table

    from surfvis.core.flag_chi2 import flag_chi2

    ms = tmp_path / "flagme.ms"
    shutil.copytree(tiny_ms["path"], ms)

    with table(str(ms), readonly=True, ack=False) as t:
        before = t.getcol("FLAG")
    assert not before.any(), "fixture starts unflagged"

    flag_chi2(ms, rcol="RESIDUAL", wcol="WEIGHT_SPECTRUM", flag_above=5.0, nthreads=1)

    with table(str(ms), readonly=True, ack=False) as t:
        after = t.getcol("FLAG")
        ant1, ant2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        flag_row = t.getcol("FLAG_ROW")

    p, q = tiny_ms["hot"]
    hot_rows = (ant1 == p) & (ant2 == q)
    assert after[hot_rows].all(), "every hot-baseline visibility should be flagged"
    assert not after[~hot_rows].all(), "quiet baselines should not be wholly flagged"
    assert flag_row[hot_rows].all(), "FLAG_ROW follows FLAG"


@needs_ms
def test_flag_chi2_respects_protected_antennas(tiny_ms, tmp_path):
    """--respect-ants leaves every baseline touching those antennas untouched."""
    from casacore.tables import table

    from surfvis.core.flag_chi2 import flag_chi2

    ms = tmp_path / "respect.ms"
    shutil.copytree(tiny_ms["path"], ms)
    p, q = tiny_ms["hot"]

    flag_chi2(ms, rcol="RESIDUAL", flag_above=5.0, nthreads=1, respect_ants=[p])

    with table(str(ms), readonly=True, ack=False) as t:
        after = t.getcol("FLAG")
        ant1, ant2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")

    touching = (ant1 == p) | (ant2 == p)
    assert not after[touching].any(), "protected antenna's baselines must be untouched"
