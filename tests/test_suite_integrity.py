"""Guards against a silently emptied suite.

Everything touching a Measurement Set needs the ``[full]`` extras. Without
them ``test_commands.py`` skips at *import*, which pytest reports as a single
skip for the whole module -- a lightweight run says "10 passed, 1 skipped",
not the 5 tests that did not run. CI installs the lightweight package by
design, so a green CI run proves less than it looks, and tests could be
deleted or quietly defanged without anything noticing.

These checks are static: they parse the test files rather than inspecting the
running session, so they hold however pytest was invoked, and they run in both
install modes. Update the counts deliberately when adding or removing a test --
that edit is the point, it makes the change visible in review.
"""

import ast
from pathlib import Path

TESTS_DIR = Path(__file__).parent
MARKERS = {"needs_ms"}

# module -> (total test functions, of which need a Measurement Set)
EXPECTED = {
    "test_commands.py": (5, 5),
    "test_install.py": (2, 0),
    "test_roundtrip.py": (4, 0),
    "test_suite_integrity.py": (3, 0),
}


def _tests_in(path: Path) -> list[ast.FunctionDef]:
    tree = ast.parse(path.read_text())
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]


def _is_marked(fn: ast.FunctionDef) -> bool:
    return any(isinstance(d, ast.Name) and d.id in MARKERS for d in fn.decorator_list)


def test_every_test_module_is_accounted_for():
    """A new test file must be added to EXPECTED, not left to skip unnoticed."""
    found = {p.name for p in TESTS_DIR.glob("test_*.py")}
    assert found == set(EXPECTED), (
        f"test modules changed: only in tree {sorted(found - set(EXPECTED))}, "
        f"only in EXPECTED {sorted(set(EXPECTED) - found)}"
    )


def test_no_tests_have_disappeared():
    """Counts are pinned per module so a deletion localises to one file."""
    actual = {}
    for name in EXPECTED:
        fns = _tests_in(TESTS_DIR / name)
        actual[name] = (len(fns), sum(_is_marked(f) for f in fns))
    assert actual == EXPECTED


def test_tests_needing_a_measurement_set_are_marked():
    """An unmarked heavy test errors on a lightweight install instead of skipping.

    The module-level ``importorskip`` hides that today, but it is one edit away
    from not doing so, and the marker is what makes the intent explicit.
    """
    for name, (_, heavy) in EXPECTED.items():
        if heavy == 0:
            continue
        unmarked = [f.name for f in _tests_in(TESTS_DIR / name) if not _is_marked(f)]
        assert not unmarked, f"{name}: needs a needs_ms marker: {unmarked}"
