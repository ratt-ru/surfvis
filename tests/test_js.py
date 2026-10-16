"""Run the browser code's unit tests (tests/js/*.test.mjs) with node, if it is installed.

Only the pure modules are tested this way: decoding, colours and view maths.
node is not in the container image, so this skips there.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

JS_TESTS = Path(__file__).parent / "js"


def test_javascript_unit_tests():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    major = int(
        subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v").split(".")[0]
    )
    if major < 18:
        pytest.skip(f"node {major} has no node:test")
    files = sorted(str(p) for p in JS_TESTS.glob("*.test.mjs"))
    result = subprocess.run([node, "--test", *files], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
