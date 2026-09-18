"""Round-trip tests: cli/*.py -> cabs/*.yml -> regenerated cli/*.py.

These tests are how this project guarantees that the source under cli/ and the
generated cab YAMLs under cabs/ stay in agreement. They run hip-cargo's
generate-cabs then generate-function, and assert the regenerated source is
byte-identical to the original (both are ruff-formatted with the same config).

Add a new test case here for every command you add under src/surfvis/cli/.
"""

import tempfile
from pathlib import Path

from hip_cargo.core.generate_cabs import generate_cabs
from hip_cargo.core.generate_function import generate_function


def _assert_roundtrip(command_name: str) -> None:
    """Round-trip cli/<command>.py through a cab and back; assert byte-identical."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cab_dir = tmpdir / "cabs"
        cab_dir.mkdir()

        cli_module = Path(f"src/surfvis/cli/{command_name}.py")
        assert cli_module.exists(), f"Missing CLI module: {cli_module}"

        generate_cabs([cli_module], output_dir=cab_dir)

        cab_file = cab_dir / f"{command_name}.yml"
        assert cab_file.exists(), f"generate-cabs did not produce {cab_file}"

        generated_file = tmpdir / f"{command_name}_roundtrip.py"
        generate_function(
            cab_file,
            output_file=generated_file,
            config_file=Path("pyproject.toml"),
        )

        assert generated_file.exists(), "generate-function did not produce output"
        generated_code = generated_file.read_text()
        compile(generated_code, str(generated_file), "exec")

        original_lines = cli_module.read_text().splitlines()
        generated_lines = generated_code.splitlines()

        assert len(original_lines) == len(generated_lines), (
            f"Line count mismatch for {command_name}: "
            f"original has {len(original_lines)} lines, generated has {len(generated_lines)} lines"
        )
        for i, (orig, gen) in enumerate(zip(original_lines, generated_lines), 1):
            assert orig == gen, f"Line {i} differs in {command_name}:\n  Original:  {orig}\n  Generated: {gen}"


def test_roundtrip_summary() -> None:
    """The summary command must round-trip cleanly through a cab."""
    _assert_roundtrip("summary")


def test_roundtrip_surf() -> None:
    """The surf command must round-trip cleanly through a cab."""
    _assert_roundtrip("surf")


def test_roundtrip_chi2() -> None:
    """The chi2 command must round-trip cleanly through a cab."""
    _assert_roundtrip("chi2")


def test_roundtrip_flag_chi2() -> None:
    """The flag-chi2 command must round-trip cleanly through a cab."""
    _assert_roundtrip("flag_chi2")


def test_roundtrip_onboard() -> None:
    """The onboard command must round-trip cleanly through a cab."""
    _assert_roundtrip("onboard")
