"""Default pytest collection must preserve installed top-level distributions."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_default_collection_keeps_packaging_external_when_an_optional_sdk_imports_it(
    tmp_path: Path,
) -> None:
    """A real import shape, with no optional SDK installation needed for the control.

    Matplotlib imports ``packaging.version`` during root-conftest setup.  This minimal stand-in
    exercises that exact third-party import while a normal pytest collection handles a real suite
    module.  It fails if ``tests/packaging`` is exposed as top-level ``packaging``.
    """
    fake_root = tmp_path / "optional_sdk"
    module = fake_root / "matplotlib"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text(
        "from packaging.version import Version\n"
        "def use(_backend: str) -> None:\n"
        "    Version('1.0')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    # Only the deliberate stand-in dependency is injected. The product comes from
    # the installed wheel, so source files cannot conceal missing package content.
    env["PYTHONPATH"] = str(fake_root)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/packaging/test_dev_group_closure.py",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
