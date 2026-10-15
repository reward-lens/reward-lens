"""Acceptance collection must not shadow the installed ``packaging`` distribution."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import packaging
import pytest
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
DEBT_FIXES = "tests/acceptance/test_debt_e_fixes.py"
TRAIN_STACK_CONSUMER = "tests/acceptance/test_w5_1_policy.py"


def test_installed_packaging_exposes_its_version_module() -> None:
    """The import is the third-party distribution, never ``tests/packaging``."""
    installed_init = Path(
        importlib.metadata.distribution("packaging").locate_file("packaging/__init__.py")
    ).resolve()
    assert Path(packaging.__file__).resolve() == installed_init
    assert Version("1.2.3") > Version("1.2.2")


@pytest.mark.parametrize(
    "modules",
    [
        (DEBT_FIXES, TRAIN_STACK_CONSUMER),
        (TRAIN_STACK_CONSUMER, DEBT_FIXES),
    ],
)
def test_acceptance_collection_order_preserves_the_train_stack(modules: tuple[str, str]) -> None:
    """Fixture loading cannot change imports seen by subsequent acceptance modules."""
    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--import-mode=importlib",
            *modules,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
