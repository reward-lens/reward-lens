"""Keep every active release declaration on the canonical reward-lens 3.1.0 line."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERSION = "3.1.0"


def test_active_release_declarations_agree() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    frontend = json.loads((ROOT / "frontend/package.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "frontend/report/package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "frontend/package-lock.json").read_text(encoding="utf-8"))
    runtime = (ROOT / "src/reward_lens/__init__.py").read_text(encoding="utf-8")
    uv_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert pyproject["project"]["name"] == "reward-lens"
    assert pyproject["project"]["version"] == VERSION
    assert frontend["version"] == VERSION
    assert report["version"] == VERSION
    assert lock["version"] == VERSION
    assert lock["packages"]["report"]["version"] == VERSION
    assert re.search(r'^__version__ = "3\.1\.0"$', runtime, re.MULTILINE)
    assert re.search(r'(?ms)^name = "reward-lens"\nversion = "3\.1\.0"$', uv_lock)


def test_readme_uses_the_3_1_route() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<!-- 3.1.0-route -->" in readme
    assert "<!-- /3.1.0-route -->" in readme
    assert "reward-lens 3.1.0 audits" in readme
