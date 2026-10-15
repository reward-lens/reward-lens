"""`npm run build:report` reproduces the committed bundle byte for byte.

The wheel ships the committed bundle and `hatch_build.py` rebuilds it when Node is present, so the
two have to be the same bytes or the wheel a user installs is not the artifact this repo reviewed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest
from conftest import ROOT

ASSETS = ROOT / "src" / "reward_lens" / "render" / "report" / "assets"
FRONTEND = ROOT / "frontend"


def node_env() -> dict[str, str]:
    return dict(os.environ)


def test_the_committed_bundle_is_present_and_is_a_classic_script() -> None:
    js = (ASSETS / "report.js").read_text(encoding="utf-8")
    css = (ASSETS / "report.css").read_text(encoding="utf-8")
    assert js and css
    assert "export " not in js
    assert "\nimport " not in js
    assert not js.lstrip().startswith("import ")
    for needle in ("fetch(", "new Worker", "importScripts", "XMLHttpRequest"):
        assert needle not in js, f"the bundle must not contain {needle!r}"
    # The XML namespace names React passes to createElementNS are identifiers, never fetched.
    namespaces = re.compile(r"https?://(www\.w3\.org/[A-Za-z0-9/.-]*|react\.dev/errors/)")
    assert re.search(r"https?://[^\s\"'`)]+", namespaces.sub("", js)) is None


def test_npm_run_build_report_reproduces_the_committed_bundle(tmp_path) -> None:
    env = node_env()
    env["RL_REPORT_ASSETS_DIR"] = str(tmp_path)
    npm = shutil.which("npm", path=env["PATH"])
    assert npm, "Node and npm are build dependencies; npm is not on PATH"
    if not (FRONTEND / "node_modules").is_dir():
        pytest.fail("frontend/node_modules is absent; run `npm ci` in frontend/ before the build")
    done = subprocess.run(
        [npm, "run", "--silent", "build:report"],
        cwd=FRONTEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    for name in ("report.js", "report.css"):
        assert (tmp_path / name).read_bytes() == (ASSETS / name).read_bytes(), (
            f"{name} rebuilt to different bytes than the committed copy"
        )
