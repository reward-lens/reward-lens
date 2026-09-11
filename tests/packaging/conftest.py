"""Session fixtures for the packaging suite.

Everything here asserts against a *built wheel installed into a venv that holds nothing else*,
never against the source tree. That distinction is the whole point of the packet: `pip install -e .`
and a `PYTHONPATH=src` run both see files the wheel may not carry and dependencies the closure may
not declare, so a suite that runs in the source tree cannot see the two defects this suite exists
to catch (a missing data file, an undeclared import).

The wheel is built once per session and the clean venv is made once per session, because both cost
seconds and every test wants the same one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
CLEAN_VENV = ROOT / ".venv-clean"

#: Distributions that must not appear in a base install. D-58 and D-30: nothing numeric, nothing
#: that renders, nothing that was only ever needed by the v3 CLI this build replaces.
FORBIDDEN = frozenset(
    {
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "pydantic-settings",
        "rich",
        "typer",
    }
)

#: The measured base closure, frozen. `uv pip list` in a venv holding only the wheel must print
#: exactly this set, compared case-insensitively and with `_` normalised to `-` (PEP 503). Adding a
#: name here is adding a base dependency, which D-58 forbids without a decision record.
EXPECTED_CLOSURE = frozenset(
    {
        "reward-lens",
        # declared, per D-58 plus addendum A-001
        "click",
        "pyyaml",
        "pydantic",
        "jsonschema-rs",
        "rfc8785",
        "libcst",
        "coverage",
        # pulled in by pydantic
        "annotated-types",
        "pydantic-core",
        "typing-extensions",
        "typing-inspection",
    }
)


def normalise(name: str) -> str:
    """PEP 503 name normalisation, enough of it for a `pip list` comparison."""
    return name.lower().replace("_", ".").replace("-", ".").replace(".", "-")


def run(argv: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=kw.pop("timeout", 600),
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="session")
def wheel(pytestconfig: pytest.Config) -> Path:
    """Build the wheel with `uv build --no-sources --wheel` and return its path.

    `--no-sources` is what D-58 names for publishing: it makes uv ignore any `[tool.uv.sources]`
    redirection so the build is the one a user would get from PyPI rather than one wired to a
    local checkout.
    """
    if DIST.exists():
        shutil.rmtree(DIST)
    proc = run(["uv", "build", "--no-sources", "--wheel", "-o", str(DIST)])
    if proc.returncode != 0:
        pytest.fail(f"uv build failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    wheels = sorted(DIST.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {[w.name for w in wheels]}"
    return wheels[0]


@pytest.fixture(scope="session")
def clean_python(wheel: Path) -> Path:
    """A venv holding the wheel and its closure and nothing else; returns its interpreter."""
    if CLEAN_VENV.exists():
        shutil.rmtree(CLEAN_VENV)
    proc = run(["uv", "venv", str(CLEAN_VENV), "-q"])
    if proc.returncode != 0:
        pytest.fail(f"uv venv failed:\n{proc.stdout}\n{proc.stderr}")
    python = CLEAN_VENV / "bin" / "python"
    proc = run(["uv", "pip", "install", "-q", "--python", str(python), str(wheel)])
    if proc.returncode != 0:
        pytest.fail(f"installing the wheel failed:\n{proc.stdout}\n{proc.stderr}")
    return python


@pytest.fixture(scope="session")
def installed(clean_python: Path) -> dict[str, str]:
    """`uv pip list` in the clean venv, as {normalised name: version}."""
    proc = run(["uv", "pip", "list", "--python", str(clean_python), "--format", "json"])
    if proc.returncode != 0:
        pytest.fail(f"uv pip list failed:\n{proc.stdout}\n{proc.stderr}")
    return {normalise(row["name"]): row["version"] for row in json.loads(proc.stdout)}


@pytest.fixture(scope="session")
def wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


@pytest.fixture(scope="session")
def dist_info(wheel: Path) -> dict[str, str]:
    """The `.dist-info` members of the wheel, as {basename: text}."""
    out: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if ".dist-info/" in name and not name.endswith("/"):
                out[name.split("/", 1)[1]] = zf.read(name).decode("utf-8")
    return out


def import_probe(python: Path, module: str, *, extra_code: str = "") -> dict[str, object]:
    """Import `module` in a fresh interpreter and report what landed in `sys.modules`.

    Runs with `-I` so neither the user site directory nor `PYTHONPATH` can smuggle a package into
    the clean venv, and with the working directory set outside the source tree so `src/` cannot be
    picked up by accident. Both of those have made this kind of test pass while lying.
    """
    watched = sorted(
        {
            "numpy",
            "scipy",
            "pandas",
            "sklearn",
            "pydantic_settings",
            "rich",
            "typer",
            "torch",
        }
    )
    code = (
        "import json, sys\n"
        f"import {module}\n"
        f"{extra_code}\n"
        f"present = [m for m in {watched!r} if m in sys.modules]\n"
        'print("RESULT " + json.dumps({"present": present}))\n'
    )
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    proc = subprocess.run(  # noqa: S603
        [str(python), "-I", "-c", code],
        cwd=str(Path(sys.prefix).parent if Path(sys.prefix).exists() else "/"),
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    tail = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "present": json.loads(tail[-1][len("RESULT ") :])["present"] if tail else None,
    }
