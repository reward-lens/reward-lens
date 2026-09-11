"""Optional exit checks for shipped report assets and console scripts.

Set ``REWARD_LENS_EXIT_CHECK=1`` to run these checks in an isolated staging checkout. The
report bundle and both command entry points are present in this curated source tree. The command
list is read from the public CLI registry, so this test does not depend on a private fleet ledger.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from .conftest import DIST, ROOT

if os.environ.get("REWARD_LENS_EXIT_CHECK") != "1":
    pytest.skip(
        "wave-1 exit check; set REWARD_LENS_EXIT_CHECK=1",
        allow_module_level=True,
    )

#: The report bundle, by its path inside the wheel. Interfaces section 7: the wheel carries a built
#: bundle so that no end user ever runs `npm`.
BUNDLE_MEMBERS = (
    "reward_lens/render/report/assets/report.js",
    "reward_lens/render/report/assets/report.css",
)

#: Both console scripts of `[project.scripts]`. `rlens` is not an alias anyone may drop: it is in
#: the published interface, so it is checked with the same weight as the long name.
ENTRY_POINTS = ("reward-lens", "rlens")

#: The fourteen verbs of the public command tree. The public registry is the one source of truth,
#: so a command removed from the CLI makes this wheel check fail without a private ledger.
from reward_lens.cli.registry import ORDER

CONTRACTED_VERBS: tuple[str, ...] = ORDER

#: What the copied tree needs for `uv build` to produce the same wheel. `frontend/` is deliberately
#: absent: without `frontend/package.json` the build hook takes its "nothing to build" branch, so
#: the negative build below succeeds and yields a wheel with no bundle, which is the artefact the
#: test needs. With `frontend/` present the hook would either rebuild the bundle or refuse, and
#: either way the test would stop measuring what it claims to measure.
BUILDABLE = ("pyproject.toml", "hatch_build.py", "src", "schema", "README.md", "LICENSE")


def _env(**extra: str) -> dict[str, str]:
    """The ambient environment with the two variables that can smuggle the source tree removed."""
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(extra)
    return env


def _run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env if env is not None else _env(),
    )


def _ok(proc: subprocess.CompletedProcess[str], what: str) -> subprocess.CompletedProcess[str]:
    if proc.returncode != 0:
        pytest.fail(f"{what} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc


def assert_bundle_in_wheel(names: list[str]) -> None:
    """The bundle check, factored out so the negative test below can run the same code."""
    missing = [member for member in BUNDLE_MEMBERS if member not in names]
    assert not missing, {
        "missing": missing,
        "under_assets": [n for n in names if "/render/report/assets/" in n],
    }


def run_entry_points(bin_dir: Path, cwd: Path) -> None:
    """Run `--help` for both console scripts out of `bin_dir` and require a usage line from each.

    The return code alone is too weak. A console script whose module imports but whose `main` is
    not a command would still exit 0 on some shapes, so the usage line is what says click's entry
    point was actually reached.
    """
    for name in ENTRY_POINTS:
        script = bin_dir / name
        assert script.is_file(), f"{name} was not installed into {bin_dir}"
        proc = _ok(_run([str(script), "--help"], cwd=cwd, timeout=180), f"{name} --help")
        assert "Usage" in proc.stdout, {"script": str(script), "stdout": proc.stdout[:400]}


def _uv_venv(path: Path, *, seed: bool = False) -> Path:
    argv = ["uv", "venv", "-q", str(path)]
    if seed:
        argv.append("--seed")
    _ok(_run(argv, cwd=ROOT), f"uv venv {path.name}")
    return path / "bin" / "python"


def test_the_wheel_carries_at_least_one_example(wheel_names: list[str]) -> None:
    """Interfaces section 7 ships `reward_lens/examples/**` so that the worked examples are
    addressable from an installed distribution. A package directory holding only `__init__.py`
    satisfies `test_wheel_carries_the_packaged_data` and ships nothing, so the exit check counts
    what is in it."""
    examples = [
        name
        for name in wheel_names
        if name.startswith("reward_lens/examples/")
        and not name.endswith("/")
        and name != "reward_lens/examples/__init__.py"
    ]
    assert examples, {
        "under_examples": [n for n in wheel_names if n.startswith("reward_lens/examples/")]
    }


def test_the_wheel_carries_the_report_bundle(wheel_names: list[str]) -> None:
    """Both halves of the bundle, not just the script: a `report.js` without its stylesheet renders
    a report that is technically present and unreadable."""
    assert_bundle_in_wheel(wheel_names)


def test_both_console_scripts_run_from_the_clean_venv(clean_python: Path, tmp_path: Path) -> None:
    """The clean venv holds the wheel and its closure and nothing else, so a script that runs here
    runs on the base install rather than on whatever a developer happens to have."""
    run_entry_points(clean_python.parent, tmp_path)


def test_uv_pip_install_runs_the_entry_point(wheel: Path, tmp_path: Path) -> None:
    """Section 8.5, first contracted path."""
    python = _uv_venv(tmp_path / "venv-uv-pip")
    _ok(
        _run(["uv", "pip", "install", "-q", "--python", str(python), str(wheel)], cwd=ROOT),
        "uv pip install",
    )
    run_entry_points(python.parent, tmp_path)


def test_pip_install_runs_the_entry_point(wheel: Path, tmp_path: Path) -> None:
    """Section 8.5, second contracted path, offline against `dist/`.

    `--no-index --find-links dist` is the whole point: the project must come out of the local
    wheel and not out of whatever PyPI happens to hold under the same name. That flag also cuts
    off pip's route to the dependencies, so the closure is seeded with uv first and the project
    itself is then removed again, leaving pip to install exactly one distribution. What this
    asserts is that pip can install the built wheel and produce working scripts; that the closure
    is the right closure is `test_closure.py`'s assertion, not this one.
    """
    python = _uv_venv(tmp_path / "venv-pip", seed=True)
    _ok(
        _run(["uv", "pip", "install", "-q", "--python", str(python), str(wheel)], cwd=ROOT),
        "seeding the closure",
    )
    _ok(
        _run([str(python), "-m", "pip", "uninstall", "-y", "-q", "reward-lens"], cwd=ROOT),
        "pip uninstall reward-lens",
    )
    _ok(
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "-q",
                "--no-index",
                "--find-links",
                str(DIST),
                "reward-lens",
            ],
            cwd=ROOT,
        ),
        "pip install --no-index --find-links dist",
    )
    run_entry_points(python.parent, tmp_path)


def test_uv_tool_install_runs_the_entry_point(wheel: Path, tmp_path: Path) -> None:
    """Section 8.5, third contracted path.

    `UV_TOOL_DIR` and `UV_TOOL_BIN_DIR` are redirected into `tmp_path` so the check neither reads
    nor writes the machine's real tool installation: a run that quietly upgraded the developer's
    own `reward-lens` would be a side effect, and one that found an older tool already on the bin
    dir would be a false pass.
    """
    bin_dir = tmp_path / "tool-bin"
    env = _env(UV_TOOL_DIR=str(tmp_path / "tool-dir"), UV_TOOL_BIN_DIR=str(bin_dir))
    _ok(
        _run(
            ["uv", "tool", "install", "-q", "--from", str(wheel), "reward-lens"],
            cwd=ROOT,
            env=env,
        ),
        "uv tool install --from",
    )
    run_entry_points(bin_dir, tmp_path)


def assert_root_help_names_the_verbs(help_text: str) -> None:
    """The root help has to list every contracted verb in its command column.

    Matching bare substrings would pass on prose instead: `open`, `import`, `runs` and `compare`
    are ordinary English and turn up in the surrounding lines. What is matched is the shape of the
    command column, an indented verb followed by its one-line summary.
    """
    assert len(CONTRACTED_VERBS) == 14, CONTRACTED_VERBS
    missing = [
        name
        for name in CONTRACTED_VERBS
        if re.search(rf"^ +{re.escape(name)} +\S", help_text, re.MULTILINE) is None
    ]
    assert not missing, {"missing": missing, "help": help_text[:800]}


def test_uvx_runs_the_entry_point(wheel: Path, dist_info: dict[str, str], tmp_path: Path) -> None:
    """Section 8.5, fourth contracted path. `uvx` installs nothing durable, so there is no bin
    directory to check afterwards; the run itself is the assertion.

    `--no-cache` is not decoration here. Without it this check ran a cached environment rather than
    the wheel the session had just built. uv keys an ephemeral `uvx` environment on the requirement
    it was handed, so `--from dist/reward_lens-<version>-py3-none-any.whl` matched an environment
    built from an earlier wheel of the same name and version, and the console script under
    `~/.cache/uv/archive-v0/` raised `ModuleNotFoundError: No module named 'reward_lens.cli.main'`
    while the wheel under test carried that module. The same path holding new bytes does not
    invalidate the entry, and that is the whole failure: the artefact being certified was never
    installed.

    Three flags fix it and they are not equivalent. `--refresh` and `--reinstall`, which implies
    `--refresh`, keep reading and rewriting the shared cache, so they ask the staleness judgement
    that just went wrong to come out right the second time, and they leave the developer's cache
    altered. `--no-cache` avoids reading from or writing to the cache at all and works in a
    temporary directory for the run, so nothing cached can be reused whatever uv makes of it, the
    second run of the day cannot differ from the first because the first left nothing behind, and
    the machine is as it was. It costs about ten seconds of fetching the closure, which is the
    price of the check meaning what it says.

    Two assertions, not one. The version line ties the running program to the wheel's own metadata.
    The root help has to name all fourteen verbs, and that is the half that catches this defect:
    the stale environment answered `--version` with the right number while carrying no CLI at all.
    """
    env = _env(UV_TOOL_DIR=str(tmp_path / "tool-dir"), UV_TOOL_BIN_DIR=str(tmp_path / "tool-bin"))

    def uvx(*argv: str) -> str:
        proc = _ok(
            _run(
                ["uvx", "--no-cache", "--from", str(wheel), "reward-lens", *argv],
                cwd=tmp_path,
                env=env,
                timeout=300,
            ),
            f"uvx --no-cache --from {wheel.name} reward-lens {' '.join(argv)}",
        )
        return proc.stdout

    declared = re.search(r"^Version: (.+)$", dist_info["METADATA"], re.MULTILINE)
    assert declared is not None, sorted(dist_info)
    version = declared.group(1).strip()
    assert uvx("--version").strip() == f"reward-lens {version}"

    help_text = uvx("--help")
    assert "Usage" in help_text, help_text[:400]
    assert_root_help_names_the_verbs(help_text)


def test_a_tree_without_report_js_fails_the_bundle_check(tmp_path: Path) -> None:
    """The negative control: without it, `assert_bundle_in_wheel` could be asserting nothing.

    A control only controls if it can fail for the reason it names, so this one manufactures the
    bundle before removing part of it. A copy of the buildable tree is made under `tmp_path`, a
    stand-in `report.js` and `report.css` are written into the assets directory, and the wheel
    built from that tree has to pass the bundle check: that is the leg that says the check can be
    satisfied at all. Then `report.js` alone is removed, the tree is rebuilt, and the same check
    has to fail on the new wheel while `report.css` is still in it: that is the leg that says one
    missing file is enough. The earlier version of this test unlinked `report.js` with
    `missing_ok=True` from a tree where the assets directory does not exist yet, so it removed
    nothing and measured the same bundle-free wheel the positive check above already fails on.

    `frontend/` is not copied, which puts the build hook on its "nothing to build" branch and keeps
    both builds green; see BUILDABLE above.
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    for entry in BUILDABLE:
        source = ROOT / entry
        assert source.exists(), f"{entry} is missing from the worktree"
        if source.is_dir():
            shutil.copytree(source, tree / entry, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, tree / entry)

    def build_names(out: Path, what: str) -> list[str]:
        _ok(
            _run(["uv", "build", "--no-sources", "--wheel", "-o", str(out)], cwd=tree),
            f"uv build of the copied tree ({what})",
        )
        built = sorted(out.glob("*.whl"))
        assert len(built) == 1, [p.name for p in built]
        with zipfile.ZipFile(built[0]) as zf:
            names = zf.namelist()
        assert "reward_lens/__init__.py" in names, f"no real wheel from the copied tree ({what})"
        return names

    # The stand-ins are one line each and the content is never read: what is under test is the
    # build configuration's willingness to carry the assets directory into the wheel, and the
    # check's response when one of the two files is gone. `exist_ok` and the overwrite keep this
    # true after P-RENDER commits a real bundle here.
    assets = tree / "src/reward_lens/render/report/assets"
    assets.mkdir(parents=True, exist_ok=True)
    js = assets / "report.js"
    css = assets / "report.css"
    js.write_text("/* stand-in for the built bundle */\n", encoding="utf-8")
    css.write_text("/* stand-in for the built stylesheet */\n", encoding="utf-8")

    names = build_names(tmp_path / "dist-with-bundle", "bundle present")
    assert_bundle_in_wheel(names)

    assert js.is_file(), f"{js} was not written into the copied tree"
    js.unlink()
    assert css.is_file(), f"{css} went with it; only report.js may be removed here"
    names = build_names(tmp_path / "dist-without-report-js", "report.js removed")
    with pytest.raises(AssertionError):
        assert_bundle_in_wheel(names)
    assert "reward_lens/render/report/assets/report.css" in names, {
        "under_assets": [n for n in names if "/render/report/assets/" in n],
    }
