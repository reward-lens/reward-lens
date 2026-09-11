"""The build hook's five cases, including the two that have to fail and the one that must not.

Every case is exercised in a tree built under pytest's `tmp_path`, never in the worktree's own
`frontend/`, which belongs to P-RENDER. Creating `frontend/package.json` here would put a file in
another packet's paths and would change what every other test in this suite measures.

Nothing here asserts the state of this checkout. An earlier test in this module did, and it went
red the hour P-RENDER's `frontend/package.json` landed: a test that reads the tree it is running in
measures whichever packet merged last, not the code under it. Branches are pinned with the
`npm` and `node_modules` parameters, or by what the tmp tree contains.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hatch_build import (  # noqa: E402
    BUNDLE,
    EDITABLE_NOTICE,
    NODE_MODULES,
    FrontendBundleMissing,
    plan_frontend_build,
)


def _tree(tmp_path: Path, *, manifest: bool, bundle: bool, node_modules: bool = False) -> Path:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "hatch_build.py").write_bytes((ROOT / "hatch_build.py").read_bytes())
    if manifest:
        (root / "frontend").mkdir()
        (root / "frontend" / "package.json").write_text(
            '{"name": "reward-lens-frontend", "scripts": {"build:report": "true"}}\n'
        )
    if node_modules:
        (root / NODE_MODULES).mkdir(parents=True, exist_ok=True)
    if bundle:
        (root / BUNDLE).parent.mkdir(parents=True, exist_ok=True)
        (root / BUNDLE).write_text("/* committed bundle */\n")
    return root


@pytest.fixture
def hook_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A second copy of `hatch_build.py`, loaded with the hatchling hook interface stubbed.

    `ReportBundleHook` is defined only when `hatchling` imports, and hatchling is a build
    requirement that is resolved in an isolated environment rather than installed into the venv the
    tests run in. Stubbing the base class is what makes `initialize` testable unconditionally; the
    alternative was a skip, and a skip on the editable path is how this defect reached three
    workers in the first place.
    """

    class _StubApp:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def display_info(self, message: str) -> None:
            self.messages.append(message)

    class _StubInterface:
        def __init__(self, root: str) -> None:
            self.root = root
            self.app = _StubApp()

    interface = types.ModuleType("hatchling.builders.hooks.plugin.interface")
    interface.BuildHookInterface = _StubInterface  # type: ignore[attr-defined]
    for name in (
        "hatchling",
        "hatchling.builders",
        "hatchling.builders.hooks",
        "hatchling.builders.hooks.plugin",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "hatchling.builders.hooks.plugin.interface", interface)

    spec = importlib.util.spec_from_file_location("hatch_build_hooked", ROOT / "hatch_build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_frontend_in_tree_is_not_a_failure(tmp_path: Path) -> None:
    """The state before P-RENDER lands. Interfaces section 7 names it explicitly."""
    action, reason = plan_frontend_build(_tree(tmp_path, manifest=False, bundle=False))
    assert action == "absent"
    assert "no frontend in tree" in reason


def test_manifest_npm_and_node_modules_means_build(tmp_path: Path) -> None:
    tree = _tree(tmp_path, manifest=True, bundle=False, node_modules=True)
    action, reason = plan_frontend_build(tree, npm="/usr/bin/npm")
    assert action == "build"
    assert "npm run build:report" in reason
    assert str(NODE_MODULES) in reason


def test_the_node_modules_branch_can_be_pinned_without_a_directory(tmp_path: Path) -> None:
    """`node_modules=True` stands in for the directory the way `npm=` stands in for PATH."""
    tree = _tree(tmp_path, manifest=True, bundle=False)
    action, _ = plan_frontend_build(tree, npm="/usr/bin/npm", node_modules=True)
    assert action == "build"


def test_node_modules_absent_falls_back_to_the_committed_bundle(tmp_path: Path) -> None:
    """The state of every fresh worktree of this repository: npm is there, its packages are not.

    Running the build script here fails with `vite: not found`, which is why npm being on PATH is
    not on its own a reason to run it.
    """
    tree = _tree(tmp_path, manifest=True, bundle=True)
    action, reason = plan_frontend_build(tree, npm="/usr/bin/npm")
    assert action == "committed"
    assert str(NODE_MODULES) in reason
    assert str(BUNDLE) in reason
    assert "npm ci" in reason


def test_node_modules_absent_and_no_bundle_refuses(tmp_path: Path) -> None:
    """No packages and no bundle is still a refusal: there is nothing to put in the wheel."""
    tree = _tree(tmp_path, manifest=True, bundle=False)
    with pytest.raises(FrontendBundleMissing) as caught:
        plan_frontend_build(tree, npm="/usr/bin/npm")
    message = str(caught.value)
    assert str(NODE_MODULES) in message
    assert str(BUNDLE) in message
    assert "npm ci" in message


def test_manifest_without_npm_falls_back_to_the_committed_bundle(tmp_path: Path) -> None:
    action, reason = plan_frontend_build(_tree(tmp_path, manifest=True, bundle=True), npm="")
    assert action == "committed"
    assert str(BUNDLE) in reason


def test_manifest_without_npm_and_without_bundle_refuses(tmp_path: Path) -> None:
    """The required refusal case. D-58: a wheel whose report is a blank page is worse than no
    wheel, so the build stops here rather than shipping one."""
    with pytest.raises(FrontendBundleMissing) as caught:
        plan_frontend_build(_tree(tmp_path, manifest=True, bundle=False), npm="")
    message = str(caught.value)
    assert "frontend/package.json" in message
    assert str(BUNDLE) in message


def test_the_refusal_exits_non_zero_with_one_line(tmp_path: Path) -> None:
    """The same refusal as a process, because that is how a build reports it. PATH is emptied so
    `shutil.which("npm")` genuinely finds nothing; the interpreter is invoked by absolute path so
    that it still starts."""
    root = _tree(tmp_path, manifest=True, bundle=False)
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(root / "hatch_build.py"), str(root)],
        capture_output=True,
        text=True,
        env={"PATH": "", "HOME": str(tmp_path)},
        timeout=120,
    )
    assert proc.returncode != 0, proc.stdout
    lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, proc.stderr
    assert lines[0].startswith("error: ")
    assert "commit the bundle" in lines[0]


def test_an_editable_install_says_so_and_continues(
    tmp_path: Path, hook_module: types.ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """Case 5. The tree refuses under either PATH: with npm it is `npm ci` that is missing,
    without it Node, and neither is a reason to take `uv sync` down with it."""
    root = _tree(tmp_path, manifest=True, bundle=False)
    hook = hook_module.ReportBundleHook(str(root))

    assert hook.initialize("editable", {}) is None

    captured = capsys.readouterr()
    lines = [ln for ln in captured.err.splitlines() if ln.strip()]
    assert len(lines) == 1, captured.err
    assert lines[0].startswith(EDITABLE_NOTICE)
    assert str(BUNDLE) in lines[0]
    assert hook.app.messages == []


def test_a_standard_build_still_refuses(tmp_path: Path, hook_module: types.ModuleType) -> None:
    """The wheel that ships gets no such indulgence."""
    root = _tree(tmp_path, manifest=True, bundle=False)
    hook = hook_module.ReportBundleHook(str(root))
    with pytest.raises(hook_module.FrontendBundleMissing):
        hook.initialize("standard", {})


def test_a_bundle_that_is_there_is_reported_to_the_build(
    tmp_path: Path, hook_module: types.ModuleType
) -> None:
    """The other half of case 5: when there is a bundle, an editable install is an ordinary one."""
    root = _tree(tmp_path, manifest=True, bundle=True)
    hook = hook_module.ReportBundleHook(str(root))
    hook.initialize("editable", {})
    assert len(hook.app.messages) == 1
    assert str(BUNDLE) in hook.app.messages[0]
