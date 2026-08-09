"""The hatchling build hook: produce the report bundle, or say plainly why there is none.

D-58 puts the frontend on the build side of the line and nowhere else: Node is a development
dependency and never an end-user requirement, so the wheel carries a built bundle and no user ever
runs `npm`. That is also why the backend is hatchling rather than `uv_build`, which is pure Python
only by its own documentation and therefore cannot run a build step at all.

Five cases, and everything after the second is what makes the hook usable in a checkout.

1. No `frontend/package.json` in the tree. That is the state before P-RENDER lands. The hook
   records "no frontend in tree" and the build succeeds. A build that failed here would block every
   other packet on one that has not started.
2. `frontend/package.json` exists, `npm` is on PATH and `frontend/node_modules` is installed. Run
   `npm run build:report` and let its failure be the build's failure.
3. `frontend/package.json` and `npm` are there and `frontend/node_modules` is not. Every fresh git
   worktree of this repository is in that state, and `npm run build:report` in it dies with
   `vite: not found` before it reads a line of source: what is missing is the build script's own
   tools, not anything about the project. So the committed bundle is the answer here as well, and
   the reason line names the absent directory and the `npm ci` that fills it. With no bundle to
   serve the hook still refuses, for the reason in case 4.
4. `frontend/package.json` exists and `npm` is not on PATH. Then the committed bundle is the only
   remaining source of the file. If it is there, use it and say so. If it is not, **fail**, loudly,
   with one line naming the missing path and the two ways out. Shipping a wheel whose report is a
   blank page because the build step was quietly skipped is worse than not shipping: the failure
   shows up at a reader's machine, hours later, as an empty `<div>`.
5. An editable install with nothing to serve. `uv sync` builds the project in order to install it
   in place, so a refusal from case 3 or 4 takes the whole development environment down with it,
   and an editable install ships to no one. There the hook prints one line to stderr and continues.
   The blank page that case 4 refuses to ship is still refused: the report renderer raises its own
   typed error when a render is attempted without a bundle. `version == "standard"`, the wheel that
   does ship, keeps the refusal.

The decision is a plain function so that it can be tested without a build. `python hatch_build.py`
runs it against the current tree and exits non-zero on the refusal, which is what the packaging
suite exercises.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Written by `npm run build:report`; committed by P-RENDER so a machine without Node can still
#: build a wheel. Interfaces section 7.
BUNDLE = Path("src/reward_lens/render/report/assets/report.js")
FRONTEND_MANIFEST = Path("frontend/package.json")
#: Where `npm ci` puts the build script's own tools. Present in a checkout that has installed
#: them, absent in a fresh git worktree, and the difference between cases 2 and 3.
NODE_MODULES = FRONTEND_MANIFEST.parent / "node_modules"
BUILD_SCRIPT = "build:report"

#: Case 5, as the one line an editable install prints. A constant because the test asserts on it
#: and a reader greps for it.
EDITABLE_NOTICE = "hatch_build: editable install without a frontend bundle:"


class FrontendBundleMissing(RuntimeError):
    """Cases 3 and 4: a frontend to build, no way to build it here, and no committed bundle."""


def plan_frontend_build(
    root: Path, *, npm: str | None = None, node_modules: bool | None = None
) -> tuple[str, str]:
    """Decide what to do about the report bundle. Returns (action, one-line reason).

    ``action`` is one of ``"absent"`` (nothing to build), ``"build"`` (run npm) or ``"committed"``
    (a bundle is already in the tree and this machine cannot rebuild it). Raises
    `FrontendBundleMissing` in the state where none of those is true.

    ``npm`` and ``node_modules`` are passed in rather than looked up inside so a test can pin any
    branch without editing PATH for the whole process or installing anything. ``npm`` stands in for
    `shutil.which("npm")`, so ``""`` means "not on PATH"; ``node_modules`` stands in for the
    existence of `frontend/node_modules` under ``root``.
    """
    manifest = root / FRONTEND_MANIFEST
    if not manifest.is_file():
        return (
            "absent",
            f"no frontend in tree ({FRONTEND_MANIFEST} does not exist); nothing to build",
        )
    bundle = root / BUNDLE
    resolved = npm if npm is not None else shutil.which("npm")
    if resolved:
        installed = node_modules if node_modules is not None else (root / NODE_MODULES).is_dir()
        if installed:
            return "build", (
                f"{FRONTEND_MANIFEST} present, npm at {resolved} and {NODE_MODULES} installed; "
                f"running npm run {BUILD_SCRIPT}"
            )
        if bundle.is_file():
            return "committed", (
                f"npm is at {resolved} but {NODE_MODULES} is absent, so `npm run {BUILD_SCRIPT}` "
                f"would fail on its own missing tools; using the committed bundle at {BUNDLE} "
                f"(run `npm ci` in {FRONTEND_MANIFEST.parent} to rebuild it instead)"
            )
        raise FrontendBundleMissing(
            f"{FRONTEND_MANIFEST} exists, {NODE_MODULES} is absent and {BUNDLE} is not committed: "
            f"run `npm ci` in {FRONTEND_MANIFEST.parent} and rerun, or commit the bundle built by "
            f"`npm run {BUILD_SCRIPT}`"
        )
    if bundle.is_file():
        return "committed", f"npm not on PATH; using the committed bundle at {BUNDLE}"
    raise FrontendBundleMissing(
        f"{FRONTEND_MANIFEST} exists, npm is not on PATH and {BUNDLE} is not committed: "
        f"install Node and rerun, or commit the bundle built by `npm run {BUILD_SCRIPT}`"
    )


def run_frontend_build(
    root: Path, *, npm: str | None = None, node_modules: bool | None = None
) -> str:
    """Apply `plan_frontend_build` and return the reason line that was acted on."""
    action, reason = plan_frontend_build(root, npm=npm, node_modules=node_modules)
    if action == "build":
        executable = npm if npm is not None else (shutil.which("npm") or "npm")
        proc = subprocess.run(  # noqa: S603
            [executable, "run", BUILD_SCRIPT],
            cwd=str(root / FRONTEND_MANIFEST.parent),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"`npm run {BUILD_SCRIPT}` failed with {proc.returncode}:\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        if not (root / BUNDLE).is_file():
            raise FrontendBundleMissing(
                f"`npm run {BUILD_SCRIPT}` succeeded but wrote no {BUNDLE}: the build script and "
                "the path the wheel ships have diverged"
            )
    return reason


try:  # hatchling is present during a build and absent when this module is imported by a test
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ModuleNotFoundError:  # pragma: no cover - exercised by every non-build import
    pass
else:

    class ReportBundleHook(BuildHookInterface):  # type: ignore[misc]
        """Runs before the wheel is assembled."""

        PLUGIN_NAME = "custom"

        def initialize(self, version: str, build_data: dict) -> None:  # noqa: ARG002
            try:
                reason = run_frontend_build(Path(self.root))
            except (FrontendBundleMissing, RuntimeError) as exc:
                if version != "editable":
                    raise
                # Case 5. One line, whatever the exception held: the npm failure carries a whole
                # log, and a wall of it on the way to an install that works is noise.
                print(f"{EDITABLE_NOTICE} {' '.join(str(exc).split())}", file=sys.stderr)
                return
            self.app.display_info(f"[reward-lens] report bundle: {reason}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent
    try:
        reason = run_frontend_build(root)
    except (FrontendBundleMissing, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(reason)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
