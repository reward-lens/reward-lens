#!/usr/bin/env python3
"""Run every test path in a release profile without cross-suite conftest collisions."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from collections import defaultdict


def groups(matrix: pathlib.Path, profile: str) -> list[list[str]]:
    data = json.loads(matrix.read_text())
    suites = data["profiles"][profile]["suites"]
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for name in suites:
        if name in seen:
            raise ValueError(f"duplicate release suite: {name}")
        seen.add(name)
        path = pathlib.Path(name)
        if not path.exists():
            raise FileNotFoundError(name)
        if name in {"tests/conftest.py", "tests/w2_3_typing"}:
            # Import/typing closure inputs, not standalone pytest test modules.
            # The latter has no test_*.py; its files are read by typing tests.
            continue
        if path.is_dir():
            if name == "tests/port":
                # Two port subtrees contain the same test module basename.
                # pytest's import mode collides if they are collected together.
                for child in sorted(path.iterdir()):
                    if child.is_dir() and list(child.rglob("test_*.py")):
                        grouped[str(child)].append(str(child))
                for child in sorted(path.glob("test_*.py")):
                    grouped[name].append(str(child))
            else:
                grouped[name].append(name)
        elif path.name.startswith("test_") and path.suffix == ".py":
            grouped[str(path.parent)].append(name)
        else:
            raise ValueError(f"not a runnable test suite: {name}")
    return [sorted(paths) for _, paths in sorted(grouped.items())]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument(
        "--matrix", type=pathlib.Path, default=pathlib.Path("release/validation-matrix.json")
    )
    parser.add_argument("--list", action="store_true", help="validate and list commands only")
    args = parser.parse_args()
    commands = [
        [sys.executable, "-m", "pytest", "-q", "-ra", *suite]
        for suite in groups(args.matrix, args.profile)
    ]
    environment = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONUSERBASE",
        "PYTEST_ADDOPTS",
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["REWARD_LENS_EXIT_CHECK"] = "1"
    environment["REWARD_LENS_SLOW"] = "1"
    with tempfile.TemporaryDirectory(prefix="rlm-") as scratch:
        environment["TMPDIR"] = scratch
        for command in commands:
            print("RUN", " ".join(command), flush=True)
            if not args.list:
                subprocess.run(command, check=True, env=environment)
    print(f"release matrix {args.profile}: {len(commands)} isolated pytest groups completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
