"""Estimator discovery imports library modules, never executable module entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap


def _in_fresh_process(body: str) -> dict[str, object]:
    """Run the public loader before any reward-lens module has been imported."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def _in_fresh_source_process(body: str) -> dict[str, object]:
    """Run a new interpreter against this checkout's source tree.

    The installed test interpreter deliberately runs with ``-I`` in the public-loader control
    above, so it cannot see a worktree's uninstalled source. This companion control must exercise
    the entry point being changed here.
    """
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        env=environment,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def test_proficiency_module_import_is_inert_in_a_fresh_process() -> None:
    """The executable proficiency entry point must not parse its importer's argv."""
    result = _in_fresh_source_process(
        """
        import importlib
        import json

        module = importlib.import_module("reward_lens.reference.proficiency.__main__")
        print(json.dumps({"name": module.__name__, "has_main": callable(module.main)}))
        """
    )

    assert result == {
        "name": "reward_lens.reference.proficiency.__main__",
        "has_main": True,
    }


def test_proficiency_module_still_executes_its_cli() -> None:
    """The import guard must not change ``python -m`` execution."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "reward_lens.reference.proficiency", "--help"],
        capture_output=True,
        env=environment,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "reward-lens proficiency" in result.stdout
    assert "publish-charter" in result.stdout
    assert "run" in result.stdout


def test_ladder_filters_every_non_vendor_command_entrypoint_in_a_fresh_process() -> None:
    """``python -m`` modules must not run while the report loads estimators.

    Command modules are still excluded from estimator discovery even though their imports are now
    inert. The assertion derives the expected modules from the walk, so a later command module
    cannot silently reopen that front door.
    """
    result = _in_fresh_process(
        """
        import json
        import pkgutil
        import reward_lens
        from reward_lens.access.estimators import load_estimator_ladder
        from reward_lens.core.quantity import ESTIMATORS

        command_modules = sorted(
            module.name
            for module in pkgutil.walk_packages(reward_lens.__path__, "reward_lens.")
            if module.name.endswith(".__main__")
        )
        ladder = load_estimator_ladder()
        print(json.dumps({
            "origin": reward_lens.__file__,
            "registered": len(ESTIMATORS),
            "broken": list(ladder.broken),
            "entrypoints_skipped": list(ladder.entrypoints_skipped),
            "command_modules": command_modules,
        }, sort_keys=True))
        """
    )

    assert result["origin"]
    assert result["registered"] > 0
    assert result["command_modules"]
    assert result["entrypoints_skipped"] == result["command_modules"]
    assert all(".__main__" not in detail for detail in result["broken"])
