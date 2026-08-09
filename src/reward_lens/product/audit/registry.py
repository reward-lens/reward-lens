"""Panel discovery: every `reward_lens.instruments.*` module that exposes a module-level `PANEL`.

Frozen with `instruments/base.py` in wave 1. A wave-2 instrument packet creates its own package
under `reward_lens.instruments`, names a `PANEL`, and appears in the audit without a line changing
here. A module that will not import is skipped rather than fatal: the audit's job is to run every
panel it can and write an honest absence for the rest, and a plugin that fails to import is a
missing panel, not a crashed run.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from reward_lens.instruments.base import Panel

__all__ = ["discover", "import_failures"]

PACKAGE = "reward_lens.instruments"

#: Module name -> the exception text, for every module under the package that would not import.
#: Read by the audit so that a broken plugin becomes a `COULD_NOT_CHECK` hole rather than silence.
import_failures: dict[str, str] = {}


def discover(package: str = PACKAGE) -> tuple[Panel, ...]:
    """Every `PANEL` under `package`, in module-name order. Never raises."""
    import_failures.clear()
    try:
        root: Any = importlib.import_module(package)
    except Exception as failure:  # pragma: no cover - the package ships with the wheel
        import_failures[package] = f"{type(failure).__name__}: {failure}"
        return ()
    found: list[Panel] = []
    for info in sorted(pkgutil.iter_modules(getattr(root, "__path__", [])), key=lambda i: i.name):
        if info.name.startswith("_") or info.name == "base":
            continue
        name = f"{package}.{info.name}"
        try:
            module = importlib.import_module(name)
        except Exception as failure:
            import_failures[name] = f"{type(failure).__name__}: {failure}"
            continue
        panel = getattr(module, "PANEL", None)
        if isinstance(panel, Panel):
            found.append(panel)
    return tuple(found)
