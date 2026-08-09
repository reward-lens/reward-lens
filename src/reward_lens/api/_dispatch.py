"""The lazy seam between the public surface and the engines under it.

A pointer table, resolved on demand, exactly like `reward_lens.examples.REGISTRY` and
`reward_lens.product.compare.registry.RUNGS`. Nothing here imports an engine at import time, so
`import reward_lens.api` costs the contracts and the standard library and nothing else. A verb
whose engine is absent is not an error: the verb builds the honest record instead.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any, Callable

__all__ = ["ENGINES", "engine", "load"]

#: Verb -> `"module:function"`. A packet lands its engine by creating the module; nothing here
#: changes. The pointer is resolved on demand and never cached, so a module injected into
#: `sys.modules` by a test or a plugin is found the same way a wheel's module is.
ENGINES: dict[str, str] = {
    "audit": "reward_lens.product.audit:run",
    "dry_run": "reward_lens.product.audit:plan",
    "trace": "reward_lens.product.trace:run",
    "compare": "reward_lens.product.compare.verb:run",
    "forecast_issue": "reward_lens.product.forecast:issue",
    "forecast_resolve": "reward_lens.product.forecast:resolve",
    "forecast_ledger": "reward_lens.product.forecast:ledger",
    "improve": "reward_lens.product.improve:run",
    "export": "reward_lens.product.export:run",
    "doctor": "reward_lens.product.access:doctor",
}


def load(target: str) -> Callable[..., Any] | None:
    """Resolve `"module:function"` to a callable, or `None` if this build does not hold it.

    `sys.modules` is consulted first: a module a test builds by hand has no `__spec__`, which is
    the one input `importlib.util.find_spec` refuses rather than reports. A missing module, a
    missing attribute and an attribute that is not callable are the same answer, because each of
    them means the same thing to the caller: the instrument is not in this build.
    """
    module_name, _, attribute = target.partition(":")
    if not attribute:
        return None
    module = sys.modules.get(module_name)
    if module is None:
        try:
            if importlib.util.find_spec(module_name) is None:
                return None
            module = importlib.import_module(module_name)
        except Exception:
            return None
    found = getattr(module, attribute, None)
    return found if callable(found) else None


def engine(verb: str) -> Callable[..., Any] | None:
    """The engine behind one verb, or `None` while its packet has not landed."""
    target = ENGINES.get(verb)
    return load(target) if target else None


def context(path: Any = None) -> dict[str, Any]:
    """The `project` and `sandbox` an engine is called with, each `None` where its packet is not
    in this build. Resolved through the same seam so that neither import is paid for here."""
    project = None
    opener = load("reward_lens.store:Project")
    if opener is not None and path is not None:
        try:
            project = opener.open(path)  # type: ignore[attr-defined]
        except Exception:
            project = None
    sandbox = None
    factory = load("reward_lens.execution:default_sandbox")
    if factory is not None:
        try:
            sandbox = factory()
        except Exception:
            sandbox = None
    return {"project": project, "sandbox": sandbox}
