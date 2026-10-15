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
from pathlib import Path
from types import SimpleNamespace
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
    "improve": "reward_lens.product.improve.session:run",
    "export": "reward_lens.product.export:run",
    "import_record": "reward_lens.product.ingest:run",
    "doctor": "reward_lens.product.access:doctor",
    "seeker_propose": "reward_lens.product.audit.proposal:run_seeker_proposal",
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


def forecast_context(path: Any = None) -> dict[str, Any]:
    """Resolve an explicit Forecast root or refuse instead of falling back to cwd."""
    resolved = context(path)
    if path is None or resolved["project"] is not None:
        return resolved

    root = Path(path)
    ledger = root / ".reward-lens" / "forecast-ledger.jsonl"
    try:
        # Forecast also supports an existing standalone ledger with no project config.
        if (root / "rewardlens.yaml").exists() or not ledger.is_file():
            raise ValueError("no opened project or standalone Forecast ledger")
        ledger.resolve().relative_to(root.resolve())
        if not ledger.read_bytes().strip():
            raise ValueError("standalone Forecast ledger is empty")
        reader = load("reward_lens.product.forecast:ledger_machine_state")
        if reader is None:
            raise ValueError("Forecast ledger reader is unavailable")
        standalone = SimpleNamespace(root=root)
        reader(standalone)  # Validate the existing sealed ledger before accepting its root.
        resolved["project"] = standalone
        return resolved
    except (OSError, ValueError) as error:
        from reward_lens.contracts import UsageError

        raise UsageError(
            code="RL0003",
            message=f"project={path} cannot be opened: {error}",
            remediation="initialize the requested reward-lens project or select an existing Forecast ledger",
            context={"field": "project", "path": str(path)},
        ) from error


def call_expected_refusal(
    found: Callable[..., Any], request: Any, *, verb: str, **kwargs: Any
) -> Any:
    """Turn declared engine refusals into the public request error family."""
    try:
        return found(request, **kwargs)
    except ValueError as error:
        message = str(error)
        name = message.split(":", 1)[0].split(" ", 1)[0]
        code = "RL0003"
        if verb.startswith("forecast_"):
            from reward_lens.forecast.evidence import ForecastRefusal

            if not isinstance(error, ForecastRefusal):
                raise
            field = {
                "FORECAST_FUTURE_INPUT": "record",
                "FORECAST_BASELINE_LEAKAGE": "baseline_provenance",
                "FORECAST_FUTURE_BET": "forecast",
                "FORECAST_RESOLUTION_MISMATCH": "forecast",
                "FORECAST_MEASUREMENT_TAMPER": "forecast",
                "FORECAST_CAMPAIGN_EVIDENCE": "record",
            }.get(name, "forecast")
            if name == "FORECAST_RESOLUTION_MISMATCH":
                if "outcome" in message:
                    field = "outcome"
                elif "sealed_digest=" in message and (
                    "is not in" in message or "already resolved" in message
                ):
                    code, field = "RL0002", "forecast_id"
        elif verb == "improve":
            from reward_lens.product.improve.session import ImproveRefusal

            if not isinstance(error, ImproveRefusal):
                raise
            if "path=" in message:
                field = "path"
            elif "candidate" in message:
                field = "candidate"
            elif "forecast" in message:
                field = "forecast"
            elif "describe" in message:
                field = "describe"
            elif name == "IMPROVE_TRADEOFF_UNRESOLVED":
                field = "mark_pair"
            else:
                field = "resume"
        else:
            raise

        from reward_lens.contracts import UsageError

        raise UsageError(
            code=code,
            message=f"{field}: {message}",
            remediation=f"correct the named {field} input and retry reward-lens {verb.split('_')[0]}",
            context={"field": field, "refusal": name, "detail": message},
        ) from error
