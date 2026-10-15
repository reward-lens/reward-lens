"""Discovery-based tracker registry; the export verb imports this seam, never backend names."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from reward_lens.contracts import CapabilityUnavailable

from .base import Availability, RecordSummary, TrackerBackend, TrackerRef

_IGNORED = frozenset({"base", "registry", "_local"})


def _unavailable(name: str, availability: Availability) -> CapabilityUnavailable:
    credential = availability.credential or "tracker configuration"
    return CapabilityUnavailable(
        code="RL0720",
        message=f"tracker '{name}' is unavailable: {credential} is required",
        remediation=availability.detail or f"configure {credential} before exporting",
        context={"tracker": name, "credential": credential},
    )


def _backend(module: ModuleType) -> TrackerBackend:
    factory = getattr(module, "backend", None)
    if not callable(factory):
        raise TypeError(f"tracker module {module.__name__!r} has no backend() factory")
    candidate = factory()
    for attribute in ("name", "available", "write"):
        if not hasattr(candidate, attribute):
            raise TypeError(f"tracker module {module.__name__!r} backend lacks {attribute}")
    return candidate


def discover() -> dict[str, TrackerBackend]:
    from . import __path__ as tracker_paths

    found: dict[str, TrackerBackend] = {}
    prefix = __name__.rsplit(".", 1)[0] + "."
    for info in pkgutil.iter_modules(tracker_paths, prefix):
        name = info.name.rsplit(".", 1)[-1]
        if name.startswith("_") or name in _IGNORED:
            continue
        candidate = _backend(importlib.import_module(info.name))
        if candidate.name in found:
            raise TypeError(f"duplicate tracker backend name {candidate.name!r}")
        found[candidate.name] = candidate
    return found


def export_to(name: str, *, summary: RecordSummary, report_url: str) -> TrackerRef:
    normalized = name.strip().lower()
    backend = discover().get(normalized)
    if backend is None:
        raise _unavailable(
            normalized or name,
            Availability.unavailable(
                "registered tracker backend", "choose a discovered tracker backend"
            ),
        )
    availability = backend.available()
    if not availability.is_available:
        raise _unavailable(backend.name, availability)
    return backend.write(summary, report_url)
