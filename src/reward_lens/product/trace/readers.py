"""The `Reader` protocol and the registry (wave-2 interfaces, section 5).

Discovery is by module name, the way `product.audit.registry.discover()` finds its panels: every
module under `reward_lens.product.trace` is imported in module-name order and its module-level
`READER` is collected. A packet that ships a reader ships a module; it does not call anything here
and it does not edit this file or `__init__.py`. That is the whole registration protocol, and it is
why P-TRACE-2 can land four readers without touching the spine.

A module that will not import is remembered in `import_failures` rather than raised, so a broken
plugin costs the reader it carries and nothing else.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .record import Policy, TraceRecord

__all__ = ["PACKAGE", "Reader", "ReaderNotFound", "discover", "import_failures", "reader_for"]

PACKAGE = "reward_lens.product.trace"

#: Module name -> the exception text, for every module under the package that would not import.
import_failures: dict[str, str] = {}

#: Modules that are the spine itself rather than a reader. They carry no `READER` either, so this
#: is a cost saving and not a rule.
_SPINE = frozenset({"readers", "record", "integrity", "errors", "entries", "follow", "answer"})


@runtime_checkable
class Reader(Protocol):
    """One artifact shape, read into a `TraceRecord`.

    `can_read` is cheap and must not raise: it answers whether this reader recognises the path.
    `read` is the work, and `policy` is declared by the caller rather than guessed, because a
    reading that silently tolerated a gap is not a reading anyone can check.
    """

    name: str

    def can_read(self, path: Any) -> bool:  # pragma: no cover - protocol
        ...

    def read(self, path: Any, *, policy: Policy = "strict") -> TraceRecord:  # pragma: no cover
        ...


class ReaderNotFound(LookupError):
    """No reader in this build recognises that artifact."""


def discover(package: str = PACKAGE) -> tuple[Reader, ...]:
    """Every module-level `READER` under `package`, in module-name order. Never raises."""
    import_failures.clear()
    try:
        root: Any = importlib.import_module(package)
    except Exception as failure:  # pragma: no cover - the package ships with the wheel
        import_failures[package] = f"{type(failure).__name__}: {failure}"
        return ()
    found: list[Reader] = []
    for info in sorted(pkgutil.iter_modules(getattr(root, "__path__", [])), key=lambda i: i.name):
        if info.name.startswith("_") or info.name in _SPINE:
            continue
        name = f"{package}.{info.name}"
        try:
            module = importlib.import_module(name)
        except Exception as failure:
            import_failures[name] = f"{type(failure).__name__}: {failure}"
            continue
        reader = getattr(module, "READER", None)
        if reader is not None and isinstance(reader, Reader):
            found.append(reader)
    return tuple(found)


def reader_for(path: Any, *, name: str | None = None) -> Reader:
    """The first reader that recognises `path`, or the one asked for by name."""
    readers = discover()
    if name is not None:
        for reader in readers:
            if reader.name == name:
                return reader
        raise ReaderNotFound(f"no reader named {name!r}; this build holds " + _names(readers))
    candidate = Path(path) if not isinstance(path, Path) else path
    for reader in readers:
        try:
            if reader.can_read(candidate):
                return reader
        except Exception as failure:  # noqa: BLE001 - a reader that cannot answer has no opinion
            import_failures[reader.name] = f"{type(failure).__name__}: {failure}"
    raise ReaderNotFound(
        f"no reader in this build recognises {candidate}; this build holds " + _names(readers)
    )


def _names(readers: tuple[Reader, ...]) -> str:
    return ", ".join(reader.name for reader in readers) or "none"
