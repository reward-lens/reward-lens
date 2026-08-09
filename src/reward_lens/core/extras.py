"""Optional dependency groups, and a guard that names one that actually exists.

The core installs with no compiled dependency, so a subsystem that needs torch has to say so at
the boundary rather than dying inside itself with ``ModuleNotFoundError: No module named 'torch'``.
That error is technically accurate and practically useless: it does not say which extra installs
torch, and it appears identically whether the user is missing an extra or has a broken
environment.

The failure this replaces was worse than an unhelpful message. ``loops/integrations/base.py``
raised ``Install reward-lens[trl]`` for three frameworks, and none of ``trl``, ``verl`` or
``openrlhf`` was declared in ``pyproject.toml``, so the instruction the error gave could not be
followed. An error naming an extra that does not exist is a dead end with a helpful tone.

So the mapping below is the single source of truth for what an extra is called, and
``tests/acceptance/test_w0_3_dependencies.py`` asserts it matches ``pyproject.toml`` exactly. A
typo in an extra name is then a failing test rather than a user stuck in a loop.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from reward_lens.core.errors import RewardLensError

#: Extra name -> an importable module that is present if and only if the extra is installed.
#: The probe is one module rather than the whole group because the group is only ever needed as a
#: whole: an environment with torch but not transformers is broken in a way this cannot diagnose
#: and should not try to.
EXTRA_PROBE: dict[str, str] = {
    "trace": "pyarrow",
    # No runtime dependency yet: no module under src/ imports an HTTP client, and declaring one on
    # the strength of a plan is what this file's own history warns against. The name is declared in
    # pyproject.toml so the instruction the error gives can be followed.
    "judge": "",
    "train": "torch",
    "sigstore": "sigstore",
    # SALib, not coverage: A-002 made coverage a base dependency, so the old probe was
    # satisfied on every base install; no other extra ships SALib and it depends on numpy.
    "verifier": "SALib",
    "white-box": "torch",
    "organisms": "peft",
    "viz": "matplotlib",
    "record": "pyarrow",
}

#: What each extra is for, in the words a user needs to decide whether they want it.
EXTRA_PURPOSE: dict[str, str] = {
    "trace": "reading a run's rollouts out of a Parquet trace",
    "judge": "calling a model judge through a provider",
    "train": "reward models, and attaching the tap to a live TRL training run",
    "sigstore": "signing a record with Sigstore rather than the built-in Ed25519",
    "verifier": "the verifier series: coverage, mutation, metamorphic relations, sensitivity",
    "white-box": "reading a model's activations and gradients",
    "organisms": "planting and training model organisms",
    "viz": "rendering figures",
    "record": "safetensors tensor shards and the numeric side of the record store",
}


class ExtraRequiredError(RewardLensError, ImportError):
    """A subsystem needs an optional dependency group that is not installed.

    Subclasses ``ImportError`` as well as ``RewardLensError`` so that ``except ImportError``
    around an optional import keeps working, which is the shape most callers already have.
    """


def require_extra(extra: str, *, subsystem: str) -> None:
    """Raise `ExtraRequiredError` if ``extra`` is not installed, naming what to install.

    Called at the top of a subsystem's ``__init__``, so importing any module beneath it fails at
    the boundary with an actionable message instead of somewhere in the middle with a bare
    ``ModuleNotFoundError``.

    ``extra`` must be a key of `EXTRA_PROBE`; an unknown one is a `KeyError` at import time rather
    than a message pointing at an extra nobody can install.
    """
    probe = EXTRA_PROBE[extra]
    if not probe:
        return
    try:
        found = importlib.util.find_spec(probe) is not None
    except Exception:
        # find_spec runs the finders, and a finder can raise: a half-removed distribution, a
        # broken namespace package, or a meta_path hook standing in for an absent dependency.
        # A probe that cannot answer is not a probe that says yes.
        found = False
    if found:
        return
    raise ExtraRequiredError(
        f"{subsystem} needs the optional {extra!r} extra, which is not installed. "
        f"Install it with:  pip install 'reward-lens[{extra}]'  "
        f"({EXTRA_PURPOSE[extra]}). "
        f"The core install is deliberately free of compiled dependencies, so most of the "
        f"instrument catalogue, including the whole grader card, runs without this."
    )


# ---------------------------------------------------------------------------
# The lazy-module seam
# ---------------------------------------------------------------------------


class _LazyModule:
    """A stand-in for a module, imported on first attribute access.

    This exists for one measured reason. `uvx` downloads the base closure on every cold
    invocation, so D-58 keeps numpy out of it; but six modules in the inherited tree carry
    ``import numpy as np`` at module scope and use ``np.`` in a dozen places each, and the three
    wave-1 instruments import two of those modules directly. Rewriting eighty-eight call sites to
    import inside their functions would be eighty-eight chances to move a line that had a reason to
    be where it was. Swapping one import line for ``np = lazy_module("numpy")`` moves none of them:
    every ``np.asarray`` still reads the same, and the import happens the first time one runs.

    The proxy resolves to the real module and caches it, so the cost is one dictionary lookup per
    attribute after the first. What it does not support is subclassing a name off it at module
    scope (``class X(np.ndarray)``), which would need the module at definition time; nothing in
    this tree does that, and it would be an honest failure rather than a silent one.
    """

    def __init__(self, name: str, *, extra: str | None = None, subsystem: str | None = None) -> None:
        self.__dict__["_name"] = name
        self.__dict__["_extra"] = extra
        self.__dict__["_subsystem"] = subsystem or name
        self.__dict__["_module"] = None

    def _load(self) -> Any:
        module = self.__dict__["_module"]
        if module is None:
            extra = self.__dict__["_extra"]
            if extra:
                require_extra(extra, subsystem=self.__dict__["_subsystem"])
            module = importlib.import_module(self.__dict__["_name"])
            self.__dict__["_module"] = module
        return module

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._load(), attribute)

    def __dir__(self) -> list[str]:
        return dir(self._load())

    def __repr__(self) -> str:
        state = "imported" if self.__dict__["_module"] is not None else "not yet imported"
        return f"<lazy module {self.__dict__['_name']!r}, {state}>"


def lazy_module(name: str, *, extra: str | None = None, subsystem: str | None = None) -> Any:
    """Return a proxy for ``name`` that imports it on first use.

    ``extra``, when given, is checked through `require_extra` at that first use, so a module that
    needs an optional numeric stack raises the message naming the extra rather than a bare
    ``ModuleNotFoundError`` from somewhere in the middle of a calculation.
    """
    return _LazyModule(name, extra=extra, subsystem=subsystem)


__all__ = [
    "EXTRA_PROBE",
    "EXTRA_PURPOSE",
    "ExtraRequiredError",
    "lazy_module",
    "require_extra",
]

