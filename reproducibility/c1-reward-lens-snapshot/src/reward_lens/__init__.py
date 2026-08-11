"""reward-lens: the reference instrument for the science of reward misspecification.

reward-lens 2.0 is one kernel, sixteen sciences, and three gates. The kernel is a set of subsystems
(``core``, ``stats``, ``runtime``, ``signals``, ``data``, ``concepts``, ``interventions``,
``geometry``, ``measure``, ``attribution``, ``organisms``, ``dynamics``, ``loops``, ``studies``,
``artifacts``); the sciences are studies over it; the gates (calibration, gauge, registration)
are runtime policy in the stats and evidence layer.

Import discipline: this top-level module imports nothing. ``import reward_lens`` and
``import reward_lens.core`` and ``import reward_lens.stats`` pull nothing heavier than numpy, so
the pure epistemics layer is usable without torch, and a test asserts that in a fresh subprocess
rather than trusting it. Anything that touches models is imported directly from its subsystem
(``from reward_lens.signals import load_signal``), which is where the white-box extra is required.

This used to be a lazy accessor holding twenty-one v1 names, so that ``reward_lens.RewardModel``
resolved without ``import reward_lens`` pulling torch. The v1 corpus and its flat public API
retired together once the E-parity suite had met its two-release deprecation condition, and the
accessor went with them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "3.0.0"


#: The subpackages that are public surface. Naming a subpackage here does not import it: the
#: entries below are resolved by the import system on ``from reward_lens import <name>``, and the
#: ``TYPE_CHECKING`` block exists only so that static analysis can see them. ``import reward_lens``
#: still pulls nothing heavier than the standard library, and
#: ``test_import_reward_lens_does_not_import_torch`` still holds.
#:
#: One consequence worth stating rather than discovering. ``from reward_lens import *`` does import
#: every subpackage named here, and nine of them call ``require_extra("white-box")`` at module
#: scope while ``verifier`` calls ``require_extra("verifier")``. On a base install that star import
#: raises ``ExtraRequiredError`` naming the extra, where before it bound three names and returned.
#: The guarantee that matters is untouched: ``import reward_lens``, ``import reward_lens.core`` and
#: ``import reward_lens.stats`` still work on a base install and still pull no torch.
#:
#: Three subpackages are on disk and deliberately absent from this list.
#:
#: ``model_adapters`` imports torch at module scope. Every other subpackage here is torch-clean at
#: import, so ``from reward_lens import *`` stays free of torch; naming ``model_adapters`` would end
#: that. Reach it by its full path, ``from reward_lens.model_adapters import get_adapter``.
#:
#: ``experiments`` says in its own docstring that nothing in it is imported by the library's runtime
#: paths and that each module is run as ``python -m reward_lens.experiments.<name>``. Its ``__all__``
#: is empty on purpose. It is an entry-point directory rather than an API.
#:
#: ``spec`` has no ``__init__.py``. It is two packaged JSON catalogues that ``core.quantity`` reads,
#: not an importable package.
_SUBPACKAGES = [
    "access",
    "artifacts",
    "attribution",
    "concepts",
    "core",
    "data",
    "dynamics",
    "forecast",
    "geometry",
    "interventions",
    "loops",
    "measure",
    "monitor",
    "operate",
    "oracles",
    "organisms",
    "policy",
    "record",
    "runtime",
    "signals",
    "stats",
    "studies",
    "tap",
    "verifier",
]

if TYPE_CHECKING:  # help static analysis without importing torch at runtime
    from reward_lens import access as access
    from reward_lens import artifacts as artifacts
    from reward_lens import attribution as attribution
    from reward_lens import concepts as concepts
    from reward_lens import core as core
    from reward_lens import data as data
    from reward_lens import dynamics as dynamics
    from reward_lens import forecast as forecast
    from reward_lens import geometry as geometry
    from reward_lens import interventions as interventions
    from reward_lens import loops as loops
    from reward_lens import measure as measure
    from reward_lens import monitor as monitor
    from reward_lens import operate as operate
    from reward_lens import oracles as oracles
    from reward_lens import organisms as organisms
    from reward_lens import policy as policy
    from reward_lens import record as record
    from reward_lens import runtime as runtime
    from reward_lens import signals as signals
    from reward_lens import stats as stats
    from reward_lens import studies as studies
    from reward_lens import tap as tap
    from reward_lens import verifier as verifier

__all__ = ["__version__", *_SUBPACKAGES]
