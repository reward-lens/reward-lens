"""The measurement layer: observables that turn a signal into Evidence.

Everything here runs through one runner. An :class:`~reward_lens.measure.base.Observable` declares the
capability it needs and the gauge status of what it returns; :func:`~reward_lens.measure.base.run`
enforces the capability and the frame requirement before it lets the observable touch a signal, and
the resulting :class:`~reward_lens.core.Evidence` carries its own trust level out. The battery
(``reward_lens.measure.battery``) holds the eleven white-box instruments; the index library
(``reward_lens.measure.indices``) holds the eighteen scalar diagnostics. Import from the subsystem you
need: ``from reward_lens.measure import base`` for the runner, ``from reward_lens.measure.battery
import DirectLinearAttribution`` for an instrument.
"""

#: The runner and the seventeen subsystems, so that the paths this package's docstring tells you to
#: import from are public paths rather than reaches into private ones. Naming them here does not
#: import them: ``from reward_lens.measure import battery`` resolves through the import system, and
#: this ``__init__`` still imports nothing, which is what keeps ``import reward_lens.measure`` cheap.
__all__ = [
    "base",
    "battery",
    "card",
    "composition",
    "controls",
    "decision",
    "efficiency",
    "estimator",
    "frontier",
    "indices",
    "labels",
    "ledger",
    "meta",
    "metrology",
    "rate",
    "reconcile",
    "selection",
    "threshold",
]
