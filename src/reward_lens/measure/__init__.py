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
