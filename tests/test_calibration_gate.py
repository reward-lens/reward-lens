"""The calibration gate goes live end to end: a scorecard makes a measurement CALIBRATED (gate 1).

This is the integration seam between the organism foundry and the measurement runner. Before an
instrument earns a scorecard entry, its output is EXPLORATORY no matter how it is computed; once a
scorecard graded against planted ground truth is registered, the same observable on a production
signal is CALIBRATED and its Evidence cites the scorecard. This test proves the wiring is real, not
aspirational, using the actual scorecard evaluate path (a synthetic dose detector graded against a
planted spurious-correlation organism).
"""

from __future__ import annotations

import pytest

from reward_lens.core.types import Capability, GaugeStatus, TrustLevel
from reward_lens.measure import base as mb
from reward_lens.organisms import gate
from reward_lens.organisms.foundry import spurious_correlation_organism
from reward_lens.organisms.scorecard import MethodScorecard, synthetic_dose_detector

_DOSES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class _Meta:
    fingerprint = "mfp:prod-signal"


class _FakeSignal:
    caps = Capability.SCORES | Capability.ACTIVATIONS
    meta = _Meta()


class _BiasBattery(mb.BaseObservable):
    name = "BiasBattery"
    gauge_status = GaugeStatus.INVARIANT

    def measure(self, ctx):
        return ctx.emit(-0.05)


@pytest.fixture(autouse=True)
def _isolate_gate():
    gate.clear()
    mb.set_calibration_provider(lambda name, subj, regime: None)
    yield
    gate.clear()
    mb.set_calibration_provider(lambda name, subj, regime: None)


def _scorecard_entry(observable: str):
    _, key = spurious_correlation_organism(rho=0.8, n=10, seed=0)
    readouts = {rho: synthetic_dose_detector(rho, n=800, seed=1, slope=6.0) for rho in _DOSES}
    return MethodScorecard(observable).evaluate(readouts, key)


def test_gate_inert_until_scorecard_registered():
    # Wiring the provider with an empty registry never upgrades anything: still EXPLORATORY.
    gate.install()
    ev = mb.run(_BiasBattery(), mb.Context(signal=_FakeSignal()))
    assert ev.trust is TrustLevel.EXPLORATORY
    assert ev.calibration is None


def test_registering_a_scorecard_makes_the_measurement_calibrated():
    entry = _scorecard_entry("BiasBattery")
    gate.register_scorecard("BiasBattery", entry)
    gate.install()

    ev = mb.run(_BiasBattery(), mb.Context(signal=_FakeSignal()))
    assert ev.trust is TrustLevel.CALIBRATED
    assert ev.is_calibrated
    # The Evidence cites the scorecard that certifies it (gate 1 traceability).
    assert ev.calibration.scorecard_entry == entry.evidence.id
    assert ev.calibration.organism_family == entry.calibration_ref.organism_family


def test_only_the_scored_observable_is_calibrated():
    entry = _scorecard_entry("BiasBattery")
    gate.register_scorecard("BiasBattery", entry)
    gate.install()

    class _Other(mb.BaseObservable):
        name = "DistortionV2"

        def measure(self, ctx):
            return ctx.emit(0.4)

    ev = mb.run(_Other(), mb.Context(signal=_FakeSignal()))
    assert ev.trust is TrustLevel.EXPLORATORY  # no scorecard for DistortionV2
