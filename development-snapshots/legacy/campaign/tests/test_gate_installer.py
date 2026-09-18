"""One-call wiring of stored scorecards into the calibration gate (`install_default_scorecards`).

The scorecard registry is per-process, so calibration earned in an earlier run only reaches the
measurement runner if something re-registers it from the store. These tests prove the installer
does exactly that: a stored scorecard makes `lookup_calibration` return a `CalibrationRef` citing
that evidence for the graded observable, an ungraded observable stays uncalibrated, and an
explicit entries mapping pins which stored evidence an observable cites.
"""

from __future__ import annotations

import pytest

from reward_lens.core import CalibrationRef, EvidenceStore, SubjectRef
from reward_lens.measure import base as mb
from reward_lens.organisms import gate
from reward_lens.organisms.foundry import spurious_correlation_organism
from reward_lens.organisms.scorecard import MethodScorecard, synthetic_dose_detector

_DOSES = [0.5, 0.75, 1.0]


@pytest.fixture(autouse=True)
def _isolate_gate():
    gate.clear()
    mb.set_calibration_provider(lambda name, subj, regime: None)
    yield
    gate.clear()
    mb.set_calibration_provider(lambda name, subj, regime: None)


def _scorecard_entry(observable: str, *, detector_seed: int = 1):
    _, key = spurious_correlation_organism(rho=0.8, n=10, seed=0)
    readouts = {
        rho: synthetic_dose_detector(rho, n=200, seed=detector_seed, slope=6.0) for rho in _DOSES
    }
    return MethodScorecard(observable).evaluate(readouts, key)


def test_installer_makes_the_stored_scorecard_visible_to_the_runner(tmp_path):
    entry = _scorecard_entry("BiasBattery")
    store = EvidenceStore(tmp_path)
    store.append(entry.evidence)

    registered = gate.install_default_scorecards(store)
    assert registered == 1

    ref = mb.lookup_calibration("BiasBattery", SubjectRef(), {})
    assert isinstance(ref, CalibrationRef)
    assert ref.scorecard_entry == entry.evidence.id
    assert ref.organism_family == entry.calibration_ref.organism_family
    assert ref.operating_point == entry.calibration_ref.operating_point
    # An instrument that never earned a scorecard stays uncalibrated.
    assert mb.lookup_calibration("DistortionV2", SubjectRef(), {}) is None


def test_regime_aware_lookup_survives_the_store_round_trip(tmp_path):
    entry = _scorecard_entry("BiasBattery")
    store = EvidenceStore(tmp_path)
    store.append(entry.evidence)
    gate.install_default_scorecards(store)

    family = entry.calibration_ref.organism_family
    ref = mb.lookup_calibration("BiasBattery", SubjectRef(), {"organism_family": family})
    assert ref is not None
    assert ref.organism_family == family


def test_explicit_entries_pin_which_evidence_is_cited(tmp_path):
    # Two gradings of the same instrument on the same family; the mapping pins one of them
    # regardless of which the scan would have preferred.
    first = _scorecard_entry("BiasBattery", detector_seed=1)
    second = _scorecard_entry("BiasBattery", detector_seed=2)
    assert first.evidence.id != second.evidence.id
    store = EvidenceStore(tmp_path)
    store.append(first.evidence)
    store.append(second.evidence)

    registered = gate.install_default_scorecards(
        store, entries={"BiasBattery": first.evidence.id}
    )
    assert registered == 1
    ref = mb.lookup_calibration("BiasBattery", SubjectRef(), {})
    assert ref.scorecard_entry == first.evidence.id


def test_entries_refuse_a_scorecard_for_a_different_instrument(tmp_path):
    # A scorecard only calibrates the instrument it graded; pinning it under another name
    # would let a measurement cite calibration it never earned.
    entry = _scorecard_entry("BiasBattery")
    store = EvidenceStore(tmp_path)
    store.append(entry.evidence)

    with pytest.raises(ValueError, match="BiasBattery"):
        gate.install_default_scorecards(store, entries={"DistortionV2": entry.evidence.id})
