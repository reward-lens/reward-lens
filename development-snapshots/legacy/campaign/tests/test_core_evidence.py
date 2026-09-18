"""M0 acceptance tests for `reward_lens.core`: Evidence, the store DAG, and the gates.

These lock the milestone-M0 acceptance criteria as committed, green tests (section 4.4): an
Evidence round-trips through the store with a resolvable parent DAG; the gate logic downgrades an
uncalibrated and unregistered Evidence to EXPLORATORY and climbs the ladder as the gate inputs
are supplied; and the pure layer imports without torch. The ess-on-a-clone-view criterion lives
in ``test_stats_ess.py``; this file covers the core half.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from reward_lens.core import (
    CalibrationRef,
    EvidenceStore,
    GaugeStatus,
    ModelFP,
    Provenance,
    ProvenanceError,
    SubjectRef,
    TrustLevel,
    Uncertainty,
    make_evidence,
)


def _subject() -> SubjectRef:
    return SubjectRef(signals=(ModelFP("mfp:demo"),), dataset="ds:demo", readout="reward")


def test_pure_layer_imports_without_torch():
    # The gates and evidence engine must be usable without torch (section 4.1): everything imports
    # them, and they carry no model dependency. This must be checked in a fresh interpreter, not by
    # inspecting sys.modules in-process: pytest's collection phase imports every test module up
    # front, and the runtime/signals/geometry test modules import torch, so by the time this test
    # runs torch is already present in this process regardless of what the core import pulls.
    code = "import reward_lens.core, reward_lens.stats, sys; assert 'torch' not in sys.modules"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"core/stats import pulled torch:\n{proc.stderr}"


def test_gate_ladder_is_computed_not_set():
    subj = _subject()
    cal = CalibrationRef(scorecard_entry="ev:score", organism_family="planted")

    exploratory = make_evidence(observable="X", observable_version="1", subject=subj, value=1.0)
    calibrated = make_evidence(
        observable="X", observable_version="1", subject=subj, value=1.0, calibration=cal
    )
    registered = make_evidence(
        observable="X", observable_version="1", subject=subj, value=1.0, registered=True
    )
    adjudicated = make_evidence(
        observable="X",
        observable_version="1",
        subject=subj,
        value=1.0,
        registered=True,
        calibration=cal,
        adjudicated=True,
    )

    assert exploratory.trust is TrustLevel.EXPLORATORY
    assert calibrated.trust is TrustLevel.CALIBRATED
    assert registered.trust is TrustLevel.REGISTERED
    assert adjudicated.trust is TrustLevel.ADJUDICATED
    # Adjudicated requires the full set: the flag alone does not grant it.
    flag_only = make_evidence(
        observable="X", observable_version="1", subject=subj, value=1.0, adjudicated=True
    )
    assert flag_only.trust is TrustLevel.EXPLORATORY


def test_uncalibrated_registered_still_carries_no_calibration():
    # A REGISTERED number without a scorecard must still visibly carry calibration=None so a
    # card renders it as unvalidated; the two axes stay independent (section 1.3).
    ev = make_evidence(
        observable="Chi", observable_version="1", subject=_subject(), value=0.4, registered=True
    )
    assert ev.trust is TrustLevel.REGISTERED
    assert ev.calibration is None
    assert not ev.is_calibrated


def test_evidence_roundtrips_with_array_payload(tmp_path):
    subj = _subject()
    payload = {"depths": np.arange(200, dtype=np.float32), "peak": 0.73, "tag": "lens"}
    ev = make_evidence(
        observable="LensCrystallization",
        observable_version="1.0",
        subject=subj,
        value=payload,
        uncertainty=Uncertainty(ci_low=0.6, ci_high=0.8, ci_level=0.95, n=30, n_effective=6.0),
        gauge=GaugeStatus.INVARIANT,
    )
    store = EvidenceStore(tmp_path)
    store.append(ev)

    reloaded = EvidenceStore(tmp_path)  # fresh index from disk
    got = reloaded.get(ev.id)
    assert np.allclose(got.value["depths"], np.arange(200))
    assert got.value["peak"] == 0.73
    assert got.uncertainty.n_effective == 6.0
    assert got.gauge is GaugeStatus.INVARIANT


def test_store_is_a_dag_and_rejects_orphans(tmp_path):
    subj = _subject()
    store = EvidenceStore(tmp_path)
    leaf = make_evidence(observable="A", observable_version="1", subject=subj, value=1.0)
    store.append(leaf)
    derived = make_evidence(
        observable="B",
        observable_version="1",
        subject=subj,
        value=2.0,
        provenance=Provenance(parents=(leaf.id,)),
    )
    store.append(derived)
    assert store.parents(derived)[0].id == leaf.id
    assert store.ancestors(derived)[0].id == leaf.id

    orphan = make_evidence(
        observable="C",
        observable_version="1",
        subject=subj,
        value=3.0,
        provenance=Provenance(parents=("ev:doesnotexist",)),
    )
    with pytest.raises(ProvenanceError):
        store.append(orphan)


def test_content_ids_dedupe(tmp_path):
    # Identical measurements from identical inputs share an id (the store is a deduplicating DAG).
    subj = _subject()
    a = make_evidence(observable="A", observable_version="1", subject=subj, value=1.0)
    b = make_evidence(observable="A", observable_version="1", subject=subj, value=1.0)
    assert a.id == b.id
    store = EvidenceStore(tmp_path)
    store.append(a)
    store.append(b)  # idempotent
    assert len(store) == 1


def test_find_filters_and_latest(tmp_path):
    subj = _subject()
    store = EvidenceStore(tmp_path)
    store.append(make_evidence(observable="A", observable_version="1", subject=subj, value=1.0))
    store.append(make_evidence(observable="B", observable_version="1", subject=subj, value=2.0))
    assert len(store.find(observable="A")) == 1
    assert len(store.find(signal="mfp:demo")) == 2
    assert store.find(observable="A")[0].value == 1.0


def _tear_last_line(store: EvidenceStore) -> bytes:
    """Simulate a writer killed mid-append: keep only half of the file's final line."""
    raw = store.jsonl.read_bytes()
    body = raw.rstrip(b"\n")
    cut = body.rfind(b"\n") + 1  # start of the final line (0 when it is the only line)
    torn = body[cut : cut + (len(body) - cut) // 2]
    store.jsonl.write_bytes(raw[:cut] + torn)
    return torn


def test_torn_final_line_is_quarantined_and_append_resumes(tmp_path):
    # A process killed mid-append leaves a truncated last line. Loading the store must not
    # raise on it: the tail is preserved in evidence.jsonl.partial, the file is cut back to
    # the last complete envelope, and the interrupted append can simply be replayed.
    subj = _subject()
    store = EvidenceStore(tmp_path)
    kept = make_evidence(observable="A", observable_version="1", subject=subj, value=1.0)
    lost = make_evidence(observable="B", observable_version="1", subject=subj, value=2.0)
    store.append(kept)
    store.append(lost)
    torn = _tear_last_line(store)

    with pytest.warns(RuntimeWarning, match="torn final line"):
        reloaded = EvidenceStore(tmp_path)
    assert kept.id in reloaded
    assert lost.id not in reloaded
    assert (tmp_path / "evidence.jsonl.partial").read_bytes() == torn + b"\n"
    assert reloaded.jsonl.read_bytes().endswith(b"\n")

    # The re-run records the lost measurement again and a fresh load sees both, cleanly.
    reloaded.append(lost)
    assert {ev.id for ev in EvidenceStore(tmp_path)} == {kept.id, lost.id}


def test_torn_only_line_leaves_an_empty_but_loadable_store(tmp_path):
    # The torn tail can be the store's only line; quarantining it must yield an empty store.
    subj = _subject()
    store = EvidenceStore(tmp_path)
    ev = make_evidence(observable="A", observable_version="1", subject=subj, value=1.0)
    store.append(ev)
    _tear_last_line(store)

    with pytest.warns(RuntimeWarning, match="torn final line"):
        reloaded = EvidenceStore(tmp_path)
    assert len(reloaded) == 0
    reloaded.append(ev)
    assert ev.id in EvidenceStore(tmp_path)


def test_malformed_line_that_is_not_the_tail_still_raises(tmp_path):
    # Only the final line can be the residue of a torn append. Damage anywhere else is real
    # corruption and must stay loud rather than be silently quarantined.
    import json

    subj = _subject()
    store = EvidenceStore(tmp_path)
    store.append(make_evidence(observable="A", observable_version="1", subject=subj, value=1.0))
    store.append(make_evidence(observable="B", observable_version="1", subject=subj, value=2.0))
    lines = store.jsonl.read_bytes().splitlines(keepends=True)
    lines[0] = lines[0][: len(lines[0]) // 2].rstrip(b"\n") + b"\n"
    store.jsonl.write_bytes(b"".join(lines))

    with pytest.raises(json.JSONDecodeError):
        EvidenceStore(tmp_path)
    assert not (tmp_path / "evidence.jsonl.partial").exists()
