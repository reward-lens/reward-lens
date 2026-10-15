"""Candidate-bound calibration evidence is calculated from redacted pair bytes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest


def _panel():
    from reward_lens.reference import load
    from reward_lens.reference.calibration import (
        CalibrationConfig,
        CalibrationLevel,
        TransferStatement,
        calibrate,
    )
    from reward_lens.reference.load import load_diagnostic_v3_material

    material = load("diagnostic-v3")
    return calibrate(
        material=material,
        reference_view=load_diagnostic_v3_material(material).view,
        unit="score-delta",
        config=CalibrationConfig(
            alpha=Decimal("0.05"),
            beta=Decimal("0.05"),
            minimum_blank_replicates=4,
            minimum_level_replicates=3,
            maximum_relative_standard_uncertainty=Decimal("0.10"),
        ),
        blank_measurements=(Decimal("0.00"), Decimal("0.01"), Decimal("-0.01"), Decimal("0.00")),
        levels=(
            CalibrationLevel(Decimal("0.05"), (Decimal("0.04"), Decimal("0.05"), Decimal("0.06"))),
            CalibrationLevel(Decimal("0.20"), (Decimal("0.19"), Decimal("0.20"), Decimal("0.21"))),
            CalibrationLevel(Decimal("0.40"), (Decimal("0.39"), Decimal("0.40"), Decimal("0.41"))),
        ),
        synthetic=True,
        transfer=TransferStatement(synthetic=True),
    )


def _candidate(
    root: Path, *, right: tuple[str, ...] = ("0.10", "0.80"), include_observation: bool = True
):
    from reward_lens.commons.flywheel import Consent, PrivatePairRequest, Redaction, promote, retain
    from reward_lens.product.improve.session import ImproveSession
    from reward_lens.reference.calibration.candidate import CandidateCalibrationInput

    root.mkdir(parents=True, exist_ok=True)
    left_payload: dict[str, object] = {"owner": "alice@example.test"}
    right_payload: dict[str, object] = {"owner": "alice@example.test"}
    if include_observation:
        left_payload["calibration_observation"] = {
            "procedure": "paired-score-delta/v1",
            "unit": "score-delta",
            "measurements": ["0.00", "0.00"],
        }
        right_payload["calibration_observation"] = {
            "procedure": "paired-score-delta/v1",
            "unit": "score-delta",
            "measurements": list(right),
        }
    left_path, right_path = root / "left.json", root / "right.json"
    left_path.write_text(json.dumps(left_payload, sort_keys=True) + "\n", encoding="utf-8")
    right_path.write_text(json.dumps(right_payload, sort_keys=True) + "\n", encoding="utf-8")
    session = ImproveSession.describe(root, "candidate calibration provenance")
    pair = retain(
        PrivatePairRequest(
            improve_session=session.mark_pair(
                left_path, right_path, "paired measurement delta"
            ).state_path,
            pair_index=0,
            owner_id="alice",
            affected_panel="candidate-calibration",
            private_store=root / "private",
        )
    ).pair
    candidate = promote(
        pair,
        consent=Consent(pair_id=pair.id, owner_id="alice", explicit=True),
        redaction=Redaction(
            {"alice@example.test": "[OWNER]"},
            sensitivity_inventory=("alice@example.test",),
            declared_complete=True,
        ),
    )
    return CandidateCalibrationInput.from_candidate_reference(candidate)


def test_candidate_bytes_drive_panel_bound_calculation_and_reopen(tmp_path) -> None:
    from reward_lens.reference.calibration.candidate import (
        calibrate_candidate,
        open_candidate_calibration,
        persist_candidate_calibration,
    )

    panel, candidate = _panel(), _candidate(tmp_path)
    evidence = calibrate_candidate(panel, candidate)
    path = persist_candidate_calibration(evidence, tmp_path, panel=panel, candidate=candidate)
    reopened = open_candidate_calibration(path, panel=panel, candidate=candidate)

    assert evidence.measured_deltas == (Decimal("0.10"), Decimal("0.80"))
    assert evidence.transfer_status == "TRANSFER_FAILURE"
    assert evidence.later_interval.lower_bound == Decimal("0.10")
    assert evidence.later_interval.upper_bound == Decimal("0.80")
    assert reopened == evidence


def test_direct_forged_candidate_input_without_an_artifact_is_refused() -> None:
    from reward_lens.reference.calibration.candidate import (
        CANDIDATE_CALIBRATION_UNSUPPORTED,
        CandidateCalibrationInput,
        CandidateCalibrationRefusal,
        calibrate_candidate,
    )

    left = (
        b'{"calibration_observation":{"measurements":["0.00","0.00"],'
        b'"procedure":"paired-score-delta/v1","unit":"score-delta"},"owner":"[OWNER]"}\n'
    )
    right = (
        b'{"calibration_observation":{"measurements":["0.10","0.80"],'
        b'"procedure":"paired-score-delta/v1","unit":"score-delta"},"owner":"[OWNER]"}\n'
    )
    candidate = CandidateCalibrationInput(
        candidate_id="CRM-forged",
        candidate_content_digest="sha256:" + hashlib.sha256(left + b"\n" + right).hexdigest(),
        redaction_digest="sha256:" + "1" * 64,
        source_private_pair_id="PP-forged",
        source_identity="sha256:" + "2" * 64,
        source_left_digest="sha256:" + "3" * 64,
        source_right_digest="sha256:" + "4" * 64,
        consent_digest="sha256:" + "5" * 64,
        left_bytes=left,
        right_bytes=right,
    )
    with pytest.raises(CandidateCalibrationRefusal, match=CANDIDATE_CALIBRATION_UNSUPPORTED):
        calibrate_candidate(_panel(), candidate)


def test_reopen_refuses_evidence_when_candidate_measurements_are_substituted(tmp_path) -> None:
    from reward_lens.reference.calibration.candidate import (
        CANDIDATE_CALIBRATION_UNSUPPORTED,
        CandidateCalibrationRefusal,
        calibrate_candidate,
        open_candidate_calibration,
        persist_candidate_calibration,
    )

    panel = _panel()
    original = _candidate(tmp_path / "original", right=("0.10", "0.80"))
    substituted = _candidate(tmp_path / "substituted", right=("0.30", "0.35"))
    evidence_path = persist_candidate_calibration(
        calibrate_candidate(panel, original),
        tmp_path,
        panel=panel,
        candidate=original,
    )

    with pytest.raises(CandidateCalibrationRefusal, match=CANDIDATE_CALIBRATION_UNSUPPORTED):
        open_candidate_calibration(evidence_path, panel=panel, candidate=substituted)


def test_reopen_refuses_a_panel_with_only_its_method_changed(tmp_path) -> None:
    from reward_lens.reference.calibration.candidate import (
        CANDIDATE_CALIBRATION_UNSUPPORTED,
        CandidateCalibrationRefusal,
        calibrate_candidate,
        open_candidate_calibration,
        persist_candidate_calibration,
    )

    panel, candidate = _panel(), _candidate(tmp_path / "candidate")
    evidence_path = persist_candidate_calibration(
        calibrate_candidate(panel, candidate),
        tmp_path,
        panel=panel,
        candidate=candidate,
    )
    changed_method = replace(panel, method="paired-score-delta/v2")

    with pytest.raises(CandidateCalibrationRefusal, match=CANDIDATE_CALIBRATION_UNSUPPORTED):
        open_candidate_calibration(evidence_path, panel=changed_method, candidate=candidate)
