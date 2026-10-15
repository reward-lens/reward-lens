"""Acceptance coverage for the consent-gated private-pair flywheel."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest


def _qualified_panel():
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


def _marked_improve_pair(
    root: Path,
    *,
    left_measurements: tuple[str, ...] = ("0.00", "0.00"),
    right_measurements: tuple[str, ...] = ("0.10", "0.80"),
    include_observation: bool = True,
) -> Path:
    """Produce the accepted public pair record that the flywheel consumes."""
    import json

    from reward_lens.product.improve.session import ImproveSession

    root.mkdir(parents=True, exist_ok=True)
    left, right = root / "before.json", root / "after.json"
    left_payload: dict[str, object] = {
        "owner": "alice@example.test",
        "reward": "same-score shortcut",
    }
    right_payload: dict[str, object] = {"owner": "alice@example.test", "reward": "verified result"}
    if include_observation:
        left_payload["calibration_observation"] = {
            "procedure": "paired-score-delta/v1",
            "unit": "score-delta",
            "measurements": list(left_measurements),
        }
        right_payload["calibration_observation"] = {
            "procedure": "paired-score-delta/v1",
            "unit": "score-delta",
            "measurements": list(right_measurements),
        }
    left.write_text(json.dumps(left_payload, sort_keys=True) + "\n", encoding="utf-8")
    right.write_text(json.dumps(right_payload, sort_keys=True) + "\n", encoding="utf-8")
    session = ImproveSession.describe(root, "correctly identify verified results")
    return session.mark_pair(
        left, right, "the same score hides a different verified result"
    ).state_path


def _complete_redaction():
    from reward_lens.commons.flywheel import Redaction

    replacements = {
        "alice@example.test": "[OWNER]",
        "same-score shortcut": "[REDACTED-REWARD]",
    }
    try:
        return Redaction(
            replacements=replacements,
            sensitivity_inventory=("alice@example.test", "same-score shortcut"),
            declared_complete=True,
        )
    except TypeError:
        return Redaction(replacements)


def test_public_request_retains_private_regression_and_reopens_exact_bytes_by_default(
    tmp_path: Path,
) -> None:
    from reward_lens.commons.flywheel import PrivatePairRequest, retain

    session_path = _marked_improve_pair(tmp_path)
    result = retain(
        PrivatePairRequest(
            improve_session=session_path,
            pair_index=0,
            owner_id="alice",
            affected_panel="reward-panel-v3",
            private_store=tmp_path / "owner-private-store",
        )
    )

    assert result.code == "PRIVATE_REGRESSION_SAVED"
    assert result.pair.regression_case is True
    assert result.pair.private is True
    assert not (tmp_path / "owner-private-store" / "candidate-reference").exists()
    reopened = result.pair.reopen()
    assert reopened.left_bytes == result.pair.left_bytes
    assert reopened.right_bytes == result.pair.right_bytes
    assert b"alice@example.test" in reopened.left_bytes


def test_reopen_refuses_a_tampered_private_payload(tmp_path: Path) -> None:
    from reward_lens.commons.flywheel import FlywheelRefusal, PrivatePairRequest, retain

    pair = retain(
        PrivatePairRequest(
            improve_session=_marked_improve_pair(tmp_path),
            pair_index=0,
            owner_id="alice",
            affected_panel="reward-panel-v3",
            private_store=tmp_path / "private",
        )
    ).pair
    pair.left_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(FlywheelRefusal, match="PRIVATE_PAIR_INTEGRITY_FAILURE"):
        pair.reopen()


def test_promotion_requires_explicit_consent_and_redaction_and_never_leaks_private_text(
    tmp_path: Path,
) -> None:
    from reward_lens.commons.flywheel import (
        CONSENT_ABSENT,
        Consent,
        FlywheelRefusal,
        PrivatePairRequest,
        promote,
        retain,
    )

    pair = retain(
        PrivatePairRequest(
            improve_session=_marked_improve_pair(tmp_path),
            pair_index=0,
            owner_id="alice",
            affected_panel="reward-panel-v3",
            private_store=tmp_path / "private",
        )
    ).pair

    with pytest.raises(FlywheelRefusal, match=CONSENT_ABSENT):
        promote(pair, consent=None, redaction=None)
    with pytest.raises(FlywheelRefusal, match="REDACTION_REQUIRED"):
        promote(
            pair, consent=Consent(pair_id=pair.id, owner_id="alice", explicit=True), redaction=None
        )

    candidate = promote(
        pair,
        consent=Consent(pair_id=pair.id, owner_id="alice", explicit=True),
        redaction=_complete_redaction(),
    )
    assert candidate.kind == "CANDIDATE_REFERENCE_MATERIAL"
    assert candidate.certificate is None
    assert "alice@example.test" not in candidate.render()
    assert "[OWNER]" in candidate.render()
    assert "same-score shortcut" not in candidate.render()


def test_promotion_refuses_an_incomplete_sensitivity_inventory(tmp_path: Path) -> None:
    from reward_lens.commons.flywheel import (
        Consent,
        FlywheelRefusal,
        PrivatePairRequest,
        Redaction,
        promote,
        retain,
    )

    pair = retain(
        PrivatePairRequest(
            improve_session=_marked_improve_pair(tmp_path),
            pair_index=0,
            owner_id="alice",
            affected_panel="reward-panel-v3",
            private_store=tmp_path / "private",
        )
    ).pair
    incomplete = Redaction({"alice@example.test": "[OWNER]"})

    with pytest.raises(FlywheelRefusal, match="REDACTION_INVENTORY_INCOMPLETE"):
        promote(
            pair,
            consent=Consent(pair_id=pair.id, owner_id="alice", explicit=True),
            redaction=incomplete,
        )


def test_recalibration_binds_a_real_panel_and_makes_widening_transfer_failure_visible(
    tmp_path: Path,
) -> None:
    from reward_lens.commons.flywheel import (
        Consent,
        PrivatePairRequest,
        promote,
        recalibrate,
        retain,
    )

    panel = _qualified_panel()
    pair = retain(
        PrivatePairRequest(
            improve_session=_marked_improve_pair(tmp_path),
            pair_index=0,
            owner_id="alice",
            affected_panel="reward-panel-v3",
            private_store=tmp_path / "private",
        )
    ).pair
    candidate = promote(
        pair,
        consent=Consent(pair_id=pair.id, owner_id="alice", explicit=True),
        redaction=_complete_redaction(),
    )
    later = recalibrate(
        panel=panel,
        candidate=candidate,
    )

    assert later.prior.lower == Decimal("0.20")
    assert later.prior.upper == Decimal("0.40")
    assert later.later.lower == Decimal("0.10")
    assert later.later.upper == Decimal("0.80")
    assert later.transfer_status == "TRANSFER_FAILURE"
    assert later.changed
    assert later.candidate_id == candidate.id
    assert later.candidate_content_digest == candidate.content_digest
    assert later.evidence_path.is_file()
    assert "[0.20, 0.40]" in later.affected_panel_display
    assert "[0.10, 0.80]" in later.affected_panel_display


def test_recalibration_refuses_missing_prior_or_later_intervals(tmp_path: Path) -> None:
    from reward_lens.commons.flywheel import (
        CALIBRATION_INTERVALS_REQUIRED,
        Consent,
        FlywheelRefusal,
        PrivatePairRequest,
        promote,
        recalibrate,
        retain,
    )

    pair = retain(
        PrivatePairRequest(
            improve_session=_marked_improve_pair(tmp_path),
            pair_index=0,
            owner_id="alice",
            affected_panel="reward-panel-v3",
            private_store=tmp_path / "private",
        )
    ).pair
    candidate = promote(
        pair,
        consent=Consent(pair_id=pair.id, owner_id="alice", explicit=True),
        redaction=_complete_redaction(),
    )
    with pytest.raises(FlywheelRefusal, match=CALIBRATION_INTERVALS_REQUIRED):
        recalibrate(
            panel=None,
            candidate=candidate,
        )


def test_candidate_substitution_changes_measured_input_or_refuses_unsupported_data(
    tmp_path: Path,
) -> None:
    from reward_lens.commons.flywheel import (
        Consent,
        FlywheelRefusal,
        PrivatePairRequest,
        promote,
        recalibrate,
        retain,
    )

    panel = _qualified_panel()

    def candidate_for(root: Path, **pair_kwargs):
        pair = retain(
            PrivatePairRequest(
                improve_session=_marked_improve_pair(root, **pair_kwargs),
                pair_index=0,
                owner_id="alice",
                affected_panel="reward-panel-v3",
                private_store=root / "private",
            )
        ).pair
        return promote(
            pair,
            consent=Consent(pair_id=pair.id, owner_id="alice", explicit=True),
            redaction=_complete_redaction(),
        )

    widened_candidate = candidate_for(tmp_path / "widened")
    confirmed_candidate = candidate_for(
        tmp_path / "confirmed",
        right_measurements=("0.30", "0.35"),
    )
    widened = recalibrate(panel=panel, candidate=widened_candidate)
    confirmed = recalibrate(panel=panel, candidate=confirmed_candidate)

    assert widened.candidate_content_digest != confirmed.candidate_content_digest
    assert widened.later == type(widened.later)(Decimal("0.10"), Decimal("0.80"), "score-delta")
    assert confirmed.later == type(confirmed.later)(Decimal("0.30"), Decimal("0.35"), "score-delta")
    assert widened.evidence_path.read_bytes() != confirmed.evidence_path.read_bytes()

    unsupported = candidate_for(tmp_path / "unsupported", include_observation=False)
    with pytest.raises(FlywheelRefusal, match="CANDIDATE_CALIBRATION_UNSUPPORTED"):
        recalibrate(panel=panel, candidate=unsupported)


def test_recalibration_refuses_a_forged_candidate_dataclass_and_digest(tmp_path: Path) -> None:
    from reward_lens.commons.flywheel import (
        Consent,
        FlywheelRefusal,
        PrivatePairRequest,
        promote,
        recalibrate,
        retain,
    )

    pair = retain(
        PrivatePairRequest(
            improve_session=_marked_improve_pair(tmp_path),
            pair_index=0,
            owner_id="alice",
            affected_panel="reward-panel-v3",
            private_store=tmp_path / "private",
        )
    ).pair
    candidate = promote(
        pair,
        consent=Consent(pair_id=pair.id, owner_id="alice", explicit=True),
        redaction=_complete_redaction(),
    )
    forged = replace(candidate, source_private_pair_id="PP-forged")

    with pytest.raises(FlywheelRefusal, match="CANDIDATE_ARTIFACT_INVALID"):
        recalibrate(panel=_qualified_panel(), candidate=forged)
