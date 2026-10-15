"""Controls for the sealed, calibrated local D-50 proficiency round."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from decimal import Decimal

import pytest

from reward_lens.reference.proficiency import (
    DetectorResult,
    ProficiencyCharter,
    ProficiencyRefusal,
    SealedCharter,
    build_four_way_table,
    load_sealed_charter,
    main,
    persist_round,
    publish_local_charter,
    run_local_round,
)


def _sealed_charter(tmp_path) -> SealedCharter:
    return publish_local_charter(tmp_path / "charter.json")


def test_unsealed_charter_refuses_before_any_detector_execution(monkeypatch) -> None:
    import reward_lens.reference.proficiency as proficiency

    called = False

    def recorder_should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("the detector must not run before charter publication")

    monkeypatch.setattr(proficiency, "_recorder_result", recorder_should_not_run)

    with pytest.raises(ProficiencyRefusal, match="sealed charter"):
        run_local_round(ProficiencyCharter.local_default())
    assert not called


def test_local_round_executes_calibrated_recorder_and_seeker_black_box_and_retains_loser(
    tmp_path,
) -> None:
    sealed = _sealed_charter(tmp_path)
    report = run_local_round(sealed)

    assert {row.code for row in report.actual_results} == {"D-REC", "D-BBX"}
    assert all(row.provenance["execution"] == "local" for row in report.actual_results)
    assert all(row.unit == sealed.charter.calibration.unit for row in report.actual_results)
    assert report.result_for("D-BBX").provenance["seeker_actor_record_digest"]
    assert report.external_rows[0].bucket == "five_external_unavailable"
    assert report.losing_method.code in {"D-REC", "D-BBX"}
    assert report.losing_method.label
    assert report.monitor_check.status == "absent"


def test_public_projection_is_coded_and_private_reports_are_separate_mode_0600(tmp_path) -> None:
    report = run_local_round(_sealed_charter(tmp_path))
    public_path, private_path = persist_round(
        report,
        public_destination=tmp_path / "public.json",
        private_destination=tmp_path / "private.json",
    )

    public = json.loads(public_path.read_text())
    rendered = public_path.read_text()
    assert "recorder" not in rendered
    assert "black-box baseline" not in rendered
    assert set(public["coded_distribution"]) == {"D-REC", "D-BBX"}
    assert "recorder" in private_path.read_text()
    assert os.stat(private_path).st_mode & 0o777 == 0o600


def test_four_way_table_refuses_fabricated_rows_and_two_executed_results(tmp_path) -> None:
    report = run_local_round(_sealed_charter(tmp_path))
    fake = tuple(
        DetectorResult(
            code=f"FAKE-{index}",
            participant="invented",
            value=Decimal("0.70"),
            unit=report.charter.calibration.unit,
            stated_standard_uncertainty=Decimal("0.10"),
            expanded_uncertainty=Decimal("0.20"),
            executed_at="2026-09-26T12:01:00Z",
            provenance={"claimed": "no execution"},
        )
        for index in range(4)
    )

    with pytest.raises(ProficiencyRefusal, match="sealed round"):
        build_four_way_table(fake)
    sealed_round = persist_round(
        report,
        public_destination=tmp_path / "public.json",
        private_destination=tmp_path / "private.json",
    )
    with pytest.raises(ProficiencyRefusal, match="four actual detector results"):
        build_four_way_table(sealed_round)


def test_persistence_refuses_an_unrecorded_seeker_measurement(tmp_path) -> None:
    report = run_local_round(_sealed_charter(tmp_path))
    black_box = report.result_for("D-BBX")
    unrecorded = replace(black_box, execution_receipt_digest="")
    tampered = replace(report, results=(report.result_for("D-REC"), unrecorded))

    with pytest.raises(ProficiencyRefusal, match="unbound execution receipt"):
        persist_round(
            tampered,
            public_destination=tmp_path / "public.json",
            private_destination=tmp_path / "private.json",
        )


def test_sealed_charter_is_durable_before_results_and_binds_calibration(tmp_path) -> None:
    sealed = _sealed_charter(tmp_path)
    report = run_local_round(load_sealed_charter(sealed.path))
    public_path, _ = persist_round(
        report,
        public_destination=tmp_path / "public.json",
        private_destination=tmp_path / "private.json",
    )
    saved = json.loads(public_path.read_text())

    assert saved["charter_seal"]["digest"] == sealed.digest
    assert saved["charter"]["published_at"] < saved["coded_distribution"]["D-REC"]["executed_at"]
    assert saved["charter"]["calibration"]["qualified"] is True
    assert saved["external_rows"][0]["bucket"] == "five_external_unavailable"


def test_rehashed_forged_calibration_binding_is_refused_against_authoritative_panel(
    tmp_path,
) -> None:
    import reward_lens.reference.proficiency as proficiency

    sealed = _sealed_charter(tmp_path)
    document = json.loads(sealed.path.read_text())
    document["charter"]["calibration"]["unit"] = "forged-unit"
    document["digest"] = proficiency._digest(document["charter"])
    sealed.path.write_text(json.dumps(document))

    with pytest.raises(ProficiencyRefusal, match="authoritative local P-CALIB binding"):
        load_sealed_charter(sealed.path)


def test_persistence_refuses_four_self_hashed_fake_rows_with_no_native_execution(tmp_path) -> None:
    import reward_lens.reference.proficiency as proficiency

    report = run_local_round(_sealed_charter(tmp_path))
    base = report.result_for("D-REC")
    fake_rows = []
    fake_scores = []
    for index in range(4):
        unsigned = replace(
            base,
            code=f"FAKE-{index}",
            participant="invented",
            provenance={"execution": "local", "claimed": "no native record"},
            execution_receipt_digest="",
        )
        fake_rows.append(
            replace(unsigned, execution_receipt_digest=proficiency._digest(unsigned.receipt_body()))
        )
        fake_scores.append(replace(report.scores[0], code=f"FAKE-{index}"))
    fake_report = replace(
        report,
        results=tuple(fake_rows),
        scores=tuple(fake_scores),
        private_reports={row.code: {"participant": "invented"} for row in fake_rows},
    )

    with pytest.raises(ProficiencyRefusal, match="native execution"):
        persist_round(
            fake_report,
            public_destination=tmp_path / "forged-public.json",
            private_destination=tmp_path / "forged-private.json",
        )


def test_cli_caller_publishes_then_runs_to_distinct_evidence_files(tmp_path) -> None:
    charter = tmp_path / "charter.json"
    public = tmp_path / "public.json"
    private = tmp_path / "private.json"

    assert main(["publish-charter", "--output", str(charter)]) == 0
    assert (
        main(
            [
                "run",
                "--charter",
                str(charter),
                "--public-output",
                str(public),
                "--private-output",
                str(private),
            ]
        )
        == 0
    )
    assert public.is_file() and private.is_file()
    assert "participant" not in public.read_text()


def test_declared_four_local_procedures_execute_real_detectors_and_score_receipts(tmp_path):
    sealed = publish_local_charter(
        tmp_path / "four.json", participants=("D-REC", "D-BBX", "D-CUSUM", "D-BOCPD")
    )
    assert sealed.charter.participants == ("D-REC", "D-BBX", "D-CUSUM", "D-BOCPD")
    report = run_local_round(sealed)
    evidence = persist_round(
        report,
        public_destination=tmp_path / "public.json",
        private_destination=tmp_path / "private.json",
    )
    rows = build_four_way_table(evidence)
    assert [row["code"] for row in rows] == list(sealed.charter.participants)
    assert all(result.receipt_valid for result in report.actual_results)
    for code in ("D-CUSUM", "D-BOCPD"):
        result = report.result_for(code)
        assert result.provenance["method"] == "reward_lens.instruments.detection.detect"
        assert result.provenance["n"] == 2
        assert result.value == Decimal(result.provenance["detected"]) / Decimal(
            result.provenance["n"]
        )
    assert os.stat(evidence.private_path).st_mode & 0o777 == 0o600


def test_four_charter_refuses_three_missing_or_invented_participants(tmp_path):
    for codes in [("D-REC", "D-BBX", "D-CUSUM"), ("D-REC", "D-BBX", "FAKE-1", "FAKE-2")]:
        with pytest.raises(ProficiencyRefusal, match="supported actual local procedures"):
            publish_local_charter(tmp_path / "bad.json", participants=codes)


def test_persist_round_refuses_jointly_forged_score_table(tmp_path):
    report = run_local_round(_sealed_charter(tmp_path))
    score = replace(report.scores[0], z=report.scores[0].z + Decimal("1"))
    altered = replace(report, scores=(score, *report.scores[1:]))
    with pytest.raises(ProficiencyRefusal, match="scores do not match"):
        persist_round(
            altered,
            public_destination=tmp_path / "public.json",
            private_destination=tmp_path / "private.json",
        )
    assert not (tmp_path / "public.json").exists()


def test_original_two_charter_body_and_four_way_refusal_remain(tmp_path):
    sealed = _sealed_charter(tmp_path)
    assert "participants" not in sealed.charter.body()
    assert sealed.charter.participants == ("D-REC", "D-BBX")
    report = run_local_round(sealed)
    evidence = persist_round(
        report,
        public_destination=tmp_path / "public.json",
        private_destination=tmp_path / "private.json",
    )
    with pytest.raises(ProficiencyRefusal, match="four actual detector results"):
        build_four_way_table(evidence)
