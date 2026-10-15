"""Acceptance checks for the artifact-derived E1 assay."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest


def _golden() -> dict[str, int]:
    root = Path(__file__).parents[2]
    return json.loads((root / "fleet/golden/wave-3/e1/counts.json").read_text())["counts"]


@pytest.fixture
def public_e1_root() -> Path:
    configured = os.environ.get("REWARD_LENS_E1_ASSAY_ROOT")
    if not configured:
        pytest.fail(
            "set REWARD_LENS_E1_ASSAY_ROOT to run the required public E1 artifact reconstruction"
        )
    return Path(configured)


@pytest.fixture
def small_e1_root(tmp_path: Path) -> Path:
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    analysis = tmp_path / "e1_analysis"
    analysis.mkdir()
    (analysis / "COMPOSITION.md").write_text(
        "# Reward weights: [thinking_format, training_passed]\n"
        "objective is `1 · thinking_format + 4 · training_passed`\n",
        encoding="utf-8",
    )
    for name in ("REGISTERED.md", "RECORD_CONTENTS.md", "GATE2.md"):
        (analysis / name).write_text(f"synthetic {name}\n", encoding="utf-8")
    rows = [
        {
            "step": 0,
            "prompt": "p",
            "completion": "a",
            "advantage": 0.0,
            "thinking_format": 1.0,
            "training_passed": 1.0,
            "source_eval_file": "a",
            "sample_id": 0,
            "reward_hacked": True,
            "actually_solved": False,
        },
        {
            "step": 0,
            "prompt": "p",
            "completion": "b",
            "advantage": 0.0,
            "thinking_format": 0.0,
            "training_passed": 0.0,
            "source_eval_file": "a",
            "sample_id": 0,
            "reward_hacked": True,
            "actually_solved": False,
        },
        {
            "step": 1,
            "prompt": "p",
            "completion": "c",
            "advantage": 0.0,
            "thinking_format": 1.0,
            "training_passed": 0.0,
            "source_eval_file": "b",
            "sample_id": 0,
            "reward_hacked": False,
            "actually_solved": True,
        },
        {
            "step": 1,
            "prompt": "p",
            "completion": "d",
            "advantage": 0.0,
            "thinking_format": 1.0,
            "training_passed": 0.0,
            "source_eval_file": "b",
            "sample_id": 0,
            "reward_hacked": False,
            "actually_solved": False,
        },
    ]
    path = tmp_path / ".e1-cache" / "extract"
    path.mkdir(parents=True)
    parquet.write_table(arrow.Table.from_pylist(rows), path / "kl0.0_rollouts.parquet")
    return tmp_path


def test_build_reconstructs_all_controller_recorded_counts(public_e1_root: Path) -> None:
    from reward_lens.examples.e1 import build, counts

    measured = counts(build(public_e1_root))
    assert measured.as_dict() == _golden()
    assert measured.groups_with_contrast != measured.registered_analysis_support


def test_assay_carries_reproducible_commands_and_explicit_absences(public_e1_root: Path) -> None:
    from reward_lens.examples.e1 import build

    assay = build(public_e1_root)
    assert set(assay.reproduce) == set(_golden())
    assert all("e1_analysis/" in citation for citation in assay.citations)
    assert "not measured" in assay.absences["applied_update_weight"].lower()
    assert "not measured" in assay.absences["gradient"].lower()
    assert "not measured" in assay.absences["effective_sample_size"].lower()


def test_missing_recorded_artifact_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from reward_lens.examples.e1 import build

    monkeypatch.setenv("REWARD_LENS_E1_ASSAY_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="e1_analysis"):
        build()


def test_missing_source_root_refuses_implicit_local_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reward_lens.examples.e1 import build

    monkeypatch.delenv("REWARD_LENS_E1_ASSAY_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="REWARD_LENS_E1_ASSAY_ROOT"):
        build()


def test_small_public_shape_fixture_reconstructs_without_private_rows(small_e1_root: Path) -> None:
    from reward_lens.examples.e1 import build, counts

    assay = build(small_e1_root)
    assert counts(assay).as_dict() == {
        "groups": 2,
        "groups_with_contrast": 1,
        "accepted_hacks": 1,
        "solutions": 1,
        "functions_summed": 2,
        "advantage_components": 2,
        "registered_analysis_support": 0,
    }
    assert all("--source-root" in command for command in assay.reproduce.values())
    assert len(assay.source_digests()) == 5


def test_public_join_receipt_keeps_new_derivation_provenance(
    small_e1_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from reward_lens.examples.e1 import build, counts

    original = small_e1_root / ".e1-cache/extract/kl0.0_rollouts.parquet"
    output = small_e1_root / "public_join_v1.parquet"
    shutil.copyfile(original, output)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    analysis = {
        f"e1_analysis/{name}": sha(small_e1_root / "e1_analysis" / name)
        for name in ("COMPOSITION.md", "REGISTERED.md", "RECORD_CONTENTS.md", "GATE2.md")
    }
    receipt = {
        "schema": "reward-lens-e1-public-join/1",
        "output": output.name,
        "output_sha256": sha(output),
        "rows": 4,
        "counts": {**counts(build(small_e1_root)).as_dict(), "native_grid_support": 0},
        "analysis_sha256": analysis,
        "inputs_sha256": {"published.parquet": "a" * 64, "published.eval": "b" * 64},
        "source_dataset": "synthetic",
        "source_revision": "synthetic-revision",
        "historical_extraction_sha256": None,
    }
    (small_e1_root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assay = build(small_e1_root, derivation_receipt=small_e1_root / "receipt.json")
    assert assay.rollout_path == output
    assert assay.trace["derivation"] == "reward-lens-e1-public-join/1"
    assert "receipt.json" in assay.source_digests()
    assert "--derivation-receipt" in assay.reproduce["groups"]
    record = assay.contract_assay()
    entry = record.measurement.reward_statistics[0]
    assert "caller-attested" in " ".join(entry.limitations)
    assert record.subject.reward_system.id == "e1-reconstruction-inputs"
    assert record.subject.version.id == "receipt-inputs"
    assert "verified public parquet" not in entry.method.procedure
    assert (
        record.measurement.reward_statistics[0].witness.inputs["receipt.json"]
        == assay.source_digests()["receipt.json"]
    )
    from reward_lens.examples.e1.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reward-lens-e1",
            "--source-root",
            str(small_e1_root),
            "--derivation-receipt",
            str(small_e1_root / "receipt.json"),
            "--field",
            "groups",
        ],
    )
    assert main() == 0
    assert capsys.readouterr().out.strip() == "2"
    receipt["counts"]["groups"] = 99
    (small_e1_root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt counts disagree"):
        build(small_e1_root, derivation_receipt=small_e1_root / "receipt.json")


def test_cli_accepts_explicit_source_root(
    small_e1_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from reward_lens.examples.e1.__main__ import main

    monkeypatch.setattr(
        sys, "argv", ["reward-lens-e1", "--source-root", str(small_e1_root), "--field", "groups"]
    )
    assert main() == 0
    assert capsys.readouterr().out.strip() == "2"


def test_cli_reports_missing_source_root(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from reward_lens.examples.e1.__main__ import main

    monkeypatch.delenv("REWARD_LENS_E1_ASSAY_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", ["reward-lens-e1", "--field", "groups"])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert "REWARD_LENS_E1_ASSAY_ROOT" in capsys.readouterr().err


def test_packaged_golden_matches_repository_reference() -> None:
    from reward_lens.examples.e1 import load_golden

    assert load_golden() == _golden()


def test_golden_disagreement_fails_instead_of_rewriting_golden(public_e1_root: Path) -> None:
    from reward_lens.examples.e1 import assert_matches_golden, build, counts

    corrupted = _golden()
    corrupted["groups"] += 1
    with pytest.raises(AssertionError, match="groups"):
        assert_matches_golden(counts(build(public_e1_root)), corrupted)


def test_report_is_a_saved_canonical_assay_not_a_summary_dict(
    tmp_path: Path, public_e1_root: Path
) -> None:
    import rfc8785

    from reward_lens import api, contracts
    from reward_lens.examples.e1 import build, persist_assay_bundle

    bundle = persist_assay_bundle(build(public_e1_root), tmp_path / "bundle")
    reopened = api.open_record(bundle.assay_json)
    assert isinstance(reopened, contracts.Assay)
    assert contracts.digest(reopened) == reopened.assay_id
    assert rfc8785.dumps(reopened.to_dict()) == bundle.assay_json.read_bytes()
    assert bundle.report_html.is_file()
    assert json.loads(bundle.manifest.read_text())["assay_id"] == reopened.assay_id


def test_sdk_assay_carries_real_count_evidence_and_telemetry_absences(public_e1_root: Path) -> None:
    from reward_lens import contracts
    from reward_lens.examples.e1 import build

    record = build(public_e1_root).contract_assay()
    contracts.validate_record(record.to_dict())
    assert build(public_e1_root).canonical()["$schema"] == contracts.SCHEMA_URL
    evidence = record.measurement.reward_statistics[0]
    assert evidence.result["counts"] == _golden()
    assert {entry.entry_id for entry in record.measurement.trace} == {
        "trace.applied_update_weight",
        "trace.effective_sample_size",
        "trace.gradient",
    }


def test_source_has_no_frozen_counts_or_preflight_citation() -> None:
    root = Path(__file__).parents[2] / "src/reward_lens/examples/e1"
    source = "\n".join(path.read_text() for path in root.glob("*.py"))
    assert "COUNTS =" not in source
    assert "e1_pre" + "flight/" not in source
