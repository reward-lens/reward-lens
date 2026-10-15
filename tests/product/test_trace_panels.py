"""The public Audit bridge carries one source trace through real registered panels."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from reward_lens import api
from reward_lens.product.audit.run import run as audit_run
from reward_lens.store import runs_root

REPO = Path(__file__).resolve().parents[2]
REAL_RUN_DIR = REPO / "tests" / "fixtures" / "readers" / "verifiers"
GRPO_RUN_DIR = REPO / "tests" / "fixtures" / "grpo_run" / "short"


@pytest.fixture(autouse=True)
def _private_run_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep each Audit manifest assertion out of the process-wide packet store."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def _grader(directory: Path) -> Path:
    path = directory / "grader.py"
    path.write_text("def score(task, response):\n    return 1.0\n", encoding="utf-8")
    return path


def _copied_run(directory: Path, *, run_id: str | None = None) -> Path:
    copied = directory / "copied-real-run"
    shutil.copytree(REAL_RUN_DIR, copied)
    if run_id is not None:
        metadata = json.loads((copied / "metadata.json").read_text(encoding="utf-8"))
        metadata["run_id"] = run_id
        (copied / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return copied


def _entry(record, entry_id: str):
    return next(entry for entry in record.entries() if entry.entry_id == entry_id)


def _manifest_paths() -> set[Path]:
    """Snapshot this test's private run store before one owned audit starts."""
    return set(runs_root().glob("*/manifest.json"))


def _test_manifest(before: set[Path]) -> dict[str, object]:
    """Read exactly the manifest created by this test, never another packet's run."""
    created = _manifest_paths() - before
    assert len(created) == 1
    return json.loads(created.pop().read_text(encoding="utf-8"))


def _seed_unrelated_trace_free_manifest(tmp_path: Path) -> None:
    """Keep ``execution_identity.trace = null`` a supported unrelated-run state."""
    control = runs_root() / f"unrelated-{tmp_path.name}" / "manifest.json"
    control.parent.mkdir(parents=True, exist_ok=True)
    control.write_text(json.dumps({"execution_identity": {"trace": None}}), encoding="utf-8")


def _positive_grpo_run(directory: Path, name: str, *, effect: bool = False) -> Path:
    """Copy a real recorded run and add only source-recorded telemetry fields."""
    copied = directory / name
    shutil.copytree(GRPO_RUN_DIR, copied)
    run = next(copied.glob("runs/*/run.json"))
    original_parent = run.parent
    run_id = f"run:{name}"
    document = json.loads(run.read_text(encoding="utf-8"))
    document["id"] = run_id
    document["lineage"]["extra"]["config"]["reward_weights"] = [1.0]
    run.write_text(json.dumps(document), encoding="utf-8")
    manifest = original_parent / "manifest.json"
    manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_doc["run_id"] = run_id
    manifest.write_text(json.dumps(manifest_doc), encoding="utf-8")
    run_parent = original_parent.parent / run_id.replace(":", "_")
    original_parent.rename(run_parent)
    trajectory_file = next(run_parent.glob("steps/*/trajectories.jsonl"))
    trajectories = [
        json.loads(line) for line in trajectory_file.read_text(encoding="utf-8").splitlines()
    ]
    for index, item in enumerate(trajectories):
        item["run_id"] = run_id
        if effect and index >= len(trajectories) // 2:
            item["advantage"] += 5.0
        payload = json.loads(item["payload"])
        call = payload["scores"]["fields"]["grader_call"]["fields"]
        call["latency_s"] = 0.01
        call["cost_usd"] = 0.02
        item["payload"] = json.dumps(payload)
    trajectory_file.write_text(
        "\n".join(json.dumps(item) for item in trajectories) + "\n", encoding="utf-8"
    )
    for source in run_parent.glob("steps/*/*.jsonl"):
        if source == trajectory_file:
            continue
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["run_id"] = run_id
        source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    # Row ids and the exact recorded advantages come from the canonical reader before this
    # envelope is attached.  The test does not manufacture a loss or acceptance from outcomes.
    from reward_lens.product.trace import reader_for

    trace = reader_for(copied).read(copied)
    first_step = [row for row in trace.rows if row.step == 0]
    document = json.loads(run_parent.joinpath("run.json").read_text(encoding="utf-8"))
    document["lineage"]["extra"]["trace_metadata"] = {
        "accepted": {":".join(row.row_id.split(":")[-2:]): True for row in trace.rows},
        "price_source": {"name": "fixture recorded price", "retrieved_on": "2026-08-04"},
        "planned_contrast": {
            "standard_error": 0.1,
            "alpha": 0.05,
            "power": 0.8,
            "n": 96,
            "clusters": 24,
        },
        "composition": {
            "steps": {
                "0": {
                    "trainer_profile": "grpo_ppo_token_mean_kl_v1",
                    "aggregation_denominator": len(first_step),
                    "minimum_denominator": 0.0,
                    "clip_low": 0.8,
                    "clip_high": 1.2,
                    "kl_coefficient": 0.0,
                    "observed_loss": -sum(row.advantage for row in first_step) / len(first_step),
                    "tokens": [
                        {
                            "row_id": row.row_id,
                            "mask": 1.0,
                            "log_probability": 0.0,
                            "old_log_probability": 0.0,
                            "importance_ratio": 1.0,
                            "advantage": row.advantage,
                            "kl_term": 0.0,
                            "auxiliary_term": 0.0,
                        }
                        for row in first_step
                    ],
                }
            }
        },
    }
    run_parent.joinpath("run.json").write_text(json.dumps(document), encoding="utf-8")
    return copied


def test_audit_opens_one_copied_real_run_and_persists_trace_scoped_panel_absences(
    tmp_path: Path,
) -> None:
    source = _copied_run(tmp_path)
    record = audit_run(api.AuditRequest(path=_grader(tmp_path), run_dir=source))

    # The Statistics and Composition public panels consumed the canonical trace.  This particular
    # source does not contain their required score-tree/loss telemetry, so their saved result is a
    # scoped absence rather than a fabricated numerical calculation.
    statistics = _entry(record, "reward_statistics.semantics")
    composition = _entry(record, "trace.composition.scalar_loss")
    assert statistics.kind == "absence"
    assert "corpus.trace" in statistics.absence.missing_access
    assert composition.kind == "absence"
    assert "scalar-loss telemetry" in composition.absence.missing_access

    assay = next((tmp_path / "assays").glob("*.assay.json"))
    saved = json.loads(assay.read_text(encoding="utf-8"))
    saved_ids = {
        entry["entry_id"]
        for section in saved["measurement"].values()
        if isinstance(section, list)
        for entry in section
    }
    assert {statistics.entry_id, composition.entry_id} <= saved_ids


def test_audit_execution_identity_binds_trace_bytes_and_declarations(tmp_path: Path) -> None:
    source = _copied_run(tmp_path)
    _seed_unrelated_trace_free_manifest(tmp_path)
    before = _manifest_paths()
    audit_run(api.AuditRequest(path=_grader(tmp_path), run_dir=source))

    # The source digest belongs to this exact new run.  Other valid executions may have a null
    # trace identity, so they are deliberately not interpreted as this audit's manifest.
    execution = _test_manifest(before)["execution_identity"]
    assert execution["trace"]["source"] == str(source)
    assert execution["trace"]["source_digest"].startswith("sha256:")
    assert execution["trace_declarations"] == {
        "reward_functions": [],
        "extra_columns": [],
        "weights": {},
        "group_column": None,
        "group_size": None,
    }


def test_audit_loads_a_disjoint_registered_detector_reference_once(tmp_path: Path) -> None:
    target = _copied_run(tmp_path)
    partitions: dict[str, list[dict[str, object]]] = {}
    for name in (
        "calibration_controls",
        "calibration_effects",
        "assessment_controls",
        "assessment_effects",
    ):
        run = _copied_run(tmp_path / name, run_id=name)
        item: dict[str, object] = {"run_dir": str(run)}
        if name.endswith("effects"):
            item.update({"onset_index": 0, "level": 1.0})
        partitions[name] = [item]
    reference = tmp_path / "detector-reference.json"
    reference.write_text(
        json.dumps(
            {
                "identifier": "offline-reference",
                "series": "reward_recorded",
                "registered_arl": 100.0,
                "effect_size": 1.0,
                "alpha": 0.05,
                "beta": 0.2,
                **partitions,
            }
        ),
        encoding="utf-8",
    )

    _seed_unrelated_trace_free_manifest(tmp_path)
    before = _manifest_paths()
    audit_run(
        api.AuditRequest(path=_grader(tmp_path), run_dir=target, detection_reference=reference)
    )
    manifest = _test_manifest(before)
    binding = manifest["execution_identity"]["detection_reference"]
    assert binding["source"] == str(reference)
    assert binding["source_digest"].startswith("sha256:")
    assert binding["spec_digest"].startswith("sha256:")
    assert len(binding["trace_sources"]) == 4
    assert {item["partition"] for item in binding["trace_sources"]} == set(partitions)
    assert all(item["source_digest"].startswith("sha256:") for item in binding["trace_sources"])


def test_sdk_audit_persists_actual_recorded_input_calculations_for_all_trace_panels(
    tmp_path: Path,
) -> None:
    target = _positive_grpo_run(tmp_path, "target")
    sources = {
        "calibration_controls": _positive_grpo_run(tmp_path, "cal-control"),
        "calibration_effects": _positive_grpo_run(tmp_path, "cal-effect", effect=True),
        "assessment_controls": _positive_grpo_run(tmp_path, "assessment-control"),
        "assessment_effects": _positive_grpo_run(tmp_path, "assessment-effect", effect=True),
    }
    partitions: dict[str, list[dict[str, object]]] = {}
    for partition, source in sources.items():
        item: dict[str, object] = {"run_dir": str(source)}
        if partition.endswith("effects"):
            item.update({"onset_index": 48, "level": 5.0})
        partitions[partition] = [item]
    reference = tmp_path / "registered-reference.json"
    reference.write_text(
        json.dumps(
            {
                "identifier": "fixture-independent-four-way",
                "series": "advantage",
                "registered_arl": 10.0,
                "effect_size": 5.0,
                "alpha": 0.1,
                "beta": 0.5,
                **partitions,
            }
        ),
        encoding="utf-8",
    )

    record = audit_run(
        api.AuditRequest(
            path=_grader(tmp_path),
            run_dir=target,
            detection_reference=reference,
            name="positive-trace",
        )
    )
    assert _entry(record, "signal.contrast_fraction").kind == "estimate"
    assert _entry(record, "reward_statistics.price.per_call").kind == "estimate"
    scalar_loss = _entry(record, "trace.composition.scalar_loss")
    assert scalar_loss.kind == "check" and scalar_loss.check.passed is True
    assert {
        entry.kind for entry in record.entries() if entry.entry_id.startswith("trace.detection.")
    } == {"detection"}
    assert _entry(record, "trace.composition.gradient").kind == "absence"

    saved = next((tmp_path / "assays").glob("*.assay.json"))
    saved_entries = {
        entry["entry_id"]
        for section in json.loads(saved.read_text(encoding="utf-8"))["measurement"].values()
        if isinstance(section, list)
        for entry in section
    }
    assert {
        "signal.contrast_fraction",
        "reward_statistics.price.per_call",
        "trace.composition.scalar_loss",
        "trace.detection.one_sided_cusum",
    } <= saved_entries
