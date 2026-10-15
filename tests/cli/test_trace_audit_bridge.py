"""The terminal Audit route forwards its one declared training artifact."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _positive_grpo_run(directory: Path, name: str, *, effect: bool = False) -> Path:
    """Reuse the SDK route's source-only real-run construction for the public CLI."""
    source = REPO / "tests" / "product" / "test_trace_panels.py"
    spec = importlib.util.spec_from_file_location("trace_panel_fixture", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._positive_grpo_run(directory, name, effect=effect)


def test_audit_cli_delivers_saved_trace_scoped_absences_from_a_copied_run_dir(
    cli, tmp_path: Path
) -> None:
    run_dir = tmp_path / "copied-real-run"
    shutil.copytree(REPO / "tests" / "fixtures" / "readers" / "verifiers", run_dir)
    grader = tmp_path / "grader.py"
    grader.write_text("def score(task, response):\n    return 1.0\n", encoding="utf-8")

    result = cli("audit", grader.name, "--run-dir", run_dir.name, "--format", "json", cwd=tmp_path)
    assert result.exit_code == 2, (result.stdout, result.stderr)
    document = json.loads(result.stdout)
    assay = tmp_path / document["artifacts"]["assay"]
    saved = json.loads(assay.read_text(encoding="utf-8"))
    entries = {
        entry["entry_id"]: entry
        for section in saved["measurement"].values()
        if isinstance(section, list)
        for entry in section
    }
    assert entries["reward_statistics.semantics"]["kind"] == "absence"
    assert "corpus.trace" in entries["reward_statistics.semantics"]["absence"]["missing_access"]
    assert entries["trace.composition.scalar_loss"]["kind"] == "absence"


def test_audit_cli_persists_actual_recorded_trace_calculations_and_report_surfaces(
    cli, tmp_path: Path
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
    grader = tmp_path / "grader.py"
    grader.write_text("def score(task, response):\n    return 1.0\n", encoding="utf-8")

    result = cli(
        "audit",
        grader.name,
        "--run-dir",
        target.name,
        "--detection-reference",
        reference.name,
        "--format",
        "json",
        cwd=tmp_path,
    )
    # The audit remains unresolved because unsupported update-level claims are explicit holes;
    # its saved artifact nevertheless contains every recorded calculation requested above.
    assert result.exit_code == 2, (result.stdout, result.stderr)
    document = json.loads(result.stdout)
    assay = tmp_path / document["artifacts"]["assay"]
    report = tmp_path / document["artifacts"]["report"]
    assert report.is_file() and "reward-lens" in report.read_text(encoding="utf-8")
    saved = json.loads(assay.read_text(encoding="utf-8"))
    entries = {
        entry["entry_id"]: entry
        for section in saved["measurement"].values()
        if isinstance(section, list)
        for entry in section
    }
    assert entries["signal.contrast_fraction"]["kind"] == "estimate"
    assert entries["reward_statistics.price.per_call"]["kind"] == "estimate"
    assert entries["trace.composition.scalar_loss"]["check"]["passed"] is True
    assert entries["trace.detection.one_sided_cusum"]["kind"] == "detection"
    assert entries["trace.detection.bocpd_run_length"]["kind"] == "detection"
    assert entries["trace.composition.gradient"]["kind"] == "absence"
