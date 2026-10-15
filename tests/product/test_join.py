"""D-47's per-family column, from one product implementation."""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from click.testing import CliRunner

from reward_lens import contracts
from reward_lens.product.compare.scorecard import Estimate
from reward_lens.product.join import (
    HEADINGS,
    JOIN_EXTENSION,
    Absence,
    canonical_rows,
    has_join_column,
    join,
)


def estimate(value: str, unit: str = "proportion") -> Estimate:
    point = Decimal(value)
    return Estimate(point, (point, point), n=1, unit=unit)


def record(*, run: bool = False, join_rows: list[dict] | None = None) -> dict:
    result = {
        "audit": {"exploit_families": ["format-string", "tool-loop"]},
        "signal": estimate("0.7"),
    }
    if run:
        result["run_record"] = {"id": "run-17"}
    if join_rows is not None:
        result["join"] = join_rows
    return result


def test_frozen_headings_and_every_missing_cell_stays_in_the_row() -> None:
    rendered = join(record())

    assert HEADINGS == (
        "selected in your run",
        "applied update weight",
        "fix changes pressure",
        "cost of the fix",
    )
    assert [row.family for row in rendered.rows] == ["format-string", "tool-loop"]
    for row in rendered.rows:
        for cell in (
            row.selected_in_your_run,
            row.applied_update_weight,
            row.fix_changes_pressure,
            row.cost_of_the_fix,
        ):
            assert isinstance(cell, Absence)
            assert cell.reason and cell.remedy


def test_missing_run_record_uses_one_middle_absence_and_audit_still_ships() -> None:
    rendered = join(record())

    row = rendered.rows[0]
    assert row.applied_update_weight is row.fix_changes_pressure
    assert "run record" in row.applied_update_weight.reason
    assert row.applied_update_weight.remedy
    assert len(rendered.rows) == 2


@pytest.mark.parametrize("invalid", [0, 1, "true", "false"])
def test_selection_cell_refuses_non_boolean_labels(invalid: object) -> None:
    row = join(
        record(
            run=True,
            join_rows=[
                {
                    "family": "format-string",
                    "selected_in_your_run": invalid,
                    "evidence": {"selection": "trace.selection.format-string"},
                }
            ],
        )
    ).rows[0]
    assert isinstance(row.selected_in_your_run, Absence)
    assert "selection in this run is unavailable" in row.selected_in_your_run.reason


def test_per_family_evidence_preserves_score_advantage_loss_and_update_boundaries() -> None:
    rendered = join(
        record(
            run=True,
            join_rows=[
                {
                    "family": "format-string",
                    "selected_in_your_run": True,
                    "applied_update_weight": estimate("0.4", "update weight"),
                    "fix_changes_pressure": estimate("0.0", "advantage delta"),
                    "cost_of_the_fix": estimate("0.1", "legitimate-task cost"),
                    "score": estimate("-0.2", "score delta"),
                    "loss": estimate("0.3", "loss"),
                    "evidence": {
                        "selection": "trace.selection.format-string",
                        "update": "trace.update.format-string",
                        "pressure": "compare.rung3.format-string",
                        "cost": "compare.capability.format-string",
                    },
                }
            ],
        )
    )

    row = rendered.rows[0]
    assert row.selected_in_your_run is True
    assert row.applied_update_weight.value == Decimal("0.4")
    assert row.fix_changes_pressure.value == Decimal("0.0")
    assert row.cost_of_the_fix.value == Decimal("0.1")
    assert isinstance(rendered.rows[1].applied_update_weight, Absence)
    assert rendered.signal.value == Decimal("0.7")

    score_and_loss_only = join(
        record(
            run=True,
            join_rows=[
                {
                    "family": "format-string",
                    "score": estimate("-0.2", "score delta"),
                    "loss": estimate("0.3", "loss"),
                }
            ],
        )
    ).rows[0]
    assert isinstance(score_and_loss_only.applied_update_weight, Absence)
    assert isinstance(score_and_loss_only.fix_changes_pressure, Absence)


def test_join_column_contract_requires_saved_full_identity_rows() -> None:
    available_run_without_column = record(run=True)

    assert has_join_column(available_run_without_column) is False
    assert has_join_column(identity_record()) is True


def test_enabled_audit_trace_saved_record_cli_and_report_call_the_one_product_join(
    tmp_path, monkeypatch
) -> None:
    from reward_lens import api
    from reward_lens.cli.verbs.join import join as join_command
    from reward_lens.product import join as product_join

    project = tmp_path / "project"
    project.mkdir()
    (project / "grader.py").write_text(
        "def score(task, response):\n    return 1.0 if response == '2' else 0.0\n",
        encoding="utf-8",
    )
    (project / "tasks.jsonl").write_text(
        '{"id":"task-1","prompt":"find a response"}\n', encoding="utf-8"
    )
    (project / "outcome").mkdir()
    (project / "outcome" / "check.py").write_text(
        "def run_outcome(task, response):\n    return {'passed': False}\n", encoding="utf-8"
    )
    (project / "rewardlens.yaml").write_text(
        "reward:\n  entry: grader.py\n  kind: plain\ntasks:\n  path: tasks.jsonl\n  prompt: prompt\n"
        "outcome:\n  kind: protected_test_suite\n  path: outcome\nsuccess: name a non-prime\n",
        encoding="utf-8",
    )
    seeker = tmp_path / "seeker.json"
    seeker.write_text(
        '{"mode":"agent","model":"fixture","max_proposals":1,"recorded_proposals":["2"]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    source = tmp_path / "real-run"
    fixture = Path(__file__).parents[1] / "fixtures" / "grpo_run" / "short"
    shutil.copytree(fixture, source)
    calls = []
    original = product_join.join

    def observed(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(product_join, "join", observed)
    _audit = api.audit(
        api.AuditRequest(
            path=project, run_dir=source, seeker="agent", seeker_config=seeker, name="join-enabled"
        )
    )
    source_path = project / "assays" / "join-enabled.assay.json"
    report_path = project / "assays" / "join-enabled.assay.html"
    assert source_path.is_file() and report_path.is_file()
    saved = json.loads(source_path.read_text(encoding="utf-8"))
    assert JOIN_EXTENSION in saved["extensions"]
    assert saved["extensions"][JOIN_EXTENSION]["trace"]["complete"] is True
    trace_entries = {entry["entry_id"]: entry for entry in saved["measurement"]["trace"]}
    fiveway = trace_entries["trace.fiveway"]["result"]["answer"]
    for reference in [*fiveway["evidence"], *fiveway["remediation_evidence"]]:
        target = trace_entries.get(reference["entry_id"])
        assert target is not None
        from reward_lens import contracts

        assert contracts.digest(target) == reference["digest"]
    produced = original(saved)
    # The accepted Seeker route now reaches the audited Python reward.  This project deliberately
    # lacks a qualified supervised execution declaration, so its scored proposal remains an
    # honest no-witness control rather than evidence for the positive Join journey.
    actor_scores = saved["extensions"]["https://reward-lens.dev/extensions/seeker"]["actor_record"][
        "scores"
    ]
    assert actor_scores[0]["verdict"] == "scored"
    assert actor_scores[0]["outcome_evidence"]["state"] == "UNQUALIFIED"
    assert produced.rows == []
    output = CliRunner().invoke(join_command, [str(source_path)])
    assert output.exit_code == 0, output.output
    assert all(heading in output.output for heading in HEADINGS)

    html = report_path.read_bytes()
    assert all(heading.encode() in html for heading in HEADINGS)
    assert (
        len(calls) >= 2
    )  # audit's saved report and the public CLI both reached this implementation


def _write_matching_closed_trace_run(root: Path, *, source_digest: str) -> Path:
    """Build a recorded, closed Trace input for the public Seeker target.

    This deliberately edits the raw run receipt before the accepted Trace reader
    closes it. It never writes an Assay entry, exploit witness, or Join extension.
    """
    fixture = Path(__file__).parents[1] / "fixtures" / "grpo_run" / "short"
    trace_root = root / "closed-trace"
    shutil.copytree(fixture, trace_root)
    run_path = next((trace_root / "runs").glob("*/run.json"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["lineage"]["extra"]["grader_source_digest"] = source_digest
    run_path.write_text(json.dumps(run), encoding="utf-8")

    trajectories_path = next((trace_root / "runs").glob("*/steps/*/trajectories.jsonl"))
    trajectories = [
        json.loads(line) for line in trajectories_path.read_text(encoding="utf-8").splitlines()
    ]
    selected = trajectories[0]
    selected["task_ref"] = "target"
    selected["applied_weight"] = 0.25
    selected["applied_weight_source"] = "recorded"
    selected_payload = json.loads(selected["payload"])
    selected_payload["applied_weight"] = 0.25
    selected_payload["applied_weight_source"] = "recorded"
    selected["payload"] = json.dumps(selected_payload, ensure_ascii=False)
    trajectories_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trajectories),
        encoding="utf-8",
    )

    turns_path = next((trace_root / "runs").glob("*/steps/*/turns.jsonl"))
    turns = [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines()]
    for turn in turns:
        if turn["trajectory_id"] == selected["trajectory_id"] and turn["role"] == "assistant":
            turn["text"] = "paid-but-wrong"
            payload = json.loads(turn["payload"])
            payload["text"] = "paid-but-wrong"
            turn["payload"] = json.dumps(payload, ensure_ascii=False)
    turns_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in turns), encoding="utf-8"
    )
    return trace_root


def _write_qualified_public_project(root: Path) -> tuple[Path, Path, str]:
    """Write the shared Seeker public fixture: controls, target, and recorded actor reply."""
    controls, targets = root / "outcome" / "controls", root / "outcome" / "targets"
    controls.mkdir(parents=True)
    targets.mkdir()
    for path in (controls / "good.case", controls / "wrong.case", targets / "target.case"):
        path.write_text(path.name, encoding="utf-8")
    reward_source = (
        "def score(task, response):\n    return 1.0 if response == 'paid-but-wrong' else 0.0\n"
    )
    (root / "grader.py").write_text(reward_source, encoding="utf-8")
    task = {"id": "target", "prompt": "return the reference", "reference": "correct"}
    (root / "tasks.jsonl").write_text(json.dumps(task) + "\n", encoding="utf-8")
    evaluator = root / "outcome" / "owner_evaluator.py"
    evaluator.write_text(
        "def score(task, response):\n    return 1.0 if response == task['reference'] else 0.0\n",
        encoding="utf-8",
    )
    from reward_lens.graders.python_grader import PythonGrader

    reward_digest = contracts.digest({"source": reward_source})
    execution = {
        "version": 1,
        "evaluator": {"path": "outcome/owner_evaluator.py", "entrypoint": "score"},
        "check": {
            "owner": "assay-owner",
            "source_digest": PythonGrader.from_path(evaluator).describe().source_digest,
            "interface": "reward_lens.graders.python_grader.PythonGrader.score/v1",
            "partition_id": "controls-v1",
        },
        "candidate_owner": "candidate",
        "audited_grader": {"owner": "candidate", "source_digest": reward_digest},
        "subject_ref": "sha256:" + "0" * 64,
        "controls": {
            "partition_path": "outcome/controls",
            "partition_id": "controls-v1",
            "known_good": [{"item_id": "good.case", "task": task, "response": "correct"}],
            "known_wrong": [{"item_id": "wrong.case", "task": task, "response": "wrong"}],
        },
        "target": {
            "partition_path": "outcome/targets",
            "partition_id": "target-v1",
            "tasks": [{"item_id": "target.case", "task": task}],
        },
    }
    (root / "outcome" / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
    (root / "rewardlens.yaml").write_text(
        "reward:\n  entry: grader.py\n  kind: plain\ntasks:\n  path: tasks.jsonl\n  prompt: prompt\n"
        "outcome:\n  kind: protected_test_suite\n  path: outcome/controls\n  execution: outcome/execution.json\n"
        "success: name a non-prime\n",
        encoding="utf-8",
    )
    seeker = root / "seeker.json"
    seeker.write_text(
        json.dumps(
            {
                "mode": "agent",
                "model": "sdk-recorded",
                "max_proposals": 1,
                "recorded_proposals": ["paid-but-wrong"],
            }
        ),
        encoding="utf-8",
    )
    return root, seeker, reward_digest


def test_public_sdk_join_positive_uses_a_real_closed_trace_and_qualified_actor(
    tmp_path, monkeypatch
) -> None:
    """The promised positive is an Audit-produced saved row, never an injected Join extension."""
    from reward_lens import api
    from reward_lens.cli.verbs.join import join as join_command
    from reward_lens.contracts import ProjectConfig
    from reward_lens.product.trace import reader_for
    from reward_lens.render.projections.views import report as report_projection
    from reward_lens.store import Project

    project_root, seeker, source_digest = _write_qualified_public_project(tmp_path / "project")
    trace_root = _write_matching_closed_trace_run(tmp_path, source_digest=source_digest)
    trace = reader_for(trace_root).read(trace_root, policy="strict")
    trace_entry = next(entry for entry in trace.entries() if entry.entry_id == "trace.row.0")
    assert trace_entry.result["task_id"] == "target"
    assert trace_entry.result["response_codec"] == "rfc8785-response-text/1"
    assert trace_entry.result["grader_source"]["grader_source_digest"] == source_digest
    assert trace_entry.result["applied_weight"] == 0.25
    assert trace_entry.result["applied_weight_source"] == "recorded"

    from reward_lens.product.audit import instruments, registry
    from reward_lens.product.audit.run import _planned_project_method_ids
    from reward_lens.store.digests import method_set

    project = Project(
        project_root,
        ProjectConfig.model_validate(
            {
                "reward": {"entry": "grader.py"},
                "tasks": {"path": "tasks.jsonl", "prompt": "prompt"},
                "outcome": {
                    "kind": "protected_test_suite",
                    "path": "outcome/controls",
                    "execution": "outcome/execution.json",
                },
                "success": "name a non-prime",
            }
        ),
    )
    execution_path = project_root / "outcome" / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    planning_corpus = instruments.build_corpus(
        tasks_path=project_root / "tasks.jsonl",
        responses_path=None,
        trace=trace,
    )
    planned_methods = _planned_project_method_ids(
        registry.discover(),
        trace_record=trace,
        response_bank=False,
        project=project,
        corpus=planning_corpus,
    )
    assert "soundness.relations.injected_defect@1.0.0" in planned_methods
    execution["subject_ref"] = project.version(methods=planned_methods).digest()
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    result = api.audit(
        api.AuditRequest(
            path=project_root,
            run_dir=trace_root,
            seeker="agent",
            seeker_config=seeker,
            name="join-positive",
        )
    )
    assert method_set(result.to_dict()) == planned_methods
    assert result.subject.version.digest == execution["subject_ref"]
    saved_path = project_root / "assays" / "join-positive.assay.json"
    report_path = project_root / "assays" / "join-positive.assay.html"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert result.to_dict() == saved
    reopened = project.open_record("join-positive")
    assert reopened.to_dict() == saved
    actor = saved["extensions"]["https://reward-lens.dev/extensions/seeker"]["actor_record"]
    assert actor["outcome_protocol"]["state"] == "QUALIFIED"
    assert actor["scores"][0]["outcome_evidence"]["state"] == "FAIL"
    payload = saved["extensions"][JOIN_EXTENSION]
    assert payload["audited_grader_source_digest"] == source_digest
    assert saved["join"][0]["family"] == "exploits.seeker"
    assert saved["join"][0]["applied_update_weight"] == 0.25
    joined = join(reopened)
    assert len(joined.rows) == 1
    assert joined.rows[0].selected_in_your_run is True
    assert joined.rows[0].applied_update_weight.value == Decimal("0.25")
    assert joined.rows[0].evidence["applied_update_weight"] == "trace.row.0"
    expected_missing = {
        name: {"reason": cell.reason, "remedy": cell.remedy}
        for name, cell in {
            "selected": joined.rows[0].selected_in_your_run,
            "applied_update_weight": joined.rows[0].applied_update_weight,
            "fix_changes_pressure": joined.rows[0].fix_changes_pressure,
            "fix_cost": joined.rows[0].cost_of_the_fix,
        }.items()
        if isinstance(cell, Absence)
    }
    assert expected_missing
    assert saved["join"][0]["missing"] == expected_missing
    output = CliRunner().invoke(join_command, [str(saved_path)])
    assert output.exit_code == 0, output.output
    assert "trace.row.0" in output.output
    assert b'href="#assay-entry-trace.row.0"' in report_path.read_bytes()
    report_text = report_path.read_text(encoding="utf-8")
    assert report_text.count('data-join-state="absent"') >= len(expected_missing)
    for absence in expected_missing.values():
        assert absence["reason"] in report_text
        assert absence["remedy"] in report_text
    projection = report_projection(reopened)
    assert projection.fields["join"] == saved["join"]
    assert "/join" in projection.source_pointers


def identity_record(*, mismatch: bool = False, dangling: bool = False) -> dict:
    identity = {
        "task_id": "task-17",
        "response_id": "response-4",
        "family": "format-string",
        "subject_ref": "sha256:" + "a" * 64,
    }
    source_identity = dict(identity)
    if mismatch:
        source_identity.update(task_id="other-task", response_id="other-response")
    entry_subject = "sha256:" + "b" * 64 if mismatch else identity["subject_ref"]
    entries = [
        {
            "entry_id": ident,
            "kind": "estimate",
            "scope": "reconstructed_training_pressure",
            "subject_ref": entry_subject,
            "value": 0.4 if ident == "trace.update" else 0.0,
            "unit": "proportion",
            "n": 1,
            "uncertainty": {"interval": [0.4, 0.4] if ident == "trace.update" else [0.0, 0.0]},
            "result": {**source_identity, "selected": True},
        }
        for ident in (
            "trace.selection",
            "trace.update",
            "compare.pressure",
            "compare.cost",
            "signal.shared",
        )
    ]
    evidence = {
        "selected_in_your_run": "trace.selection",
        "applied_update_weight": "trace.update",
        "fix_changes_pressure": "compare.pressure",
        "cost_of_the_fix": "compare.cost",
    }
    if dangling:
        evidence["fix_changes_pressure"] = "missing.entry"
    return {
        "subject": {"version": {"digest": identity["subject_ref"]}},
        "measurement": {"trace": entries[:2], "reward_statistics": entries[2:]},
        "extensions": {
            JOIN_EXTENSION: {
                "audit_families": [identity],
                "rows": [{**identity, "evidence": evidence, "trace_row_matched": True}],
                "signal_evidence": "signal.shared",
            }
        },
    }


def test_full_identity_and_resolved_entry_evidence_make_join_cells_available() -> None:
    rendered = join(identity_record())

    row = rendered.rows[0]
    assert row.selected_in_your_run is True
    assert row.applied_update_weight.value == Decimal("0.4")
    assert row.evidence["applied_update_weight"] == "trace.update"
    assert rendered.signal.value == Decimal("0.0")


def test_canonical_rows_keep_measured_values_and_name_missing_cells() -> None:
    row = canonical_rows(join(identity_record()))[0]

    assert row.family == "format-string"
    assert row.selected == "yes"
    assert row.applied_update_weight == 0.4
    assert row.fix_changes_pressure == 0.0
    assert row.fix_cost == "0.0 proportion"
    assert row.missing is None


def test_canonical_rows_persist_each_absent_cell_reason_and_remedy() -> None:
    document = identity_record(dangling=True)
    document["extensions"][JOIN_EXTENSION]["rows"][0]["evidence"].pop("cost_of_the_fix")
    joined = join(document)
    row = canonical_rows(joined)[0]

    expected = {
        "fix_changes_pressure": {
            "reason": joined.rows[0].fix_changes_pressure.reason,
            "remedy": joined.rows[0].fix_changes_pressure.remedy,
        },
        "fix_cost": {
            "reason": joined.rows[0].cost_of_the_fix.reason,
            "remedy": joined.rows[0].cost_of_the_fix.remedy,
        },
    }
    persisted = row.model_dump(mode="json")["missing"]
    assert persisted == expected
    assert "selected" not in persisted
    assert "applied_update_weight" not in persisted


def test_join_row_accepts_legacy_missing_string_and_validates_named_cell_map() -> None:
    import jsonschema_rs

    base = {
        "family": "format-string",
        "selected": "yes",
        "applied_update_weight": 0.4,
        "fix_changes_pressure": None,
        "fix_cost": None,
    }
    legacy = contracts.JoinRow.model_validate(
        {**base, "missing": "a run record: reward-lens trace <run>"}
    )
    assert legacy.missing == "a run record: reward-lens trace <run>"
    named = contracts.JoinRow.model_validate(
        {
            **base,
            "missing": {
                "fix_changes_pressure": {"reason": "no compare reading", "remedy": "run compare"},
                "fix_cost": {"reason": "no held-out result", "remedy": "run held-out evaluation"},
            },
        }
    )
    assert named.missing.fix_changes_pressure.reason == "no compare reading"
    assert named.missing.fix_cost.remedy == "run held-out evaluation"
    with pytest.raises(Exception):
        contracts.JoinRow.model_validate(
            {**base, "missing": {"unknown": {"reason": "x", "remedy": "y"}}}
        )
    with pytest.raises(Exception):
        contracts.JoinRow.model_validate({**base, "missing": {"fix_cost": {"reason": "x"}}})
    invalid_named_maps = [
        {**base, "missing": {}},
        {
            **base,
            "missing": {
                "selected": {"reason": "selected was not measured", "remedy": "measure selection"}
            },
        },
    ]
    join_item_schema = {
        **contracts.assay_schema()["properties"]["join"]["items"],
        "$defs": contracts.assay_schema()["$defs"],
    }
    validator = jsonschema_rs.validator_for(join_item_schema)
    for invalid in invalid_named_maps:
        with pytest.raises(Exception):
            contracts.JoinRow.model_validate(invalid)
        assert not validator.is_valid(invalid)


def test_wrong_task_response_or_subject_identity_becomes_per_cell_absences() -> None:
    row = join(identity_record(mismatch=True)).rows[0]

    assert isinstance(row.selected_in_your_run, Absence)
    assert isinstance(row.applied_update_weight, Absence)
    assert "identity" in row.applied_update_weight.reason
    assert row.applied_update_weight.remedy

    mismatched_join_identity = identity_record()
    mismatched_join_identity["extensions"][JOIN_EXTENSION]["rows"][0]["response_id"] = (
        "other-response"
    )
    assert isinstance(join(mismatched_join_identity).rows[0].applied_update_weight, Absence)


def test_dangling_evidence_identifier_becomes_an_absence_and_never_a_link() -> None:
    row = join(identity_record(dangling=True)).rows[0]

    assert isinstance(row.fix_changes_pressure, Absence)
    assert "does not resolve" in row.fix_changes_pressure.reason


def test_c061_trace_row_requires_exact_response_text_and_declared_grader_source() -> None:
    document = identity_record()
    payload = document["extensions"][JOIN_EXTENSION]
    row = payload["rows"][0]
    binding = {
        "task_id": "task-17",
        "response_digest": "sha256:" + "c" * 64,
        "response_codec": "rfc8785-response-text/1",
        "grader_source_digest": "sha256:" + "d" * 64,
    }
    row["trace_binding"] = binding
    row["trace_subject_ref"] = "sha256:" + "e" * 64
    row["trace_run_identity"] = "sha256:" + "f" * 64
    row["evidence"]["applied_update_weight"] = "trace.row.0"
    document["measurement"]["trace"].append(
        {
            "entry_id": "trace.row.0",
            "kind": "witness",
            "scope": "reconstructed_training_pressure",
            "subject_ref": row["trace_subject_ref"],
            "result": {
                "row_id": "recorded-row",
                "run_identity": row["trace_run_identity"],
                **binding,
                "grader_source": {"grader_source_digest": binding["grader_source_digest"]},
                "applied_weight": 0.25,
                "applied_weight_source": "recorded",
            },
        }
    )

    matched = join(document).rows[0]
    assert matched.applied_update_weight.value == Decimal("0.25")
    assert matched.evidence["applied_update_weight"] == "trace.row.0"

    document["measurement"]["trace"][-1]["result"]["response_digest"] = "sha256:" + "0" * 64
    mismatched = join(document).rows[0]
    assert isinstance(mismatched.applied_update_weight, Absence)
    assert "response digest" in mismatched.applied_update_weight.reason
    assert mismatched.applied_update_weight.remedy

    document["measurement"]["trace"][-1]["result"]["response_digest"] = binding["response_digest"]
    document["measurement"]["trace"][-1]["result"]["response_codec"] = "rfc8785-json/1"
    wrong_codec = join(document).rows[0]
    assert isinstance(wrong_codec.applied_update_weight, Absence)
    assert "codec" in wrong_codec.applied_update_weight.reason
    assert wrong_codec.applied_update_weight.remedy


def test_report_links_every_resolved_cell_and_the_shared_signal_to_saved_entry_ids() -> None:
    from reward_lens.render.report.join import render_join

    html = render_join(identity_record())
    assert 'href="#assay-entry-trace.update"' in html
    assert 'data-evidence-entry="trace.update"' in html
    assert 'href="#assay-entry-signal.shared"' in html
