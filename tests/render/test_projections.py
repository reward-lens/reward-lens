"""Projection contract tests from persisted Assay producers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _at(document: object, pointer: str) -> object:
    current = document
    for token in pointer.removeprefix("/").split("/") if pointer else ():
        token = token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def _persisted_assay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use Audit, policy finalization, trace/five-way, and Join producers, then reopen bytes."""
    from reward_lens import api
    from reward_lens.acceptance import apply_and_persist, load_policy
    from reward_lens.contracts import Assay

    subject = tmp_path / "grader.py"
    subject.write_text("def score(task, response):\n    return 1.0\n", encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"id":"task-1","prompt":"find a response"}\n', encoding="utf-8")
    seeker = tmp_path / "seeker.json"
    seeker.write_text(
        '{"mode":"agent","model":"fixture","max_proposals":1,"recorded_proposals":["paid-but-wrong"]}',
        encoding="utf-8",
    )
    source = tmp_path / "real-run"
    fixture = Path(__file__).parents[1] / "fixtures" / "grpo_run" / "short"
    shutil.copytree(fixture, source)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assembled = api.audit(
        api.AuditRequest(
            path=subject,
            tasks=tasks,
            run_dir=source,
            seeker="agent",
            seeker_config=seeker,
            name="projection-source",
        )
    )
    policy = load_policy(
        Path(__file__).parents[2] / "src/reward_lens/acceptance/policies/default-evaluator-v1.json"
    )
    saved = apply_and_persist(
        assembled,
        policy,
        project=None,
        name="projection-finalized",
        fallback_dir=tmp_path / "finalized",
    )
    assay_path = saved.paths["assay"]
    reopened = Assay.model_validate(json.loads(assay_path.read_text(encoding="utf-8")))
    return reopened, json.loads(assay_path.read_text(encoding="utf-8"))


def _persisted_case_assay(tmp_path: Path):
    """Run public Rung 3, then reopen the exact bundle that supplied Case evidence."""
    from reward_lens import api
    from reward_lens.contracts import Assay

    run_dir = tmp_path / "closed-run"
    fixture = Path(__file__).parents[1] / "fixtures" / "grpo_run" / "short"
    shutil.copytree(fixture, run_dir)
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps({"id": "baseline", "weights": {"length_reward": "1"}, "offset": "0"}),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps({"id": "candidate", "weights": {"length_reward": "1"}, "offset": "-3"}),
        encoding="utf-8",
    )
    record = api.compare(
        api.CompareRequest(
            baseline="baseline",
            candidate="candidate",
            rungs=(3,),
            run_dir=run_dir,
            baseline_config=baseline,
            candidate_config=candidate,
            name="projection-case",
        )
    )
    assay_path = tmp_path / "assays" / "projection-case.assay.json"
    reopened = Assay.model_validate_json(assay_path.read_bytes())
    assert reopened.assay_id == record.assay_id
    return reopened, json.loads(assay_path.read_text(encoding="utf-8"))


def test_regenerate_names_the_missing_case_producer_field(tmp_path, monkeypatch) -> None:
    from reward_lens.render.projections import ProjectionFieldAbsent, regenerate

    record, _document = _persisted_assay(tmp_path, monkeypatch)

    with pytest.raises(ProjectionFieldAbsent) as raised:
        regenerate(record)

    assert raised.value.code == "RL0640"
    assert raised.value.context == {
        "view": "case",
        "field": "decisive_pair",
        "pointer": "/extensions/https:~1~1reward-lens.github.io~1case~1v1/decisive_pair",
    }


def test_case_refuses_an_aggregate_cost_without_a_candidate_cost_producer(
    tmp_path, monkeypatch
) -> None:
    from reward_lens.render.projections import ProjectionFieldAbsent, case

    record, _document = _persisted_assay(tmp_path, monkeypatch)
    with pytest.raises(ProjectionFieldAbsent) as raised:
        case(record)

    assert raised.value.code == "RL0640"
    assert raised.value.context["view"] == "case"
    assert raised.value.context["field"] == "decisive_pair"


def test_case_lifts_only_actual_public_replay_evidence_from_the_reopened_bundle(tmp_path) -> None:
    from reward_lens.render.projections import Projection, case

    record, document = _persisted_case_assay(tmp_path)
    projection = case(record)

    assert type(projection) is Projection
    assert projection.fields["assay_id"] == record.assay_id
    assert len(projection.fields["decisive_pair"]) == 2
    assert len(projection.fields["candidate_costs"]) == 2
    for value, pointer in zip(projection.fields.values(), projection.source_pointers, strict=True):
        assert _at(document, pointer) == value


def test_available_views_share_one_projection_type_and_resolve_every_pointer(
    tmp_path, monkeypatch
) -> None:
    from reward_lens.render.projections import Projection, gate, incident, registry, report

    record, document = _persisted_assay(tmp_path, monkeypatch)
    views = (gate(record), report(record), incident(record), registry(record))

    assert all(type(view) is Projection for view in views)
    for projection in views:
        assert projection.fields["assay_id"] in projection.summary
        assert len(projection.fields) == len(projection.source_pointers)
        for value, pointer in zip(
            projection.fields.values(), projection.source_pointers, strict=True
        ):
            assert _at(document, pointer) == value


def test_gate_lifts_the_saved_policy_verdict_and_findings(tmp_path, monkeypatch) -> None:
    from reward_lens.render.projections import gate

    record, _document = _persisted_assay(tmp_path, monkeypatch)
    ci = gate(record)

    assert {"verdict", "policy_id", "policy_digest", "open_findings"} <= set(ci.fields)


def test_report_calls_the_one_join_implementation(tmp_path, monkeypatch) -> None:
    from reward_lens.product import join as join_module
    from reward_lens.render.projections import report

    record, _document = _persisted_assay(tmp_path, monkeypatch)
    calls = []
    original = join_module.join

    def observed(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(join_module, "join", observed)
    stranger = report(record)

    assert "join" in stranger.fields and calls


def test_incident_lifts_saved_trace_statistics_and_fiveway_evidence(tmp_path, monkeypatch) -> None:
    from reward_lens.render.projections import incident

    record, _document = _persisted_assay(tmp_path, monkeypatch)
    investigator = incident(record)

    assert {"trace", "reward_statistics", "fiveway_answer", "fiveway_evidence"} <= set(
        investigator.fields
    )


def test_registry_lifts_persisted_assay_identity(tmp_path, monkeypatch) -> None:
    from reward_lens.render.projections import registry

    record, _document = _persisted_assay(tmp_path, monkeypatch)
    entry = registry(record)

    assert {"assay_id", "subject", "decision"} <= set(entry.fields)


def test_projection_field_absence_names_view_and_field_from_a_real_saved_audit(
    tmp_path, monkeypatch
) -> None:
    from reward_lens import api
    from reward_lens.render.projections import ProjectionFieldAbsent, incident

    subject = tmp_path / "grader.py"
    subject.write_text("def score(task, response):\n    return 1.0\n", encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"id":"task-1","prompt":"find a response"}\n', encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    record = api.audit(api.AuditRequest(path=subject, tasks=tasks, name="projection-no-trace"))
    with pytest.raises(ProjectionFieldAbsent) as raised:
        incident(record)

    error = raised.value
    assert error.code == "RL0640"
    assert error.exit_status == 4
    assert "incident" in error.message.lower()
    assert "fiveway" in error.message


def test_no_per_view_schema_exists() -> None:
    source = Path(__file__).parents[2] / "src/reward_lens/render/projections"
    names = (
        "Case" + "View",
        "Gate" + "View",
        "Report" + "View",
        "Incident" + "View",
        "Registry" + "View",
    )
    for name in names:
        found = subprocess.run(
            ["/usr/bin/grep", "-R", "--", name, str(source)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert found.returncode == 1, found.stdout
