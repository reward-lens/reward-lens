"""Public-contract negatives for the resumable improve loop.

These fail until P-IMPROVE lands its owned engine and the neutral request seam
can carry the public CLI/SDK inputs.  They are deliberately narrow: they prove
the missing request-to-engine contract rather than a local helper.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


def _live_improve_candidate(
    tmp_path: Path, *, route: str, policy: str
) -> tuple[Path, dict[str, object]]:
    """Build executable Compare inputs, not a hand-written record fixture."""
    helpers = runpy.run_path(
        str(Path(__file__).parents[1] / "compare" / "test_public_execution.py")
    )
    root = helpers["_project"](tmp_path / f"{route}-{policy}-project")
    protected = root / "candidate-protected"
    protected.mkdir()
    (protected / "held.py").write_text("candidate may not read this\n", encoding="utf-8")
    grader = root / "planner-grader.py"
    grader.write_text(
        "def score(task, response):\n    return {'left': 1.0, 'right': 0.0}[response]\n",
        encoding="utf-8",
    )
    candidate: dict[str, object] = {
        "acceptance_probe": {
            "root": str(protected),
            "item": "held.py",
            "partition_id": "improve-protected",
        },
        "planner_result": {
            "grader": str(grader),
            "task": {"id": "planner", "prompt": "separate the candidates"},
            "left_response": "left",
            "right_response": "right",
        },
    }
    if route == "selection":
        bank, config = helpers["_write_selection_stress_inputs"](root)
        policy_path = root / "qualified-policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "id": "improve-selection-qualified-v1",
                    "version": "1.0.0",
                    "actions": ["use_for_selection"],
                    "default_interval": "0.95",
                    "multiplicity": {
                        "rule": "benjamini_hochberg_fdr",
                        "family": "improve integration",
                    },
                    "checks": [
                        {
                            "key": "rung2.complete",
                            "source": "entry_complete",
                            "section": "reward_statistics",
                            "entry_id": "reward_statistics.selection_stress",
                            "fresh_for_seconds": 604800,
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        candidate.update(
            {
                "planner": {
                    "explanations": [
                        {"kind": "selection_response", "statement": "selection changes"},
                        {"kind": "advantage_pressure", "statement": "advantage changes"},
                    ],
                    "availability": ["response_bank", "outcome_check"],
                    "budget_usd": "0",
                    "retry_limit": 2,
                },
                "compare": {
                    "baseline": "before",
                    "candidate": "after",
                    "rungs": [2],
                    "bank": str(bank),
                    "selection_config": str(config),
                    "name": "improve-selection",
                },
                "policy": str(policy_path),
            }
        )
        candidate["planner_result"]["pair"] = ["selection_response", "advantage_pressure"]  # type: ignore[index]
    else:
        config = helpers["_write_transfer_config"](root)
        policy_path = root / f"{policy}-policy.json"
        source = (
            Path(__file__).parents[3]
            / "src/reward_lens/acceptance/policies"
            / ("default-evaluator-v1.json" if policy == "rejected" else "default-learning-v1.json")
        )
        policy_path.write_bytes(source.read_bytes())
        candidate.update(
            {
                "planner": {
                    "explanations": [
                        {"kind": "horizon_behavior", "statement": "horizon changes"},
                        {"kind": "transfer_generalisation", "statement": "transfer changes"},
                    ],
                    "availability": ["held_out_partitions"],
                    "budget_usd": "0",
                    "retry_limit": 2,
                },
                "compare": {
                    "baseline": "before",
                    "candidate": "after",
                    "rungs": [6],
                    "transfer_config": str(config),
                    "name": f"improve-{policy}",
                },
                "policy": str(policy_path),
            }
        )
        candidate["planner_result"]["pair"] = ["horizon_behavior", "transfer_generalisation"]  # type: ignore[index]
    return root, candidate


def test_improve_request_carries_describe_pair_and_resume_inputs(tmp_path: Path) -> None:
    """The fourth entrance and real pair interaction must reach the engine."""
    from reward_lens.api import ImproveRequest

    request = ImproveRequest(
        path=tmp_path,
        describe="prefer answers with a concrete verifier",
        candidate_source="owner_edit",
        candidate={"reward": "verifier"},
        resume="improve-0001",
        mark_pair=("a.json", "b.json"),
        why="the candidates differ on interaction handling",
    )

    assert request.describe == "prefer answers with a concrete verifier"
    assert request.mark_pair == (Path("a.json"), Path("b.json"))


def test_improve_session_persists_six_states_and_can_resume(tmp_path: Path) -> None:
    """A partial plan must be recoverable as state, not recreated from a result."""
    from reward_lens.product.improve.session import ImproveSession

    session = ImproveSession.describe(
        tmp_path,
        "prefer answers with a concrete verifier",
    )
    assert session.state == "describe"
    restored = ImproveSession.resume(session.state_path)
    assert restored.state == "describe"


def test_unresolved_tradeoff_is_a_named_refusal_with_unresolved_pairs(tmp_path: Path) -> None:
    """No feasible discriminator is an owner decision, never a collapsed winner."""
    from reward_lens.product.improve.session import ImproveRefusal, ImproveSession

    session = ImproveSession.describe(tmp_path, "behaviour")
    with pytest.raises(ImproveRefusal, match=r"IMPROVE_TRADEOFF_UNRESOLVED.*unresolved_pairs"):
        session.require_resolved_tradeoff((("horizon_behavior", "transfer_generalisation"),))


def test_mark_pair_retains_distinct_version_bytes_as_a_regression_case(tmp_path: Path) -> None:
    """A pair is evidence about two files, rather than two caller-supplied names."""
    from reward_lens.product.improve.session import ImproveSession

    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text('{"reward":"left"}\n', encoding="utf-8")
    right.write_text('{"reward":"right"}\n', encoding="utf-8")
    stored = ImproveSession.describe(tmp_path, "behaviour").mark_pair(
        left, right, "the two versions reward different verifier behaviour"
    )
    pair = stored.payload["pairs"][0]
    assert pair["regression_case"] is True
    assert pair["left"]["sha256"].startswith("sha256:")
    assert pair["left"]["sha256"] != pair["right"]["sha256"]
    assert ImproveSession.resume(stored.state_path).payload["pairs"] == [pair]


def test_identical_pair_is_refused_without_creating_a_regression_case(tmp_path: Path) -> None:
    from reward_lens.product.improve.session import ImproveRefusal, ImproveSession

    source = tmp_path / "version.json"
    source.write_text('{"reward":"same"}\n', encoding="utf-8")
    duplicate = tmp_path / "same-bytes.json"
    duplicate.write_bytes(source.read_bytes())
    session = ImproveSession.describe(tmp_path, "behaviour")
    with pytest.raises(ImproveRefusal, match="identical bytes"):
        session.mark_pair(source, duplicate, "must differ")


@pytest.mark.parametrize(
    ("route", "policy", "expected"),
    [
        ("selection", "qualified", "qualified"),
        ("transfer", "rejected", "rejected"),
        ("transfer", "unresolved", "unresolved"),
    ],
)
def test_public_sdk_persists_and_reopens_each_policy_outcome(
    tmp_path: Path,
    route: str,
    policy: str,
    expected: str,
) -> None:
    """A real Compare run, immutable policy and Audit bundle determine each state."""
    from reward_lens import api

    root, candidate = _live_improve_candidate(tmp_path, route=route, policy=policy)
    record = api.improve(
        api.ImproveRequest(
            path=root,
            describe="prefer independently checked behaviour",
            candidate_source="owner_edit",
            candidate=candidate,
            name=policy,
        )
    )
    saved = root / "assays" / f"improve-{policy}.assay.json"
    reopened = api.open_record(saved)
    extension = reopened.extensions["https://reward-lens.github.io/extensions/improve/v1"]

    assert record.decision.state == reopened.decision.state == expected
    assert reopened.decision.policy is not None
    assert reopened.experiments[-1].informative is True
    assert extension["planner_result"]["informative"] is True
    assert Path(extension["planner_result"]["artifact"]).is_file()
    assert extension["acceptance_boundary"]["access_code"] == "ACCEPTANCE_ACCESS_DENIED"


def test_public_cli_uses_the_same_persisted_qualified_route(tmp_path: Path) -> None:
    """The supported terminal entrance reaches Compare, Policy and the saved Audit record."""
    from click.testing import CliRunner

    from reward_lens.cli.main import cli

    root, candidate = _live_improve_candidate(tmp_path, route="selection", policy="qualified")
    candidate_path = root / "candidate.json"
    candidate_path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "improve",
            str(root),
            "--describe",
            "prefer checked behaviour",
            "--candidate-source",
            "owner_edit",
            "--candidate",
            str(candidate_path),
            "--name",
            "cli-qualified",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    from reward_lens import api

    reopened = api.open_record(root / payload["artifacts"]["assay"])
    assert reopened.decision.state == "qualified"
    assert (
        reopened.extensions["https://reward-lens.github.io/extensions/improve/v1"][
            "planner_result"
        ]["informative"]
        is True
    )


@pytest.mark.parametrize("at_state", ["propose", "plan"])
def test_public_resume_corrects_only_the_incomplete_stage_without_duplicate_planner_effects(
    tmp_path: Path,
    at_state: str,
) -> None:
    """A corrected request resumes the saved declaration and one planner registration."""
    from reward_lens import api, contracts
    from reward_lens.product.improve import state

    root, candidate = _live_improve_candidate(tmp_path, route="transfer", policy="rejected")
    invalid = json.loads(json.dumps(candidate))
    if at_state == "propose":
        invalid["planner"].pop("availability")
    else:
        invalid["compare"]["transfer_config"] = str(root / "not-yet-corrected.json")
    with pytest.raises(contracts.UsageError) as caught:
        api.improve(
            api.ImproveRequest(
                path=root,
                describe="resume an incomplete request",
                candidate_source="owner_edit",
                candidate=invalid,
                name="resume",
            )
        )
    assert caught.value.code == "RL0003" and caught.value.context["field"] == "candidate"
    partial = state.read(root / ".reward-lens" / "improve" / "current.json")
    assert partial["state"] == at_state

    record = api.improve(
        api.ImproveRequest(
            path=root,
            resume="current",
            candidate_source="owner_edit",
            candidate=candidate,
            name="resume",
        )
    )
    ledger = [
        json.loads(line)
        for line in (root / ".reward-lens" / "improve" / "planner.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert record.decision.state == "rejected"
    assert [entry["kind"] for entry in ledger] == ["choice", "result"]
    assert record.experiments[-1].informative is True


def test_public_resume_execute_reuses_a_committed_planner_result(tmp_path: Path) -> None:
    """An interruption after ``record_result`` does not execute the paired grader twice."""
    from reward_lens import api
    from reward_lens.product.improve import state

    root, candidate = _live_improve_candidate(tmp_path, route="transfer", policy="rejected")
    first = api.improve(
        api.ImproveRequest(
            path=root,
            describe="recover committed evidence",
            candidate_source="owner_edit",
            candidate=candidate,
            name="execute-resume",
        )
    )
    path = root / ".reward-lens" / "improve" / "current.json"
    interrupted = state.read(path)
    execution = dict(interrupted["execution"])
    execution.pop("planner_result")
    state.write(
        path, {**interrupted, "state": "execute", "execution": execution, "experiments": []}
    )

    resumed = api.improve(
        api.ImproveRequest(
            path=root,
            resume="current",
            candidate_source="owner_edit",
            candidate=candidate,
            name="execute-resume",
        )
    )
    ledger = [
        json.loads(line)
        for line in (root / ".reward-lens" / "improve" / "planner.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert resumed.assay_id == first.assay_id
    assert [entry["kind"] for entry in ledger] == ["choice", "result"]


def test_public_sdk_persists_immutable_expectation_before_actual_compare(tmp_path, monkeypatch):
    """Observe the real public Compare boundary, preserving its actual implementation."""
    from reward_lens import api
    from reward_lens.product.improve import state

    root, candidate = _live_improve_candidate(tmp_path, route="selection", policy="qualified")
    original_compare = api.compare
    observed = []

    def observe(request):
        payload = state.read(root / ".reward-lens/improve/current.json")
        registration = payload.get("preregistration")
        assert registration is not None, (
            "actual Compare started before immutable expectation was persisted"
        )
        path = Path(registration["path"])
        raw = path.read_bytes()
        import hashlib

        assert registration["digest"] == "sha256:" + hashlib.sha256(raw).hexdigest()
        declared = json.loads(raw)
        assert declared["planner_choice"]["expected_outcomes"]
        assert declared["compare_request"] == request.model_dump(mode="json")
        assert not list((root / "assays").glob("improve-selection.assay.json"))
        observed.append(raw)
        return original_compare(request)

    monkeypatch.setattr(api, "compare", observe)
    result = api.improve(
        api.ImproveRequest(
            path=root,
            describe="register before experiment",
            candidate_source="owner_edit",
            candidate=candidate,
            name="preregister",
        )
    )
    assert len(observed) == 1
    extension = result.extensions["https://reward-lens.github.io/extensions/improve/v1"]
    registration = extension["preregistration"]
    assert Path(registration["path"]).read_bytes() == observed[0]
    assert result.experiments[-1].registered_at == registration["registered_at"]
    from datetime import datetime

    assert datetime.fromisoformat(registration["registered_at_precise"]) < datetime.fromisoformat(
        extension["experiment_ordering"]["started_at"]
    )
    assert (
        extension["experiment_ordering"]["completed_at"]
        > extension["experiment_ordering"]["started_at"]
    )


@pytest.mark.parametrize(
    "fault",
    (
        "registration_bytes",
        "registration_timestamp",
        "compare_input_bytes",
        "candidate_declaration",
    ),
)
def test_public_resume_refuses_mutating_preregistered_inputs_before_experiment(
    tmp_path, monkeypatch, fault
):
    from reward_lens import api, contracts
    from reward_lens.product.improve import state

    root, candidate = _live_improve_candidate(tmp_path, route="selection", policy="qualified")
    calls = []

    def interrupt(request):
        calls.append(request)
        raise RuntimeError("controlled interruption immediately before actual Compare")

    monkeypatch.setattr(api, "compare", interrupt)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        api.improve(
            api.ImproveRequest(
                path=root,
                describe="freeze expectation",
                candidate_source="owner_edit",
                candidate=candidate,
            )
        )
    state_path = root / ".reward-lens/improve/current.json"
    payload = state.read(state_path)
    registration = payload["preregistration"]
    registered_path = Path(registration["path"])
    original = registered_path.read_bytes()
    corrected = None
    if fault == "registration_bytes":
        registered_path.write_bytes(original + b" ")
    elif fault == "registration_timestamp":
        payload["preregistration"]["registered_at"] = "2000-01-01T00:00:00Z"
        state.write(state_path, payload)
    elif fault == "compare_input_bytes":
        config = Path(candidate["compare"]["selection_config"])
        config.write_bytes(config.read_bytes() + b" ")
    else:
        corrected = json.loads(json.dumps(candidate))
        corrected["compare"]["name"] = "different-comparison"
    with pytest.raises(contracts.UsageError, match="(preregister|immutable)") as caught:
        api.improve(
            api.ImproveRequest(
                path=root,
                resume="current",
                candidate_source="owner_edit" if corrected else None,
                candidate=corrected,
            )
        )
    assert caught.value.code == "RL0003" and caught.value.context["field"] == "resume"
    assert len(calls) == 1
    assert not (root / "assays/improve-selection.assay.json").exists()


def _t3_improve_request(tmp_path):
    from reward_lens import api, contracts
    from reward_lens.product.improve.session import _compare_request
    from reward_lens.store import Project

    root, candidate = _live_improve_candidate(tmp_path, route="selection", policy="qualified")
    base = api.ImproveRequest(
        path=root,
        describe="prospective contrast check",
        candidate_source="owner_edit",
        candidate=candidate,
        name="forecast",
        seed=41,
        offline=True,
        max_budget_usd="0.00",
        non_interactive=True,
    )
    import inspect

    options = (
        {"outer_request": base}
        if "outer_request" in inspect.signature(_compare_request).parameters
        else {}
    )
    compare, _, _ = _compare_request(root, candidate, project=Project.open(root), **options)
    config = json.loads(Path(candidate["compare"]["selection_config"]).read_text())
    helpers = runpy.run_path(str(Path(__file__).parents[1] / "test_forecast.py"))
    issue = helpers["_issue_row"]("T3")
    issue.update(checkpoint=0, source_cutoff=0, feature_cutoff=0, model_selection_cutoff=0)
    for value in issue["baseline_provenance"].values():
        value["cutoff"] = 0
    prediction = root / "prospective-contrast.json"
    prediction.write_text(json.dumps({"spec": {"horizon": 1}, "issues": [issue]}))
    forecast = {
        "record": str(prediction),
        "claim": "next registered window reward-group contrast",
        "target": "T3",
        "outcome": {
            "section": "signal",
            "entry_id": "signal.contrast_fraction",
            "measurand": "fraction of declared reward groups with any within-group applied-reward contrast",
            "unit": "fraction",
            "sampling_unit": "task",
        },
        "context": {
            "subject_ref": config["outcome"]["subject_ref"],
            "task_set": Project.open(root).version().to_subject_digests().task_distribution,
            "window": {"checkpoint": 0, "start": 1, "end": 1},
            "configuration_digest": contracts.digest(compare.model_dump(mode="json")),
        },
    }
    return root, base.model_copy(update={"forecast": forecast}), forecast


def test_public_forecast_issues_before_compare_and_missing_numeric_outcome_stays_absent(
    tmp_path, monkeypatch
):
    from reward_lens import api

    root, request, forecast = _t3_improve_request(tmp_path)
    original_compare = api.compare
    observed = []

    def compare(actual):
        ledger = root / ".reward-lens/forecast-ledger.jsonl"
        assert ledger.exists(), "real public forecast issue did not run before actual Compare"
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert [row["kind"] for row in rows] == ["header", "issue"]
        frozen = json.loads(rows[1]["issued_bytes"])
        observed.append(rows[1]["issued_bytes"])
        assert frozen["target"] == "T3"
        # A changed caller-side predictor file cannot change the already issued bytes.
        Path(forecast["record"]).write_text('{"issues":[]}')
        assert actual.seed == request.seed and actual.offline == request.offline
        assert actual.sandbox == request.sandbox and actual.require_tier == request.require_tier
        assert actual.max_budget_usd == request.max_budget_usd
        return original_compare(actual)

    monkeypatch.setattr(api, "compare", compare)
    result = api.improve(request)
    extension = result.extensions["https://reward-lens.github.io/extensions/improve/v1"]
    receipt = extension["forecast"]
    assert receipt["status"] == "COULD_NOT_CHECK"
    assert receipt["subject_ref"] == result.subject.version.digest
    entry = next(
        entry
        for entry in result.measurement.forecast
        if entry.entry_id == "forecast.improve_t3_outcome"
    )
    assert entry.kind == "absence" and entry.absence.state == "COULD_NOT_CHECK"
    assert entry.subject_ref == result.subject.version.digest
    rows = [
        json.loads(line)
        for line in (root / ".reward-lens/forecast-ledger.jsonl").read_text().splitlines()
    ]
    assert len(observed) == 1 and rows[1]["issued_bytes"] == observed[0]
    assert not any(row["kind"] == "resolve" for row in rows)
    reopened = api.open_record(root / "assays/improve-forecast.assay.json")
    assert (
        reopened.assay_id == result.assay_id
        and reopened.extensions["https://reward-lens.github.io/extensions/improve/v1"]["forecast"]
        == receipt
    )
    assert extension["request_metadata"]["seed"] == 41
    assert extension["request_metadata"]["max_budget_usd"] == "0.00"
    assert result.cost.usd == "0.00"


@pytest.mark.parametrize(
    "fault",
    (
        "literal_outcome",
        "wrong_target",
        "wrong_window",
        "wrong_configuration",
        "future_cutoff",
        "wrong_subject",
    ),
)
def test_public_forecast_refuses_unbound_or_future_inputs(tmp_path, monkeypatch, fault):
    from reward_lens import api, contracts

    root, request, forecast = _t3_improve_request(tmp_path)
    if fault == "literal_outcome":
        forecast["outcome"]["value"] = 0.5
    elif fault == "wrong_target":
        forecast["target"] = "T1"
    elif fault == "wrong_window":
        forecast["context"]["window"]["start"] = 0
    elif fault == "wrong_configuration":
        forecast["context"]["configuration_digest"] = "sha256:" + "0" * 64
    elif fault == "future_cutoff":
        source = Path(forecast["record"])
        document = json.loads(source.read_text())
        document["issues"][0]["feature_cutoff"] = 1
        source.write_text(json.dumps(document))
    else:
        forecast["context"]["subject_ref"] = "sha256:" + "0" * 64
    called = []
    original_compare = api.compare

    def compare(actual):
        called.append(actual)
        return original_compare(actual)

    monkeypatch.setattr(api, "compare", compare)
    with pytest.raises(contracts.UsageError) as caught:
        api.improve(request)
    assert caught.value.code == "RL0003"
    assert caught.value.context["field"] == "forecast"
    if fault != "wrong_subject":
        assert called == []


def test_public_outer_zero_budget_cannot_execute_paid_registered_rung(tmp_path, monkeypatch):
    from reward_lens import api, contracts

    root, candidate = _live_improve_candidate(tmp_path, route="selection", policy="qualified")
    candidate["planner"].update(
        explanations=[
            {"kind": "update_mechanism", "statement": "one update"},
            {"kind": "horizon_behavior", "statement": "later behavior"},
        ],
        availability=["pinned_trainer", "checkpoint", "gpu", "horizon_budget"],
        budget_usd="100",
    )
    called = []
    monkeypatch.setattr(api, "compare", lambda request: called.append(request))
    with pytest.raises(contracts.UsageError) as caught:
        api.improve(
            api.ImproveRequest(
                path=root,
                describe="outer budget is binding",
                candidate_source="owner_edit",
                candidate=candidate,
                max_budget_usd="0.00",
            )
        )
    assert caught.value.code == "RL0003" and caught.value.context["field"] == "mark_pair"
    assert called == []
    assert not (root / ".reward-lens/forecast-ledger.jsonl").exists()


def test_public_forecast_resume_recovers_one_committed_issue_without_reissuing(
    tmp_path, monkeypatch
):
    from reward_lens import api

    root, request, forecast = _t3_improve_request(tmp_path)
    original = api.forecast_issue
    calls = []

    def interrupt(actual):
        calls.append(actual)
        original(actual)
        raise RuntimeError("controlled interruption after committed Forecast issue")

    monkeypatch.setattr(api, "forecast_issue", interrupt)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        api.improve(request)
    monkeypatch.setattr(api, "forecast_issue", original)
    resumed = request.model_copy(update={"describe": None, "resume": "current"})
    result = api.improve(resumed)
    receipt = result.extensions["https://reward-lens.github.io/extensions/improve/v1"]["forecast"]
    assert receipt["issue"]["recovered_after_interruption"] is True
    assert len(calls) == 1
    rows = [
        json.loads(line)
        for line in (root / ".reward-lens/forecast-ledger.jsonl").read_text().splitlines()
    ]
    assert [row["kind"] for row in rows] == ["header", "issue"]
    assert receipt["status"] == "COULD_NOT_CHECK"


def test_public_cli_forecast_uses_actual_loop_and_preserves_shared_metadata(tmp_path):
    from click.testing import CliRunner

    from reward_lens import api
    from reward_lens.cli.main import cli

    root, request, forecast = _t3_improve_request(tmp_path)
    candidate_file = root / "candidate.json"
    forecast_file = root / "forecast.json"
    candidate_file.write_text(json.dumps(dict(request.candidate)))
    forecast_file.write_text(json.dumps(forecast))
    result = CliRunner().invoke(
        cli,
        [
            "improve",
            str(root),
            "--describe",
            request.describe,
            "--candidate-source",
            "owner_edit",
            "--candidate",
            str(candidate_file),
            "--forecast",
            str(forecast_file),
            "--seed",
            "41",
            "--max-budget-usd",
            "0.00",
            "--non-interactive",
            "--name",
            "forecast",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    record = api.open_record(root / json.loads(result.output)["artifacts"]["assay"])
    extension = record.extensions["https://reward-lens.github.io/extensions/improve/v1"]
    assert extension["request_metadata"]["seed"] == 41
    assert extension["request_metadata"]["non_interactive"] is True
    assert extension["forecast"]["status"] == "COULD_NOT_CHECK"
    assert not any(
        json.loads(line)["kind"] == "resolve"
        for line in (root / ".reward-lens/forecast-ledger.jsonl").read_text().splitlines()
    )


def test_public_outer_request_metadata_is_preserved_in_actual_compare(tmp_path, monkeypatch):
    from reward_lens import api

    root, candidate = _live_improve_candidate(tmp_path, route="selection", policy="qualified")
    original = api.compare
    actual = []

    def observe(request):
        actual.append(request)
        return original(request)

    monkeypatch.setattr(api, "compare", observe)
    request = api.ImproveRequest(
        path=root,
        describe="keep explicit outer metadata",
        candidate_source="owner_edit",
        candidate=candidate,
        seed=73,
        offline=False,
        require_tier="L1",
        max_budget_usd="0.00",
        non_interactive=True,
    )
    record = api.improve(request)
    assert len(actual) == 1
    for name in ("seed", "offline", "sandbox", "require_tier", "max_budget_usd", "non_interactive"):
        assert getattr(actual[0], name) == getattr(request, name)
    assert record.provenance.seed == 73 and record.provenance.offline is False
    assert (
        record.extensions["https://reward-lens.github.io/extensions/improve/v1"][
            "request_metadata"
        ]["require_tier"]
        == "L1"
    )


def _measured_t3_improve_request(tmp_path):
    root, request, forecast = _t3_improve_request(tmp_path)
    candidate = dict(request.candidate)
    candidate["forecast_measurement"] = {
        "profile": "finite-bank-selector-recorded-groups/1",
        "capture_path": str(root / ".reward-lens/improve/recorded-groups"),
    }
    return root, request.model_copy(update={"candidate": candidate}), forecast


def test_public_measured_t3_resolves_actual_reward_groups_and_four_baselines(tmp_path, monkeypatch):
    from datetime import datetime

    from reward_lens import api
    from reward_lens.product.improve import state

    root, request, forecast = _measured_t3_improve_request(tmp_path)
    actual_compare = api.compare
    observations = []

    def compare(actual):
        payload = state.read(root / ".reward-lens/improve/current.json")
        registration = payload.get("forecast_measurement_registration")
        assert registration is not None, (
            "actual experiment started before native measurement registration"
        )
        bound = json.loads(Path(registration["path"]).read_bytes())
        assert bound["forecast_id"] == payload["forecast_issue"]["sealed_digest"]
        assert datetime.fromisoformat(bound["registered_at"]) > datetime.fromisoformat(
            payload["forecast_issue"]["observed_at"]
        )
        assert not Path(bound["capture_path"]).exists()
        observations.append(actual.model_dump(mode="json"))
        return actual_compare(actual)

    monkeypatch.setattr(api, "compare", compare)
    record = api.improve(request)
    receipt = record.extensions["https://reward-lens.github.io/extensions/improve/v1"]["forecast"]
    assert receipt["status"] == "RESOLVED", (
        "actual numeric T3 was not resolved from measured groups"
    )
    assert len(observations) == 1
    measured = receipt["measured_outcome"]
    assert measured["profile"] == "recorded-reward-groups/1"
    assert measured["target"] == "T3" and measured["sampling_unit"] == "task"
    assert measured["outcome_fraction"] == [1, 1]
    assert measured["steps"][0]["eligible_groups"] == measured["steps"][0]["contrasted_groups"] == 1
    assert measured["context"] == forecast["context"]
    assert receipt["subject_ref"] == record.subject.version.digest
    assert "no trainer execution" in receipt["claim_boundary"]
    assert datetime.fromisoformat(measured["collection_started_at"]) > datetime.fromisoformat(
        json.loads(Path(measured["registration"]).read_text())["registered_at"]
    )
    assert datetime.fromisoformat(measured["collection_finished_at"]) <= datetime.fromisoformat(
        measured["measured_at"]
    )
    ledger_path = root / ".reward-lens/forecast-ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["header", "issue", "resolve"]
    resolution = rows[-1]
    assert set(resolution["baseline_scores"]) == {
        "persistence",
        "climatology",
        "drift",
        "covariance",
    }
    assert (
        resolution["sealed_digest"] == measured["forecast_id"] == receipt["issue"]["sealed_digest"]
    )
    assert resolution["outcome"] == measured["outcome"]
    capture_root = Path(measured["capture_path"])
    from reward_lens.product.trace import follow

    captured = [row for chunk in follow(capture_root, policy="strict") for row in chunk.rows]
    assert len(captured) == 4
    assert all(row["reward_applied"] == row["scoring_envelope"]["score"] for row in captured)
    from reward_lens.graders.base import EvidenceEnvelope

    assert all(
        row["scoring_envelope_digest"]
        == EvidenceEnvelope.from_dict(row["scoring_envelope"]).digest()
        for row in captured
    )
    assert all(
        row["subject_ref"] == record.subject.version.digest
        and row["task_set"] == forecast["context"]["task_set"]
        for row in captured
    )
    reopened = api.open_record(root / "assays/improve-forecast.assay.json")
    assert reopened.assay_id == record.assay_id
    entry = next(
        entry
        for entry in reopened.measurement.forecast
        if entry.entry_id == "forecast.improve_t3_score"
    )
    assert entry.kind == "estimate" and entry.state == "complete"
    assert entry.subject_ref == record.subject.version.digest
    assert entry.result["baseline_scores"] == resolution["baseline_scores"]
    assert entry.result["outcome"] == measured["outcome"]
    assert entry.result["skill_claim"] is False
    assert record.cost.usd == "0.00"


@pytest.mark.parametrize("fault", ("profile", "preloaded_capture", "literal_outcome"))
def test_public_measured_profile_refuses_caller_shortcuts_before_issue(tmp_path, fault):
    from reward_lens import api, contracts

    root, request, forecast = _measured_t3_improve_request(tmp_path)
    candidate = dict(request.candidate)
    declaration = candidate["forecast_measurement"]
    if fault == "profile":
        declaration["profile"] = "caller-scalar"
    elif fault == "preloaded_capture":
        path = Path(declaration["capture_path"])
        path.mkdir(parents=True)
        (path / "outcome.json").write_text('{"value":1}')
    else:
        declaration["value"] = 1.0
    with pytest.raises(contracts.UsageError) as caught:
        api.improve(request.model_copy(update={"candidate": candidate}))
    detail = {
        "profile": "recorded reward groups require the explicit finite-bank selector profile and rung 2 inputs",
        "preloaded_capture": "recorded-group capture must be a new absolute directory beneath the project",
        "literal_outcome": "forecast.measurement has undeclared or missing fields",
    }[fault]
    refusal = f"IMPROVE_RESUME_REQUIRED: {detail}"
    assert caught.value.code == "RL0003"
    assert caught.value.exit_code == 4
    assert caught.value.message == f"improve: {refusal}"
    assert caught.value.context == {
        "field": "improve",
        "refusal": "IMPROVE_RESUME_REQUIRED",
        "detail": refusal,
    }
    assert not (root / ".reward-lens/forecast-ledger.jsonl").exists()
    assert not (root / "assays/improve-selection.assay.json").exists()


def test_public_measured_source_tamper_refuses_resolution_and_resume_spends_no_second_comparison(
    tmp_path, monkeypatch
):
    from reward_lens import api
    from reward_lens.forecast.evidence import ForecastRefusal
    from reward_lens.product.forecast import outcomes

    root, request, forecast = _measured_t3_improve_request(tmp_path)
    original_capture = outcomes.capture_measurement
    original_compare = api.compare
    compared = []
    captured = []

    def compare(actual):
        compared.append(actual)
        return original_compare(actual)

    def break_capture(*args, **kwargs):
        result = original_capture(*args, **kwargs)
        chunk = result.parent / "chunk_00001.jsonl"
        original = chunk.read_bytes()
        captured.append((chunk, original))
        chunk.write_bytes(original + b" ")
        return result

    monkeypatch.setattr(api, "compare", compare)
    monkeypatch.setattr(outcomes, "capture_measurement", break_capture)
    with pytest.raises(ForecastRefusal, match="TAMPER"):
        api.improve(request)
    ledger = root / ".reward-lens/forecast-ledger.jsonl"
    assert [json.loads(line)["kind"] for line in ledger.read_text().splitlines()] == [
        "header",
        "issue",
    ]
    assert not (root / "assays/improve-forecast.assay.json").exists()
    assert len(compared) == 1 and len(captured) == 1
    chunk, original = captured[0]
    chunk.write_bytes(original)
    monkeypatch.setattr(outcomes, "capture_measurement", original_capture)
    resumed = api.improve(request.model_copy(update={"describe": None, "resume": "current"}))
    receipt = resumed.extensions["https://reward-lens.github.io/extensions/improve/v1"]["forecast"]
    assert receipt["status"] == "RESOLVED"
    assert len(compared) == 1 and len(captured) == 1
    assert [json.loads(line)["kind"] for line in ledger.read_text().splitlines()] == [
        "header",
        "issue",
        "resolve",
    ]
    planner = [
        json.loads(line)
        for line in (root / ".reward-lens/improve/planner.jsonl").read_text().splitlines()
    ]
    assert [row["kind"] for row in planner] == ["choice", "result"]


def test_public_measured_resolution_interruption_reuses_source_and_original_ledger_row(
    tmp_path, monkeypatch
):
    from reward_lens import api

    root, request, forecast = _measured_t3_improve_request(tmp_path)
    original = api.forecast_resolve
    calls = []

    def interrupt(actual):
        calls.append(actual)
        original(actual)
        raise RuntimeError("controlled interruption after actual native resolution")

    monkeypatch.setattr(api, "forecast_resolve", interrupt)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        api.improve(request)
    monkeypatch.setattr(api, "forecast_resolve", original)
    resumed = api.improve(request.model_copy(update={"describe": None, "resume": "current"}))
    receipt = resumed.extensions["https://reward-lens.github.io/extensions/improve/v1"]["forecast"]
    assert receipt["status"] == "RESOLVED"
    assert len(calls) == 1
    rows = [
        json.loads(line)
        for line in (root / ".reward-lens/forecast-ledger.jsonl").read_text().splitlines()
    ]
    assert [row["kind"] for row in rows] == ["header", "issue", "resolve"]
