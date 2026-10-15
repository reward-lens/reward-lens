"""Known-answer controls for the rule-table experiment planner."""

from __future__ import annotations

import json
import subprocess
import time
from decimal import Decimal

import pytest

from reward_lens.commons.corpus import Case, CaseConclusion, CorpusStore
from reward_lens.commons.corpus.protocol import PlanningAnnotation, load_planner_cases
from reward_lens.contracts import Assay, Counters, digest
from reward_lens.execution import Limits, Result
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.product.improve import (
    MANDATORY_ORDERINGS,
    DiscriminationCell,
    EvaluatorContract,
    ExecutionEvidence,
    Explanation,
    HmacEvidenceAuthenticator,
    PlannerLedger,
    PlannerRefusal,
    RungCost,
    TrustedEvaluator,
    load_discrimination_table,
    plan,
    record_result,
)
from reward_lens.store import Project
from tests.store.conftest import build_record, write_project
from tests.store.test_project import align


def _live(*kinds: str) -> tuple[Explanation, ...]:
    return tuple(Explanation(kind, f"test explanation for {kind}") for kind in kinds)


def _availability(*access: str) -> frozenset[str]:
    return frozenset(access)


R1 = ("frozen_samples", "two_versions", "response_bank")
R2 = ("response_bank", "outcome_check")


TRUSTED = TrustedEvaluator(
    contract=EvaluatorContract(
        evaluator_id="known-answer-evaluator",
        implementation_digest="sha256:" + "a" * 64,
        rule_id="paired-grader-envelope-v1",
        key_id="known-answer-key-v1",
    ),
    authenticator=HmacEvidenceAuthenticator(
        b"known answer evaluator key", issuer="known-answer-evaluator"
    ),
)


def _ledger(tmp_path):
    return PlannerLedger(tmp_path / "planner-ledger.jsonl", evaluators=(TRUSTED,))


def _saved_planner_corpus(tmp_path):
    project = Project.open(write_project(tmp_path / "case-project"))
    assay = Assay.model_validate(align(build_record(), project))
    project.record(assay, name="measured")
    corpus = CorpusStore(tmp_path / "case-corpus")
    saved = corpus.save(
        Case(
            id="planner-saved-case",
            failure_or_shortfall="The observed selection shift required a declared replay.",
            mechanism_evidence=("A frozen replay changed the selection response.",),
            candidate_revisions=("Replay selection under the declared cost ceiling.",),
            intended_task_effects=("Preserve the recorded target task behavior.",),
            costs=("Replay cost: 0.00 USD and no provider calls.",),
            rejected_alternatives=("Do not infer an update mechanism from selection alone.",),
            stopping_conditions=("Stop when the replay separates the two accounts.",),
            conclusion=CaseConclusion.REVISION_WARRANTED,
        ),
        project=project,
        record_name="measured",
    )
    annotation = PlanningAnnotation(
        case_id=saved.id,
        case_digest=digest(saved.to_dict()),
        source_assay_id=saved.source.assay_id,
        explanation_kinds=("score_treatment", "selection_response"),
        rung=3,
        mechanism_evidence=saved.mechanism_evidence,
        costs=saved.costs,
    )
    return corpus, saved, annotation, project


def _equal_cost_table(*, second_rank: int = 1):
    costs = tuple(
        RungCost(
            rung=rung,
            rank=second_rank if rung == 2 else (1 if rung == 1 else rung + 1),
            money_usd=Decimal("0"),
            samples=10,
            minutes=1,
            access=("tie-access",),
        )
        for rung in range(1, 7)
    )
    pair = ("score_treatment", "selection_response")
    cells = tuple(
        DiscriminationCell(
            pair=pair,
            rung=rung,
            separates=rung in {1, 2},
            measurement=f"tied rung {rung}",
            required_access=("tie-access",),
        )
        for rung in range(1, 7)
    )
    return costs, cells


def _planned(tmp_path):
    ledger = _ledger(tmp_path)
    choice = plan(
        _live("score_treatment", "selection_response"),
        availability=_availability(*R1, *R2),
        budget=Decimal("1.00"),
        orderings=MANDATORY_ORDERINGS,
        ledger=ledger,
        evaluator_id=TRUSTED.contract.evaluator_id,
    )
    return ledger, choice


class _LocalSandbox:
    """Real local subprocess sandbox for the planner's producer-control fixture."""

    tier = "T0"

    def run(self, argv, *, limits, cwd, env, stdin=None):
        started = time.monotonic()
        completed = subprocess.run(
            argv, cwd=cwd, env=env, input=stdin, capture_output=True, timeout=limits.wall_s
        )
        return Result(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            counters=Counters(wall_s=round(time.monotonic() - started, 3), calls=1),
            breach=None,
            tier=self.tier,
        )


def _produce_result_artifact(tmp_path, choice, *, left_score=0.9, right_score=0.1):
    """Execute the existing PythonGrader producer twice and persist its complete envelopes."""
    grader_source = tmp_path / "planner_producer.py"
    grader_source.write_text(
        "def score(task, response):\n"
        "    return {'left': "
        + repr(left_score)
        + ", 'right': "
        + repr(right_score)
        + "}[response]\n",
        encoding="utf-8",
    )
    grader = PythonGrader.from_path(grader_source)
    sandbox, limits = _LocalSandbox(), Limits(wall_s=10, cpu_s=10)
    task = {"id": "planner-known-answer", "prompt": "executed paired planner control"}
    left = grader.score(task, "left", sandbox=sandbox, limits=limits)
    right = grader.score(task, "right", sandbox=sandbox, limits=limits)
    artifact = tmp_path / f"paired-result-{choice.registration_id[-12:]}.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "reward-lens.paired-grader-envelope.v1",
                "choice_digest": choice.choice_digest,
                "evaluator_contract_digest": choice.evaluator_contract.contract_digest,
                "pair": ["score_treatment", "selection_response"],
                "left": {"envelope": left.to_dict(), "envelope_digest": left.digest()},
                "right": {"envelope": right.to_dict(), "envelope_digest": right.digest()},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return artifact


def _evidence(tmp_path, choice, evaluator=TRUSTED, *, left_score=0.9, right_score=0.1):
    return ExecutionEvidence.issue(
        choice_digest=choice.choice_digest,
        artifact_path=_produce_result_artifact(
            tmp_path, choice, left_score=left_score, right_score=right_score
        ),
        evaluator=evaluator,
    )


def test_table_has_every_unordered_kind_pair_at_every_rung_and_declared_costs():
    costs, cells = load_discrimination_table()
    assert tuple(cost.rung for cost in costs) == (1, 2, 3, 4, 5, 6)
    assert len({cost.rank for cost in costs}) == 6
    kinds = {kind for cell in cells for kind in cell.pair}
    pairs = {tuple(sorted((left, right))) for left in kinds for right in kinds if left < right}
    assert {(cell.rung, cell.pair) for cell in cells} == {
        (rung, pair) for rung in range(1, 7) for pair in pairs
    }


def test_complete_choice_uses_cheapest_feasible_separator_and_pre_registers_expectation(tmp_path):
    choice = plan(
        _live("score_treatment", "selection_response"),
        availability=_availability(*R1, *R2),
        budget=Decimal("1.00"),
        orderings=MANDATORY_ORDERINGS,
        corpus=load_planner_cases(),
        ledger=_ledger(tmp_path),
    )
    assert choice.kind == "complete"
    assert choice.rung == 1
    assert choice.expected_outcomes
    assert choice.alternatives
    assert choice.prior_case is not None
    assert "EIG" not in repr(choice)


def test_saved_corpus_annotation_is_reopened_and_persisted_as_planner_provenance(tmp_path):
    corpus, saved, annotation, _project = _saved_planner_corpus(tmp_path)
    choice = plan(
        _live("score_treatment", "selection_response"),
        availability=_availability(*R1, *R2),
        budget=Decimal("1.00"),
        orderings=MANDATORY_ORDERINGS,
        ledger=_ledger(tmp_path),
        corpus=corpus,
        planning_annotations=(annotation,),
    )
    assert choice.prior_case is not None
    assert choice.prior_case.id == saved.id
    assert choice.prior_case.rung == 3
    assert choice.prior_case.case_digest == annotation.case_digest
    assert choice.prior_case.source_digest == annotation.source_assay_id
    assert choice.prior_case.annotation_digest == annotation.annotation_digest
    assert "declared mechanism evidence" in choice.prior_case.analogy
    persisted = _ledger(tmp_path).path.read_text(encoding="utf-8")
    assert annotation.case_digest in persisted
    assert annotation.source_assay_id in persisted


def test_saved_corpus_adapter_refuses_missing_tampered_or_changed_case_references(tmp_path):
    corpus, saved, annotation, project = _saved_planner_corpus(tmp_path)
    missing = PlanningAnnotation(
        case_id="missing-case",
        case_digest=annotation.case_digest,
        source_assay_id=annotation.source_assay_id,
        explanation_kinds=annotation.explanation_kinds,
        rung=annotation.rung,
        mechanism_evidence=annotation.mechanism_evidence,
        costs=annotation.costs,
    )
    with pytest.raises(ValueError, match="not persisted"):
        load_planner_cases(corpus=corpus, annotations=(missing,))
    path = corpus.path_for(saved.id)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["case"]["mechanism_evidence"] = ["changed after planning annotation"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="case digest"):
        load_planner_cases(corpus=corpus, annotations=(annotation,))
    path.write_text(
        json.dumps({"case": saved.to_dict(), "source": saved.source.to_dict()}), encoding="utf-8"
    )
    project.path_for("measured").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source assay"):
        load_planner_cases(corpus=corpus, annotations=(annotation,))


def test_saved_corpus_adapter_never_reads_protected_acceptance_mapping(tmp_path, monkeypatch):
    corpus, _saved, annotation, _project = _saved_planner_corpus(tmp_path)
    monkeypatch.setattr(
        CorpusStore,
        "search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("protected search read")),
    )
    assert (
        load_planner_cases(corpus=corpus, annotations=(annotation,))[0].annotation_digest
        == annotation.annotation_digest
    )


def test_default_packaged_planner_seed_remains_stable_without_a_corpus_store():
    seed = load_planner_cases()
    assert [case.id for case in seed] == ["seed-score-selection", "seed-advantage-update"]
    assert seed[0].rung == 1


def test_saved_corpus_support_changes_only_a_true_equal_cost_selection_order(tmp_path, monkeypatch):
    from dataclasses import replace

    import reward_lens.product.improve.planner as planner_module

    monkeypatch.setattr(planner_module, "load_discrimination_table", lambda: _equal_cost_table())
    common = dict(
        live=_live("score_treatment", "selection_response"),
        availability=_availability("tie-access"),
        budget=Decimal("1.00"),
        orderings=MANDATORY_ORDERINGS,
    )
    seed_choice = plan(ledger=_ledger(tmp_path / "seed"), **common)
    corpus, _saved, annotation, _project = _saved_planner_corpus(tmp_path / "saved")
    saved_choice = plan(
        ledger=_ledger(tmp_path / "saved"),
        corpus=corpus,
        planning_annotations=(replace(annotation, rung=2),),
        **common,
    )
    assert seed_choice.rung == 1
    assert [item.rung for item in seed_choice.alternatives[:2]] == [1, 2]
    assert saved_choice.rung == 2
    assert [item.rung for item in saved_choice.alternatives[:2]] == [2, 1]
    assert saved_choice.selection_rule == "declared_cost_coverage_then_prior_case_support_v1"
    assert saved_choice.compared_prior_support == ((1, ()), (2, ("planner-saved-case",)))
    preexecution = _ledger(tmp_path / "saved").path.read_text(encoding="utf-8")
    assert "declared_cost_coverage_then_prior_case_support_v1" in preexecution
    assert "planner-saved-case" in preexecution


def test_cheaper_feasible_separator_beats_corpus_supported_later_rung(tmp_path, monkeypatch):
    from dataclasses import replace

    import reward_lens.product.improve.planner as planner_module

    monkeypatch.setattr(
        planner_module, "load_discrimination_table", lambda: _equal_cost_table(second_rank=2)
    )
    corpus, _saved, annotation, _project = _saved_planner_corpus(tmp_path)
    choice = plan(
        _live("score_treatment", "selection_response"),
        _availability("tie-access"),
        Decimal("1.00"),
        orderings=MANDATORY_ORDERINGS,
        ledger=_ledger(tmp_path),
        corpus=corpus,
        planning_annotations=(replace(annotation, rung=2),),
    )
    assert choice.rung == 1
    assert choice.compared_prior_support == ((1, ()),)


def test_partial_choice_rejects_cheaper_non_discriminator_and_names_unresolved_pairs(tmp_path):
    choice = plan(
        _live("selection_response", "advantage_pressure", "update_mechanism"),
        availability=_availability(*R1, *R2, "run_record"),
        budget=Decimal("1.00"),
        orderings=MANDATORY_ORDERINGS,
        ledger=_ledger(tmp_path),
    )
    assert choice.kind == "partial"
    assert choice.rung == 2
    assert choice.unresolved_pairs
    assert any(item.rung == 1 and not item.separates for item in choice.alternatives)


def test_no_feasible_discriminator_names_missing_live_access(tmp_path):
    choice = plan(
        _live("update_mechanism", "horizon_behavior"),
        availability=frozenset(),
        budget=Decimal("100.00"),
        orderings=MANDATORY_ORDERINGS,
        ledger=_ledger(tmp_path),
    )
    assert choice.kind == "unresolved"
    assert choice.refusal is not None
    assert choice.refusal.code == "PLANNER_NO_FEASIBLE_DISCRIMINATOR"
    assert "pinned_trainer" in choice.missing_access


def test_unavailable_expensive_rungs_are_not_treated_as_available_access(tmp_path):
    choice = plan(
        _live("update_mechanism", "horizon_behavior"),
        availability=_availability("pinned_trainer", "checkpoint", "gpu", "horizon_budget"),
        budget=Decimal("0"),
        orderings=MANDATORY_ORDERINGS,
        ledger=_ledger(tmp_path),
    )
    assert choice.kind == "unresolved"
    assert choice.missing_access == ()
    assert {item.rung for item in choice.alternatives if item.over_budget} >= {4, 5}


@pytest.mark.parametrize(
    "orderings, required",
    [
        (
            (
                "neural_intervention",
                "replay_nondeterminism_environment_identity",
                "masks_and_loss_aggregation",
                "reward_model_feature",
            ),
            "replay_nondeterminism_environment_identity",
        ),
        (
            (
                "replay_nondeterminism_environment_identity",
                "neural_intervention",
                "reward_model_feature",
            ),
            "masks_and_loss_aggregation",
        ),
    ],
)
def test_mandatory_orderings_refuse_a_skipped_precondition(orderings, required, tmp_path):
    with pytest.raises(PlannerRefusal, match="PLANNER_ORDERING_REQUIRED") as raised:
        plan(
            _live("score_treatment", "selection_response"),
            availability=_availability(*R1, *R2),
            budget=Decimal("1.00"),
            orderings=orderings,
            ledger=_ledger(tmp_path),
        )
    assert required in str(raised.value)


def test_inconclusive_retries_are_bounded_by_the_same_experiment_signature_after_reload(tmp_path):
    ledger, first = _planned(tmp_path)
    assert first.signature is not None
    record_result(
        first,
        ledger,
        evidence=_evidence(tmp_path, first, left_score=0.5, right_score=0.5),
        analogy_survived=False,
    )
    reloaded = PlannerLedger.open(ledger.path, evaluators=(TRUSTED,))
    second = plan(
        _live("score_treatment", "selection_response"),
        availability=_availability(*R1, *R2),
        budget=Decimal("1.00"),
        orderings=MANDATORY_ORDERINGS,
        ledger=reloaded,
    )
    record_result(
        second,
        reloaded,
        evidence=_evidence(tmp_path, second, left_score=0.5, right_score=0.5),
        analogy_survived=False,
    )
    with pytest.raises(PlannerRefusal, match="PLANNER_RETRY_HISTORY_INCONSISTENT"):
        plan(
            _live("score_treatment", "selection_response"),
            _availability(*R1, *R2),
            Decimal("1.00"),
            orderings=MANDATORY_ORDERINGS,
            ledger=PlannerLedger.open(ledger.path, evaluators=(TRUSTED,)),
            retry_limit=3,
        )
    with pytest.raises(PlannerRefusal, match="PLANNER_RETRY_EXHAUSTED") as raised:
        plan(
            _live("score_treatment", "selection_response"),
            _availability(*R1, *R2),
            Decimal("1.00"),
            orderings=MANDATORY_ORDERINGS,
            ledger=PlannerLedger.open(ledger.path, evaluators=(TRUSTED,)),
            retry_limit=2,
        )
    assert first.signature in str(raised.value)


def test_registered_trusted_evaluator_derives_elimination_from_concrete_artifact(tmp_path):
    ledger, first = _planned(tmp_path)
    artifact = _produce_result_artifact(tmp_path, first)
    entry = record_result(
        first,
        ledger,
        evidence=ExecutionEvidence.issue(
            choice_digest=first.choice_digest, artifact_path=artifact, evaluator=TRUSTED
        ),
        analogy_survived=True,
    )
    assert entry.informative is True
    assert entry.ruled_out == ("selection_response",)
    assert first.evaluator_contract == TRUSTED.contract
    assert first.choice_digest == entry.evidence.choice_digest
    assert entry.evidence.artifact_digest.startswith("sha256:")
    assert TRUSTED.contract.implementation_digest in ledger.path.read_text(encoding="utf-8")
    assert len(PlannerLedger.open(ledger.path, evaluators=(TRUSTED,)).entries) == 2
    reverse_root = tmp_path / "reverse-producer"
    reverse_root.mkdir()
    reverse_ledger, reverse_choice = _planned(reverse_root)
    reverse = _produce_result_artifact(
        reverse_root, reverse_choice, left_score=0.1, right_score=0.9
    )
    reverse_entry = record_result(
        reverse_choice,
        reverse_ledger,
        evidence=ExecutionEvidence.issue(
            choice_digest=reverse_choice.choice_digest, artifact_path=reverse, evaluator=TRUSTED
        ),
        analogy_survived=True,
    )
    assert reverse_entry.ruled_out == ("score_treatment",)


def test_malicious_first_result_and_unregistered_or_altered_records_are_refused(tmp_path):
    ledger, choice = _planned(tmp_path)
    from dataclasses import replace

    candidate = TrustedEvaluator(
        contract=EvaluatorContract(
            "candidate", "sha256:" + "c" * 64, "paired-grader-envelope-v1", "candidate-key"
        ),
        authenticator=HmacEvidenceAuthenticator(
            b"candidate controlled evaluator key", issuer="candidate"
        ),
    )
    malicious = _evidence(tmp_path, choice, candidate)
    with pytest.raises(PlannerRefusal, match="PLANNER_EVIDENCE_UNAUTHENTICATED"):
        record_result(choice, ledger, evidence=malicious, analogy_survived=True)
    wrong_plan = replace(_evidence(tmp_path, choice), choice_digest="sha256:" + "e" * 64)
    with pytest.raises(PlannerRefusal, match="PLANNER_EVIDENCE_CHOICE_MISMATCH"):
        record_result(choice, ledger, evidence=wrong_plan, analogy_survived=True)
    with pytest.raises(PlannerRefusal, match="PLANNER_ARTIFACT_CHANGED"):
        evidence = _evidence(tmp_path, choice)
        artifact = _produce_result_artifact(tmp_path, choice)
        artifact.write_text(
            artifact.read_text(encoding="utf-8").replace('"score": 0.9', '"score": 0.8'),
            encoding="utf-8",
        )
        record_result(choice, ledger, evidence=evidence, analogy_survived=True)
    with pytest.raises(PlannerRefusal, match="PLANNER_EVALUATOR_UNTRUSTED"):
        plan(
            _live("score_treatment", "selection_response"),
            _availability(*R1, *R2),
            Decimal("1.00"),
            orderings=MANDATORY_ORDERINGS,
            ledger=ledger,
            evaluator_id="candidate",
        )
    with pytest.raises(PlannerRefusal, match="PLANNER_CHOICE_UNREGISTERED"):
        record_result(
            choice.__class__(**{**choice.__dict__, "registration_id": None}),
            ledger,
            evidence=_evidence(tmp_path, choice),
            analogy_survived=True,
        )
    with pytest.raises(PlannerRefusal, match="PLANNER_CHOICE_DIGEST_MISMATCH"):
        record_result(
            replace(choice, expected_outcomes=("written after a result",)),
            ledger,
            evidence=_evidence(tmp_path, choice),
            analogy_survived=True,
        )
    outside_path = _produce_result_artifact(tmp_path, choice)
    outside_path.write_text(
        outside_path.read_text(encoding="utf-8")
        .replace("score_treatment", "advantage_pressure")
        .replace("selection_response", "update_mechanism"),
        encoding="utf-8",
    )
    outside = ExecutionEvidence.issue(
        choice_digest=choice.choice_digest, artifact_path=outside_path, evaluator=TRUSTED
    )
    with pytest.raises(PlannerRefusal, match="PLANNER_ELIMINATION_UNSUPPORTED"):
        record_result(choice, ledger, evidence=outside, analogy_survived=True)


def test_missing_or_malformed_artifact_cannot_turn_a_caller_claim_into_an_elimination(tmp_path):
    ledger, choice = _planned(tmp_path)
    missing = tmp_path / "no-file-or-execution.json"
    with pytest.raises(FileNotFoundError):
        ExecutionEvidence.issue(
            choice_digest=choice.choice_digest, artifact_path=missing, evaluator=TRUSTED
        )
    malformed = tmp_path / "all-zero-shaped-digest.json"
    malformed.write_text(
        json.dumps(
            {
                "schema": "reward-lens.paired-grader-envelope.v1",
                "choice_digest": choice.choice_digest,
                "evaluator_contract_digest": choice.evaluator_contract.contract_digest,
                "pair": ["score_treatment", "selection_response"],
                "left": {
                    "score": 1.0,
                    "witness": {"result_digest": "sha256:" + "0" * 64, "item_id": "forged"},
                },
                "right": {
                    "score": 0.0,
                    "witness": {"result_digest": "sha256:" + "1" * 64, "item_id": "forged"},
                },
            }
        ),
        encoding="utf-8",
    )
    evidence = ExecutionEvidence.issue(
        choice_digest=choice.choice_digest, artifact_path=malformed, evaluator=TRUSTED
    )
    with pytest.raises(PlannerRefusal, match="PLANNER_ARTIFACT_INVALID"):
        record_result(choice, ledger, evidence=evidence, analogy_survived=True)


def test_non_scored_python_grader_envelope_cannot_eliminate_an_explanation(tmp_path):
    ledger, choice = _planned(tmp_path)
    artifact = _produce_result_artifact(tmp_path, choice, left_score=None, right_score=0.1)
    evidence = ExecutionEvidence.issue(
        choice_digest=choice.choice_digest, artifact_path=artifact, evaluator=TRUSTED
    )
    with pytest.raises(PlannerRefusal, match="PLANNER_ARTIFACT_INVALID"):
        record_result(choice, ledger, evidence=evidence, analogy_survived=True)


def test_changed_serialized_producer_envelope_fails_its_own_digest(tmp_path):
    ledger, choice = _planned(tmp_path)
    artifact = _produce_result_artifact(tmp_path, choice)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["left"]["envelope"]["score"] = 0.8
    artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    evidence = ExecutionEvidence.issue(
        choice_digest=choice.choice_digest, artifact_path=artifact, evaluator=TRUSTED
    )
    with pytest.raises(PlannerRefusal, match="PLANNER_ARTIFACT_CHANGED"):
        record_result(choice, ledger, evidence=evidence, analogy_survived=True)


def test_reload_refuses_a_tampered_persisted_pre_execution_expectation(tmp_path):
    ledger, _choice = _planned(tmp_path)
    original = ledger.path.read_text(encoding="utf-8")
    assert "score frozen samples" in original
    ledger.path.write_text(
        original.replace("score frozen samples", "written after an observed result", 1),
        encoding="utf-8",
    )
    with pytest.raises(PlannerRefusal, match="PLANNER_LEDGER_TAMPERED"):
        PlannerLedger.open(ledger.path, evaluators=(TRUSTED,))


def test_same_public_corpus_interface_changes_declared_tie_order_and_pins_dataset(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    import reward_lens.product.improve.planner as module
    from reward_lens.commons.corpus import Corpus

    monkeypatch.setattr(module, "load_discrimination_table", _equal_cost_table)
    calls = []
    original = Corpus.cases_like

    def observed(self, query, *, actor_context):
        calls.append((self, query, actor_context))
        return original(self, query, actor_context=actor_context)

    monkeypatch.setattr(Corpus, "cases_like", observed)
    common = dict(
        live=_live("score_treatment", "selection_response"),
        availability=_availability("tie-access"),
        budget=Decimal("1.00"),
        orderings=MANDATORY_ORDERINGS,
    )
    seed = Corpus()
    seed_choice = plan(ledger=_ledger(tmp_path / "seed-facade"), corpus=seed, **common)
    store, _saved, annotation, _project = _saved_planner_corpus(tmp_path / "replacement")
    replacement = Corpus(store, planning_annotations=(replace(annotation, rung=2),))
    replacement_ledger = _ledger(tmp_path / "replacement-ledger")
    choice = plan(ledger=replacement_ledger, corpus=replacement, **common)
    assert [self for self, _query, _actor in calls] == [seed, replacement]
    assert seed_choice.rung == 1 and choice.rung == 2
    assert [item.rung for item in seed_choice.alternatives[:2]] == [1, 2]
    assert [item.rung for item in choice.alternatives[:2]] == [2, 1]
    assert choice.corpus_provenance["kind"] == "registered_store"
    assert seed_choice.corpus_provenance["kind"] == "accepted_seed"
    assert (
        choice.corpus_provenance["dataset_digest"]
        != seed_choice.corpus_provenance["dataset_digest"]
    )
    assert choice.prior_case.source_digest == annotation.source_assay_id
    assert choice.prior_case.case_digest == annotation.case_digest
    assert choice.prior_case.annotation_digest == replace(annotation, rung=2).annotation_digest
    saved = replacement_ledger.path.read_text()
    assert choice.corpus_provenance["dataset_digest"] in saved
    assert annotation.case_digest in saved and annotation.source_assay_id in saved


def test_public_facade_store_planning_requires_annotations_and_refuses_tamper(tmp_path):
    from reward_lens.commons.corpus import Corpus

    common = dict(
        live=_live("score_treatment", "selection_response"),
        availability=_availability(*R1),
        budget=Decimal("1.00"),
        orderings=MANDATORY_ORDERINGS,
    )
    store, saved, annotation, project = _saved_planner_corpus(tmp_path)
    with pytest.raises(ValueError, match="explicit planning annotations"):
        plan(ledger=_ledger(tmp_path / "unannotated"), corpus=Corpus(store), **common)
    document = json.loads(store.path_for(saved.id).read_text())
    document["case"]["mechanism_evidence"] = ["changed after annotation"]
    store.path_for(saved.id).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="case digest"):
        plan(
            ledger=_ledger(tmp_path / "changed"),
            corpus=Corpus(store, planning_annotations=(annotation,)),
            **common,
        )


def _write_public_cost_declaration(path, *, tied=True):
    costs, _cells = load_discrimination_table()
    rows = [
        dict(
            rung=cost.rung,
            rank=cost.rank,
            money_usd=str(cost.money_usd),
            samples=cost.samples,
            minutes=cost.minutes,
        )
        for cost in costs
    ]
    if tied:
        rows[1].update(
            rank=rows[0]["rank"],
            money_usd=rows[0]["money_usd"],
            samples=rows[0]["samples"],
            minutes=rows[0]["minutes"],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "costs": rows}, indent=2) + "\n")
    return path


def test_public_declared_cost_tie_changes_saved_corpus_order_and_reopens_exact_bytes(tmp_path):
    import hashlib
    from dataclasses import replace

    from reward_lens.commons.corpus import Corpus

    declaration = _write_public_cost_declaration(tmp_path / "costs.json")
    raw = declaration.read_bytes()
    common = dict(
        live=_live("score_treatment", "selection_response"),
        availability=_availability(*R1, *R2),
        budget=Decimal("0"),
        orderings=MANDATORY_ORDERINGS,
        cost_table=declaration,
    )
    observed = []
    for rung in (1, 2):
        store, saved, annotation, _project = _saved_planner_corpus(tmp_path / f"dataset-{rung}")
        annotation = replace(annotation, rung=rung)
        corpus = Corpus(store, planning_annotations=(annotation,))
        ledger = _ledger(tmp_path / f"plan-{rung}")
        choice = plan(corpus=corpus, ledger=ledger, **common)
        reopened = (
            PlannerLedger.open(ledger.path, evaluators=(TRUSTED,)).entries[0].payload["choice"]
        )
        assert choice.rung == rung
        assert [item.rung for item in choice.alternatives[:2]] == ([1, 2] if rung == 1 else [2, 1])
        provenance = reopened["cost_table_provenance"]
        assert provenance == choice.cost_table_provenance
        assert provenance["source_bytes_utf8"].encode() == raw
        assert provenance["source_digest"] == "sha256:" + hashlib.sha256(raw).hexdigest()
        assert provenance["meaning"] == "declared_local_planning_costs_not_measured_performance"
        assert (
            reopened["corpus_provenance"]["dataset_digest"]
            == choice.corpus_provenance["dataset_digest"]
        )
        assert reopened["prior_case"]["case_digest"] == annotation.case_digest
        assert reopened["prior_case"]["source_digest"] == annotation.source_assay_id
        assert reopened["prior_case"]["annotation_digest"] == annotation.annotation_digest
        assert str(declaration) not in json.dumps(reopened)
        assert str(saved.source.path) not in json.dumps(reopened)
        assert {item.rung: item.cost.access for item in choice.alternatives} == {
            row.rung: row.access for row in load_discrimination_table()[0]
        }
        observed.append(choice)
    assert (
        observed[0].corpus_provenance["dataset_digest"]
        != observed[1].corpus_provenance["dataset_digest"]
    )


def test_declared_costs_preserve_cheaper_first_and_default_payload(tmp_path):
    from dataclasses import replace

    from reward_lens.commons.corpus import Corpus

    store, _saved, annotation, _project = _saved_planner_corpus(tmp_path / "dataset")
    corpus = Corpus(store, planning_annotations=(replace(annotation, rung=2),))
    common = dict(
        live=_live("score_treatment", "selection_response"),
        availability=_availability(*R1, *R2),
        budget=Decimal("0"),
        orderings=MANDATORY_ORDERINGS,
        corpus=corpus,
    )
    default = plan(ledger=_ledger(tmp_path / "default"), **common)
    assert default.rung == 1 and default.cost_table_provenance is None
    assert "cost_table_provenance" not in _ledger(tmp_path / "default").entries[0].payload["choice"]
    declaration = _write_public_cost_declaration(tmp_path / "unequal.json", tied=False)
    cheaper = plan(ledger=_ledger(tmp_path / "cheaper"), cost_table=declaration, **common)
    assert cheaper.rung == 1
    assert [item.rung for item in cheaper.alternatives] == [
        item.rung for item in default.alternatives
    ]
    # A tied prior cannot replace a missing required outcome check.
    declaration = _write_public_cost_declaration(tmp_path / "tie.json")
    unavailable = plan(
        ledger=_ledger(tmp_path / "unavailable"),
        cost_table=declaration,
        **{**common, "availability": R1},
    )
    assert unavailable.rung == 1
    assert next(item for item in unavailable.alternatives if item.rung == 2).missing_access == (
        "outcome_check",
    )
    over_budget = json.loads(declaration.read_text())
    over_budget["costs"][1]["money_usd"] = "1.00"
    declaration.write_text(json.dumps(over_budget))
    choice = plan(ledger=_ledger(tmp_path / "over-budget"), cost_table=declaration, **common)
    assert choice.rung == 1
    assert next(item for item in choice.alternatives if item.rung == 2).over_budget
    unresolved = plan(
        ledger=_ledger(tmp_path / "unresolved"),
        cost_table=declaration,
        **{**common, "availability": ()},
    )
    assert unresolved.kind == "unresolved" and unresolved.rung is None


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "torn_json",
        "corrupt_utf8",
        "duplicate_rung",
        "unknown_access",
        "version_bool",
        "rank_zero",
        "rank_bool",
        "negative_samples",
        "minutes_float",
        "money_nan",
        "money_infinity",
        "negative_money",
        "money_float",
        "duplicate_key",
        "oversized",
    ],
)
def test_malformed_public_cost_declaration_refuses_before_corpus_or_registration(tmp_path, defect):
    from reward_lens.commons.corpus import Corpus

    class UnreadableCorpus(Corpus):
        def cases_like(self, *args, **kwargs):
            raise AssertionError("invalid declaration reached corpus inspection")

    declaration = _write_public_cost_declaration(tmp_path / "cost.json")
    value = json.loads(declaration.read_text())
    if defect == "missing":
        declaration.unlink()
    elif defect == "torn_json":
        declaration.write_text('{"version":1,"costs":[')
    elif defect == "corrupt_utf8":
        declaration.write_bytes(b"\xff")
    elif defect == "duplicate_key":
        declaration.write_text('{"version":1,"version":1,"costs":[]}')
    elif defect == "oversized":
        declaration.write_bytes(b" " * 65537)
    else:
        if defect == "duplicate_rung":
            value["costs"][1]["rung"] = 1
        elif defect == "unknown_access":
            value["costs"][0]["access"] = []
        elif defect == "version_bool":
            value["version"] = True
        elif defect == "rank_zero":
            value["costs"][0]["rank"] = 0
        elif defect == "rank_bool":
            value["costs"][0]["rank"] = True
        elif defect == "negative_samples":
            value["costs"][0]["samples"] = -1
        elif defect == "minutes_float":
            value["costs"][0]["minutes"] = 1.5
        elif defect == "money_nan":
            value["costs"][0]["money_usd"] = "NaN"
        elif defect == "money_infinity":
            value["costs"][0]["money_usd"] = "Infinity"
        elif defect == "negative_money":
            value["costs"][0]["money_usd"] = "-1"
        elif defect == "money_float":
            value["costs"][0]["money_usd"] = 1.0
        declaration.write_text(json.dumps(value))
    ledger = _ledger(tmp_path / "ledger")
    with pytest.raises(PlannerRefusal) as refusal:
        plan(
            _live("score_treatment", "selection_response"),
            R1,
            Decimal("0"),
            orderings=MANDATORY_ORDERINGS,
            ledger=ledger,
            corpus=UnreadableCorpus(),
            cost_table=declaration,
        )
    assert refusal.value.code == "PLANNER_COST_TABLE_INVALID"
    assert not ledger.path.exists() and not ledger.entries


def test_torn_cost_read_and_change_during_corpus_lookup_refuse_before_registration(
    tmp_path, monkeypatch
):
    from reward_lens.commons.corpus import Corpus

    declaration = _write_public_cost_declaration(tmp_path / "cost.json")
    import reward_lens.product.improve.planner as module

    original = module._read_cost_bytes
    calls = []

    def torn(self):
        raw = original(self)
        if self == declaration:
            calls.append(self)
            return raw if len(calls) == 1 else raw + b" "
        return raw

    monkeypatch.setattr(module, "_read_cost_bytes", torn)
    ledger = _ledger(tmp_path / "torn-ledger")
    with pytest.raises(PlannerRefusal, match="PLANNER_COST_TABLE_CHANGED"):
        plan(
            _live("score_treatment", "selection_response"),
            R1,
            Decimal("0"),
            orderings=MANDATORY_ORDERINGS,
            ledger=ledger,
            cost_table=declaration,
        )
    assert not ledger.path.exists()
    monkeypatch.setattr(module, "_read_cost_bytes", original)

    class ChangedCorpus(Corpus):
        def cases_like(self, *args, **kwargs):
            declaration.write_bytes(declaration.read_bytes() + b" ")
            return super().cases_like(*args, **kwargs)

    ledger = _ledger(tmp_path / "changed-ledger")
    with pytest.raises(PlannerRefusal, match="PLANNER_COST_TABLE_CHANGED"):
        plan(
            _live("score_treatment", "selection_response"),
            R1,
            Decimal("0"),
            orderings=MANDATORY_ORDERINGS,
            ledger=ledger,
            corpus=ChangedCorpus(),
            cost_table=declaration,
        )
    assert not ledger.path.exists()


def test_sealed_cost_bytes_survive_external_change_but_ledger_corruption_refuses(tmp_path):
    declaration = _write_public_cost_declaration(tmp_path / "cost.json")
    ledger = _ledger(tmp_path / "ledger")
    choice = plan(
        _live("score_treatment", "selection_response"),
        _availability(*R1, *R2),
        Decimal("0"),
        orderings=MANDATORY_ORDERINGS,
        ledger=ledger,
        cost_table=declaration,
    )
    sealed = choice.cost_table_provenance["source_bytes_utf8"]
    declaration.write_text("corrupt later external file")
    reopened = PlannerLedger.open(ledger.path, evaluators=(TRUSTED,))
    assert (
        reopened.entries[0].payload["choice"]["cost_table_provenance"]["source_bytes_utf8"]
        == sealed
    )
    # Even a rehashed ledger row must preserve the inner byte identity.
    row = json.loads(ledger.path.read_text())
    registered = row["payload"]["choice"]
    registered["cost_table_provenance"]["source_bytes_utf8"] += " "
    row["payload"]["choice_digest"] = digest(registered)
    row.pop("entry_digest")
    row["entry_digest"] = digest(row)
    ledger.path.write_text(json.dumps(row) + "\n")
    with pytest.raises(PlannerRefusal, match="PLANNER_LEDGER_TAMPERED"):
        PlannerLedger.open(ledger.path, evaluators=(TRUSTED,))


def test_declared_cost_tie_partial_choice_keeps_unresolved_pairs(tmp_path):
    from dataclasses import replace

    from reward_lens.commons.corpus import Corpus

    declaration = _write_public_cost_declaration(tmp_path / "cost.json")
    store, _saved, annotation, _project = _saved_planner_corpus(tmp_path / "dataset")
    corpus = Corpus(store, planning_annotations=(replace(annotation, rung=2),))
    choice = plan(
        _live("score_treatment", "selection_response", "advantage_pressure"),
        _availability(*R1, *R2),
        Decimal("0"),
        orderings=MANDATORY_ORDERINGS,
        ledger=_ledger(tmp_path / "ledger"),
        corpus=corpus,
        cost_table=declaration,
    )
    assert choice.kind == "partial" and choice.rung == 2
    assert choice.unresolved_pairs == (("advantage_pressure", "score_treatment"),)
    assert len(choice.separated_pairs) == 2


def test_deep_cost_json_refuses_with_named_error_before_corpus_or_registration(tmp_path):
    from reward_lens.commons.corpus import Corpus

    class UnreadableCorpus(Corpus):
        def cases_like(self, *args, **kwargs):
            raise AssertionError("deep declaration reached corpus inspection")

    raw = b'{"version":1,"costs":' + b"[" * 1200 + b"]" * 1200 + b"}"
    assert len(raw) == 2422 and len(raw) < 65536
    declaration = tmp_path / "deep-costs.json"
    declaration.write_bytes(raw)
    ledger = _ledger(tmp_path / "ledger")
    with pytest.raises(PlannerRefusal) as refusal:
        plan(
            _live("score_treatment", "selection_response"),
            R1,
            Decimal("0"),
            orderings=MANDATORY_ORDERINGS,
            ledger=ledger,
            corpus=UnreadableCorpus(),
            cost_table=declaration,
        )
    assert refusal.value.code == "PLANNER_COST_TABLE_INVALID"
    assert not ledger.path.exists() and not ledger.entries
