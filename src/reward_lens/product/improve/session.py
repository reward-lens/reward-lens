"""The resumable six-state improve loop, without manufacturing measurements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from reward_lens import contracts

from . import record, state


class ImproveRefusal(ValueError):
    pass


_SOURCES = frozenset({"owner_edit", "agent_draft", "generator_output", "supported_intervention"})
_EXTENSION = "https://reward-lens.github.io/extensions/improve/v1"


@dataclass(frozen=True)
class ImproveSession:
    state_path: Path
    payload: dict[str, Any]

    @property
    def state(self) -> str:
        return str(self.payload["state"])

    @classmethod
    def describe(
        cls, root: Path | str, behaviour: str, *, session_id: str = "current"
    ) -> "ImproveSession":
        if not isinstance(behaviour, str) or not behaviour.strip():
            raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: describe is required")
        path = state.session_path(root, session_id)
        payload = {
            "state": "describe",
            "description": behaviour.strip(),
            "candidates": [],
            "pairs": [],
            "experiments": [],
        }
        state.write(path, payload)
        return cls(path, payload)

    @classmethod
    def resume(cls, path: Path | str) -> "ImproveSession":
        resolved = Path(path)
        return cls(resolved, state.read(resolved))

    def _transition(self, target: str, **fields: Any) -> "ImproveSession":
        order = {name: index for index, name in enumerate(state.STATES)}
        if order[target] != order[self.state] + 1:
            raise ImproveRefusal(f"IMPROVE_RESUME_REQUIRED: state={self.state} target={target}")
        payload = {**self.payload, **fields, "state": target}
        state.write(self.state_path, payload)
        return ImproveSession(self.state_path, payload)

    def propose(self, *, source: str, candidate: dict[str, Any]) -> "ImproveSession":
        if source not in _SOURCES or not candidate:
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: candidate_source and candidate are required"
            )
        candidates = [
            *self.payload.get("candidates", ()),
            {"source": source, "candidate": candidate},
        ]
        return self._transition("propose", candidates=candidates)

    def plan(self, choice: Any) -> "ImproveSession":
        if getattr(choice, "kind", None) == "unresolved":
            self.require_resolved_tradeoff(tuple(getattr(choice, "unresolved_pairs", ())))
        return self._transition("plan", plan=record.choice_payload(choice))

    def execute(self, evidence: dict[str, Any]) -> "ImproveSession":
        if not evidence:
            raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: execution evidence is required")
        return self._transition("execute", execution=evidence)

    def decide(self, outcome: str, *, reason: str) -> "ImproveSession":
        if outcome not in {"qualified", "all_rejected", "unresolved_tradeoff"} or not reason:
            raise ImproveRefusal("IMPROVE_TRADEOFF_UNRESOLVED: outcome and reason are required")
        return self._transition("decide", outcome={"state": outcome, "reason": reason})

    def export(self, experiment: dict[str, Any]) -> "ImproveSession":
        if self.state != "decide":
            raise ImproveRefusal(f"IMPROVE_RESUME_REQUIRED: state={self.state}")
        return ImproveSession(
            self.state_path, record.attach(self.state_path, experiment=experiment)
        )

    def mark_pair(self, left: Path | str, right: Path | str, why: str) -> "ImproveSession":
        try:
            payload = record.mark_pair(self.state_path, left=left, right=right, why=why)
        except ValueError as error:
            raise ImproveRefusal(str(error)) from error
        return ImproveSession(self.state_path, payload)

    def require_resolved_tradeoff(self, unresolved_pairs: tuple[tuple[str, str], ...]) -> None:
        if unresolved_pairs:
            raise ImproveRefusal(
                f"IMPROVE_TRADEOFF_UNRESOLVED: unresolved_pairs={unresolved_pairs}"
            )


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ImproveRefusal(f"IMPROVE_RESUME_REQUIRED: candidate.{field} must be an object")
    return dict(value)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImproveRefusal(
            f"IMPROVE_RESUME_REQUIRED: candidate.{field} must be a non-empty string"
        )
    return value.strip()


def _path_beneath(root: Path, value: object, field: str, *, directory: bool) -> Path:
    raw = Path(_text(value, field))
    candidate = raw.resolve()
    root = root.resolve()
    if root not in candidate.parents and candidate != root:
        raise ImproveRefusal(f"IMPROVE_RESUME_REQUIRED: candidate.{field} must stay beneath path")
    if directory and not candidate.is_dir():
        raise ImproveRefusal(f"IMPROVE_RESUME_REQUIRED: candidate.{field} must name a directory")
    if not directory and not candidate.is_file():
        raise ImproveRefusal(f"IMPROVE_RESUME_REQUIRED: candidate.{field} must name a file")
    return candidate


def _planner_choice(
    root: Path, session: ImproveSession, candidate: Mapping[str, Any], request: Any
):
    """Register the actual rule-table choice before calling Compare.

    The registration signer authenticates the planner ledger's local record of
    its rule-table decision.  It is not an outcome grader, and this function
    never records a comparison result as a planner elimination.
    """
    from . import MANDATORY_ORDERINGS, Explanation, plan

    planner = _mapping(candidate.get("planner"), "planner")
    rows = planner.get("explanations")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate.planner.explanations needs two live explanations"
        )
    live = tuple(
        Explanation(
            _text(_mapping(row, "planner.explanations").get("kind"), "planner.explanations.kind"),
            _text(
                _mapping(row, "planner.explanations").get("statement"),
                "planner.explanations.statement",
            ),
        )
        for row in rows
    )
    availability = planner.get("availability")
    if not isinstance(availability, list) or any(
        not isinstance(item, str) for item in availability
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate.planner.availability must be a string list"
        )
    try:
        budget = Decimal(_text(planner.get("budget_usd"), "planner.budget_usd"))
    except (InvalidOperation, ValueError) as error:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate.planner.budget_usd must be a decimal"
        ) from error
    if budget < 0:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate.planner.budget_usd must be non-negative"
        )
    if request.max_budget_usd is not None:
        budget = min(budget, Decimal(str(request.max_budget_usd)))
    retry_limit = planner.get("retry_limit", 2)
    if type(retry_limit) is not int or retry_limit < 0:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate.planner.retry_limit must be a non-negative integer"
        )
    session, ledger, evaluator = _planner_runtime(root, session)
    recovered = _untransitioned_choice(ledger, evaluator)
    if recovered is not None:
        return recovered, session, ledger, evaluator
    return (
        plan(
            live,
            tuple(availability),
            budget,
            orderings=MANDATORY_ORDERINGS,
            ledger=ledger,
            retry_limit=retry_limit,
        ),
        session,
        ledger,
        evaluator,
    )


def _planner_runtime(root: Path, session: ImproveSession):
    """Reopen the session-bound planner ledger and its preconfigured evaluator.

    The runtime contract is written before the first planner registration.  If
    a process dies after the append-only registration but before the state
    transition, resumption can find that exact registration instead of adding a
    second choice to the ledger.
    """
    from . import EvaluatorContract, HmacEvidenceAuthenticator, PlannerLedger, TrustedEvaluator

    key = contracts.digest(
        {"session": str(session.state_path), "description": session.payload["description"]}
    ).encode("ascii")
    runtime = session.payload.get("planner_runtime")
    if isinstance(runtime, Mapping) and isinstance(runtime.get("contract"), Mapping):
        contract = EvaluatorContract(**dict(runtime["contract"]))
    elif isinstance(session.payload.get("plan"), Mapping) and isinstance(
        session.payload["plan"].get("evaluator_contract"), Mapping
    ):
        # Sessions written by the first public route used the fixed legacy
        # contract.  Keep them resumable; new sessions receive an identity
        # unique to their durable state file below.
        contract = EvaluatorContract(**dict(session.payload["plan"]["evaluator_contract"]))
    else:
        contract = EvaluatorContract(
            evaluator_id="improve-planner-registration",
            implementation_digest=contracts.digest(
                {"rule_table": "discrimination.yaml", "version": 1}
            ),
            rule_id="paired-grader-envelope-v1",
            key_id="local-improve-planner-registration-" + key.decode("ascii")[:16],
        )
        payload = {**session.payload, "planner_runtime": {"contract": contract.payload()}}
        state.write(session.state_path, payload)
        session = ImproveSession(session.state_path, payload)
    evaluator = TrustedEvaluator(
        contract, HmacEvidenceAuthenticator(key=key, issuer=contract.evaluator_id)
    )
    ledger = PlannerLedger.open(
        root / ".reward-lens" / "improve" / "planner.jsonl", evaluators=(evaluator,)
    )
    return session, ledger, evaluator


def _untransitioned_choice(ledger: Any, evaluator: Any):
    """Return a pre-existing choice for this session's unique runtime only."""
    matches = [
        entry.payload
        for entry in ledger.entries
        if entry.kind == "choice"
        and entry.payload["choice"].get("evaluator_contract") == evaluator.contract.payload()
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: session has ambiguous registered planner choices"
        )
    stored = matches[0]
    return _restore_choice(
        {
            **stored["choice"],
            "registration_id": stored["registration_id"],
            "choice_digest": stored["choice_digest"],
        }
    )


def _restore_choice(payload: Mapping[str, Any]):
    """Rehydrate the public planner dataclasses without issuing another choice."""
    from reward_lens.commons.corpus.protocol import PlannerCase

    from . import EvaluatorContract, PlanAlternative, PlanChoice, PlannerRefusalInfo, RungCost

    def cost(raw: Mapping[str, Any] | None):
        return (
            None
            if raw is None
            else RungCost(
                raw["rung"],
                raw["rank"],
                Decimal(raw["money_usd"]),
                raw["samples"],
                raw["minutes"],
                tuple(raw["access"]),
            )
        )

    prior_raw = payload.get("prior_case")
    prior = (
        None
        if prior_raw is None
        else PlannerCase(
            prior_raw["id"],
            tuple(prior_raw["explanation_kinds"]),
            prior_raw["rung"],
            prior_raw["analogy"],
            prior_raw.get("case_digest"),
            prior_raw.get("source_digest"),
            prior_raw.get("annotation_digest"),
        )
    )
    contract_raw = payload.get("evaluator_contract")
    contract = None if contract_raw is None else EvaluatorContract(**contract_raw)
    refusal_raw = payload.get("refusal")
    refusal = None if refusal_raw is None else PlannerRefusalInfo(**refusal_raw)
    alternatives = tuple(
        PlanAlternative(
            row["rung"],
            cost(row["cost"]),
            tuple(tuple(pair) for pair in row["separates_pairs"]),
            tuple(tuple(pair) for pair in row["unresolved_pairs"]),
            row["feasible"],
            tuple(row["missing_access"]),
            row["over_budget"],
            tuple(row["prior_support_case_ids"]),
        )
        for row in payload["alternatives"]
    )
    return PlanChoice(
        payload["kind"],
        payload["rung"],
        cost(payload["cost"]),
        tuple(tuple(pair) for pair in payload["separated_pairs"]),
        tuple(tuple(pair) for pair in payload["unresolved_pairs"]),
        alternatives,
        tuple(payload["expected_outcomes"]),
        payload["selection_rule"],
        tuple((row["rung"], tuple(row["case_ids"])) for row in payload["compared_prior_support"]),
        tuple(payload["missing_access"]),
        payload["missing_measurement"],
        prior,
        contract,
        refusal,
        payload["registration_id"],
        payload["choice_digest"],
        corpus_provenance=payload.get("corpus_provenance"),
        cost_table_provenance=payload.get("cost_table_provenance"),
    )


def _candidate_denial(root: Path, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Run P-OUTCOME's containment probe, retaining its named result verbatim."""
    from reward_lens.outcome import attempt_protected_read, certify_separation
    from reward_lens.partitions import Partition, PartitionKind

    declared = _mapping(candidate.get("acceptance_probe"), "acceptance_probe")
    protected_root = _path_beneath(
        root, declared.get("root"), "acceptance_probe.root", directory=True
    )
    item = _text(declared.get("item"), "acceptance_probe.item")
    partition_id = _text(declared.get("partition_id"), "acceptance_probe.partition_id")
    public_root = root / ".reward-lens" / "improve" / "candidate-visible"
    public_root.mkdir(parents=True, exist_ok=True)
    partition = Partition.from_dir(
        partition_id,
        PartitionKind.PROTECTED_SUITE,
        protected_root,
        log_path=root / ".reward-lens" / "improve" / "acceptance-access.jsonl",
        public_root=public_root,
    )
    try:
        proof = attempt_protected_read(
            partition,
            item,
            cwd=root / ".reward-lens" / "improve" / "candidate-probe",
        )
        certified = certify_separation(proof)
    except Exception as error:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate acceptance boundary was not established: "
            + str(error)
        ) from error
    return dict(certified)


def _compare_request(
    root: Path, candidate: Mapping[str, Any], *, project: Any, outer_request: Any = None
):
    """Use the public Compare SDK, which owns the actual comparison execution."""
    from reward_lens import api

    compare = _mapping(candidate.get("compare"), "compare")
    baseline = _text(compare.get("baseline"), "compare.baseline")
    proposed = _text(compare.get("candidate"), "compare.candidate")
    rungs = compare.get("rungs")
    if (
        not isinstance(rungs, list)
        or not rungs
        or any(type(rung) is not int or rung not in range(1, 7) for rung in rungs)
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate.compare.rungs must be non-empty integers from 1 to 6"
        )
    fields = {
        name: _path_beneath(root, value, f"compare.{name}", directory=False)
        for name, value in compare.items()
        if name
        in {
            "bank",
            "run_dir",
            "baseline_config",
            "candidate_config",
            "selection_config",
            "controlled_update_config",
            "short_fork_config",
            "transfer_config",
        }
    }
    metadata = {} if outer_request is None else _request_metadata(outer_request)
    if outer_request is not None and any(
        name in compare and compare[name] != value for name, value in metadata.items()
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate Compare metadata conflicts with outer request"
        )
    request = api.CompareRequest(
        baseline=baseline,
        candidate=proposed,
        project=root if project is not None else None,
        rungs=tuple(rungs),
        **(
            {"offline": True, "seed": int(compare.get("seed", 20260911))}
            if outer_request is None
            else metadata
        ),
        name=_text(compare.get("name", "improve-comparison"), "compare.name"),
        **fields,
    )
    return request, tuple(rungs), compare


def _compare(root: Path, candidate: Mapping[str, Any], *, project: Any):
    from reward_lens import api

    request, rungs, compare = _compare_request(root, candidate, project=project)
    return api.compare(request), rungs, compare


def _experiment(
    choice: Any,
    comparison: contracts.Assay,
    candidates: list[dict[str, Any]],
    registration: Mapping[str, Any],
) -> contracts.Experiment:
    if choice.rung is None:
        raise ImproveRefusal(
            "IMPROVE_TRADEOFF_UNRESOLVED: unresolved_pairs=" + repr(choice.unresolved_pairs)
        )
    expectation = "; ".join(choice.expected_outcomes)
    return contracts.Experiment(
        id=f"improve.rung_{choice.rung}",
        rung=choice.rung,
        scope="evaluator_comparison",
        registered_expectation=expectation
        or "the registered planner choice names a comparison outcome",
        result=comparison.assay_id,
        informative=None,
        registered_at=registration["registered_at"],
        candidates=[f"candidate.{index + 1}" for index, _ in enumerate(candidates)],
        rejected_rungs=[],
    )


def _record_planner_result(
    root: Path,
    session: ImproveSession,
    candidate: Mapping[str, Any],
    choice: Any,
    ledger: Any,
    evaluator: Any,
    comparison: contracts.Assay,
    request: Any,
) -> dict[str, Any]:
    """Execute the declared paired evaluator and bind its envelopes to this Compare run."""
    import json

    from reward_lens.execution import DEFAULT_LIMITS, default_sandbox
    from reward_lens.graders.python_grader import PythonGrader

    from . import ExecutionEvidence, record_result

    existing = next(
        (
            entry.payload
            for entry in ledger.entries
            if entry.kind == "result" and entry.payload["registration_id"] == choice.registration_id
        ),
        None,
    )
    if existing is not None:
        # PlannerLedger validated both the digest chain and this evidence on
        # open.  A retry must therefore reuse the committed evaluator result,
        # rather than run the grader a second time.
        return {
            "informative": existing["informative"],
            "artifact": existing["evidence"]["artifact_path"],
            "retry_limit": next(
                entry.payload["retry_limit"]
                for entry in ledger.entries
                if entry.kind == "choice"
                and entry.payload["registration_id"] == choice.registration_id
            ),
            "inconclusive_attempts": ledger.inconclusive_attempts(choice.signature),
        }
    declared = _mapping(candidate.get("planner_result"), "planner_result")
    grader_path = _path_beneath(
        root, declared.get("grader"), "planner_result.grader", directory=False
    )
    task = _mapping(declared.get("task"), "planner_result.task")
    if "id" not in task or "prompt" not in task:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate.planner_result.task needs id and prompt"
        )
    pair = tuple(declared.get("pair", ()))
    if len(pair) != 2 or tuple(sorted(pair)) not in choice.separated_pairs:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate.planner_result.pair must be a registered separated pair"
        )
    if "left_response" not in declared or "right_response" not in declared:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: candidate.planner_result needs left_response and right_response"
        )
    grader, sandbox = (
        PythonGrader.from_path(grader_path),
        default_sandbox(require_tier=request.require_tier),
    )
    left = grader.score(
        task, declared["left_response"], sandbox=sandbox, limits=DEFAULT_LIMITS, seed=request.seed
    )
    right = grader.score(
        task, declared["right_response"], sandbox=sandbox, limits=DEFAULT_LIMITS, seed=request.seed
    )
    artifact = (
        root
        / ".reward-lens"
        / "improve"
        / f"planner-result-{choice.registration_id.rsplit(':', 1)[-1][-12:]}.json"
    )
    artifact.write_text(
        json.dumps(
            {
                "schema": "reward-lens.paired-grader-envelope.v1",
                "choice_digest": choice.choice_digest,
                "evaluator_contract_digest": choice.evaluator_contract.contract_digest,
                "pair": list(pair),
                "comparison_assay_id": comparison.assay_id,
                "left": {"envelope": left.to_dict(), "envelope_digest": left.digest()},
                "right": {"envelope": right.to_dict(), "envelope_digest": right.digest()},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    evidence = ExecutionEvidence.issue(
        choice_digest=choice.choice_digest, artifact_path=artifact, evaluator=evaluator
    )
    result = record_result(
        choice,
        ledger,
        evidence=evidence,
        analogy_survived=False if choice.prior_case is not None else None,
    )
    retry_limit = next(
        entry.payload["retry_limit"]
        for entry in ledger.entries
        if entry.kind == "choice" and entry.payload["registration_id"] == choice.registration_id
    )
    return {
        "informative": result.informative,
        "artifact": str(artifact),
        "retry_limit": retry_limit,
        "inconclusive_attempts": ledger.inconclusive_attempts(choice.signature),
    }


def _persist_improve_record(
    record: contracts.Assay,
    *,
    root: Path,
    project: Any,
    candidate: Mapping[str, Any],
    name: str,
) -> tuple[contracts.Assay, dict[str, Path]]:
    """Persist a new improve record through the released Audit writer only."""
    policy_path = candidate.get("policy")
    if policy_path is not None:
        from reward_lens.acceptance import apply_and_persist

        persisted = apply_and_persist(
            record,
            _path_beneath(root, policy_path, "policy", directory=False),
            # Compare has already registered the exact subject version in the
            # project store.  An improve record adds its planner, boundary and
            # policy evidence to that executed record; attempting a second
            # Project.record call would correctly be rejected as a rewrite.
            # The Audit writer's explicit fallback bundle is therefore the
            # durable linked record for this composite workflow.
            project=None,
            name=name,
            fallback_dir=root / "assays",
        )
        return persisted.record, dict(persisted.paths)
    from reward_lens.product.audit.run import persist_assay_bundle

    return persist_assay_bundle(record, project=None, name=name, fallback_dir=root / "assays")


def _request_metadata(request: Any) -> dict[str, Any]:
    return {
        name: getattr(request, name)
        for name in (
            "offline",
            "sandbox",
            "require_tier",
            "seed",
            "max_budget_usd",
            "non_interactive",
        )
    }


def _measurement_refusal(detail: str) -> contracts.UsageError:
    refusal = "IMPROVE_RESUME_REQUIRED: " + detail
    return contracts.UsageError(
        code="RL0003",
        message="improve: " + refusal,
        remediation="correct the declared forecast measurement before issuing its forecast",
        context={"field": "improve", "refusal": "IMPROVE_RESUME_REQUIRED", "detail": refusal},
    )


def _closed(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        if field == "measurement":
            raise _measurement_refusal("forecast.measurement has undeclared or missing fields")
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast." + field + " has undeclared or missing fields"
        )
    return dict(value)


def _forecast_plan(
    root: Path, session: ImproveSession, request: Any, compare_request: Any
) -> dict[str, Any] | None:
    import hashlib
    import json
    import os

    supplied = getattr(request, "forecast", None)
    stored = session.payload.get("forecast_plan")
    if stored is not None:
        if supplied is not None and dict(supplied) != stored["declaration"]:
            raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: registered forecast declaration changed")
        if (
            "sha256:" + hashlib.sha256(Path(stored["frozen_record"]).read_bytes()).hexdigest()
            != stored["record_digest"]
        ):
            raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: frozen forecast input digest changed")
        return dict(stored)
    if supplied is None:
        return None
    declaration = _closed(supplied, {"record", "claim", "target", "outcome", "context"}, "plan")
    outcome = _closed(
        declaration["outcome"],
        {"section", "entry_id", "measurand", "unit", "sampling_unit"},
        "outcome",
    )
    if (
        declaration["target"] != "T3"
        or outcome["section"] != "signal"
        or outcome["entry_id"] != "signal.contrast_fraction"
        or outcome["measurand"]
        != "fraction of declared reward groups with any within-group applied-reward contrast"
        or outcome["unit"] != "fraction"
        or outcome["sampling_unit"] != "task"
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast requires the exact T3 reward-group contrast measurand"
        )
    _text(declaration["claim"], "forecast.claim")
    _text(outcome["entry_id"], "forecast.outcome.entry_id")
    context = _closed(
        declaration["context"],
        {"subject_ref", "task_set", "window", "configuration_digest"},
        "context",
    )
    window = _closed(context["window"], {"checkpoint", "start", "end"}, "window")
    if (
        any(type(v) is not int or v < 0 for v in window.values())
        or window["start"] != window["checkpoint"] + 1
        or window["end"] != window["start"]
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast requires one exact next-step contrast window"
        )
    if context["configuration_digest"] != contracts.digest(compare_request.model_dump(mode="json")):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast configuration differs from registered Compare request"
        )
    for key in ("subject_ref", "task_set"):
        import re

        if not isinstance(context[key], str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", context[key]
        ):
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: forecast context requires authoritative subject/task digests"
            )
    from reward_lens.product.compare.registry import Rung
    from reward_lens.product.compare.verb import _authoritative_factory_subject
    from reward_lens.store import Project

    subject = _authoritative_factory_subject(
        source=Path(compare_request.bank or root),
        project=Project.open(root),
        rungs=tuple(Rung(rung) for rung in compare_request.rungs),
    )
    if (
        context["subject_ref"] != subject.version.digest
        or context["task_set"] != subject.context.task_set
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast subject/task context differs from planned Compare subject"
        )
    source = _path_beneath(root, declaration["record"], "forecast.record", directory=False)
    raw = source.read_bytes()
    document = json.loads(raw)
    issues = document.get("issues")
    if (
        not isinstance(issues, list)
        or len(issues) != 1
        or issues[0].get("target") != "T3"
        or issues[0].get("checkpoint") != window["checkpoint"]
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast input must declare one matching T3 checkpoint"
        )
    from reward_lens.forecast.evidence import ForecastRefusal
    from reward_lens.product.forecast import _issued_from_input, _spec_from_mapping

    spec = _spec_from_mapping(document.get("spec"))
    if spec.horizon != window["end"] - window["start"] + 1:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast spec horizon differs from the registered window"
        )
    try:
        _issued_from_input(spec, issues[0], declaration["claim"])
    except ForecastRefusal as error:
        raise ImproveRefusal(f"IMPROVE_RESUME_REQUIRED: forecast input refused: {error}") from error
    destination = session.state_path.parent / (
        "forecast-input-" + contracts.digest(declaration)[-16:] + ".json"
    )
    if destination.exists():
        if destination.read_bytes() != raw:
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: frozen forecast input cannot be replaced"
            )
    else:
        import os

        with destination.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    plan = {
        "declaration": declaration,
        "frozen_record": str(destination),
        "record_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }
    measurement = session.payload["candidates"][-1]["candidate"].get("forecast_measurement")
    if measurement is not None:
        measurement = _closed(measurement, {"profile", "capture_path"}, "measurement")
        if (
            measurement["profile"] != "finite-bank-selector-recorded-groups/1"
            or tuple(compare_request.rungs) != (2,)
            or compare_request.bank is None
            or compare_request.selection_config is None
        ):
            raise _measurement_refusal(
                "recorded reward groups require the explicit finite-bank selector profile and rung 2 inputs"
            )
        capture = Path(_text(measurement["capture_path"], "forecast_measurement.capture_path"))
        if (
            not capture.is_absolute()
            or not capture.resolve().is_relative_to(root)
            or os.path.lexists(capture)
        ):
            raise _measurement_refusal(
                "recorded-group capture must be a new absolute directory beneath the project"
            )
        config = json.loads(compare_request.selection_config.read_bytes())
        selector = _closed(config.get("selector"), {"path", "entrypoint"}, "selector")
        selector_path = Path(selector["path"])
        if not selector_path.is_absolute():
            selector_path = compare_request.selection_config.parent / selector_path
        selector_path = _path_beneath(
            root, str(selector_path), "forecast_measurement.selector", directory=False
        )
        if selector["entrypoint"] != "score":
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: recorded groups require the supported score selector entrypoint"
            )
        task_rows = config.get("outcome", {}).get("target", {}).get("tasks")
        if not isinstance(task_rows, Mapping) or not task_rows:
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: recorded reward groups need actual declared task inputs"
            )
        plan["measurement_source"] = {
            **measurement,
            "capture_path": str(capture.resolve()),
            "selector_path": str(selector_path),
            "selector_sha256": "sha256:" + hashlib.sha256(selector_path.read_bytes()).hexdigest(),
            "bank_path": str(compare_request.bank),
            "bank_sha256": "sha256:"
            + hashlib.sha256(compare_request.bank.read_bytes()).hexdigest(),
            "tasks": dict(task_rows),
            "run_id": issues[0]["source_name"],
            "claim_boundary": "controlled recorded reward groups; no trainer execution, optimizer update or behavior claim",
        }
    return plan


def _issue_forecast(root: Path, session: ImproveSession, request: Any) -> ImproveSession:
    plan = session.payload.get("forecast_plan")
    if plan is None or session.payload.get("forecast_issue") is not None:
        return session
    from reward_lens import api
    from reward_lens.product.forecast import _load, _path

    project = api._dispatch.context(root)["project"]
    before = _load(_path(project))
    intent = session.payload.get("forecast_intent")
    if intent is None:
        compare_name = session.payload["candidates"][-1]["candidate"]["compare"].get(
            "name", "improve-comparison"
        )
        if (root / "assays" / (compare_name + ".assay.json")).exists():
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: forecast cannot be issued after its comparison exists"
            )
        intent = {
            "ledger_rows": len(before),
            "previous_digest": before[-1]["row_digest"] if before else None,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = {**session.payload, "forecast_intent": intent}
        state.write(session.state_path, payload)
        session = ImproveSession(session.state_path, payload)
    if len(before) < intent["ledger_rows"] or (
        intent["ledger_rows"]
        and before[intent["ledger_rows"] - 1]["row_digest"] != intent["previous_digest"]
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast ledger prefix changed during interrupted issue"
        )
    added = before[intent["ledger_rows"] :]
    issue_assay_id = None
    if not added:
        issue_assay = api.forecast_issue(
            api.ForecastIssueRequest(
                claim=plan["declaration"]["claim"],
                record=Path(plan["frozen_record"]),
                project=root,
                offline=request.offline,
                seed=request.seed,
                non_interactive=request.non_interactive,
            )
        )
        issue_assay_id = issue_assay.assay_id
        added = _load(_path(project))[intent["ledger_rows"] :]
    issued = [row for row in added if row["kind"] == "issue"]
    import json

    from reward_lens.product.forecast import _issued_from_input, _spec_from_mapping

    document = json.loads(Path(plan["frozen_record"]).read_bytes())
    expected = _issued_from_input(
        _spec_from_mapping(document.get("spec")),
        document["issues"][0],
        plan["declaration"]["claim"],
    )
    if (
        len(issued) != 1
        or issued[0]["sealed_digest"] != expected.sealed_digest
        or any(row["kind"] not in {"header", "issue"} for row in added)
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: actual Forecast issue does not bind the registered single issue"
        )
    receipt = {
        "sealed_digest": issued[0]["sealed_digest"],
        "row_digest": issued[0]["row_digest"],
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "assay_id": issue_assay_id,
        "recovered_after_interruption": issue_assay_id is None,
    }
    payload = {**session.payload, "forecast_issue": receipt}
    state.write(session.state_path, payload)
    return ImproveSession(session.state_path, payload)


def _register_forecast_measurement(root: Path, session: ImproveSession) -> ImproveSession:
    plan = session.payload.get("forecast_plan")
    if plan is None or "measurement_source" not in plan:
        return session
    import hashlib

    from reward_lens import api
    from reward_lens.bank import load_bank
    from reward_lens.product.forecast.outcomes import _registration, register_measurement

    project = api._dispatch.context(root)["project"]
    source = plan["measurement_source"]
    bank_path = Path(source["bank_path"])
    if "sha256:" + hashlib.sha256(bank_path.read_bytes()).hexdigest() != source["bank_sha256"]:
        raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: preregistered reward-group bank changed")
    bank = load_bank(bank_path, name=bank_path.stem)
    expected_groups = {
        str(task_id): sorted(response.digest for response in bank.responses_for(task_id))
        for task_id in bank.tasks
    }
    if set(expected_groups) != set(source["tasks"]):
        raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: measured group has no declared task")
    issue = session.payload["forecast_issue"]
    existing = session.payload.get("forecast_measurement_registration")
    if existing is not None:
        bound = _registration(project, issue["sealed_digest"], existing["path"])
        if (
            bound["digest"] != existing["digest"]
            or bound["context"] != plan["declaration"]["context"]
        ):
            raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: measurement registration changed")
        return session
    # A durable intent lets resume recover an original registration if the
    # producer committed just before the session write was interrupted.
    intent = session.payload.get("measurement_registration_intent")
    if intent is None:
        intent = {
            "forecast_id": issue["sealed_digest"],
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = {**session.payload, "measurement_registration_intent": intent}
        state.write(session.state_path, payload)
        session = ImproveSession(session.state_path, payload)
        path = register_measurement(
            project,
            issue["sealed_digest"],
            context=plan["declaration"]["context"],
            capture_path=Path(source["capture_path"]),
            expected_groups=expected_groups,
        )
    else:
        path = (
            root
            / ".reward-lens/forecast-measurements"
            / (issue["sealed_digest"][7:] + ".registration.json")
        )
        if not path.exists():
            path = register_measurement(
                project,
                issue["sealed_digest"],
                context=plan["declaration"]["context"],
                capture_path=Path(source["capture_path"]),
                expected_groups=expected_groups,
            )
    bound = _registration(project, issue["sealed_digest"], path)
    if (
        bound["context"] != plan["declaration"]["context"]
        or bound["capture_path"] != plan["measurement_source"]["capture_path"]
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: recovered measurement registration differs from the immutable plan"
        )
    payload = {
        **session.payload,
        "forecast_measurement_registration": {
            "path": str(path),
            "digest": bound["digest"],
            "registered_at": bound["registered_at"],
        },
    }
    state.write(session.state_path, payload)
    return ImproveSession(session.state_path, payload)


def _compare_with_measurement(
    root: Path, session: ImproveSession, request: Any, compare_request: Any
):
    from reward_lens import api

    plan = session.payload.get("forecast_plan")
    if plan is None or "measurement_source" not in plan:
        return api.compare(compare_request), session
    import hashlib

    from reward_lens.bank import load_bank
    from reward_lens.execution import DEFAULT_LIMITS, default_sandbox
    from reward_lens.graders.python_grader import PythonGrader
    from reward_lens.product.forecast.outcomes import capture_measurement, derive_outcome

    project = api._dispatch.context(root)["project"]
    source = plan["measurement_source"]
    registration = session.payload["forecast_measurement_registration"]
    captured = Path(source["capture_path"]) / "FORECAST_CAPTURE.json"
    saved = session.payload.get("comparison")
    if captured.exists():
        if not isinstance(saved, Mapping):
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: committed measurement lacks its actual comparison receipt"
            )
        comparison = api.open_record(Path(saved["assay"]))
        derive_outcome(
            project,
            session.payload["forecast_issue"]["sealed_digest"],
            registration=Path(registration["path"]),
        )
        return comparison, session
    comparison = None

    def collect():
        nonlocal comparison, session
        if isinstance(saved, Mapping):
            comparison = api.open_record(Path(saved["assay"]))
        else:
            comparison = api.compare(compare_request)
            path = root / "assays" / (str(compare_request.name) + ".assay.json")
            payload = {
                **session.payload,
                "comparison": {"assay": str(path), "assay_id": comparison.assay_id},
            }
            state.write(session.state_path, payload)
            session = ImproveSession(session.state_path, payload)
        context = plan["declaration"]["context"]
        if (
            comparison.subject.version.digest != context["subject_ref"]
            or comparison.subject.context.task_set != context["task_set"]
        ):
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: actual recorded-group experiment has a different subject/task set"
            )
        for path_key, hash_key in (
            ("selector_path", "selector_sha256"),
            ("bank_path", "bank_sha256"),
        ):
            if (
                "sha256:" + hashlib.sha256(Path(source[path_key]).read_bytes()).hexdigest()
                != source[hash_key]
            ):
                raise ImproveRefusal(
                    "IMPROVE_RESUME_REQUIRED: preregistered reward-group source changed"
                )
        grader = PythonGrader.from_path(Path(source["selector_path"]))
        sandbox = default_sandbox(require_tier=request.require_tier)
        bank = load_bank(Path(source["bank_path"]), name=Path(source["bank_path"]).stem)
        rows = []
        for task_id in bank.tasks:
            task = source["tasks"].get(str(task_id))
            if not isinstance(task, Mapping):
                raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: measured group has no declared task")
            for response in bank.responses_for(task_id):
                envelope = grader.score(
                    dict(task),
                    response.text,
                    sandbox=sandbox,
                    limits=DEFAULT_LIMITS,
                    project_dir=root,
                    seed=request.seed,
                )
                if envelope.verdict != "scored" or envelope.score is None:
                    raise ImproveRefusal(
                        "IMPROVE_RESUME_REQUIRED: actual reward-group selector execution refused"
                    )
                rows.append(
                    {
                        "step": context["window"]["start"],
                        "row_id": contracts.digest(
                            {
                                "response_digest": response.digest,
                                "task_id": str(task_id),
                                "step": context["window"]["start"],
                            }
                        ),
                        "response_digest": response.digest,
                        "task_id": str(task_id),
                        "group_id": "recorded-group-" + str(task_id),
                        "reward_applied": float(envelope.score),
                        "components": {"selector_reward": float(envelope.score)},
                        "accepted": True,
                        "run_id": source["run_id"],
                        "subject_ref": comparison.subject.version.digest,
                        "task_set": comparison.subject.context.task_set,
                        "configuration_digest": contracts.digest(
                            compare_request.model_dump(mode="json")
                        ),
                        "scoring_envelope": envelope.to_dict(),
                        "scoring_envelope_digest": envelope.digest(),
                        "reward_application_profile": source["profile"],
                        "claim_boundary": source["claim_boundary"],
                    }
                )
        return rows

    capture_measurement(
        project,
        session.payload["forecast_issue"]["sealed_digest"],
        registration=Path(registration["path"]),
        collector=collect,
    )
    payload = {**session.payload, "forecast_measurement_capture": str(captured)}
    state.write(session.state_path, payload)
    return comparison, ImproveSession(session.state_path, payload)


def _resolve_forecast(
    root: Path, session: ImproveSession, request: Any, comparison: contracts.Assay
):
    import hashlib

    plan = session.payload.get("forecast_plan")
    if plan is None:
        return None, None
    issue = session.payload["forecast_issue"]
    declared = plan["declaration"]
    context = declared["context"]
    if (
        comparison.subject.version.digest != context["subject_ref"]
        or comparison.subject.context.task_set != context["task_set"]
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast outcome subject/task differs from preregistered context"
        )
    if datetime.fromisoformat(
        session.payload["experiment_ordering"]["started_at"]
    ) <= datetime.fromisoformat(issue["observed_at"]):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast was not issued before actual comparison"
        )
    path = Path(session.payload["execution"]["comparison_assay"])
    from reward_lens import api

    reopened = api.open_record(path)
    if reopened.assay_id != comparison.assay_id:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: forecast outcome differs from persisted actual Compare record"
        )
    wanted = declared["outcome"]
    matches = [
        entry
        for entry in getattr(reopened.measurement, wanted["section"])
        if entry.entry_id == wanted["entry_id"]
    ]
    entry = matches[0] if len(matches) == 1 else None
    measurement = None
    if "measurement_source" in plan:
        from reward_lens.product.forecast.outcomes import derive_outcome, verify_outcome

        project = api._dispatch.context(root)["project"]
        registration = session.payload["forecast_measurement_registration"]
        measurement_path = session.state_path.parent / (
            "measured-outcome-" + issue["sealed_digest"][-16:] + ".json"
        )
        if measurement_path.exists():
            import json

            measurement = verify_outcome(
                project, issue["sealed_digest"], json.loads(measurement_path.read_bytes())
            )
        else:
            measurement = derive_outcome(
                project, issue["sealed_digest"], registration=Path(registration["path"])
            )
            state.write_once(measurement_path, measurement)
        if (
            measurement["context"] != context
            or measurement["registration_digest"] != registration["digest"]
            or measurement["native_entry_id"] != wanted["entry_id"]
            or measurement["sampling_unit"] != wanted["sampling_unit"]
            or measurement["measurand"] != wanted["measurand"]
        ):
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: verified native outcome differs from the registered target/context"
            )
        entry = contracts.Entry.model_validate(measurement["steps"][0]["entry"])
    receipt = {
        "plan": plan,
        "issue": issue,
        "comparison_assay_id": reopened.assay_id,
        "comparison_artifact": str(path),
        "comparison_artifact_digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "subject_ref": context["subject_ref"],
        "target": "T3",
        "skill_claim": False,
    }
    ledger = api.forecast_ledger(
        api.ForecastLedgerRequest(
            project=root,
            offline=request.offline,
            seed=request.seed,
            non_interactive=request.non_interactive,
        )
    )
    receipt["ledger_assay_id"] = ledger.assay_id
    if measurement is not None:
        receipt["measured_outcome"] = measurement
        receipt["claim_boundary"] = plan["measurement_source"]["claim_boundary"]
    if (
        entry is None
        or entry.kind != "estimate"
        or entry.state != "complete"
        or entry.value is None
    ):
        receipt["status"] = "COULD_NOT_CHECK"
        absent = contracts.absence(
            "forecast",
            "forecast.improve_t3_outcome",
            wanted["measurand"],
            "actual Compare has no complete numeric entry for the preregistered T3 contrast window",
            "execute a compatible reward-group contrast producer for the same subject, task set, configuration and next window",
            ("T3 forecast resolution",),
            subject_ref=context["subject_ref"],
            provenance=contracts.EntryProvenance(
                started=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                duration_s=0.0,
                sandbox_tier="T0",
                offline=request.offline,
            ),
            scope="evaluator_comparison",
        )
        absent.absence.state = "COULD_NOT_CHECK"
        return receipt, absent
    if (
        entry.subject_ref != context["subject_ref"]
        or any(getattr(entry, key) != wanted[key] for key in ("measurand", "unit", "sampling_unit"))
        or not 0 <= float(entry.value) <= 1
        or entry.n is None
        or entry.n < 1
        or datetime.fromisoformat(entry.provenance.started.replace("Z", "+00:00"))
        < datetime.fromisoformat(issue["observed_at"]).replace(microsecond=0)
        or (
            measurement is None
            and (
                (entry.result or {}).get("window") != context["window"]
                or (entry.result or {}).get("configuration_digest")
                != context["configuration_digest"]
            )
        )
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: actual T3 outcome lacks its exact registered semantic/window/configuration binding"
        )
    from reward_lens.product.forecast import _load, _path

    project = api._dispatch.context(root)["project"]
    rows = _load(_path(project))
    issued_row = next(
        (
            row
            for row in rows
            if row["kind"] == "issue" and row["sealed_digest"] == issue["sealed_digest"]
        ),
        None,
    )
    if issued_row is None or issued_row["row_digest"] != issue["row_digest"]:
        raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: committed forecast issue receipt changed")
    resolved = next(
        (
            row
            for row in rows
            if row["kind"] == "resolve" and row["sealed_digest"] == issue["sealed_digest"]
        ),
        None,
    )
    if resolved is None:
        api.forecast_resolve(
            api.ForecastResolveRequest(
                forecast_id=issue["sealed_digest"],
                outcome=measurement["outcome"] if measurement is not None else str(entry.value),
                project=root,
                seed=request.seed,
                offline=request.offline,
                non_interactive=request.non_interactive,
            )
        )
        rows = _load(_path(project))
        resolved = next(
            row
            for row in rows
            if row["kind"] == "resolve" and row["sealed_digest"] == issue["sealed_digest"]
        )
    elif Decimal(resolved["outcome"]) != Decimal(
        measurement["outcome"] if measurement is not None else str(entry.value)
    ):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: committed forecast resolution differs from actual outcome"
        )
    ledger = api.forecast_ledger(
        api.ForecastLedgerRequest(
            project=root,
            offline=request.offline,
            seed=request.seed,
            non_interactive=request.non_interactive,
        )
    )
    receipt.update(
        {
            "status": "RESOLVED",
            "entry_id": entry.entry_id,
            "entry_digest": contracts.digest(entry),
            "resolve": resolved,
            "ledger_assay_id": ledger.assay_id,
        }
    )
    from reward_lens.forecast.evidence import BASELINES

    if set(resolved["baseline_scores"]) != set(BASELINES):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: resolved forecast lost one of four baseline scores"
        )
    score_method = contracts.Method(
        id="forecast.improve_t3_score",
        version="1.0.0",
        params_digest=contracts.digest(
            {
                "forecast_sealed_digest": issue["sealed_digest"],
                "measurement_digest": measurement["digest"]
                if measurement is not None
                else contracts.digest(entry),
                "context": context,
            }
        ),
        procedure="reopen and verify the registered next-step reward-group measurement, resolve its frozen T3 issue with the existing proper-score engine, and retain the four baseline scores",
    )
    scored_entry = contracts.estimate(
        "forecast",
        "forecast.improve_t3_score",
        "proper score of the prospective registered T3 reward-group contrast forecast",
        float(resolved["score"]),
        "CRPS/WIS",
        1,
        "registered window",
        contracts.Uncertainty(
            method=contracts.NO_INTERVAL,
            level=0.95,
            reason="one prospectively scored window; no sequential skill claim for T3",
            clusters=1,
        ),
        subject_ref=context["subject_ref"],
        provenance=contracts.EntryProvenance(
            started=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_s=0.0,
            sandbox_tier="T0",
            offline=request.offline,
        ),
        method=score_method,
        result={
            "target": "T3",
            "outcome": measurement["outcome"] if measurement is not None else str(entry.value),
            "baseline_scores": resolved["baseline_scores"],
            "hardest_difference": resolved["hardest_difference"],
            "forecast_sealed_digest": issue["sealed_digest"],
            "comparison_assay_id": comparison.assay_id,
            "context": context,
            "skill_claim": False,
        },
        scope="evaluator_comparison",
    )
    return receipt, scored_entry


def run(request: Any, *, project: Any = None, sandbox: Any = None) -> contracts.Assay:
    """Execute the persisted improve route through planner, outcome, Compare and Audit storage.

    Candidate descriptions carry declarations and file references only.  Results are taken from
    the executed Compare record and a policy decision, when the owner supplied a policy, is made
    by the accepted policy module before the final record is written and reopened.
    """
    root = Path(request.path).resolve()
    metadata = _request_metadata(request)
    if request.require_tier is not None:
        from reward_lens.execution import default_sandbox

        default_sandbox(require_tier=request.require_tier)
    if not root.is_dir():
        raise ImproveRefusal(
            f"IMPROVE_RESUME_REQUIRED: path={root} must be an existing project directory"
        )
    if request.describe is not None and request.resume is not None:
        raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: use either describe or resume, not both")
    if request.describe is not None:
        session = ImproveSession.describe(root, request.describe, session_id="current")
    elif request.resume is not None:
        session = ImproveSession.resume(state.session_path(root, request.resume))
    else:
        raise ImproveRefusal("IMPROVE_RESUME_REQUIRED: describe or resume is required")
    saved_metadata = session.payload.get("request_metadata")
    if saved_metadata is not None and saved_metadata != metadata:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: resume cannot replace registered request metadata"
        )
    registration = session.payload.get("preregistration")
    if isinstance(registration, Mapping):
        try:
            record.require_registration(dict(registration))
        except (OSError, KeyError, ValueError) as error:
            raise ImproveRefusal(str(error)) from error
    elif session.state in {"execute", "decide", "export"}:
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: executed session lacks a pre-experiment registration"
        )
    if session.state == "export":
        from reward_lens import api

        return api.open_record(Path(session.payload["experiments"][-1]["assay"]))
    if request.mark_pair is not None:
        left, right = (
            value if Path(value).is_absolute() else root / value for value in request.mark_pair
        )
        session = session.mark_pair(left, right, request.why or "")
    if session.state == "describe":
        if request.candidate is None or request.candidate_source is None:
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: candidate_source and candidate are required to execute improve"
            )
        candidate = dict(request.candidate)
        session = session.propose(source=request.candidate_source, candidate=candidate)
    else:
        if not session.payload.get("candidates"):
            raise ImproveRefusal(
                "IMPROVE_RESUME_REQUIRED: stored session has no candidate declaration"
            )
        stored = session.payload["candidates"][-1]
        candidate = dict(stored["candidate"])
        if request.candidate is not None:
            corrected = dict(request.candidate)
            if isinstance(registration, Mapping) and corrected != candidate:
                raise ImproveRefusal(
                    "IMPROVE_RESUME_REQUIRED: resume cannot replace preregistered execution inputs"
                )
            if session.state != "propose" and corrected.get("planner") != candidate.get("planner"):
                raise ImproveRefusal(
                    "IMPROVE_RESUME_REQUIRED: resume cannot replace the registered planner declaration"
                )
            if corrected != candidate:
                revised = {
                    **session.payload,
                    "candidates": [
                        *session.payload["candidates"][:-1],
                        {**stored, "candidate": corrected},
                    ],
                    "candidate_revisions": [
                        *session.payload.get("candidate_revisions", ()),
                        candidate,
                    ],
                }
                state.write(session.state_path, revised)
                session = ImproveSession(session.state_path, revised)
                candidate = corrected
    if session.state == "propose":
        choice, session, ledger, evaluator = _planner_choice(root, session, candidate, request)
        if choice.kind != "complete":
            session.require_resolved_tradeoff(tuple(choice.unresolved_pairs))
        session = session.plan(choice)
    else:
        choice = _restore_choice(_mapping(session.payload.get("plan"), "stored_plan"))
        session, ledger, evaluator = _planner_runtime(root, session)
    if session.state == "decide":
        pending = session.payload.get("pending_export")
        if not isinstance(pending, Mapping):
            compare_name = _text(
                _mapping(candidate.get("compare"), "compare").get("name", "improve-comparison"),
                "compare.name",
            )
            recovered = (
                root
                / "assays"
                / f"improve-{getattr(request, 'name', None) or compare_name}.assay.json"
            )
            from reward_lens import api

            reopened = api.open_record(recovered)
            pending = {"assay": str(recovered), "assay_id": reopened.assay_id}
            payload = {**session.payload, "pending_export": dict(pending)}
            state.write(session.state_path, payload)
            session = ImproveSession(session.state_path, payload)
        session.export(dict(pending))
        from reward_lens import api

        return api.open_record(Path(pending["assay"]))
    if session.state == "plan":
        compare_payload = _mapping(candidate.get("compare"), "compare")
        saved_comparison = session.payload.get("comparison")
        if isinstance(saved_comparison, Mapping):
            from reward_lens import api

            comparison = api.open_record(Path(saved_comparison["assay"]))
            rungs = tuple(compare_payload.get("rungs", ()))
            if (session.payload.get("forecast_plan") or {}).get("measurement_source") is not None:
                compare_request, _, _ = _compare_request(
                    root, candidate, project=project, outer_request=request
                )
                session = _register_forecast_measurement(root, session)
                comparison, session = _compare_with_measurement(
                    root, session, request, compare_request
                )
                payload = {
                    **session.payload,
                    "experiment_ordering": {
                        **session.payload["experiment_ordering"],
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                state.write(session.state_path, payload)
                session = ImproveSession(session.state_path, payload)
        else:
            compare_request, rungs, compare_payload = _compare_request(
                root, candidate, project=project, outer_request=request
            )
            forecast_plan = _forecast_plan(root, session, request, compare_request)
            if choice.rung not in rungs:
                raise ImproveRefusal(
                    "IMPROVE_TRADEOFF_UNRESOLVED: comparison must execute the registered discriminator"
                )
            try:
                registration = record.preregister(
                    session.state_path,
                    choice=choice,
                    compare_request=compare_request,
                    candidate=dict(candidate),
                    forecast_plan=forecast_plan,
                )
            except (OSError, KeyError, ValueError) as error:
                raise ImproveRefusal(str(error)) from error
            payload = {
                **session.payload,
                "preregistration": registration,
                "request_metadata": metadata,
                "forecast_plan": forecast_plan,
            }
            state.write(session.state_path, payload)
            session = ImproveSession(session.state_path, payload)
            from reward_lens import api

            session = _issue_forecast(root, session, request)
            session = _register_forecast_measurement(root, session)
            payload = {
                **session.payload,
                "experiment_ordering": {"started_at": datetime.now(timezone.utc).isoformat()},
            }
            state.write(session.state_path, payload)
            session = ImproveSession(session.state_path, payload)
            comparison, session = _compare_with_measurement(root, session, request, compare_request)
            payload = {
                **session.payload,
                "experiment_ordering": {
                    **session.payload["experiment_ordering"],
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            state.write(session.state_path, payload)
            session = ImproveSession(session.state_path, payload)
        if choice.rung not in rungs:
            raise ImproveRefusal(
                "IMPROVE_TRADEOFF_UNRESOLVED: planner chose rung="
                + str(choice.rung)
                + "; candidate.compare.rungs must execute the registered discriminator"
            )
        comparison_path = root / "assays" / f"{compare_payload['name']}.assay.json"
        if not isinstance(saved_comparison, Mapping):
            payload = {
                **session.payload,
                "comparison": {"assay": str(comparison_path), "assay_id": comparison.assay_id},
            }
            state.write(session.state_path, payload)
            session = ImproveSession(session.state_path, payload)
        saved_boundary = session.payload.get("acceptance_boundary")
        if isinstance(saved_boundary, Mapping):
            boundary = dict(saved_boundary)
        else:
            boundary = _candidate_denial(root, candidate)
            payload = {**session.payload, "acceptance_boundary": boundary}
            state.write(session.state_path, payload)
            session = ImproveSession(session.state_path, payload)
        session = session.execute(
            {
                "comparison_assay_id": comparison.assay_id,
                "comparison_assay": str(comparison_path),
                "acceptance_boundary": boundary,
            }
        )
    else:
        from reward_lens import api

        comparison = api.open_record(Path(session.payload["execution"]["comparison_assay"]))
        compare_payload = _mapping(candidate.get("compare"), "compare")
        boundary = dict(session.payload["execution"]["acceptance_boundary"])
    if session.state == "execute":
        planner_result = _record_planner_result(
            root, session, candidate, choice, ledger, evaluator, comparison, request
        )
        payload = {
            **session.payload,
            "execution": {**session.payload["execution"], "planner_result": planner_result},
        }
        state.write(session.state_path, payload)
        session = ImproveSession(session.state_path, payload)
    result = comparison.model_copy(deep=True)
    forecast_receipt, forecast_entry = _resolve_forecast(root, session, request, comparison)
    if forecast_entry is not None:
        result.measurement.forecast = [forecast_entry]

    registration = session.payload.get("preregistration")
    if not isinstance(registration, Mapping):
        raise ImproveRefusal(
            "IMPROVE_RESUME_REQUIRED: executed session lacks a pre-experiment registration"
        )
    try:
        record.require_registration(dict(registration))
    except (OSError, KeyError, ValueError) as error:
        raise ImproveRefusal(str(error)) from error
    experiment = _experiment(choice, result, session.payload["candidates"], registration)
    planner_result = _mapping(
        session.payload["execution"].get("planner_result"), "execution.planner_result"
    )
    experiment.informative = bool(planner_result["informative"])
    result.experiments = [*(result.experiments or []), experiment]
    result.extensions = dict(result.extensions or {})
    result.extensions[_EXTENSION] = {
        "session_path": str(session.state_path),
        "planner_choice": record.choice_payload(choice),
        "preregistration": dict(registration),
        "experiment_ordering": dict(session.payload["experiment_ordering"]),
        "candidate_sources": [row["source"] for row in session.payload["candidates"]],
        "non_equivalent_pairs": list(session.payload.get("pairs", ())),
        "acceptance_boundary": boundary,
        "comparison_assay_id": comparison.assay_id,
        "planner_result": planner_result,
        "request_metadata": metadata,
        "forecast": forecast_receipt,
    }
    result.producer = contracts.Producer(
        tool=result.producer.tool,
        version=result.producer.version,
        method_set=contracts.digest(
            {"methods": sorted({entry.method.id for entry in result.entries()})}
        ),
    )
    result.holes_from_entries()
    result.assay_id = contracts.digest(result)
    final_name = "improve-" + (
        str(getattr(request, "name", None) or compare_payload.get("name", "run"))
    )
    final_path = root / "assays" / f"{final_name}.assay.json"
    if final_path.is_file():
        from reward_lens import api

        reopened = api.open_record(final_path)
        extension = (reopened.extensions or {}).get(_EXTENSION, {})
        if extension.get("session_path") == str(session.state_path):
            persisted, paths = reopened, {"assay": final_path}
        else:
            persisted, paths = _persist_improve_record(
                result, root=root, project=project, candidate=candidate, name=final_name
            )
    else:
        persisted, paths = _persist_improve_record(
            result, root=root, project=project, candidate=candidate, name=final_name
        )
    if persisted.decision.state == "qualified":
        outcome, reason = "qualified", "policy qualified the persisted comparison evidence"
    elif persisted.decision.state == "rejected":
        outcome, reason = "all_rejected", "; ".join(persisted.decision.reasons)
    else:
        outcome, reason = "unresolved_tradeoff", "; ".join(persisted.decision.reasons)
    if session.state == "execute":
        session = session.decide(outcome, reason=reason)
        pending = {
            "assay": str(paths["assay"]),
            "assay_id": persisted.assay_id,
            **{key: str(paths[key]) for key in ("manifest", "report") if key in paths},
        }
        session = ImproveSession(session.state_path, {**session.payload, "pending_export": pending})
        state.write(session.state_path, session.payload)
    if session.state == "decide":
        session.export(dict(session.payload["pending_export"]))
    return persisted
