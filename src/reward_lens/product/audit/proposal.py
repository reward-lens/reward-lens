"""One public caller proposal, executed by Audit's normal persistence route."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reward_lens import contracts
from reward_lens.api.requests import AuditRequest, SeekerProposalRequest
from reward_lens.api.results import SeekerProposalResult
from reward_lens.execution import Limits, default_sandbox
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.instruments.seeker.actors.agent import AgentActor, propose_tool
from reward_lens.instruments.seeker.protocol import GraderScorer
from reward_lens.outcome.project_execution import (
    ProjectOutcomeExecution,
    load_project_outcome_execution,
)
from reward_lens.store import Project

from . import registry
from .run import _planned_project_method_ids, _supervised_outcome_checker, resolve, run

__all__ = ["run_seeker_proposal"]


def _refuse(field: str, detail: str) -> contracts.UsageError:
    return contracts.UsageError(
        code="RL0003",
        message=f"seeker proposal {detail}",
        remediation="supply one declared project target task and a candidate response",
        context={"field": field, "detail": detail},
    )


def _declared_task(subject: Any, *, task_id: str) -> dict[str, Any]:
    """Read the declared project task bytes and select exactly one named task."""
    path = getattr(subject, "tasks_path", None)
    if path is None or not Path(path).is_file():
        raise _refuse("task_id", "needs a readable declared project task set")
    try:
        rows = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line]
    except (OSError, ValueError) as error:
        raise _refuse("task_id", f"cannot read the declared task set: {error}") from error
    matching = [row for row in rows if isinstance(row, dict) and str(row.get("id", "")) == task_id]
    if len(matching) != 1:
        raise _refuse("task_id", "must name exactly one task in the declared project task set")
    return dict(matching[0])


def _saved_path(project: Project, assay_id: str) -> Path:
    """Locate the bytes just persisted by Audit's sole writer, never reconstruct them."""
    for candidate in project.assays_dir.glob("*.assay.json"):
        try:
            payload = json.loads(candidate.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("assay_id") == assay_id:
            return candidate
    raise RuntimeError("Audit returned an assay whose persisted bytes could not be reopened")


def run_seeker_proposal(request: SeekerProposalRequest) -> SeekerProposalResult:
    """Score one response through the declared supervisor and normal saved Audit record."""
    project = Project.open(request.project_path)
    audit_request = AuditRequest(path=request.project_path, run_dir=request.run_dir)
    sandbox = default_sandbox()
    subject = resolve(audit_request, project)
    expected_subject_ref = project.version(
        methods=_planned_project_method_ids(registry.discover())
    ).digest()
    loaded = load_project_outcome_execution(
        project,
        expected_subject_ref=expected_subject_ref,
        sandbox=sandbox,
        limits=Limits(network=False),
    )
    if not isinstance(loaded, ProjectOutcomeExecution) or not loaded.qualified:
        raise _refuse("task_id", "needs a qualified declared project outcome target")
    task = _declared_task(subject, task_id=request.task_id)
    if contracts.digest(task) not in loaded.target_items_by_task_digest:
        raise _refuse("task_id", "is not an exact declared project outcome target")
    scorer = GraderScorer(
        PythonGrader.from_path(subject.grader_path, subject.entrypoint),
        outcome_check=_supervised_outcome_checker(loaded),
        sandbox=sandbox,
        limits=Limits(network=False),
    )

    def caller(session: str):
        def offer(_pending: Mapping[str, Any]) -> dict[str, Any]:
            return propose_tool(
                {
                    "session": session,
                    "task_id": request.task_id,
                    "response": request.response,
                    "rationale": request.rationale,
                }
            )

        return offer

    actor = AgentActor(scorer=scorer, caller=caller)
    record = run(
        audit_request,
        project=project,
        sandbox=sandbox,
        seeker_actor=actor,
        seeker_tasks=(task,),
        seeker_intent=getattr(project.config, "success", ""),
    )
    record_path = _saved_path(project, record.assay_id)
    reopened = Project.open(project.root).open_record(record_path.name.removesuffix(".assay.json"))
    seeker_entry = next(
        (
            entry
            for entry in reopened.measurement.exploits
            if entry.entry_id.startswith("exploits.seeker.")
            and isinstance(entry.result, dict)
            and isinstance(entry.result.get("actor_record"), dict)
        ),
        None,
    )
    if seeker_entry is None:
        raise RuntimeError("the saved Audit record did not retain seeker evidence")
    actor_record = dict(seeker_entry.result["actor_record"])
    scores = actor_record.get("scores")
    if not isinstance(scores, list) or len(scores) != 1 or not isinstance(scores[0], dict):
        raise RuntimeError("the saved Audit record did not retain exactly one actor proposal")
    score = scores[0]
    evidence = {
        "assay_id": reopened.assay_id,
        "actor": actor_record.get("actor"),
        "proposal_id": score.get("proposal", {}).get("proposal_id"),
        "outcome_evidence": score.get("outcome_evidence"),
    }
    return SeekerProposalResult(
        record_path=str(record_path),
        summary="saved one qualified seeker proposal",
        evidence=evidence,
    )
