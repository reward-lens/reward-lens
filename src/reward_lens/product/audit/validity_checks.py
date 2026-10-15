"""Executed, deliberately narrow dynamic and task-set validity controls for audit.

Each completed entry reports the exact invocations observed. An inability to force a wall or
resource breach, missing known-good case, or unqualified outcome is an absence, never a pass.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reward_lens import contracts
from reward_lens.execution import Limits
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.instruments.base import RunContext
from reward_lens.outcome.resolution import OutcomeState

from . import absences, instruments


def _dependencies(suffix: str) -> list[str]:
    """Bind each result to every project dimension whose bytes its procedure reads."""
    dynamic = {item[0] for item in absences.DYNAMIC_CHECKS}
    if suffix in dynamic:
        return [
            "digest:source",
            "digest:environment",
            "digest:scorer_config",
            "digest:task_distribution",
            "digest:samples",
        ]
    if suffix == "task_validity":
        return ["digest:source", "digest:task_distribution", "digest:outcome_protocol"]
    if suffix == "pass_rate_floor":
        return [
            "digest:source",
            "digest:task_distribution",
            "digest:samples",
            "digest:outcome_protocol",
        ]
    return [
        "digest:source",
        "digest:environment",
        "digest:scorer_config",
        "digest:task_distribution",
        "digest:outcome_protocol",
    ]


def _method(suffix: str, procedure: str, parameters: dict[str, Any]) -> contracts.Method:
    return contracts.Method(
        id=f"validity.{suffix}",
        version="1.0.0",
        params_digest=contracts.digest(parameters),
        procedure=procedure,
        credited_to="reward_lens.product.audit.validity_checks",
    )


def _check(
    ctx: RunContext,
    subject_ref: str,
    suffix: str,
    measurand: str,
    *,
    passed: bool,
    predicate: str,
    scope_tested: str,
    procedure: str,
    parameters: dict[str, Any],
    result: dict[str, Any],
    limitations: list[str],
    started: float,
) -> contracts.Entry:
    return contracts.Entry(
        entry_id=f"validity.{suffix}",
        section="validity",
        kind="check",
        measurand=measurand,
        method=_method(suffix, procedure, parameters),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=_dependencies(suffix),
        state="complete",
        provenance=ctx.provenance(duration_s=ctx.clock() - started),
        limitations=limitations,
        check=contracts.Check(
            predicate=predicate,
            passed=passed,
            scope_tested=scope_tested,
        ),
        result=result,
    )


def _absence(
    ctx: RunContext,
    subject_ref: str,
    suffix: str,
    measurand: str,
    claim: str,
    missing: str,
    remedy: str,
    *,
    started: float,
    need: str | None = None,
    limitations: tuple[str, ...] = (),
) -> contracts.Entry:
    entry = ctx.absence(
        "validity",
        f"validity.{suffix}",
        measurand,
        missing,
        remedy,
        (claim,),
        subject_ref=subject_ref,
        depends_on=_dependencies(suffix),
        duration_s=ctx.clock() - started,
        limitations=limitations,
        method=_method(suffix, "Attempt the named control on supplied audit inputs.", {}),
    )
    return absences.needs_note(entry, need) if need else entry


def _pair(rollout: Any) -> tuple[dict[str, Any], Any]:
    return dict(rollout.inputs["task"]), rollout.inputs["response"]


def _observation(envelope: Any) -> dict[str, Any]:
    return {
        "verdict": envelope.verdict,
        "score": envelope.score,
        "breach": (envelope.observations or {}).get("sandbox", {}).get("breach"),
        "request_digest": envelope.request_digest,
    }


def _scored(*envelopes: Any) -> bool:
    return all(item.verdict == "scored" and item.score is not None for item in envelopes)


def _outcome(execution: Any, task: dict[str, Any], response: Any) -> OutcomeState | None:
    if execution is None or not getattr(execution, "qualified", False):
        return None
    item_id = execution.target_items_by_task_digest.get(contracts.digest(task))
    if item_id is None:
        return None
    try:
        return execution.resolve_for_actual(
            task=task, response=response, item_id=item_id
        ).outcome.state
    except (ValueError, TypeError, RuntimeError, OSError):
        return None


def run(
    *,
    ctx: RunContext,
    subject_ref: str,
    vut: Any,
    corpus: Any,
    project: Any,
    sandbox: Any,
    outcome_execution: Any,
    has_task_set: bool,
    has_response_bank: bool,
    grader_failure: Any,
    tasks_path: Path | None = None,
) -> list[contracts.Entry]:
    """Return four dynamic then three task-set entries, each checked or input-qualified absent."""
    entries: list[contracts.Entry] = []
    raw_tasks = (
        instruments.read_jsonl(tasks_path)
        if has_task_set and tasks_path is not None and tasks_path.is_file()
        else []
    )
    # The supervised evaluator uses PythonGrader's {id, prompt} input contract. Audit joins
    # task_id rows by adding id before scoring; use that exact payload for declared targets.
    outcome_tasks = [
        instruments.normalised_task(task, f"task-{index}") for index, task in enumerate(raw_tasks)
    ]
    project_dir = getattr(project, "root", None) if project is not None else None
    rollout_a = corpus.rollouts[0] if corpus.rollouts else None
    rollout_b = next(
        (r for r in corpus.rollouts[1:] if rollout_a is not None and _pair(r) != _pair(rollout_a)),
        None,
    )
    grader = None
    if sandbox is not None and grader_failure is None and rollout_a is not None:
        try:
            grader = PythonGrader.from_path(vut.source_path, vut.entrypoint)
        except (OSError, ValueError, RuntimeError):
            pass

    def score(rollout: Any, limits: Limits = Limits()) -> Any:
        task, response = _pair(rollout)
        return grader.score(task, response, sandbox=sandbox, limits=limits, project_dir=project_dir)

    suffix, measurand, _, claim = absences.DYNAMIC_CHECKS[0]
    started = ctx.clock()
    if grader is None:
        entries.append(
            _absence(
                ctx,
                subject_ref,
                suffix,
                measurand,
                claim,
                "a runnable local grader, sandbox, and probe input were unavailable",
                "supply a runnable local grader and sandbox",
                started=started,
            )
        )
        first = second = None
    else:
        first, second = score(rollout_a), score(rollout_a)
        if not _scored(first, second):
            entries.append(
                _absence(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    claim,
                    "the two fresh process invocations did not both return scores",
                    "supply an input the grader scores under the sandbox",
                    started=started,
                    limitations=(f"observed verdicts: {first.verdict}, {second.verdict}",),
                )
            )
        else:
            observations = [_observation(first), _observation(second)]
            entries.append(
                _check(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    passed=first.score == second.score,
                    predicate="Two independently staged invocations of the same input return one score.",
                    scope_tested=f"one input {first.request_digest}, two fresh grader processes",
                    procedure="Stage the project afresh and score the same input in two sandboxed processes.",
                    parameters={"repeats": 2, "request_digest": first.request_digest},
                    result={"observations": observations, "processes": 2},
                    limitations=[
                        "Only score-visible state transfer for this input was tested; unobserved state may persist."
                    ],
                    started=started,
                )
            )

    suffix, measurand, _, claim = absences.DYNAMIC_CHECKS[1]
    started = ctx.clock()
    if grader is None or rollout_b is None or first is None or not _scored(first):
        entries.append(
            _absence(
                ctx,
                subject_ref,
                suffix,
                measurand,
                claim,
                "two distinct scoreable episode inputs were unavailable",
                "supply two distinct responses that the grader scores",
                started=started,
            )
        )
    else:
        serial_b, serial_a = score(rollout_b), score(rollout_a)
        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent_a = pool.submit(score, rollout_a)
            concurrent_b = pool.submit(score, rollout_b)
            concurrent_a, concurrent_b = concurrent_a.result(), concurrent_b.result()
        observations = [first, serial_b, serial_a, concurrent_a, concurrent_b]
        if not _scored(*observations):
            entries.append(
                _absence(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    claim,
                    "a serial or concurrent episode did not produce a score",
                    "supply scoreable episode inputs under this sandbox",
                    started=started,
                    limitations=("verdicts: " + ", ".join(item.verdict for item in observations),),
                )
            )
        else:
            entries.append(
                _check(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    passed=(
                        first.score == serial_a.score == concurrent_a.score
                        and serial_b.score == concurrent_b.score
                    ),
                    predicate="A-B-A serial and A/B concurrent scoring agree with their per-input baselines.",
                    scope_tested="two distinct inputs, three serial and two concurrent isolated grader processes",
                    procedure="Score A, B, A in separate processes, then score A and B concurrently.",
                    parameters={"a": first.request_digest, "b": serial_b.request_digest},
                    result={"observations": [_observation(item) for item in observations]},
                    limitations=[
                        "Only these two inputs and this concurrency level were exercised."
                    ],
                    started=started,
                )
            )

    for suffix, measurand, _, claim, limits, expected in (
        (*absences.DYNAMIC_CHECKS[2], Limits(wall_s=0.001), "wall"),
        (*absences.DYNAMIC_CHECKS[3], Limits(wall_s=5.0, memory_bytes=32 << 20), "memory"),
    ):
        started = ctx.clock()
        if grader is None:
            entries.append(
                _absence(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    claim,
                    "a runnable local grader, sandbox, and probe input were unavailable",
                    "supply a runnable local grader and sandbox",
                    started=started,
                )
            )
            continue
        observed = score(rollout_a, limits)
        breach = _observation(observed)["breach"]
        if breach != expected:
            entries.append(
                _absence(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    claim,
                    f"the probe did not observe an enforced {expected} breach",
                    f"run a workload that reaches the enforced {expected} budget",
                    started=started,
                    limitations=(f"observed verdict={observed.verdict}, breach={breach}",),
                )
            )
            continue
        entries.append(
            _check(
                ctx,
                subject_ref,
                suffix,
                measurand,
                passed=(
                    observed.verdict == ("timeout" if expected == "wall" else "resource_exhausted")
                    and observed.score is None
                ),
                predicate=f"An enforced {expected} breach is reported without a numeric reward.",
                scope_tested=f"one grader invocation with {expected} limit",
                procedure=f"Run the supplied grader input under a restrictive {expected} sandbox limit.",
                parameters={"wall_s": limits.wall_s, "memory_bytes": limits.memory_bytes},
                result={"observation": _observation(observed)},
                limitations=[
                    "This tests the grader adapter's breach verdict, not trainer treatment of that verdict."
                ],
                started=started,
            )
        )

    suffix, measurand, claim = absences.TASK_SET_CHECKS[0]
    started = ctx.clock()
    floor = getattr(outcome_execution, "capability_floor", None)
    if not raw_tasks:
        entries.append(
            _absence(
                ctx,
                subject_ref,
                suffix,
                measurand,
                claim,
                "a complete task set was unavailable",
                "supply --tasks <file>",
                started=started,
                need=absences.TASK_SET,
            )
        )
        good_cases: list[tuple[dict[str, Any], Any]] = []
    elif floor is None or not getattr(outcome_execution, "qualified", False):
        entries.append(
            _absence(
                ctx,
                subject_ref,
                suffix,
                measurand,
                claim,
                "qualified, explicitly declared known-good outcome cases were unavailable",
                "declare known-good cases and qualify an independent outcome execution",
                started=started,
                need=absences.PROTECTED_CHECK,
            )
        )
        good_cases = []
    else:
        cases = list(floor.cases)
        by_digest = {contracts.digest(task): task for task in outcome_tasks}
        matched = [
            (dict(case.task), case.response)
            for case in cases
            if contracts.digest(dict(case.task)) in by_digest
        ]
        try:
            floor_resolutions = outcome_execution.resolve_capability_floor()
            states_by_case = {
                (
                    contracts.digest(dict(case.task)),
                    contracts.digest({"response": case.response}),
                ): resolution.outcome.state
                for case, resolution in zip(cases, floor_resolutions)
            }
            states = [
                states_by_case.get(
                    (contracts.digest(task), contracts.digest({"response": response}))
                )
                for task, response in matched
            ]
        except (ValueError, TypeError, RuntimeError, OSError):
            states = [None] * len(matched)
        passes = {
            contracts.digest(task)
            for (task, _), state in zip(matched, states)
            if state is OutcomeState.PASS
        }
        good_cases = [
            (task, response)
            for (task, response), state in zip(matched, states)
            if state is OutcomeState.PASS
        ]
        missing = sorted(set(by_digest) - passes)
        if any(state is None or state is OutcomeState.UNQUALIFIED for state in states):
            entries.append(
                _absence(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    claim,
                    "a declared known-good outcome case was not resolved by the qualified check",
                    "repair the independent outcome execution",
                    started=started,
                    limitations=(
                        f"resolved {sum(state is not None for state in states)} of {len(states)} cases",
                    ),
                )
            )
        else:
            entries.append(
                _check(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    passed=not missing,
                    predicate="Every supplied task has a declared candidate accepted by the independent outcome check.",
                    scope_tested=f"{len(raw_tasks)} supplied tasks and {len(matched)} declared candidates",
                    procedure="Join exact declared known-good tasks to the supplied task set and execute their protected outcome checks.",
                    parameters={"task_digests": sorted(by_digest), "candidate_count": len(matched)},
                    result={
                        "task_count": len(raw_tasks),
                        "candidate_count": len(matched),
                        "accepted_task_count": len(passes),
                        "without_passing_candidate": missing,
                    },
                    limitations=[
                        "A missing passing candidate does not prove the task is impossible."
                    ],
                    started=started,
                )
            )

    suffix, measurand, claim = absences.TASK_SET_CHECKS[1]
    started = ctx.clock()
    if not raw_tasks:
        entries.append(
            _absence(
                ctx,
                subject_ref,
                suffix,
                measurand,
                claim,
                "a complete task set was unavailable",
                "supply --tasks <file>",
                started=started,
                need=absences.TASK_SET,
            )
        )
    elif not has_response_bank or not getattr(corpus, "rollouts", ()):
        entries.append(
            _absence(
                ctx,
                subject_ref,
                suffix,
                measurand,
                claim,
                "sampled responses joined to the task set were unavailable",
                "supply --responses <file>",
                started=started,
                need=absences.SAMPLED_RESPONSES,
            )
        )
    elif not getattr(outcome_execution, "qualified", False):
        entries.append(
            _absence(
                ctx,
                subject_ref,
                suffix,
                measurand,
                claim,
                "a qualified independent outcome check was unavailable",
                "qualify an independent outcome execution",
                started=started,
                need=absences.PROTECTED_CHECK,
            )
        )
    else:
        tasks_by_id = {str(task.get("task_id") or task.get("id")): task for task in outcome_tasks}
        attempted = {key: 0 for key in tasks_by_id}
        passed = {key: 0 for key in tasks_by_id}
        unknown = 0
        for rollout in corpus.rollouts:
            task, response = _pair(rollout)
            key = str(task.get("task_id") or task.get("id"))
            if key not in tasks_by_id:
                unknown += 1
                continue
            state = _outcome(outcome_execution, tasks_by_id[key], response)
            if state is None or state is OutcomeState.UNQUALIFIED:
                unknown += 1
                continue
            attempted[key] += 1
            passed[key] += int(state is OutcomeState.PASS)
        if unknown:
            entries.append(
                _absence(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    claim,
                    "one or more sampled responses had an unqualified outcome",
                    "supply declared outcome targets for every sampled task",
                    started=started,
                    limitations=(f"unqualified responses: {unknown}",),
                )
            )
        else:
            uninformative = sorted(key for key in attempted if attempted[key] and passed[key] == 0)
            unsampled = sorted(key for key in attempted if attempted[key] == 0)
            entries.append(
                _check(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    passed=not uninformative and not unsampled,
                    predicate="Every task has a sampled response accepted by the independent outcome check.",
                    scope_tested=f"{len(corpus.rollouts)} sampled responses across {len(raw_tasks)} tasks",
                    procedure="Resolve every joined response through the qualified independent outcome check and count passes by task.",
                    parameters={
                        "response_count": len(corpus.rollouts),
                        "task_ids": sorted(tasks_by_id),
                    },
                    result={
                        "attempted_by_task": attempted,
                        "passed_by_task": passed,
                        "zero_pass_tasks": uninformative,
                        "unsampled_tasks": unsampled,
                    },
                    limitations=[
                        "A zero sampled pass rate describes this response bank, not task impossibility."
                    ],
                    started=started,
                )
            )

    suffix, measurand, claim = absences.TASK_SET_CHECKS[2]
    started = ctx.clock()
    if not raw_tasks:
        entries.append(
            _absence(
                ctx,
                subject_ref,
                suffix,
                measurand,
                claim,
                "a complete task set was unavailable",
                "supply --tasks <file>",
                started=started,
                need=absences.TASK_SET,
            )
        )
    elif not good_cases or grader is None:
        entries.append(
            _absence(
                ctx,
                subject_ref,
                suffix,
                measurand,
                claim,
                "a protected-check-passing known-good case or runnable grader was unavailable",
                "supply a declared known-good case, qualified outcome check, and runnable grader",
                started=started,
                need=absences.PROTECTED_CHECK,
            )
        )
    else:
        task, response = good_cases[0]
        task = instruments.normalised_task(task, "known-good")
        results = [
            grader.score(task, response, sandbox=sandbox, limits=Limits(), project_dir=project_dir)
            for _ in range(3)
        ]
        if not _scored(*results):
            entries.append(
                _absence(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    claim,
                    "the known-good case did not return three scores",
                    "supply a known-good case the grader scores under this sandbox",
                    started=started,
                    limitations=("verdicts: " + ", ".join(r.verdict for r in results),),
                )
            )
        else:
            entries.append(
                _check(
                    ctx,
                    subject_ref,
                    suffix,
                    measurand,
                    passed=len({r.score for r in results}) == 1,
                    predicate="Three independent scores of one outcome-passing known-good case are equal.",
                    scope_tested="one qualified known-good case, three fresh grader processes",
                    procedure="Resolve the declared candidate through the protected check, then score it three times.",
                    parameters={"k": 3, "task_digest": contracts.digest(task)},
                    result={
                        "scores": [r.score for r in results],
                        "request_digest": results[0].request_digest,
                    },
                    limitations=["Other tasks and longer repeat sequences were not tested."],
                    started=started,
                )
            )
    return entries
