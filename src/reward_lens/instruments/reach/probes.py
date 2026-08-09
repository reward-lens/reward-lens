"""The instrument: six surfaces, one entry each, and the finding that says pass or fail.

`kind: pass` in the commission's line is the finding's kind, not the entry's: `Entry.kind` has five
members and none of them is `pass`. So an unreachable surface is an entry that carries the attempts
that were made and a finding of kind `pass`, and never an omission. A surface whose attempts never
ran at all is neither: it is an absence, because unreachable has to mean attempted and refused.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from reward_lens import contracts
from reward_lens.execution import DEFAULT_LIMITS, Limits, default_sandbox
from reward_lens.instruments.base import Panel, RunContext

from .surfaces import ATTEMPT_NAMES, MEASURANDS, SURFACES, Surface, entry_id_for, finding_id_for
from .witness import WITNESS_VERSION, ReachRun, run_witness

__all__ = [
    "FINDING_IDS",
    "METHOD_VERSION",
    "ReachRequiresTaskSet",
    "ReachSurfaces",
    "findings_for",
    "probe_surfaces",
    "sandbox_for",
]

#: Bumped when the instrument changes what it computes from a witness, not when its prose moves.
#: 2.0.0 is the probe moving onto the run's own sandbox and the filesystem attempt moving off the
#: directory this instrument had to bind.
METHOD_VERSION = "2.0.0"

INSTRUMENT_ID = "reach.surfaces"

_Z = 1.959963984540054
_LEVEL = 0.95


class ReachRequiresTaskSet(contracts.UsageError):
    """RL0120. The reach panel was asked to run with no task set to hand the grader.

    The probe is an executed witness, and a grader is not executed without a task. There is no
    fallback task: inventing one would run the grader on something the caller never supplied and
    report reach against it.
    """

    def __init__(self, *, subject_name: str, tasks_path: str | None, n_tasks: int = 0) -> None:
        super().__init__(
            code="RL0120",
            message=(
                f"the reach panel runs the grader for {subject_name} against a task, and no task "
                "set was supplied"
            ),
            remediation="supply a task set: run: reward-lens audit . --tasks tasks.jsonl",
            context={
                "subject": subject_name,
                "tasks_path": tasks_path,
                "n_tasks": int(n_tasks),
            },
        )


def _wilson(k: int, n: int) -> tuple[float, float]:
    """The Wilson score interval, which is defined at k = 0 and at k = n (D-17)."""
    if n <= 0:
        return 0.0, 1.0
    proportion = k / n
    denominator = n + _Z * _Z
    centre = (k + 0.5 * _Z * _Z) / denominator
    half = (_Z * math.sqrt(n) / denominator) * math.sqrt(
        proportion * (1.0 - proportion) + _Z * _Z / (4.0 * n)
    )
    return max(0.0, round(centre - half, 6)), min(1.0, round(centre + half, 6))


def sandbox_for(ctx: RunContext) -> Any:
    """The sandbox the probe runs under: the run's own, off the `RunContext` the runner built.

    When the run carries none, `ctx.tier()` is `T0`, so a real T0 sandbox is what makes that label
    true. Reaching for `default_sandbox()` here instead would run the probe at whatever this
    machine happens to hold and label the entry with a tier the run never established.
    """
    if getattr(ctx, "sandbox", None) is not None:
        return ctx.sandbox
    return default_sandbox(require_tier="T0")


def probe_surfaces(
    subject: Any,
    ctx: RunContext,
    *,
    task: dict[str, Any],
    surfaces: Iterable[Surface] = SURFACES,
    limits: Limits = DEFAULT_LIMITS,
    outside_file: Any = None,
) -> ReachRun:
    """Run the witness once, under the run's own sandbox, and bring back a report per surface."""
    return run_witness(
        subject,
        task=task,
        surfaces=tuple(surfaces),
        limits=limits,
        sandbox=sandbox_for(ctx),
        outside_file=outside_file,
    )


@dataclass(frozen=True)
class ReachSurfaces:
    """What the graded process can touch, established by running it (section 7.3, D-41).

    The arm is the black-box one: the probe never reads the grader's source to decide a surface,
    it runs the grader and reports what the graded process got through to.
    """

    id: str = INSTRUMENT_ID
    section: str = "reach"
    surfaces: tuple[Surface, ...] = SURFACES
    limits: Limits = field(default_factory=lambda: DEFAULT_LIMITS)

    @property
    def method(self) -> contracts.Method:
        """Declared up front, so the audit's reuse check can read it before anything runs."""
        return contracts.Method(
            id=INSTRUMENT_ID,
            version=METHOD_VERSION,
            params_digest=contracts.digest(
                {
                    "surfaces": [surface.value for surface in self.surfaces],
                    "attempts": {
                        surface.value: list(ATTEMPT_NAMES[surface]) for surface in self.surfaces
                    },
                    "witness_version": WITNESS_VERSION,
                    "wall_s": float(self.limits.wall_s),
                    "network_allowed": bool(self.limits.network),
                }
            ),
            procedure=(
                "one response is handed to the grader through its own entrypoint and run in the "
                "sandbox at the tier that held; the response attempts each surface from inside "
                "the graded process and reports every attempt it made and what came of it"
            ),
        )

    def run(
        self, subject: Any, corpus: Any, ctx: RunContext
    ) -> list[contracts.Entry]:
        """One entry per surface. Exceptions are not caught here; the audit catches them."""
        tasks = tuple(getattr(corpus, "tasks", ()) or ())
        if not tasks:
            tasks_path = getattr(subject, "tasks_path", None)
            raise ReachRequiresTaskSet(
                subject_name=str(getattr(subject, "name", "the reward system")),
                tasks_path=None if tasks_path is None else str(tasks_path),
                n_tasks=0,
            )
        task = dict(tasks[0])
        started = time.monotonic()
        outcome = probe_surfaces(
            subject, ctx, task=task, surfaces=self.surfaces, limits=self.limits
        )
        elapsed = time.monotonic() - started
        subject_ref = contracts.digest({"source": str(getattr(subject, "source", ""))})
        method = self.method
        per_surface = elapsed / max(1, len(self.surfaces))
        return [
            self._entry(
                surface,
                outcome,
                ctx,
                subject=subject,
                task=task,
                subject_ref=subject_ref,
                method=method,
                duration_s=per_surface,
            )
            for surface in self.surfaces
        ]

    def _entry(
        self,
        surface: Surface,
        outcome: ReachRun,
        ctx: RunContext,
        *,
        subject: Any,
        task: dict[str, Any],
        subject_ref: str,
        method: contracts.Method,
        duration_s: float,
    ) -> contracts.Entry:
        report = outcome.reports[surface]
        entry_id = entry_id_for(surface)
        measurand = MEASURANDS[surface]
        if not report.executed:
            return ctx.absence(
                "reach",
                entry_id,
                measurand,
                (
                    f"the grader did not run the response the probe handed it, so no attempt on "
                    f"the {surface.value} surface was made: {outcome.detail}"
                ),
                (
                    "run the panel against a grader that executes the response it is given, or "
                    "supply the entrypoint that does"
                ),
                (f"no reach claim about {surface.value}",),
                subject_ref=subject_ref,
                duration_s=duration_s,
            )
        low, high = _wilson(report.k, report.n)
        result: dict[str, Any] = {
            "surface": surface.value,
            "reached": report.reached,
            "finding_kind": "fail" if report.reached else "pass",
            "attempts": [attempt.to_dict() for attempt in report.attempts],
            "attempted": report.n,
            "reached_attempts": report.k,
            "sandbox_tier": outcome.tier,
            "grader_executed_the_response": report.executed,
            "exit_code": outcome.exit_code,
        }
        if surface is Surface.NETWORK:
            result["listener_accepted"] = outcome.listener_accepted
            result["egress_uncontained"] = outcome.egress_uncontained
        entry = contracts.estimate(
            "reach",
            entry_id,
            measurand,
            round(report.k / report.n, 6),
            "fraction",
            report.n,
            "probe attempt",
            contracts.Uncertainty(method="wilson", level=_LEVEL, interval=[low, high]),
            subject_ref=subject_ref,
            provenance=ctx.provenance(duration_s=duration_s, arm="black_box"),
            method=method,
            result=result,
        )
        # `ctx.tier()` already reads the sandbox the witness ran under, because `sandbox_for` hands
        # the witness that same object. Writing the run's own tier here anyway is what makes the
        # two impossible to separate: the label on the entry is the tier of the run that produced
        # it, and no path through this module can leave a tier the run did not establish.
        entry.provenance.sandbox_tier = outcome.tier
        entry.witness = contracts.Witness(
            inputs={
                "grader": str(getattr(subject, "grader_path", "")),
                "entrypoint": str(getattr(subject, "entrypoint", "score")),
                "task_id": str(task.get("id", "")),
                "response_digest": outcome.response_digest,
                "program_digest": outcome.program_digest,
                "attempts": list(ATTEMPT_NAMES[surface]),
                "sandbox_tier": outcome.tier,
            },
            procedure="; ".join(attempt.procedure for attempt in report.attempts),
            observed=report.observed,
        )
        return entry


#: The finding ids this package raises under, in surface order. Section 8.1 has the runner take
#: the id a package declares as `FINDING_ID`; reach raises under one id per surface rather than one
#: for the package, because a reader who sees `RGX-local-1203` has to be able to tell which surface
#: was reached without opening the entry. The set is declared here for the runner to read.
FINDING_IDS: tuple[str, ...] = tuple(finding_id_for(surface) for surface in SURFACES)


def findings_for(entry: contracts.Entry) -> list[contracts.Finding]:
    """The findings for one entry this instrument produced: `pass` for a refusal, `fail` for a reach.

    Section 8.1's second form, module-level and one entry at a time, which the runner applies to
    every entry the instrument returned. It is the only form this package exposes: there is no
    `findings` method on `ReachSurfaces`, so there is no second way to harvest the same finding.
    An entry that is not a reach estimate (an absence, or anything else that lands in the section)
    produces nothing, because unreachable has to mean attempted and refused.
    """
    result = dict(entry.result or {})
    name = result.get("surface")
    if name is None:
        return []
    try:
        surface = Surface(str(name))
    except ValueError:
        return []
    reached = bool(result.get("reached"))
    message = (
        f"the graded process reached the {surface.value} surface"
        if reached
        else f"the graded process could not reach the {surface.value} surface"
    )
    return [
        contracts.Finding(
            id=finding_id_for(surface),
            rule=entry.entry_id,
            level="warning" if reached else "none",
            kind="fail" if reached else "pass",
            severity_rationale=(
                MEASURANDS[surface]
                + (
                    ", and an executed witness got through"
                    if reached
                    else ", and every attempt an executed witness made was refused"
                )
            ),
            scope="evaluator_comparison",
            entries=[entry.entry_id],
            partial_fingerprint=contracts.digest(
                {"surface": surface.value, "entry": entry.entry_id, "reached": reached}
            ).removeprefix("sha256:")[:16],
            message=message,
            arm="black_box",
        )
    ]


#: The panel `reward_lens.product.audit.registry.discover()` finds.
PANEL = Panel(section="reach", instruments=(ReachSurfaces(),))
