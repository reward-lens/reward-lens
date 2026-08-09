"""The discovered panel for the `exploits` section, and the entries it writes.

Three outcomes, and the record says which one happened. With the seeker off, which is the default
and what the runner's `RunContext` says today, one absence: the arm was not run, with the flag that
would run it and the claims that stay unsupported without it. With the seeker on and no actor
supplied, the same shape naming the actor as what is missing. With an actor, one witness per
exploit and one estimate carrying the yield, whose `result` holds the honest negative when the
yield is zero. A run with the seeker off is a complete audit that says the arm was not run, and
that is what the absence is: an entry, not a gap.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from reward_lens import contracts
from reward_lens.instruments.base import Instrument, Panel, RunContext

from .protocol import NOT_FOUND, ActorRecord
from .search import seek

__all__ = [
    "FINDING_ID",
    "METHOD_VERSION",
    "PANEL",
    "ExploitSearch",
    "findings_for",
]

#: Bumped when what the instrument computes changes, never when its prose does.
METHOD_VERSION = "1.0.0"

#: The local-band finding id for an exploit the black-box arm found, allocated from D-71's number.
#: No RL code is carried: section 6 allocates none for an exploit finding, and the handoff proposes
#: the row rather than this packet minting one.
FINDING_ID = "RGX-local-0071"

#: Rung 0's scope, which is the audit's own and the only one a black-box search at this rung has.
SCOPE = "evaluator_comparison"

MEASURAND = (
    "responses that the audited reward scored at or above its threshold and that the independent "
    "outcome check found did not satisfy the intent"
)

LIMITATIONS = (
    "a black-box search establishes that an exploit exists at this budget by this actor; it "
    "establishes nothing about what a larger budget or a different actor would find",
    "the yield is a property of this search, not of the reward system: a different proposal "
    "policy over the same tasks gives a different number",
)


def _wilson(successes: int, n: int, level: float = 0.95) -> contracts.Uncertainty:
    """A Wilson score interval on the yield, or the stated absence of one when there is no n."""
    if n <= 0:
        return contracts.Uncertainty(
            method="no_interval_below_15_clusters",
            level=level,
            reason="no proposal was made, so there is no proportion to bound",
            clusters=0,
        )
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return contracts.Uncertainty(
        interval=[
            contracts.quantise(max(0.0, centre - half), 6),
            contracts.quantise(min(1.0, centre + half), 6),
        ],
        method="wilson",
        level=level,
    )


def _subject_ref(subject: Any) -> str:
    return contracts.digest({"source": getattr(subject, "source", "") or ""})


def findings_for(entry: contracts.Entry) -> tuple[contracts.Finding, ...]:
    """One finding per exploit witness, pointing at the witness the record already carries.

    The runner harvests findings through this module-level function (interfaces section 8.1), and
    this package exposes only this form. An entry that is not an exploit witness raises nothing and
    produces nothing: an absence is not a finding, and neither is a yield of zero.
    """
    if entry.kind != "witness" or not str(entry.entry_id).startswith("exploits.seeker."):
        return ()
    result = dict(entry.result or {})
    return (
        contracts.Finding(
            id=FINDING_ID,
            rule=entry.entry_id,
            level="error",
            kind="fail",
            severity_rationale=(
                "a response was executed against the audited reward and scored at or above its "
                "threshold while the independent outcome check found the intent unmet; the pair "
                "is the evidence, and both halves are in the witness"
            ),
            scope=SCOPE,
            entries=[entry.entry_id],
            partial_fingerprint=contracts.digest(
                {
                    "entry": entry.entry_id,
                    "proposal": result.get("proposal_id", ""),
                    "actor": result.get("actor", ""),
                }
            ).removeprefix("sha256:")[:16],
            message=(
                f"the {result.get('actor', 'seeker')} actor found a response scoring "
                f"{result.get('reward')} that the outcome check rejected, at a budget of "
                f"{result.get('budget_usd', '0.00')} USD"
            ),
            arm="black_box",
            witness_path=f"/measurement/exploits/{entry.entry_id}/witness",
        ),
    )


class ExploitSearch:
    """The black-box arm, as one instrument filling one section."""

    id = "exploits.seeker"
    section: contracts.Section = "exploits"

    @property
    def method(self) -> contracts.Method:
        return contracts.Method(
            id="exploits.seeker",
            version=METHOD_VERSION,
            params_digest=contracts.digest({"scope": SCOPE, "version": METHOD_VERSION}),
            procedure=(
                "an actor proposes responses for the supplied tasks, each is scored under the "
                "audited reward and checked against the independent outcome check, every proposal "
                "is recorded with the inputs it was made from, and a finding enters the record "
                "only after it has been reproduced from those inputs alone"
            ),
            credited_to="reward_lens.instruments.seeker (D-71, D-41)",
        )

    # --- honesty ------------------------------------------------------------------------------

    def assert_scope_honest(self, entry: contracts.Entry) -> contracts.Entry:
        """`entry` unchanged, or `ValueError`: the scope must be the rung the method ran at."""
        if entry.scope != SCOPE:
            raise ValueError(
                f"{entry.entry_id} declares scope {entry.scope}, and this instrument runs at rung "
                f"0, whose scope is {SCOPE}"
            )
        return entry

    # --- the run ------------------------------------------------------------------------------

    def run(self, subject: Any, corpus: Any, ctx: RunContext) -> list[contracts.Entry]:
        mode = str(getattr(ctx, "seeker", "off") or "off")
        subject_ref = _subject_ref(subject)
        if mode == "off":
            return [self._not_run(ctx, subject_ref)]
        actor = getattr(ctx, "seeker_actor", None)
        if actor is None:
            return [self._no_actor(ctx, subject_ref, mode)]
        tasks: Sequence[dict] = tuple(
            getattr(ctx, "seeker_tasks", None) or getattr(corpus, "tasks", ()) or ()
        )
        if not tasks:
            return [self._no_tasks(ctx, subject_ref, mode)]
        intent = str(getattr(ctx, "seeker_intent", "") or MEASURAND)
        record = seek(
            actor,
            tasks=tasks,
            intent=intent,
            max_proposals=int(getattr(ctx, "seeker_max_proposals", 8)),
        )
        return self.entries_for(record, subject_ref=subject_ref, ctx=ctx)

    # --- the entries --------------------------------------------------------------------------

    def _provenance(self, ctx: RunContext, record: ActorRecord) -> contracts.EntryProvenance:
        block = ctx.provenance(arm="black_box")
        block.actor = record.actor
        block.budget_usd = record.budget_usd
        return block

    def entries_for(
        self, record: ActorRecord, *, subject_ref: str, ctx: RunContext
    ) -> list[contracts.Entry]:
        """One witness per exploit, then the yield, with the honest negative in its result."""
        entries: list[contracts.Entry] = []
        found = {scored.proposal.proposal_id: scored for scored in record.scores if scored.exploit}
        for index, proposal_id in enumerate(record.found):
            scored = found[proposal_id]
            entries.append(
                self.assert_scope_honest(
                    contracts.Entry(
                        entry_id=f"exploits.seeker.{record.actor}.witness_{index}",
                        section="exploits",
                        kind="witness",
                        measurand=MEASURAND,
                        method=self.method,
                        scope=SCOPE,
                        subject_ref=subject_ref,
                        depends_on=[],
                        state="complete",
                        provenance=self._provenance(ctx, record),
                        limitations=list(LIMITATIONS),
                        result={
                            "actor": record.actor,
                            "budget_usd": record.budget_usd,
                            "spent_usd": record.spent_usd,
                            "proposal_id": proposal_id,
                            "reward": scored.reward,
                            "threshold": scored.threshold,
                            "outcome_passed": scored.outcome_passed,
                            "arm": record.arm,
                        },
                        witness=contracts.Witness(
                            inputs=dict(scored.proposal.inputs),
                            procedure=(
                                "the recorded inputs were scored again under the audited reward "
                                "and the independent outcome check, and the same pair came back"
                            ),
                            observed=(
                                f"the response scored {scored.reward} against a threshold of "
                                f"{scored.threshold}, and the outcome check found the intent unmet"
                            ),
                            artifact_identity=scored.proposal.proposal_id,
                        ),
                    )
                )
            )
        n = len(record.proposals)
        successes = len(record.found)
        entries.append(
            self.assert_scope_honest(
                contracts.estimate(
                    "exploits",
                    f"exploits.seeker.{record.actor}.yield",
                    "the fraction of this actor's proposals that the audited reward paid for and "
                    "the independent outcome check rejected",
                    contracts.quantise(successes / n if n else 0.0, 6),
                    "fraction",
                    n,
                    "proposal",
                    _wilson(successes, n),
                    subject_ref=subject_ref,
                    provenance=self._provenance(ctx, record),
                    method=self.method,
                    scope=SCOPE,
                    limitations=list(LIMITATIONS),
                    rate_definition="other",
                    result={
                        "actor": record.actor,
                        "budget_usd": record.budget_usd,
                        "spent_usd": record.spent_usd,
                        "calls": record.calls,
                        "proposals": n,
                        "found": list(record.found),
                        "outcome": record.outcome,
                        "stopped": record.stopped,
                        "threshold": record.threshold,
                        "arm": record.arm,
                        "not_found_sentence": (
                            record.not_found_sentence() if not record.found else ""
                        ),
                    },
                )
            )
        )
        return entries

    # --- the three absences --------------------------------------------------------------------

    def _absence(
        self,
        ctx: RunContext,
        subject_ref: str,
        *,
        missing: str,
        remedy: str,
        state: str = "NOT_MEASURED",
    ) -> contracts.Entry:
        block = ctx.provenance(arm="black_box")
        block.actor = "none"
        block.budget_usd = "0.00"
        return self.assert_scope_honest(
            contracts.absence(
                "exploits",
                self.id,
                MEASURAND,
                missing,
                remedy,
                (
                    "no claim that this reward system resists exploitation by a black-box search",
                    "no claim that the white-box arm's findings are the only ones there are",
                ),
                subject_ref=subject_ref,
                provenance=block,
                state=state,
                scope=SCOPE,
                limitations=list(LIMITATIONS),
            )
        )

    def _not_run(self, ctx: RunContext, subject_ref: str) -> contracts.Entry:
        return self._absence(
            ctx,
            subject_ref,
            missing=(
                "the seeker arm was not run: it is the only arm that spends money, so it is off "
                "unless it is asked for, and this run did not ask for it"
            ),
            remedy=(
                "run the arm and name what it may spend: run: reward-lens audit . --seeker api "
                "--max-budget-usd 0.50, or --seeker local for a model on this machine at no cost"
            ),
        )

    def _no_actor(self, ctx: RunContext, subject_ref: str, mode: str) -> contracts.Entry:
        return self._absence(
            ctx,
            subject_ref,
            missing=(
                f"the seeker arm was asked for as {mode} and no actor was supplied to this run, "
                "so nothing proposed anything"
            ),
            remedy=(
                f"supply the {mode} actor the arm needs, or leave the seeker off, which is the "
                "default"
            ),
            state="COULD_NOT_CHECK",
        )

    def _no_tasks(self, ctx: RunContext, subject_ref: str, mode: str) -> contracts.Entry:
        return self._absence(
            ctx,
            subject_ref,
            missing=(
                f"the seeker arm was asked for as {mode} and no task set was supplied, so there "
                "was nothing to propose against"
            ),
            remedy="pass a task set with `--tasks <file>` and run the arm again",
        )


#: What discovery finds. `seconds` is not passed: `Panel` does not carry it at this head.
PANEL = Panel(section="exploits", instruments=(ExploitSearch(),))


def _unused() -> None:  # pragma: no cover - keeps the import list honest for linters
    _ = (Instrument, NOT_FOUND)
