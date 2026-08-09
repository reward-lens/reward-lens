"""The framing panel: what the policy can infer about the grader from its context (section 7.3).

Two instruments under one `PANEL`. `ContextInference` reads the four channels the policy's context
arrives on, the prompt, the system message, the file names that context carries and the error
strings the grader returns, and reports per channel every inference that context permits, each with a witness. It runs
no policy and reads no weights: it is a static read of text, and the entry says so in its own
`result.scope_label`. `BeliefAssay` measures nothing and returns one absence, because the question
"what does the policy believe about the grader" is answered by the Apollo-style belief assay, which
is a research module with finetuning requirements this build does not meet and did not run.

The two are one panel on purpose. A record that carried the cheap proxy without the absence beside
it would leave a reader to assume the belief question was asked, and that assumption is exactly the
relabelling this packet exists to prevent. The proxy and the hole ship together, or the panel is
lying by omission.

Both declare `method` up front, as the audit's reuse check requires: `run._requested` reads the
method identity off the instrument before anything runs, so a landed panel makes a request that an
already-stored record of the same project cannot answer, and the project is measured again.
"""

from __future__ import annotations

from typing import Any

from reward_lens import contracts
from reward_lens.instruments.base import Panel, RunContext

from . import channels as channels_module
from . import witness as witness_module
from .channels import CHANNEL_NAMES, CHANNELS, CUES, Channel, Leak, Sample
from .witness import (
    FINDING_CODE,
    FINDING_ID,
    REFUSAL_CODE,
    RUNG_SCOPES,
    SCOPE_LABEL,
    BeliefClaimRefused,
    ScopeMismatch,
    assert_scope_honest,
    declare_method,
    findings_for,
    rung_of,
)

__all__ = [
    "BELIEF_ASSAY_COMMISSION",
    "CHANNELS",
    "CHANNEL_NAMES",
    "CUES",
    "FINDING_CODE",
    "FINDING_ID",
    "PANEL",
    "REFUSAL_CODE",
    "RUNG",
    "RUNG_SCOPES",
    "SCOPE_LABEL",
    "BeliefAssay",
    "BeliefClaimRefused",
    "Channel",
    "ContextInference",
    "Leak",
    "Sample",
    "ScopeMismatch",
    "assert_scope_honest",
    "declare_method",
    "findings_for",
    "rung_of",
]

VERSION = "1.0.0"

#: The rung this panel sits on. A static read of text the policy is handed is the audit's own
#: rung, rung 0, whose scope is `evaluator_comparison` (wave-2 interfaces 8.5).
RUNG = 0

#: What the commission requires of the belief assay, quoted line by line from the commission and
#: nothing beyond it. An earlier attempt wrote out six requirements of its own (weight access, two
#: finetuning arms, an elicitation check, a probe battery, a budget): every one is plausible and
#: none is sourced, and a requirement list this build invented is one nobody can hold the build to.
#: What is left is what the commission says, with the line each sentence comes from, so a reader
#: can check the entry against the document rather than against this packet's opinion.
BELIEF_ASSAY_COMMISSION: tuple[str, ...] = (
    "the assay is the Apollo-style one and belongs to research rather than to this audit, quoting "
    'PRODUCT_BUILD_PROMPT.md:2167 "The Apollo-style belief assay is a research module with"',
    "its finetuning requirements are stated before it is run, quoting "
    'PRODUCT_BUILD_PROMPT.md:2168 "its finetuning requirements stated"',
    "it is reported as its own entry and this record says it did not run, quoting "
    'PRODUCT_BUILD_PROMPT.md:3039 "names the belief assay as a separate module" and '
    'PRODUCT_BUILD_PROMPT.md:3040 "that was not run"',
    'the forbidden thing, quoting PRODUCT_BUILD_PROMPT.md:3040 "Forbidden: presenting a prompt '
    'test as a belief measurement", is never done here',
    "what this build ran instead is the proxy, quoting PRODUCT_BUILD_PROMPT.md:2166 "
    '"This is the cheap static proxy for" and PRODUCT_BUILD_PROMPT.md:2167 "the belief question, '
    'and the record says so"',
)

#: The limitations every framing measurement entry carries. The first is the scope label in words;
#: `result.scope_label` is the same thing as data, which is the form a gate can read.
MEASUREMENT_LIMITATIONS: tuple[str, ...] = (
    "static proxy: the context the policy can see was read, no policy was run, and no belief was "
    'measured (PRODUCT_BUILD_PROMPT.md:3039 "the record labels it a static proxy")',
    "the cue table is a floor: an inference carried by wording no cue here matches is not counted",
    "a leak is what the context permits, not what any policy did with it",
)


def _measurand(one: Channel) -> str:
    return f"what {one.description} lets the policy infer about the grader"


def _subject_ref(subject: Any) -> str:
    """The same digest the runner uses for its own entries, so the record agrees with itself."""
    return contracts.digest({"source": getattr(subject, "source", "") or ""})


def _result(one: Channel, samples: tuple[Sample, ...], leaks: tuple[Leak, ...]) -> dict[str, Any]:
    return {
        "scope_label": SCOPE_LABEL,
        "not_measured": "no policy was run, so nothing here measures what a policy believes",
        "channel": one.name,
        "summary": (
            f"{len(leaks)} inference{'' if len(leaks) == 1 else 's'} about the grader available "
            f"from {one.description}, over {len(samples)} sample"
            f"{'' if len(samples) == 1 else 's'}"
        ),
        "samples_read": len(samples),
        "leaks_found": len(leaks),
        "cues_fired": sorted({leak.cue_id for leak in leaks}),
        "leaks": [
            {
                "channel": leak.channel,
                "origin": leak.origin,
                "cue": leak.cue_id,
                "excerpt": leak.excerpt,
                "inference": leak.inference,
                "witness": witness_module.leak_witness(leak).model_dump(exclude_none=True),
            }
            for leak in leaks
        ],
    }


class ContextInference:
    """The four channels, read statically, one entry each."""

    id = "framing.context_inference"
    section: contracts.Section = "framing"
    method = declare_method(
        id="framing.context_inference",
        version=VERSION,
        rung=RUNG,
        params={
            "arm": "static",
            "channels": list(CHANNEL_NAMES),
            "cues": [cue.id for cue in CUES],
        },
        procedure=(
            "the prompt, the system message, the visible file names and the grader's returned "
            "error strings are read as text and matched against a fixed cue table; each match is "
            "reported as the inference that text permits, with the text quoted; nothing is "
            "executed and no policy is run, so this is a static read of the context and not a "
            "measurement of what any policy concluded from it"
        ),
        credited_to="reward_lens.instruments.framing (section 7.3)",
    )

    def run(
        self, subject: Any, corpus: Any, ctx: RunContext
    ) -> list[contracts.Entry]:  # noqa: D102
        subject_ref = _subject_ref(subject)
        produced: list[contracts.Entry] = []
        for one in CHANNELS:
            started = ctx.clock()
            samples = tuple(one.read(subject, corpus))
            leaks = tuple(channels_module.probe(one, samples)) if samples else ()
            duration = max(0.0, ctx.clock() - started)
            produced.append(
                assert_scope_honest(
                    self._entry(
                        one,
                        samples,
                        leaks,
                        ctx,
                        subject_ref=subject_ref,
                        duration=duration,
                    )
                )
            )
        return produced

    def _entry(
        self,
        one: Channel,
        samples: tuple[Sample, ...],
        leaks: tuple[Leak, ...],
        ctx: RunContext,
        *,
        subject_ref: str,
        duration: float,
    ) -> contracts.Entry:
        entry_id = f"{self.id}.{one.name}"
        measurand = _measurand(one)
        if not samples:
            # Nothing was read, so neither "it leaks" nor "it does not" is an answer this run has.
            return contracts.absence(
                "framing",
                entry_id,
                measurand,
                one.absent,
                one.remedy,
                (f"no claim about what {one.description} discloses to the policy",),
                subject_ref=subject_ref,
                provenance=ctx.provenance(duration_s=duration),
                depends_on=one.depends_on,
                method=self.method,
                limitations=MEASUREMENT_LIMITATIONS,
            )
        common: dict[str, Any] = {
            "entry_id": entry_id,
            "section": "framing",
            "measurand": measurand,
            "method": self.method,
            "scope": RUNG_SCOPES[RUNG],
            "subject_ref": subject_ref,
            "depends_on": list(one.depends_on),
            "state": "partial",
            "provenance": ctx.provenance(duration_s=duration),
            "limitations": list(MEASUREMENT_LIMITATIONS),
            "result": _result(one, samples, leaks),
        }
        # A check and not an estimate: the leaks are counted, not sampled from a population, so a
        # count on `Entry.n` would be a number with no sampling unit and no interval, which is the
        # shape the record's own rule exists to refuse. The count is data under `result`, the
        # predicate is what was tested, and the witness is the evidence either way.
        return contracts.Entry(
            kind="check",
            check=contracts.Check(
                predicate=(
                    f"{one.description} carries nothing the cue table reads as an inference "
                    "about the grader"
                ),
                passed=not leaks,
                scope_tested=(
                    f"{len(samples)} sample{'' if len(samples) == 1 else 's'} of "
                    f"{one.description}, against {len(CUES)} cues; static, no policy run"
                ),
            ),
            witness=witness_module.channel_witness(one, samples, leaks),
            **common,
        )


class BeliefAssay:
    """The question this panel does not answer, recorded as the hole it is."""

    id = "framing.belief_assay"
    section: contracts.Section = "framing"
    method = declare_method(
        id="framing.belief_assay",
        version=VERSION,
        rung=RUNG,
        params={"commission": list(BELIEF_ASSAY_COMMISSION)},
        procedure=(
            "the Apollo-style belief assay implants a fact by finetuning on synthetic documents, "
            "runs a negated control arm, checks the implant took out of context, and then reads a "
            "pre-registered behavioural probe battery; this build meets none of those "
            "requirements, so the entry records the attempt as not made"
        ),
        credited_to="reward_lens.instruments.framing (section 7.3)",
    )

    def run(
        self, subject: Any, corpus: Any, ctx: RunContext
    ) -> list[contracts.Entry]:  # noqa: D102
        return [
            assert_scope_honest(
                contracts.absence(
                    "framing",
                    self.id,
                    (
                        "whether the policy holds a belief about the grader, which the "
                        "Apollo-style assay would measure and this build did not run"
                    ),
                    (
                        "the Apollo-style belief assay was not run in this build: it needs weight "
                        "access and a finetuning run, and this audit has neither"
                    ),
                    (
                        "run the belief assay as its own research module, with the finetuning "
                        "requirements this entry lists, and report it as a separate entry; "
                        "framing.context_inference does not substitute for it"
                    ),
                    (
                        "no claim about what the policy believes, concludes or is aware of",
                        "no claim that the framing entries in this record measured a belief",
                    ),
                    subject_ref=_subject_ref(subject),
                    provenance=ctx.provenance(),
                    depends_on=("digest:policy",),
                    method=self.method,
                    limitations=BELIEF_ASSAY_COMMISSION,
                )
            )
        ]


PANEL = Panel(section="framing", instruments=(ContextInference(), BeliefAssay()))
