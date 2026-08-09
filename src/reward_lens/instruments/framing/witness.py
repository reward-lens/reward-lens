"""The witness a leak carries, the finding it raises, and the guard that holds the label.

Three jobs, and the third is the reason this packet exists. A leak is reported with the text that
permits the inference and the inference itself, so a reader can check the reasoning rather than
trust the panel: that is `leak_witness` and `channel_witness`. A leak raises RL0231, one finding
per leak, each pointing at its own witness: that is `findings_for`. And every framing entry is held
to its scope label before it can leave the panel: that is `assert_scope_honest`.

The guard is enforcement and not documentation. This panel reads text the policy can see and never
runs a policy, so it cannot say what a policy believes; the Apollo-style belief assay answers that
question and is not in this build. A sentence in a framing entry that asserts a belief was measured
is refused with RL0003, exit 4, naming the field it was found in, and the entry is not emitted. The
label is checked as data too: `result.scope_label` has to read `static_proxy`, because Gate 20 reads
the entry and a gate cannot read a sentence that was merely intended honestly.

What the guard deliberately does not read: quoted evidence. An excerpt is the user's own prompt text
and a prompt that says "the model believes it is in an eval" is a finding, not a mislabelling, so
the evidence keys are excluded and only the record's own claim text is scanned.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence, get_args

from reward_lens import contracts, errors

from .channels import Channel, Leak, Sample

__all__ = [
    "BELIEF_ASSERTIONS",
    "EVIDENCE_KEYS",
    "FINDING_CODE",
    "FINDING_ID",
    "REFUSAL_CODE",
    "RUNG_SCOPES",
    "SCOPE_LABEL",
    "BeliefClaimRefused",
    "ScopeMismatch",
    "assert_message_honest",
    "assert_scope_honest",
    "channel_witness",
    "declare_method",
    "findings_for",
    "leak_witness",
    "rung_of",
]

#: The framing finding: what the policy can infer about the grader from its context. Allocated in
#: this package and nowhere else (wave-2 interfaces section 6).
FINDING_CODE = "RL0231"

#: The finding id the code is recorded under, in the local band the audit's own findings use.
FINDING_ID = "RGX-local-0231"

#: Wave 1's reserved input-contract refusal, exit 4, a bad field named. Reused unchanged, as
#: P-CONNECT reuses it for its override refusal; this packet allocates no refusal code of its own.
REFUSAL_CODE = "RL0003"

#: The one value `result.scope_label` may carry on a framing measurement entry.
SCOPE_LABEL = "static_proxy"

#: The six-rung ladder's scopes, in ladder order, taken from the contract itself so that this
#: package cannot drift from it: rung 0 is `evaluator_comparison`, the audit's own rung and the
#: honest scope for a static read (wave-2 interfaces 8.5).
RUNG_SCOPES: tuple[str, ...] = get_args(contracts.Scope)

#: The rung each method declared through `declare_method` sits on, keyed by the digest of the
#: params it declared it in. A `Method` carries its params as a digest and a digest cannot be read
#: back, so the rung is kept here at the moment it is digested rather than guessed from the id.
_RUNGS: dict[str, int] = {}

#: Result keys holding quoted evidence rather than the record's own claims. Never scanned.
EVIDENCE_KEYS = frozenset(
    {"excerpt", "excerpts", "origin", "origins", "inputs", "sample", "samples"}
)

_BELIEF = r"belie(?:f|fs|ve|ves|ved)"
_NEGATOR = r"\b(?:no|not|never|without|cannot|can't|nothing|none|un-?measured)\b"
#: Verbs that assert a measurement was made. Imperatives and nouns are deliberately out: "run the
#: belief assay" is an instruction and "the belief assay" is the name of a module, and a guard that
#: refused either would be refusing the honest sentence this panel is required to write.
_ASSERTS = (
    r"\b(?:measured|measures|measurement|measuring|assayed|established|establishes|shown|shows|"
    r"demonstrated|demonstrates|proved|proves|proven|found|finds)\b"
)

#: The two shapes a mislabelling takes. The first asserts that a belief was measured; the second
#: asserts the belief itself. Either one in a framing entry is a prompt test wearing the belief
#: assay's name, and either one is refused.
BELIEF_ASSERTIONS: tuple[tuple[str, str], ...] = (
    (
        rf"(?is){_BELIEF}.*?{_ASSERTS}|{_ASSERTS}.*?{_BELIEF}",
        "asserts that a belief was measured",
    ),
    (
        r"(?i)\b(?:the\s+|a\s+|any\s+)?(?:policy|model|agent)\s+(?:believes?|believed|thinks?|knows)\b",
        "asserts what the policy believes",
    ),
)


def _refusal(field: str, detail: str, **context: Any) -> contracts.UsageError:
    """The RL0003 error as the catalogue writes it, with this panel's detail in the slot for it.

    The wording is the catalogue's, not this module's, so `reward-lens explain RL0003` describes
    the fault that was actually raised. What the panel supplies is the detail the template leaves
    open: which field, and what is wrong with it.
    """
    return errors.make(REFUSAL_CODE, field=field, detail=detail, **context)


class BeliefClaimRefused(contracts.UsageError):
    """A framing entry that claims a belief measurement. Exit 4, and the field is named.

    Raised inside the panel rather than checked after it, so the mislabelled entry never reaches
    the record. The audit turns the exception into a `COULD_NOT_CHECK` hole naming this class,
    which is the honest outcome: the framing panel refused to answer rather than answering a
    question it did not ask.
    """

    def __init__(self, field: str, sentence: str, why: str) -> None:
        built = _refusal(
            field,
            (
                f"it {why}: {sentence!r}; this panel reads the context the policy can see and "
                f"runs no policy, so its entries carry the scope label {SCOPE_LABEL!r} and say "
                f"what the context permits, never what a policy believes"
            ),
            sentence=sentence,
        )
        super().__init__(
            code=built.code,
            message=built.message,
            remediation=built.remediation,
            context=dict(built.context),
            exit_code=4,
        )


class ScopeMismatch(contracts.UsageError):
    """An entry whose `scope` is not the scope of the rung its method declares. Exit 4.

    The label a reader sees and the field a gate reads are two different things, and this is the
    second. Gate 20 reads `Entry.scope`; an entry whose method climbed a rung and whose scope did
    not is a claim about a stronger design than the one that ran, so the panel refuses it.
    """

    def __init__(self, field: str, detail: str) -> None:
        built = _refusal(field, detail)
        super().__init__(
            code=built.code,
            message=built.message,
            remediation=built.remediation,
            context=dict(built.context),
            exit_code=4,
        )


def declare_method(
    *,
    id: str,
    version: str,
    procedure: str,
    credited_to: str,
    rung: int,
    params: dict[str, Any],
) -> contracts.Method:
    """A `Method` whose rung is on the record: digested into its params, and kept for the guard."""
    if not 0 <= rung < len(RUNG_SCOPES):
        raise ValueError(f"rung {rung} is not one of the {len(RUNG_SCOPES)} rungs")
    params_digest = contracts.digest({**params, "rung": rung})
    _RUNGS[params_digest] = rung
    return contracts.Method(
        id=id,
        version=version,
        params_digest=params_digest,
        procedure=procedure,
        credited_to=credited_to,
    )


def rung_of(method: contracts.Method) -> int | None:
    """The rung `method` declared, or `None` for a method this package did not declare."""
    return _RUNGS.get(method.params_digest)


def _sentences(text: str) -> Iterable[str]:
    for piece in re.split(r"[.;\n]", text):
        stripped = piece.strip()
        if stripped:
            yield stripped


def assert_message_honest(text: str, field: str) -> str:
    """`text` unchanged, or `BeliefClaimRefused` naming `field`.

    Sentence by sentence, because a paragraph that measures a proxy in one sentence and names the
    unrun assay in the next is honest, and a scan over the whole paragraph would refuse it.
    """
    for sentence in _sentences(str(text)):
        if re.search(_NEGATOR, sentence, flags=re.IGNORECASE):
            continue
        for pattern, why in BELIEF_ASSERTIONS:
            if re.search(pattern, sentence):
                raise BeliefClaimRefused(field, sentence, why)
    return text


def _walk_claims(value: Any, field: str) -> None:
    """Every claim string under `value`, quoted evidence excluded."""
    if isinstance(value, str):
        assert_message_honest(value, field)
    elif isinstance(value, dict):
        for key, inner in value.items():
            if key in EVIDENCE_KEYS:
                continue
            _walk_claims(inner, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            _walk_claims(inner, f"{field}[{index}]")


def assert_scope_honest(entry: contracts.Entry) -> contracts.Entry:
    """`entry` unchanged, or a refusal. Every framing entry passes through here.

    Two claims are checked, because a reader and a gate read different fields. `result.scope_label`
    is the sentence a reader sees; `entry.scope` is the rung a gate reads, and it has to be the
    scope of the rung the entry's own method declares.
    """
    rung = rung_of(entry.method)
    if rung is None:
        raise ScopeMismatch(
            "method.params_digest",
            "the method was not declared through framing.witness.declare_method, so the rung it "
            "sits on is not on the record and the entry's scope cannot be checked against it",
        )
    expected = RUNG_SCOPES[rung]
    if entry.scope != expected:
        raise ScopeMismatch(
            "scope",
            f"it is {entry.scope!r}, but the entry's method declares rung {rung}, whose scope is "
            f"{expected!r}",
        )
    if entry.kind != "absence":
        label = (entry.result or {}).get("scope_label")
        if label != SCOPE_LABEL:
            raise BeliefClaimRefused(
                "result.scope_label",
                str(label),
                f"is not {SCOPE_LABEL!r}, so the entry does not say what it measured",
            )
    assert_message_honest(entry.measurand, "measurand")
    assert_message_honest(entry.method.procedure, "method.procedure")
    for index, limitation in enumerate(entry.limitations):
        assert_message_honest(limitation, f"limitations[{index}]")
    if entry.result is not None:
        _walk_claims(entry.result, "result")
    if entry.witness is not None:
        assert_message_honest(entry.witness.observed, "witness.observed")
        assert_message_honest(entry.witness.procedure, "witness.procedure")
    if entry.check is not None:
        assert_message_honest(entry.check.predicate, "check.predicate")
        assert_message_honest(entry.check.scope_tested, "check.scope_tested")
    if entry.absence is not None:
        assert_message_honest(entry.absence.missing_access, "absence.missing_access")
        assert_message_honest(entry.absence.remedy, "absence.remedy")
        for index, claim in enumerate(entry.absence.affected_claims):
            assert_message_honest(claim, f"absence.affected_claims[{index}]")
    return entry


def _procedure(one: Channel) -> str:
    return (
        f"{one.description} was read as text and matched against the cue table; nothing was "
        f"executed, no policy was run, and the match is over the text the policy can see"
    )


def leak_witness(leak: Leak) -> contracts.Witness:
    """One leak as a `contracts.Witness`: the text that permits the inference, and the inference.

    A real witness and not a free-form dict, so the shape is validated where it is built rather
    than trusted where it is read: `Entry.result` is `dict[str, Any]` and validates nothing inside
    it, which is how a witness per leak becomes a witness in name only.

    `observed` carries the inference and not the excerpt. The excerpt is evidence and lives under
    `inputs`, where the guard does not scan it: a prompt that itself says what the model believes
    is a finding about that prompt, and quoting it must not read as the panel claiming a belief.
    """
    return contracts.Witness(
        inputs={
            "channel": leak.channel,
            "origin": leak.origin,
            "cue": leak.cue_id,
            "excerpt": leak.excerpt,
        },
        procedure=(
            f"the cue {leak.cue_id!r} matched the text at {leak.origin}; static read, no policy run"
        ),
        observed=f"{leak.origin}: {leak.inference}",
    )


def channel_witness(
    one: Channel, samples: Sequence[Sample], leaks: Sequence[Leak]
) -> contracts.Witness:
    """The channel's own witness: what was read, how, and every inference it permits."""
    return contracts.Witness(
        inputs={
            "channel": one.name,
            "samples": [sample.origin for sample in samples],
            "n_samples": len(samples),
            "cues": sorted({leak.cue_id for leak in leaks}),
        },
        procedure=_procedure(one),
        observed="; ".join(f"{leak.origin}: {leak.inference}" for leak in leaks)
        or f"{one.description} carried no text the cue table reads as an inference",
    )


def findings_for(entry: contracts.Entry) -> tuple[contracts.Finding, ...]:
    """One RL0231 finding per leak on this entry, each pointing at that leak's own witness.

    A finding per leak and not per channel: two disclosures in one prompt are two things to fix,
    and a single finding over both would have one witness and hide the other.
    """
    leaks = list((entry.result or {}).get("leaks", ()) or ())
    findings: list[contracts.Finding] = []
    for index, leak in enumerate(leaks):
        message = assert_message_honest(
            f"{leak['origin']} discloses the grader to the policy: {leak['inference']}",
            "message",
        )
        findings.append(
            contracts.Finding(
                id=FINDING_ID,
                rule=entry.entry_id,
                level="warning",
                kind="fail",
                severity_rationale=(
                    "a static read of the context the policy is handed, with the place named; no "
                    "policy was run, so this says what the context permits and not what any "
                    "policy did with it"
                ),
                scope="evaluator_comparison",
                entries=[entry.entry_id],
                partial_fingerprint=contracts.digest(
                    {
                        "code": FINDING_CODE,
                        "entry": entry.entry_id,
                        "cue": leak["cue"],
                        "origin": leak["origin"],
                    }
                ).removeprefix("sha256:")[:16],
                message=message,
                code=FINDING_CODE,
                arm="static",
                witness_path=f"/measurement/framing/{entry.entry_id}/result/leaks/{index}/witness",
            )
        )
    return tuple(findings)
