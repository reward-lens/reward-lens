"""Evidence-bound five-way trace answers and causal remediation choices.

The historical five labels describe the observed training record.  They are deliberately
separate from the causal remediation choice required by revision 3: scalar loss evidence does
not identify a gradient, optimizer transform, or parameter update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from reward_lens import contracts

if TYPE_CHECKING:  # pragma: no cover
    from .record import TraceRecord

__all__ = [
    "ANSWER_ENTRY_ID",
    "Answer",
    "AnswerKind",
    "EntryRef",
    "RemediationKind",
    "answer",
    "answer_entry",
    "answer_from_entry",
]


ANSWER_ENTRY_ID = "trace.fiveway"
_ANSWER_SCHEMA = "reward-lens/trace-fiveway/1"
_ANSWER_METHOD_ID = "trace.fiveway"
_ANSWER_METHOD_VERSION = "1.0.0"


class AnswerKind(str, Enum):
    """The five observational outcomes frozen in the wave-2 public interface."""

    OPTIMISED_AS_INTENDED = "OPTIMISED_AS_INTENDED"
    OPTIMISED_PROXY = "OPTIMISED_PROXY"
    SELECTED_NOT_TRAINED = "SELECTED_NOT_TRAINED"
    NO_SIGNAL = "NO_SIGNAL"
    INCONCLUSIVE_INPUTS = "INCONCLUSIVE_INPUTS"


class RemediationKind(str, Enum):
    """The distinct causal interventions section 7.4 permits a record to support."""

    EVALUATOR_OR_SURFACE = "EVALUATOR_OR_SURFACE"
    OBJECTIVE_OR_TRAINER = "OBJECTIVE_OR_TRAINER"
    MEASUREMENT_ORACLE = "MEASUREMENT_ORACLE"
    CONTAINMENT_OR_TASK_MIXTURE = "CONTAINMENT_OR_TASK_MIXTURE"
    RECORD_NEXT_TIME = "RECORD_NEXT_TIME"


@dataclass(frozen=True)
class EntryRef:
    """A stable reference to one retained entry, optionally pinned to its canonical digest."""

    entry_id: str
    digest: str | None = None

    def resolve(self, source: "TraceRecord | Iterable[contracts.Entry]") -> bool:
        entries = source.entries() if hasattr(source, "entries") else source
        return any(
            entry.entry_id == self.entry_id
            and (self.digest is None or contracts.digest(entry) == self.digest)
            for entry in entries
        )


@dataclass(frozen=True)
class Answer:
    """A checkable observation plus a separate, evidence-bound remediation choice."""

    kind: AnswerKind
    evidence: tuple[EntryRef, ...]
    distinguished_from: Mapping[AnswerKind, str]
    remediation: RemediationKind
    remediation_evidence: tuple[EntryRef, ...]
    remediation_distinguished_from: Mapping[RemediationKind, str]
    summary: str
    record: "TraceRecord | None" = field(default=None, repr=False, compare=False)
    entries: tuple[contracts.Entry, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.evidence or not self.remediation_evidence:
            raise ValueError("an answer needs non-empty observational and remediation evidence")
        if not self.summary.strip():
            raise ValueError("an answer needs a summary")
        available = self.entries or (() if self.record is None else tuple(self.record.entries()))
        for reference in (*self.evidence, *self.remediation_evidence):
            if not reference.entry_id.strip() or not reference.resolve(available):
                raise ValueError(
                    f"answer evidence reference does not resolve: {reference.entry_id!r}"
                )
        _validate_neighbours(self.kind, self.distinguished_from, AnswerKind, "observational")
        _validate_neighbours(
            self.remediation,
            self.remediation_distinguished_from,
            RemediationKind,
            "remediation",
        )


def answer(
    trace: "TraceRecord",
    *,
    evidence_entries: Sequence[contracts.Entry] = (),
    trace_entries: Sequence[contracts.Entry] | None = None,
) -> Answer:
    """Classify only evidence the canonical trace actually retains.

    An independent outcome component is required to call the observed optimisation intended or a
    proxy. A record without it remains unresolved even when its scalar-loss fields are complete.
    """
    base_entries = tuple(trace.entries()) if trace_entries is None else tuple(trace_entries)
    entries = (*base_entries, *evidence_entries)
    refs = _refs(trace, available_entries=base_entries)
    kind = _observational_kind(trace)
    remediation, remediation_refs = _remediation(entries, subject_ref=trace.subject_ref)
    return Answer(
        kind=kind,
        evidence=refs,
        distinguished_from=_observational_neighbours(kind),
        remediation=remediation,
        remediation_evidence=remediation_refs or refs,
        remediation_distinguished_from=_remediation_neighbours(remediation),
        summary=_summary(kind, remediation),
        record=trace,
        entries=entries,
    )


def answer_entry(
    trace: "TraceRecord",
    *,
    evidence_entries: Sequence[contracts.Entry] = (),
    trace_entries: Sequence[contracts.Entry] | None = None,
    provenance: contracts.EntryProvenance,
) -> contracts.Entry:
    """Serialize the answer as the ordinary, validated ``measurement.trace`` entry.

    The result is deliberately self-contained enough for a renderer to validate after reopening an
    Assay.  Every cited entry carries its canonical digest; the outer witness only says that this
    module selected and serialized the evidence already present in the record.
    """
    base_entries = tuple(trace.entries()) if trace_entries is None else tuple(trace_entries)
    result = answer(trace, evidence_entries=evidence_entries, trace_entries=base_entries)
    pool = (*base_entries, *evidence_entries)
    payload = _payload(result, pool)
    return contracts.Entry(
        entry_id=ANSWER_ENTRY_ID,
        section="trace",
        kind="witness",
        measurand="five-way observed trace answer and evidence-bound causal remediation",
        method=_answer_method(),
        scope="reconstructed_training_pressure",
        subject_ref=trace.subject_ref,
        depends_on=list(trace.depends_on),
        state="complete",
        provenance=provenance,
        limitations=[
            "the observed label is derived from retained score and scalar-loss record evidence",
            "a causal remediation is selected only from the cited instrument witnesses",
            "scalar loss evidence does not establish a gradient, optimizer transform, or parameter update",
        ],
        result=payload,
        witness=contracts.Witness(
            inputs={
                "trace_source": trace.source,
                "evidence": _ref_payload((*result.evidence, *result.remediation_evidence), pool),
            },
            procedure="classify retained trace evidence under the five-way contract and validate digest-bound causal witnesses",
            observed=result.summary,
        ),
    )


def answer_from_entry(
    entry: contracts.Entry, *, evidence_entries: Sequence[contracts.Entry]
) -> Answer:
    """Validate and deserialize a persisted ``trace.fiveway`` entry.

    This is the read-side contract for incident projection.  It refuses a payload if a cited entry
    changed, disappeared, or no longer semantically supports the selected causal remediation.
    """
    if (
        entry.entry_id != ANSWER_ENTRY_ID
        or entry.section != "trace"
        or entry.kind != "witness"
        or entry.state != "complete"
        or entry.method.id != _ANSWER_METHOD_ID
        or entry.method.version != _ANSWER_METHOD_VERSION
        or entry.method.params_digest != _answer_method().params_digest
    ):
        raise ValueError("not a validated trace.fiveway answer entry")
    result = dict(entry.result or {})
    if result.get("schema") != _ANSWER_SCHEMA or set(result) != {"schema", "answer"}:
        raise ValueError("trace.fiveway result has an unsupported schema")
    raw = result.get("answer")
    if not isinstance(raw, Mapping):
        raise ValueError("trace.fiveway result is missing its answer object")
    expected = {
        "kind",
        "evidence",
        "distinguished_from",
        "remediation",
        "remediation_evidence",
        "remediation_distinguished_from",
        "summary",
        "claim_boundary",
    }
    if set(raw) != expected:
        raise ValueError("trace.fiveway answer has unexpected or missing fields")
    if (
        raw["claim_boundary"]
        != "No gradient, optimizer transform, or parameter update is inferred from scalar trace evidence."
    ):
        raise ValueError("trace.fiveway answer has an unsupported claim boundary")
    pool = tuple(evidence_entries)
    try:
        kind = AnswerKind(str(raw["kind"]))
        remediation = RemediationKind(str(raw["remediation"]))
    except ValueError as error:
        raise ValueError("trace.fiveway answer names an unknown alternative") from error
    evidence = _parse_refs(raw["evidence"], pool)
    remediation_evidence = _parse_refs(raw["remediation_evidence"], pool)
    observed_neighbours = _parse_neighbours(raw["distinguished_from"], AnswerKind)
    remediation_neighbours = _parse_neighbours(
        raw["remediation_distinguished_from"], RemediationKind
    )
    selected, supported = _remediation(pool, subject_ref=entry.subject_ref)
    if remediation is not selected or tuple(remediation_evidence) != supported:
        raise ValueError(
            "trace.fiveway remediation is not supported by its retained instrument evidence"
        )
    return Answer(
        kind=kind,
        evidence=tuple(evidence),
        distinguished_from=observed_neighbours,
        remediation=remediation,
        remediation_evidence=tuple(remediation_evidence),
        remediation_distinguished_from=remediation_neighbours,
        summary=str(raw["summary"]),
        entries=pool,
    )


def _validate_neighbours(
    selected: Enum, values: Mapping[Enum, str], enum: type[Enum], label: str
) -> None:
    expected = set(enum) - {selected}
    if set(values) != expected:
        raise ValueError(f"{label} distinctions must name every alternative")
    if any(not str(text).strip() for text in values.values()):
        raise ValueError(f"{label} distinctions may not be blank")


def _refs(
    trace: "TraceRecord", *, available_entries: Sequence[contracts.Entry] | None = None
) -> tuple[EntryRef, ...]:
    available = {
        entry.entry_id
        for entry in (trace.entries() if available_entries is None else available_entries)
    }
    chosen = ["trace.reconstruction", "trace.integrity", "trace.access_level"]
    chosen.extend(
        {
            "reward_recorded": "trace.displayed_aggregate",
            "reward_applied": "trace.optimized_composite",
            "advantage": "trace.advantage",
            "selection": "trace.selection",
        }[name]
        for name, series in trace.series.items()
        if not series.present
    )
    return tuple(EntryRef(name) for name in chosen if name in available)


def _observational_kind(trace: "TraceRecord") -> AnswerKind:
    if _incomplete(trace):
        return AnswerKind.INCONCLUSIVE_INPUTS
    recorded = _finite_values(trace, "reward_recorded")
    applied = _finite_values(trace, "reward_applied")
    advantages = _finite_values(trace, "advantage")
    selections = _finite_values(trace, "selection", allow_empty=True)
    if _flat(recorded) and _flat(applied):
        return AnswerKind.NO_SIGNAL
    if (
        selections
        and all(value == 0.0 for value in advantages)
        and any(value > 0.0 for value in selections)
    ):
        return AnswerKind.SELECTED_NOT_TRAINED
    outcome = _component_values(trace, "independent_outcome")
    if not outcome or len(outcome) != len(applied):
        return AnswerKind.INCONCLUSIVE_INPUTS
    association = _association(applied, outcome)
    if association > 0:
        return AnswerKind.OPTIMISED_AS_INTENDED
    if association < 0:
        return AnswerKind.OPTIMISED_PROXY
    return AnswerKind.INCONCLUSIVE_INPUTS


def _answer_method() -> contracts.Method:
    return contracts.Method(
        id=_ANSWER_METHOD_ID,
        version=_ANSWER_METHOD_VERSION,
        params_digest=contracts.digest(
            {"schema": _ANSWER_SCHEMA, "selection": "digest-bound instrument evidence"}
        ),
        procedure="serialize the five observed trace alternatives and select a causal remediation only from a retained qualifying instrument witness",
    )


def _remediation(
    entries: Iterable[contracts.Entry], *, subject_ref: str
) -> tuple[RemediationKind, tuple[EntryRef, ...]]:
    """Select a remediation from a producer contract, never a caller label or scalar shortcut.

    The only currently implemented positive path is intentionally narrow.  An actual Seeker
    witness says an audited reward accepted a response while the independent outcome rejected it;
    an actual protected-suite qualification says that outcome protocol passed its known-good and
    known-wrong controls.  Together they discriminate an evaluator/surface repair from an
    objective, oracle, or containment repair.  Reach and composition records remain useful
    observations, but their existing scalar/permission fields do not make that causal choice.
    """
    pool = tuple(entries)
    qualifications = tuple(entry for entry in pool if _qualified_outcome(entry, subject_ref))
    exploits = tuple(entry for entry in pool if _reproduced_exploit(entry, subject_ref))
    if qualifications and exploits:
        return RemediationKind.EVALUATOR_OR_SURFACE, _bound_refs((qualifications[0], exploits[0]))
    return RemediationKind.RECORD_NEXT_TIME, _record_next_refs(pool)


def _qualified_outcome(entry: contracts.Entry, subject_ref: str) -> bool:
    result = dict(entry.result or {})
    return (
        entry.entry_id == "validity.outcome_qualification"
        and entry.section == "validity"
        and entry.kind == "estimate"
        and entry.state == "complete"
        and entry.method.id == "outcome.protected_suite"
        and entry.subject_ref == subject_ref
        and entry.value == 1.0
        and all(
            type(result.get(name)) is int and result[name] > 0
            for name in (
                "known_good_passed",
                "known_good_total",
                "known_wrong_failed",
                "known_wrong_total",
            )
        )
        and result["known_good_passed"] == result["known_good_total"]
        and result["known_wrong_failed"] == result["known_wrong_total"]
        and _digest_string(result.get("partition_digest"))
    )


def _reproduced_exploit(entry: contracts.Entry, subject_ref: str) -> bool:
    result = dict(entry.result or {})
    try:
        reward, threshold = float(result["reward"]), float(result["threshold"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        entry.section == "exploits"
        and entry.kind == "witness"
        and entry.state == "complete"
        and entry.scope == "evaluator_comparison"
        and entry.method.id == "exploits.seeker"
        and entry.subject_ref == subject_ref
        and entry.witness is not None
        and entry.witness.artifact_identity == result.get("proposal_id")
        and _digest_string(result.get("proposal_id"))
        and result.get("outcome_passed") is False
        and reward >= threshold
    )


def _digest_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _record_next_refs(entries: Iterable[contracts.Entry]) -> tuple[EntryRef, ...]:
    by_id = {entry.entry_id: entry for entry in entries}
    return tuple(
        EntryRef(entry_id, contracts.digest(by_id[entry_id]))
        for entry_id in ("trace.reconstruction", "trace.integrity", "trace.access_level")
        if entry_id in by_id
    )


def _bound_refs(entries: Iterable[contracts.Entry]) -> tuple[EntryRef, ...]:
    return tuple(EntryRef(entry.entry_id, contracts.digest(entry)) for entry in entries)


def _ref_payload(
    refs: Iterable[EntryRef], entries: Iterable[contracts.Entry]
) -> list[dict[str, str]]:
    pool = {entry.entry_id: entry for entry in entries}
    payload: list[dict[str, str]] = []
    for reference in refs:
        entry = pool.get(reference.entry_id)
        if entry is None:
            raise ValueError(f"answer evidence reference does not resolve: {reference.entry_id!r}")
        digest = reference.digest or contracts.digest(entry)
        if contracts.digest(entry) != digest:
            raise ValueError(f"answer evidence digest does not resolve: {reference.entry_id!r}")
        payload.append({"entry_id": reference.entry_id, "digest": digest})
    return payload


def _payload(answer_result: Answer, entries: Iterable[contracts.Entry]) -> dict[str, Any]:
    return {
        "schema": _ANSWER_SCHEMA,
        "answer": {
            "kind": answer_result.kind.value,
            "evidence": _ref_payload(answer_result.evidence, entries),
            "distinguished_from": {
                kind.value: text for kind, text in answer_result.distinguished_from.items()
            },
            "remediation": answer_result.remediation.value,
            "remediation_evidence": _ref_payload(answer_result.remediation_evidence, entries),
            "remediation_distinguished_from": {
                kind.value: text
                for kind, text in answer_result.remediation_distinguished_from.items()
            },
            "summary": answer_result.summary,
            "claim_boundary": "No gradient, optimizer transform, or parameter update is inferred from scalar trace evidence.",
        },
    }


def _parse_refs(value: object, entries: Iterable[contracts.Entry]) -> tuple[EntryRef, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("trace.fiveway answer needs non-empty digest-bound evidence")
    refs: list[EntryRef] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"entry_id", "digest"}:
            raise ValueError("trace.fiveway evidence reference has an invalid shape")
        entry_id, digest = item.get("entry_id"), item.get("digest")
        if not isinstance(entry_id, str) or not isinstance(digest, str):
            raise ValueError("trace.fiveway evidence reference must carry strings")
        reference = EntryRef(entry_id, digest)
        if not reference.resolve(entries):
            raise ValueError(f"trace.fiveway evidence reference does not resolve: {entry_id!r}")
        refs.append(reference)
    if len({(ref.entry_id, ref.digest) for ref in refs}) != len(refs):
        raise ValueError("trace.fiveway evidence references may not repeat")
    return tuple(refs)


def _parse_neighbours(value: object, enum: type[Enum]) -> dict[Enum, str]:
    if not isinstance(value, Mapping):
        raise ValueError("trace.fiveway alternatives must be an object")
    try:
        parsed = {enum(str(key)): str(text) for key, text in value.items()}
    except ValueError as error:
        raise ValueError("trace.fiveway alternatives name an unknown value") from error
    return parsed


def _incomplete(trace: "TraceRecord") -> bool:
    return (
        not trace.complete
        or trace.torn_tail is not None
        or any(not series.present for series in trace.series.values())
        # A recorded selection column may legitimately contain nulls when the trainer did not
        # select individual rows. Its *absence* matters; a null value is not a missing loss fact.
        or any(
            not _finite(value)
            for name, series in trace.series.items()
            if series.present and name != "selection"
            for value in series.values
        )
    )


def _finite_values(
    trace: "TraceRecord", name: str, *, allow_empty: bool = False
) -> tuple[float, ...]:
    values = tuple(trace.series[name].values)
    if not values and allow_empty:
        return ()
    return tuple(float(value) for value in values if _finite(value))


def _component_values(trace: "TraceRecord", name: str) -> tuple[float, ...]:
    series = trace.components.get(name)
    if series is None or not series.present or any(not _finite(value) for value in series.values):
        return ()
    return tuple(float(value) for value in series.values)


def _flat(values: tuple[float, ...]) -> bool:
    return bool(values) and all(
        math.isclose(value, values[0], rel_tol=1e-12, abs_tol=1e-12) for value in values
    )


def _association(left: tuple[float, ...], right: tuple[float, ...]) -> int:
    """Sign of a centered paired covariance; zero is deliberately unresolved."""
    mean_left = math.fsum(left) / len(left)
    mean_right = math.fsum(right) / len(right)
    covariance = math.fsum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    return 1 if covariance > 0 else -1 if covariance < 0 else 0


def _finite(value: object) -> bool:
    return value is not None and math.isfinite(float(value))


def _observational_neighbours(kind: AnswerKind) -> dict[AnswerKind, str]:
    statements = {
        AnswerKind.OPTIMISED_AS_INTENDED: "a recorded independent outcome would need to fall as the applied composite rises",
        AnswerKind.OPTIMISED_PROXY: "a recorded independent outcome would need to rise as the applied composite rises",
        AnswerKind.SELECTED_NOT_TRAINED: "recorded selection with zero advantages would need to replace the observed pressure evidence",
        AnswerKind.NO_SIGNAL: "the recorded applied and displayed composites would need to be flat across every row",
        AnswerKind.INCONCLUSIVE_INPUTS: "the missing or non-finite named series and independent outcome component would need to be recorded completely",
    }
    return {other: text for other, text in statements.items() if other is not kind}


def _remediation_neighbours(kind: RemediationKind) -> dict[RemediationKind, str]:
    statements = {
        RemediationKind.EVALUATOR_OR_SURFACE: "a digest-bound evaluator or reach witness would need to show that repairing the reachable surface resolves the observed issue",
        RemediationKind.OBJECTIVE_OR_TRAINER: "a digest-bound controlled objective or trainer witness would need to change the loss-path pressure while holding the evaluator evidence fixed",
        RemediationKind.MEASUREMENT_ORACLE: "a digest-bound qualified independent-oracle comparison would need to establish that the measurement is unreliable",
        RemediationKind.CONTAINMENT_OR_TASK_MIXTURE: "a digest-bound task or containment witness would need to establish an impossible, over-permissioned, or uninformative policy task",
        RemediationKind.RECORD_NEXT_TIME: "no retained causal witness currently separates evaluator, objective, measurement, and containment explanations",
    }
    return {other: text for other, text in statements.items() if other is not kind}


def _summary(kind: AnswerKind, remediation: RemediationKind) -> str:
    return (
        f"Observed answer: {kind.value}. Causal remediation: {remediation.value}. "
        "This trace reports score and scalar-loss-level evidence only; it does not establish a gradient, "
        "optimizer transform, or parameter update."
    )
