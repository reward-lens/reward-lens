"""The D-47 column that connects audit families, run evidence, and compare readings.

This module deliberately reads a saved Assay as data.  It does not replay a run,
infer a family, or turn a score/loss value into an update measurement.  Missing
facts stay named absences in their cells, so a report is still useful without
claiming a link that its evidence did not establish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping

from reward_lens import contracts
from reward_lens.product.compare.scorecard import Absence, Estimate

__all__ = [
    "Absence",
    "Estimate",
    "HEADINGS",
    "JOIN_EXTENSION",
    "Join",
    "JoinRow",
    "attach_join_evidence",
    "canonical_rows",
    "has_join_column",
    "join",
    "render_text",
]


HEADINGS = (
    "selected in your run",
    "applied update weight",
    "fix changes pressure",
    "cost of the fix",
)
JOIN_EXTENSION = "https://reward-lens.github.io/join/v1"

_FIELDS = (
    "selected_in_your_run",
    "applied_update_weight",
    "fix_changes_pressure",
    "cost_of_the_fix",
)
_EVIDENCE_KEY = {
    "selected_in_your_run": "selection",
    "applied_update_weight": "update",
    "fix_changes_pressure": "pressure",
    "cost_of_the_fix": "cost",
}


@dataclass(frozen=True)
class JoinRow:
    """One audit family, with each displayed fact bound to its own evidence entry."""

    family: str
    selected_in_your_run: bool | Absence
    applied_update_weight: Estimate | Absence
    fix_changes_pressure: Estimate | Absence
    cost_of_the_fix: Estimate | Absence
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise TypeError("join family must be a non-empty string")
        evidence = dict(self.evidence)
        for field_name in _FIELDS:
            cell = getattr(self, field_name)
            allowed = (
                (bool, Absence) if field_name == "selected_in_your_run" else (Estimate, Absence)
            )
            if not isinstance(cell, allowed):
                raise TypeError(f"{field_name} must be an evidence-bound value or Absence")
            if not isinstance(cell, Absence) and not evidence.get(field_name):
                raise ValueError(f"{field_name} needs an evidence identifier")
        object.__setattr__(self, "evidence", MappingProxyType(evidence))


@dataclass(frozen=True)
class Join:
    rows: list[JoinRow]
    signal: Estimate | Absence
    signal_evidence: str | None = None


def _document(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        document = to_dict()
        if isinstance(document, Mapping):
            return document
    raise TypeError("join expects a saved Assay mapping or an object with to_dict()")


def _absence(reason: str, remedy: str) -> Absence:
    return Absence(f"not measured: {reason}", remedy)


def _families(document: Mapping[str, Any]) -> list[str]:
    audit = document.get("audit")
    declared = audit.get("exploit_families") if isinstance(audit, Mapping) else None
    if declared is None:
        declared = document.get("exploit_families")
    if declared is None:
        declared = [
            row.get("family") for row in document.get("join", []) if isinstance(row, Mapping)
        ]
    found: list[str] = []
    for family in declared or ():
        if isinstance(family, str) and family and family not in found:
            found.append(family)
        elif isinstance(family, Mapping):
            name = family.get("id") or family.get("family")
            if isinstance(name, str) and name and name not in found:
                found.append(name)
    return found


def _entries(document: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    rows = document.get("join")
    if not isinstance(rows, list):
        return {}
    by_family: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("family"), str):
            by_family.setdefault(row["family"], row)
    return by_family


def _estimate(value: Any, *, field_name: str, evidence: str | None) -> Estimate | Absence:
    if isinstance(value, Estimate):
        if evidence:
            return value
        return _absence(
            f"{field_name.replace('_', ' ')} has no evidence identifier",
            "record the evidence entry that establishes this per-family reading",
        )
    if not isinstance(value, Mapping):
        return _absence(
            f"{field_name.replace('_', ' ')} is unavailable",
            "record an evidence-bound estimate for this family",
        )
    if not evidence:
        return _absence(
            f"{field_name.replace('_', ' ')} has no evidence identifier",
            "record the evidence entry that establishes this per-family reading",
        )
    try:
        interval = value["interval"]
        result = Estimate(
            Decimal(str(value["value"])),
            (Decimal(str(interval[0])), Decimal(str(interval[1]))),
            n=int(value["n"]),
            n_eff=None if value.get("n_eff") is None else Decimal(str(value["n_eff"])),
            unit=str(value.get("unit") or "proportion"),
            samples=value.get("samples"),
        )
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return _absence(
            f"{field_name.replace('_', ' ')} is not an evidence-bound estimate",
            "record value, interval, n, unit, and the evidence entry for this family",
        )
    return result


def _selected(value: Any, evidence: str | None) -> bool | Absence:
    if value is True or value == "yes":
        result = True
    elif value is False or value == "no":
        result = False
    else:
        return _absence(
            "selection in this run is unavailable",
            "record whether this family was selected and the evidence entry for that decision",
        )
    if not evidence:
        return _absence(
            "selection in this run has no evidence identifier",
            "record the trace entry that establishes the per-family selection decision",
        )
    return result


def _evidence(row: Mapping[str, Any], field_name: str) -> str | None:
    evidence = row.get("evidence")
    if not isinstance(evidence, Mapping):
        return None
    value = evidence.get(field_name) or evidence.get(_EVIDENCE_KEY[field_name])
    return value if isinstance(value, str) and value else None


def _row(family: str, evidence_row: Mapping[str, Any] | None, *, has_run: bool) -> JoinRow:
    if evidence_row is None:
        evidence_row = {}
    evidence = {
        field_name: value
        for field_name in _FIELDS
        if (value := _evidence(evidence_row, field_name))
    }
    selected = _selected(
        evidence_row.get("selected_in_your_run", evidence_row.get("selected")),
        evidence.get("selected_in_your_run"),
    )
    if not has_run:
        middle = _absence(
            "no run record is attached for this audit",
            "attach the closed trace run record for this audited reward version",
        )
        update = middle
        pressure = middle
    else:
        update = _estimate(
            evidence_row.get("applied_update_weight"),
            field_name="applied_update_weight",
            evidence=evidence.get("applied_update_weight"),
        )
        pressure = _estimate(
            evidence_row.get("fix_changes_pressure"),
            field_name="fix_changes_pressure",
            evidence=evidence.get("fix_changes_pressure"),
        )
    cost = _estimate(
        evidence_row.get("cost_of_the_fix", evidence_row.get("fix_cost")),
        field_name="cost_of_the_fix",
        evidence=evidence.get("cost_of_the_fix"),
    )
    return JoinRow(family, selected, update, pressure, cost, evidence)


def _signal(document: Mapping[str, Any]) -> Estimate | Absence:
    value = document.get("signal")
    if isinstance(value, Estimate):
        return value
    if isinstance(value, Mapping):
        return _estimate(value, field_name="signal", evidence=value.get("evidence_id"))
    return _absence(
        "the reward signal reading is unavailable",
        "record the shared signal estimate and its evidence entry",
    )


_IDENTITY_FIELDS = ("task_id", "response_id", "family", "subject_ref")
# Trace and Audit bind the one unambiguous retained assistant response with the
# shared C061 response-text identity.  The digest deliberately names only the
# recorded response text; assistant-turn position remains trace provenance.
_RESPONSE_CODEC = "rfc8785-response-text/1"
_VALID_SCOPES = frozenset(
    {
        "evaluator_comparison",
        "selection_stress",
        "reconstructed_training_pressure",
        "controlled_update_response",
        "short_fork_behavior",
        "held_out_transfer",
    }
)


def _extension(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    extensions = document.get("extensions")
    value = extensions.get(JOIN_EXTENSION) if isinstance(extensions, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _identity(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    result = {field: value.get(field) for field in _IDENTITY_FIELDS}
    return result if all(isinstance(item, str) and item for item in result.values()) else None


def _entry_index(document: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    measurement = document.get("measurement")
    if not isinstance(measurement, Mapping):
        return {}
    found: dict[str, Mapping[str, Any]] = {}
    for section, group in measurement.items():
        if not isinstance(group, list):
            continue
        for entry in group:
            if isinstance(entry, Mapping) and isinstance(entry.get("entry_id"), str):
                found.setdefault(entry["entry_id"], {**entry, "section": section})
    return found


def _seeker_identity(entry: Mapping[str, Any]) -> dict[str, str] | None:
    """Read an exploit witness's identity from its retained proposal record.

    The seeker witness is the canonical audit-side producer: its result names the
    proposal and its retained actor record names the task and response that were
    replayed.  A TraceRow does not currently retain that proposal identifier, so
    this helper deliberately does not try to synthesize a trace binding.
    """
    result = entry.get("result")
    if not isinstance(result, Mapping) or entry.get("section") != "exploits":
        return None
    proposal_id = result.get("proposal_id")
    actor_record = result.get("actor_record")
    if not isinstance(proposal_id, str) or not isinstance(actor_record, Mapping):
        return None
    scored = actor_record.get("scores")
    if not isinstance(scored, list):
        return None
    for item in scored:
        proposal = item.get("proposal") if isinstance(item, Mapping) else None
        if not isinstance(proposal, Mapping) or proposal.get("proposal_id") != proposal_id:
            continue
        task_id = proposal.get("task_id")
        response_id = proposal.get("proposal_id")
        family = (
            (entry.get("method") or {}).get("id")
            if isinstance(entry.get("method"), Mapping)
            else None
        )
        subject = entry.get("subject_ref")
        if all(
            isinstance(value, str) and value for value in (task_id, response_id, family, subject)
        ):
            return {
                "task_id": task_id,
                "response_id": response_id,
                "family": family,
                "subject_ref": subject,
            }
    return None


def _seeker_binding(
    entry: Mapping[str, Any], *, grader_source_digest: str | None
) -> dict[str, str] | None:
    """Return the retained proposal's joinable task/response/source binding.

    A proposal id is a digest of the actor request, not its response bytes.  The actor record
    retained by the accepted Seeker witness carries the proposal's separately recorded
    response-text identity.  Read that stored identity verbatim; never derive a Trace binding
    from the response text again in Join.
    """
    identity = _seeker_identity(entry)
    result = entry.get("result")
    if identity is None or not isinstance(result, Mapping):
        return None
    proposal_id = result.get("proposal_id")
    actor_record = result.get("actor_record")
    scores = actor_record.get("scores") if isinstance(actor_record, Mapping) else None
    proposal = next(
        (
            item.get("proposal")
            for item in scores or ()
            if isinstance(item, Mapping)
            and isinstance(item.get("proposal"), Mapping)
            and item["proposal"].get("proposal_id") == proposal_id
        ),
        None,
    )
    response_digest = proposal.get("response_digest") if isinstance(proposal, Mapping) else None
    response_codec = proposal.get("response_codec") if isinstance(proposal, Mapping) else None
    if not (
        isinstance(response_digest, str)
        and response_digest
        and response_codec == _RESPONSE_CODEC
        and isinstance(grader_source_digest, str)
        and grader_source_digest
    ):
        return None
    return {
        **identity,
        "response_digest": response_digest,
        "response_codec": response_codec,
        "grader_source_digest": grader_source_digest,
    }


def _trace_binding(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read the C057 canonical row witness without accepting an arbitrary evidence string."""
    result = entry.get("result")
    if (
        entry.get("section") != "trace"
        or entry.get("scope") != "reconstructed_training_pressure"
        or not isinstance(result, Mapping)
    ):
        return None
    required = ("task_id", "response_digest", "response_codec", "run_identity")
    if not all(isinstance(result.get(name), str) and result[name] for name in required):
        return None
    grader_source = result.get("grader_source")
    if not isinstance(grader_source, Mapping):
        return None
    source_digest = grader_source.get("grader_source_digest")
    if not isinstance(source_digest, str) or not source_digest:
        return None
    return {
        "task_id": result["task_id"],
        "response_digest": result["response_digest"],
        "response_codec": result["response_codec"],
        "grader_source_digest": source_digest,
        "run_identity": result["run_identity"],
        "trace_subject_ref": entry.get("subject_ref"),
    }


def _matching_trace_entry(
    binding: Mapping[str, str], entry: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    observed = _trace_binding(entry)
    if observed is None:
        return None
    fields = ("task_id", "response_digest", "response_codec", "grader_source_digest")
    return entry if all(observed[field] == binding[field] for field in fields) else None


def _entry_identity(entry: Mapping[str, Any]) -> dict[str, str] | None:
    result = entry.get("result")
    direct = _identity(result) if isinstance(result, Mapping) else None
    return direct or _seeker_identity(entry)


def _target(
    reference: Any,
    *,
    identity: Mapping[str, str] | None,
    entries: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, Absence | None]:
    if not isinstance(reference, str) or not reference:
        return None, _absence(
            "no saved Assay evidence identifier is declared",
            "record the entry id that establishes this cell",
        )
    entry = entries.get(reference)
    if entry is None:
        return None, _absence(
            f"evidence entry {reference!r} does not resolve in this saved Assay",
            "save the producing audit, trace, or compare entry in this Assay before joining it",
        )
    if entry.get("scope") not in _VALID_SCOPES:
        return None, _absence(
            f"evidence entry {reference!r} has no valid comparison scope",
            "record the entry under one declared reward-lens comparison scope",
        )
    if identity is not None:
        observed = _entry_identity(entry)
        if entry.get("subject_ref") != identity["subject_ref"] or observed != dict(identity):
            return None, _absence(
                f"evidence entry {reference!r} does not match task, response, family, and subject identity",
                "save evidence for this exact task, response, exploit family, and audited subject",
            )
    return entry, None


def _entry_estimate(entry: Mapping[str, Any]) -> Estimate | Absence:
    uncertainty = entry.get("uncertainty")
    interval = uncertainty.get("interval") if isinstance(uncertainty, Mapping) else None
    try:
        return Estimate(
            Decimal(str(entry["value"])),
            (Decimal(str(interval[0])), Decimal(str(interval[1]))),
            n=int(entry["n"]),
            unit=str(entry["unit"]),
        )
    except (KeyError, TypeError, ValueError, ArithmeticError, IndexError):
        return _absence(
            f"evidence entry {entry.get('entry_id')!r} is not a complete estimate",
            "save value, interval, n, and unit on the cited evidence entry",
        )


def _trace_update(entry: Mapping[str, Any]) -> Estimate | Absence:
    """Expose only a recorded update weight; score and advantage remain separate facts."""
    result = entry.get("result")
    if not isinstance(result, Mapping) or result.get("applied_weight_source") != "recorded":
        return _absence(
            "the matched trace row has no recorded applied update weight",
            "record the optimizer-applied row weight before the run closes",
        )
    value = result.get("applied_weight")
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return _absence(
            "the matched trace row's recorded applied update weight is unavailable",
            "record a numeric optimizer-applied row weight before the run closes",
        )
    number = Decimal(str(value))
    return Estimate(number, (number, number), n=1, unit="recorded applied update weight")


def _trace_target(
    reference: Any,
    *,
    binding: Mapping[str, str],
    entries: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, Absence | None]:
    if not isinstance(reference, str) or not reference:
        return None, _absence(
            "no canonical trace-row evidence identifier is declared",
            "save the matching trace.row entry in this Assay before joining it",
        )
    entry = entries.get(reference)
    if entry is None:
        return None, _absence(
            f"evidence entry {reference!r} does not resolve in this saved Assay",
            "save the matching trace.row entry in this Assay before joining it",
        )
    if _matching_trace_entry(binding, entry) is None:
        return None, _absence(
            f"evidence entry {reference!r} does not match task, response digest, codec, and grader source",
            "retain one canonical trace row with this exact task, response digest, and declared grader source",
        )
    return entry, None


def _strict_row(
    identity: Mapping[str, str],
    row: Mapping[str, Any] | None,
    entries: Mapping[str, Mapping[str, Any]],
) -> JoinRow:
    if row is None:
        missing = _absence(
            "no join reading has the audit finding's full task, response, family, and subject identity",
            "save trace and compare readings for this exact audit finding",
        )
        return JoinRow(identity["family"], missing, missing, missing, missing)
    evidence = row.get("evidence") if isinstance(row.get("evidence"), Mapping) else {}
    trace_binding = (
        row.get("trace_binding") if isinstance(row.get("trace_binding"), Mapping) else None
    )
    values: dict[str, bool | Estimate | Absence] = {}
    resolved: dict[str, str] = {}
    for field_name in _FIELDS:
        reference = evidence.get(field_name)
        if field_name == "applied_update_weight" and trace_binding is not None:
            entry, missing = _trace_target(reference, binding=trace_binding, entries=entries)
            if missing is not None:
                values[field_name] = missing
                continue
            assert entry is not None
            values[field_name] = _trace_update(entry)
            if not isinstance(values[field_name], Absence):
                resolved[field_name] = reference
            continue
        if field_name in {"applied_update_weight", "fix_changes_pressure"} and not row.get(
            "trace_row_matched", False
        ):
            values[field_name] = _absence(
                "the closed TraceRecord has no exact task, response digest, codec, and grader-source match",
                "retain the matching canonical trace row and declared grader source before the run closes",
            )
            continue
        entry, missing = _target(reference, identity=identity, entries=entries)
        if missing is not None:
            values[field_name] = missing
            continue
        assert entry is not None
        if field_name == "selected_in_your_run":
            selected = (
                True
                if _seeker_identity(entry) == dict(identity)
                else (
                    entry.get("result", {}).get("selected")
                    if isinstance(entry.get("result"), Mapping)
                    else None
                )
            )
            if type(selected) is not bool:
                values[field_name] = _absence(
                    f"evidence entry {reference!r} has no boolean selection reading",
                    "save whether this exact family was selected in the cited trace entry",
                )
                continue
            values[field_name] = selected
        else:
            values[field_name] = _entry_estimate(entry)
            if isinstance(values[field_name], Absence):
                continue
        resolved[field_name] = reference
    return JoinRow(
        identity["family"],
        values["selected_in_your_run"],
        values["applied_update_weight"],
        values["fix_changes_pressure"],
        values["cost_of_the_fix"],
        resolved,
    )


def _strict_join(document: Mapping[str, Any], payload: Mapping[str, Any]) -> Join:
    audits = [_identity(value) for value in payload.get("audit_families", ())]
    audit_identities = [value for value in audits if value is not None]
    rows = [_identity(value) for value in payload.get("rows", ())]
    _readings = [value for value in rows if value is not None]
    raw_rows = payload.get("rows", ())
    entries = _entry_index(document)
    result_rows = []
    for identity in audit_identities:
        matched = next(
            (
                raw
                for raw, parsed in zip(raw_rows, rows)
                if parsed == identity and isinstance(raw, Mapping)
            ),
            None,
        )
        result_rows.append(_strict_row(identity, matched, entries))
    subjects = {identity["subject_ref"] for identity in audit_identities}
    signal_entry, missing = _target(payload.get("signal_evidence"), identity=None, entries=entries)
    if missing is not None:
        signal: Estimate | Absence = missing
        signal_ref = None
    elif signal_entry is None or signal_entry.get("subject_ref") not in subjects:
        signal = _absence(
            "signal evidence does not match the saved Assay subject",
            "save the shared signal entry for this audited subject",
        )
        signal_ref = None
    else:
        signal = _entry_estimate(signal_entry)
        signal_ref = (
            str(payload.get("signal_evidence")) if not isinstance(signal, Absence) else None
        )
    return Join(result_rows, signal, signal_ref)


def join(record: Any) -> Join:
    """Return the retained D-47 column for every family the audit reports."""
    document = _document(record)
    payload = _extension(document)
    if payload is not None:
        return _strict_join(document, payload)
    rows = _entries(document)
    has_run = bool(document.get("run_record"))
    return Join(
        [_row(family, rows.get(family), has_run=has_run) for family in _families(document)],
        _signal(document),
    )


def has_join_column(record: Any) -> bool:
    """Whether every saved audit identity has a full-identity join reading.

    This is the product contract P-GATES-3 consumes; it is intentionally not a
    Gate-21 implementation or a substitute for the gate runner's audit-route
    positive and negative controls.
    """
    document = _document(record)
    payload = _extension(document)
    if payload is None:
        return False
    audits = [_identity(value) for value in payload.get("audit_families", ())]
    rows = [_identity(value) for value in payload.get("rows", ())]
    return bool(audits) and all(identity is not None and identity in rows for identity in audits)


def canonical_rows(value: Join) -> list[contracts.JoinRow]:
    """Project evidence-bound D-47 readings into the sealed Assay join schema.

    The Join extension retains every identity and entry-id anchor.  ``Assay.join`` is its compact,
    schema-level projection for pointer consumers such as Report; an absent cell stays ``None``
    with its reason in ``missing`` and is never promoted to a numeric value.
    """
    rows: list[contracts.JoinRow] = []
    for row in value.rows:
        absent = {
            name: {"reason": cell.reason, "remedy": cell.remedy}
            for name, cell in (
                ("selected", row.selected_in_your_run),
                ("applied_update_weight", row.applied_update_weight),
                ("fix_changes_pressure", row.fix_changes_pressure),
                ("fix_cost", row.cost_of_the_fix),
            )
            if isinstance(cell, Absence)
        }
        selected = (
            "yes"
            if row.selected_in_your_run is True
            else "no"
            if row.selected_in_your_run is False
            else "not_measured"
        )
        applied = row.applied_update_weight
        pressure = row.fix_changes_pressure
        cost = row.cost_of_the_fix
        payload: dict[str, Any] = {
            "family": row.family,
            "selected": selected,
            "applied_update_weight": applied.value if isinstance(applied, Estimate) else None,
            "fix_changes_pressure": pressure.value if isinstance(pressure, Estimate) else None,
            "fix_cost": f"{cost.value} {cost.unit}" if isinstance(cost, Estimate) else None,
        }
        if absent:
            payload["missing"] = absent
        rows.append(contracts.JoinRow(**payload))
    return rows


def attach_join_evidence(
    assay: Any,
    *,
    trace_record: Any,
    audited_grader_source_digest: str,
) -> Any:
    """Attach the audit's retained exploit identities before the Assay is sealed.

    This producer reads retained exploit witnesses and canonical Trace row entries already saved
    into the Assay.  It joins only equal task ids, response digests, response codecs, and explicit
    grader-source digests.  Audit and Trace subject/run references remain provenance, not equality
    constraints: they name different things.
    """
    if not isinstance(audited_grader_source_digest, str) or not audited_grader_source_digest:
        raise TypeError("attach_join_evidence needs the Audit grader source digest")
    document = _document(assay)
    if not getattr(trace_record, "complete", False):
        raise ValueError("join attachment requires the closed TraceRecord")
    entries = _entry_index(document)
    identities: list[dict[str, str]] = []
    selection: dict[tuple[str, str, str, str], str] = {}
    bindings: dict[tuple[str, str, str, str], dict[str, str]] = {}
    # Audit passes this before project-version settlement.  It is in the exact
    # raw-grader digest domain preserved by canonical Trace rows; the settled
    # project source-component digest is deliberately a different identity.
    grader_source_digest = audited_grader_source_digest
    for entry_id, entry in entries.items():
        identity = _seeker_identity(entry)
        if identity is None:
            continue
        key = tuple(identity[field] for field in _IDENTITY_FIELDS)
        if identity not in identities:
            identities.append(identity)
        selection[key] = entry_id
        binding = _seeker_binding(entry, grader_source_digest=grader_source_digest)
        if binding is not None:
            bindings[key] = binding
    normalized_rows: list[dict[str, Any]] = []
    for identity in identities:
        key = tuple(identity[field] for field in _IDENTITY_FIELDS)
        binding = bindings.get(key)
        trace_entry = next(
            (
                candidate
                for candidate in entries.values()
                if binding is not None and _matching_trace_entry(binding, candidate) is not None
            ),
            None,
        )
        evidence = {"selected_in_your_run": selection[key]}
        row: dict[str, Any] = {**identity, "evidence": evidence}
        if binding is not None:
            row["trace_binding"] = binding
        if trace_entry is not None:
            row["trace_subject_ref"] = trace_entry.get("subject_ref")
            trace_result = trace_entry.get("result")
            if isinstance(trace_result, Mapping):
                row["trace_run_identity"] = trace_result.get("run_identity")
                if trace_result.get("applied_weight_source") == "recorded":
                    evidence["applied_update_weight"] = trace_entry["entry_id"]
        else:
            # The reader retains each source's own run identity, even while an absent matched row
            # is reported per cell by the strict reader.
            row["trace_subject_ref"] = str(getattr(trace_record, "subject_ref", ""))
            run_identity = getattr(trace_record, "run_identity", None)
            row["trace_run_identity"] = (
                run_identity if isinstance(run_identity, str) and run_identity else None
            )
        normalized_rows.append(row)
    subjects = {identity["subject_ref"] for identity in identities}
    signal_evidence = next(
        (
            entry_id
            for entry_id, entry in entries.items()
            if entry.get("section") == "signal"
            and entry.get("subject_ref") in subjects
            and entry.get("scope") in _VALID_SCOPES
        ),
        None,
    )
    payload = {
        "audit_families": identities,
        "rows": normalized_rows,
        "signal_evidence": signal_evidence,
        "trace": {
            "reader": str(getattr(trace_record, "reader", "")),
            "run_id": str(getattr(trace_record, "run_id", "")),
            "run_identity": getattr(trace_record, "run_identity", None),
            "subject_ref": str(getattr(trace_record, "subject_ref", "")),
            "complete": True,
        },
        # This is the digest domain Trace uses for its executed grader identity:
        # ``contracts.digest({\"source\": exact_grader_text})``.  It is intentionally distinct
        # from the settled project version's source-component digest, so Join compares like with
        # like instead of silently treating two digest domains as interchangeable.
        "audited_grader_source_digest": audited_grader_source_digest,
    }
    extensions = dict(getattr(assay, "extensions", None) or document.get("extensions") or {})
    extensions[JOIN_EXTENSION] = payload
    if hasattr(assay, "extensions"):
        assay.extensions = extensions
        assay.join = canonical_rows(join(assay))
        return assay
    document["extensions"] = extensions
    document["join"] = [row.model_dump(mode="json") for row in canonical_rows(join(document))]
    return document


def _text_cell(value: bool | Estimate | Absence, evidence: str | None = None) -> str:
    if isinstance(value, Absence):
        return f"{value.reason}; remedy: {value.remedy}"
    if isinstance(value, Estimate):
        suffix = f" [{evidence}]" if evidence else ""
        return f"{value.value} {value.unit}{suffix}"
    return ("yes" if value else "no") + (f" [{evidence}]" if evidence else "")


def render_text(value: Join) -> str:
    """The CLI presentation only; all row construction remains in :func:`join`."""
    lines = ["family | " + " | ".join(HEADINGS)]
    for row in value.rows:
        lines.append(
            " | ".join(
                (
                    row.family,
                    _text_cell(row.selected_in_your_run, row.evidence.get("selected_in_your_run")),
                    _text_cell(
                        row.applied_update_weight, row.evidence.get("applied_update_weight")
                    ),
                    _text_cell(row.fix_changes_pressure, row.evidence.get("fix_changes_pressure")),
                    _text_cell(row.cost_of_the_fix, row.evidence.get("cost_of_the_fix")),
                )
            )
        )
    lines.append("signal | " + _text_cell(value.signal))
    return "\n".join(lines)
