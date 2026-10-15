"""Five projections that only lift values from a persisted canonical Assay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reward_lens.contracts import digest, validate_record
from reward_lens.product import join as product_join

from .types import JSONPointer, Projection, ProjectionFieldAbsent, View

_JOIN_POINTER = "/join"
_CASE_EXTENSION = "https://reward-lens.github.io/case/v1"


def _document(record: Any) -> Mapping[str, Any]:
    to_dict = getattr(record, "to_dict", None)
    document = to_dict() if callable(to_dict) else record
    if not isinstance(document, Mapping):
        raise TypeError("projections require a canonical Assay or its saved mapping")
    validate_record(dict(document))
    return document


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _at(document: Any, pointer: JSONPointer) -> Any:
    current = document
    for token in pointer.removeprefix("/").split("/") if pointer else ():
        key = _unescape(token)
        if isinstance(current, list):
            current = current[int(key)]
        elif isinstance(current, Mapping):
            current = current[key]
        else:
            raise KeyError(pointer)
    return current


def _selected(
    record: Any,
    view: View,
    selections: tuple[tuple[str, JSONPointer], ...],
) -> Projection:
    document = _document(record)
    fields: dict[str, Any] = {}
    pointers: list[JSONPointer] = []
    for field, pointer in selections:
        try:
            fields[field] = _at(document, pointer)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ProjectionFieldAbsent(view=view, field=field, pointer=pointer) from error
        pointers.append(pointer)
    assay_id = fields.get("assay_id")
    if not isinstance(assay_id, str) or not assay_id:
        raise ProjectionFieldAbsent(view=view, field="assay_id", pointer="/assay_id")
    return Projection(
        view=view,
        fields=fields,
        source_pointers=tuple(pointers),
        summary=f"{view.value} projection for saved Assay {assay_id}",
    )


def _trace_entry_pointer(document: Mapping[str, Any], entry_id: str) -> JSONPointer:
    entries = _at(document, "/measurement/trace")
    if not isinstance(entries, list):
        raise KeyError("/measurement/trace")
    for index, entry in enumerate(entries):
        if isinstance(entry, Mapping) and entry.get("entry_id") == entry_id:
            return f"/measurement/trace/{index}"
    raise KeyError(entry_id)


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _case_pointer(field: str) -> JSONPointer:
    return f"/extensions/{_escape(_CASE_EXTENSION)}/{field}"


def _entry_index(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    measurement = _at(document, "/measurement")
    if not isinstance(measurement, Mapping):
        raise KeyError("/measurement")
    entries: dict[str, Mapping[str, Any]] = {}
    for section in measurement.values():
        if not isinstance(section, list):
            continue
        for entry in section:
            if isinstance(entry, Mapping) and isinstance(entry.get("entry_id"), str):
                entries[entry["entry_id"]] = entry
    return entries


def _case_source(document: Mapping[str, Any], field: str) -> JSONPointer:
    pointer = _case_pointer(field)
    try:
        value = _at(document, pointer)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ProjectionFieldAbsent(view=View.CASE, field=field, pointer=pointer) from error
    if not isinstance(value, list) or not value:
        raise ProjectionFieldAbsent(view=View.CASE, field=field, pointer=pointer)
    return pointer


def _case_evidence_resolves(
    document: Mapping[str, Any], *, field: str, pointer: JSONPointer
) -> None:
    value = _at(document, pointer)
    entries = _entry_index(document)
    for row in value:
        if not isinstance(row, Mapping):
            raise ProjectionFieldAbsent(view=View.CASE, field=field, pointer=pointer)
        entry_id, recorded_digest = row.get("entry_id"), row.get("digest")
        target = entries.get(entry_id) if isinstance(entry_id, str) else None
        if (
            target is None
            or not isinstance(recorded_digest, str)
            or digest(target) != recorded_digest
        ):
            raise ProjectionFieldAbsent(view=View.CASE, field=field, pointer=pointer)


def _case_rows_match(
    document: Mapping[str, Any], *, pair_pointer: JSONPointer, costs_pointer: JSONPointer
) -> None:
    pairs = _at(document, pair_pointer)
    costs = _at(document, costs_pointer)
    if not isinstance(pairs, list) or len(pairs) != 2:
        raise ProjectionFieldAbsent(view=View.CASE, field="decisive_pair", pointer=pair_pointer)
    if not isinstance(costs, list) or len(costs) != 2:
        raise ProjectionFieldAbsent(view=View.CASE, field="candidate_costs", pointer=costs_pointer)
    pair_ids = {row.get("candidate_id") for row in pairs if isinstance(row, Mapping)}
    cost_ids = {row.get("candidate_id") for row in costs if isinstance(row, Mapping)}
    cost_fields = {"candidate_id", "entry_id", "digest", "wall_s", "cpu_s", "usd", "api_calls"}
    if (
        len(pair_ids) != 2
        or not all(isinstance(candidate, str) and candidate for candidate in pair_ids)
        or pair_ids != cost_ids
        or any(not isinstance(row, Mapping) or not cost_fields <= set(row) for row in costs)
    ):
        raise ProjectionFieldAbsent(view=View.CASE, field="candidate_costs", pointer=costs_pointer)


def case(record: Any) -> Projection:
    """Owner reading, only when a producer retained pair and per-candidate-cost evidence."""
    document = _document(record)
    pair_pointer = _case_source(document, "decisive_pair")
    costs_pointer = _case_source(document, "candidate_costs")
    _case_evidence_resolves(document, field="decisive_pair", pointer=pair_pointer)
    _case_evidence_resolves(document, field="candidate_costs", pointer=costs_pointer)
    _case_rows_match(document, pair_pointer=pair_pointer, costs_pointer=costs_pointer)
    return _selected(
        record,
        View.CASE,
        (
            ("assay_id", "/assay_id"),
            ("decisive_pair", pair_pointer),
            ("candidate_costs", costs_pointer),
        ),
    )


def gate(record: Any) -> Projection:
    """CI reading: the saved policy verdict and its retained finding set."""
    return _selected(
        record,
        View.GATE,
        (
            ("assay_id", "/assay_id"),
            ("verdict", "/decision/state"),
            ("policy_id", "/decision/policy/id"),
            ("policy_digest", "/decision/policy/digest"),
            ("open_findings", "/findings"),
        ),
    )


def report(record: Any) -> Projection:
    """Stranger reading, bound to P-JOIN's sole join implementation."""
    product_join.join(record)
    return _selected(
        record,
        View.REPORT,
        (
            ("assay_id", "/assay_id"),
            ("subject", "/subject"),
            ("decision", "/decision"),
            ("join", _JOIN_POINTER),
        ),
    )


def incident(record: Any) -> Projection:
    """Investigator reading: retained trace, reward statistics, and five-way evidence."""
    document = _document(record)
    try:
        fiveway = _trace_entry_pointer(document, "trace.fiveway")
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ProjectionFieldAbsent(
            view=View.INCIDENT, field="fiveway_answer", pointer="/measurement/trace/*/trace.fiveway"
        ) from error
    return _selected(
        record,
        View.INCIDENT,
        (
            ("assay_id", "/assay_id"),
            ("trace", "/measurement/trace"),
            ("reward_statistics", "/measurement/reward_statistics"),
            ("fiveway_answer", f"{fiveway}/result/answer"),
            ("fiveway_evidence", f"{fiveway}/result/answer/evidence"),
        ),
    )


def registry(record: Any) -> Projection:
    """Registry reading: stable identity, subject, decision, and reproduction provenance."""
    return _selected(
        record,
        View.REGISTRY,
        (
            ("assay_id", "/assay_id"),
            ("subject", "/subject"),
            ("decision", "/decision"),
            ("provenance", "/provenance"),
        ),
    )


def project(record: Any, view: View) -> Projection:
    """Dispatch one frozen view through its sole projection implementation."""
    try:
        selected = {
            View.CASE: case,
            View.GATE: gate,
            View.REPORT: report,
            View.INCIDENT: incident,
            View.REGISTRY: registry,
        }[View(view)](record)
    except ValueError as error:
        raise TypeError(f"unknown projection view {view!r}") from error
    return selected


def regenerate(record: Any) -> dict[View, Projection]:
    """Produce every view through the same five functions used by single-view dispatch."""
    return {view: project(record, view) for view in View}
