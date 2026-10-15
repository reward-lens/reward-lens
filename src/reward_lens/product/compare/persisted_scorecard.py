"""The sealed, lossless scorecard projection carried by a Compare Assay.

Compare's rung engines deliberately return a rich :class:`Scorecard` rather
than trying to encode six heterogeneous rows in ordinary measurement entries.
This module is the one bridge between that result and the durable Assay: the
writer preserves exactly what the engine returned, and the reader validates it
back into the same typed values.  It never tries to reconstruct a scorecard
from a selection of entries, because that would silently lose named absences,
intervals, scopes, and tradeoffs.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from reward_lens import contracts

from .registry import CompareResult, Rung
from .scorecard import (
    ROWS,
    Absence,
    CapabilityFloor,
    Estimate,
    Scorecard,
    ScorecardRow,
    Tradeoff,
)

SCORECARD_EXTENSION = "https://reward-lens.github.io/extensions/compare-scorecard/1"
_FORMAT_VERSION = 1

__all__ = [
    "SCORECARD_EXTENSION",
    "PersistedScorecardError",
    "read_persisted_scorecards",
    "retain_persisted_scorecards",
]


class PersistedScorecardError(ValueError):
    """A record does not carry a complete, self-consistent Compare scorecard."""


def _fail(detail: str) -> None:
    raise PersistedScorecardError(f"invalid persisted compare scorecard: {detail}")


def _decimal(value: Any, *, field: str) -> Decimal:
    if not isinstance(value, str):
        _fail(f"{field} must be a decimal string")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise PersistedScorecardError(
            f"invalid persisted compare scorecard: {field} is not a decimal string"
        ) from error
    if not parsed.is_finite():
        _fail(f"{field} must be finite")
    return parsed


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], *, field: str, keys: set[str]) -> None:
    observed = set(value)
    if observed != keys:
        _fail(f"{field} must contain exactly {', '.join(sorted(keys))}")


def _encode_estimate(value: Estimate) -> dict[str, Any]:
    encoded: dict[str, Any] = {
        "value": str(value.value),
        "interval": [str(value.interval[0]), str(value.interval[1])],
        "n": value.n,
        "unit": value.unit,
    }
    if value.n_eff is not None:
        encoded["n_eff"] = str(value.n_eff)
    if value.samples is not None:
        encoded["samples"] = value.samples
    return encoded


def _encode_value(value: Estimate | Absence) -> dict[str, Any]:
    if isinstance(value, CapabilityFloor):
        encoded = _encode_estimate(value)
        encoded.update(
            {
                "kind": "capability_floor",
                "partition_id": value.partition_id,
                "estimand": value.estimand,
                "floor": str(value.floor),
                "measured": _encode_estimate(value.measured),
                "held": value.held,
            }
        )
        return encoded
    if isinstance(value, Estimate):
        return {"kind": "estimate", **_encode_estimate(value)}
    if isinstance(value, Absence):
        return {"kind": "absence", "reason": value.reason, "remedy": value.remedy}
    raise TypeError("scorecard values must be Estimate, CapabilityFloor, or Absence")


def _decode_estimate(value: Mapping[str, Any], *, field: str) -> Estimate:
    allowed = {"value", "interval", "n", "n_eff", "unit", "samples"}
    if set(value) - allowed or not {"value", "interval", "n", "unit"} <= set(value):
        _fail(f"{field} has unsupported or missing estimate fields")
    interval = value["interval"]
    if not isinstance(interval, list) or len(interval) != 2:
        _fail(f"{field}.interval must have exactly two decimal endpoints")
    n = value["n"]
    samples = value.get("samples")
    if type(n) is not int or (samples is not None and type(samples) is not int):
        _fail(f"{field}.n and {field}.samples must be integers")
    if not isinstance(value["unit"], str):
        _fail(f"{field}.unit must be a string")
    return Estimate(
        value=_decimal(value["value"], field=f"{field}.value"),
        interval=(
            _decimal(interval[0], field=f"{field}.interval[0]"),
            _decimal(interval[1], field=f"{field}.interval[1]"),
        ),
        n=n,
        n_eff=(None if "n_eff" not in value else _decimal(value["n_eff"], field=f"{field}.n_eff")),
        unit=value["unit"],
        samples=samples,
    )


def _decode_value(value: Any, *, field: str) -> Estimate | Absence:
    encoded = _mapping(value, field=field)
    kind = encoded.get("kind")
    if kind == "absence":
        _require_keys(encoded, field=field, keys={"kind", "reason", "remedy"})
        return Absence(reason=encoded["reason"], remedy=encoded["remedy"])
    if kind == "estimate":
        if "kind" not in encoded:
            _fail(f"{field}.kind is required")
        return _decode_estimate(
            {key: item for key, item in encoded.items() if key != "kind"}, field=field
        )
    if kind == "capability_floor":
        required = {
            "kind",
            "value",
            "interval",
            "n",
            "unit",
            "partition_id",
            "estimand",
            "floor",
            "measured",
            "held",
        }
        allowed = required | {"n_eff", "samples"}
        if set(encoded) - allowed or not required <= set(encoded):
            _fail(f"{field} has unsupported or missing capability-floor fields")
        measured = _decode_estimate(
            _mapping(encoded["measured"], field=f"{field}.measured"), field=f"{field}.measured"
        )
        base = _decode_estimate(
            {
                key: item
                for key, item in encoded.items()
                if key not in {"kind", "partition_id", "estimand", "floor", "measured", "held"}
            },
            field=field,
        )
        if type(encoded["held"]) is not bool:
            _fail(f"{field}.held must be boolean")
        return CapabilityFloor(
            value=base.value,
            interval=base.interval,
            n=base.n,
            n_eff=base.n_eff,
            unit=base.unit,
            samples=base.samples,
            partition_id=encoded["partition_id"],
            estimand=encoded["estimand"],
            floor=_decimal(encoded["floor"], field=f"{field}.floor"),
            measured=measured,
            held=encoded["held"],
        )
    _fail(f"{field}.kind must name estimate, absence, or capability_floor")


def _scorecard_payload(
    *, record: contracts.Assay, rung: Rung, result: CompareResult
) -> dict[str, Any]:
    """Encode one actual rung result after its entries have been settled into ``record``."""
    locations = {
        entry.entry_id: (section, index, entry)
        for section in contracts.Measurement.model_fields
        for index, entry in enumerate(getattr(record.measurement, section))
    }
    evidence = []
    for entry in result.entries:
        try:
            section, index, settled = locations[entry.entry_id]
        except KeyError as error:
            raise RuntimeError(
                f"compare scorecard entry {entry.entry_id!r} was not retained in its Assay"
            ) from error
        evidence.append(
            {
                "entry_id": settled.entry_id,
                "section": section,
                "pointer": f"/measurement/{section}/{index}",
                "subject_ref": settled.subject_ref,
            }
        )
    payload: dict[str, Any] = {
        "rung": int(rung),
        "scope": result.scope,
        "rows": [
            {
                "id": row.id,
                "label": row.label,
                "scope": row.scope,
                "value": _encode_value(row.value),
            }
            for row in result.scorecard.rows
        ],
        "evidence": evidence,
    }
    if result.scorecard.tradeoff is not None:
        payload["tradeoff"] = {
            "option_a": result.scorecard.tradeoff.option_a,
            "option_b": result.scorecard.tradeoff.option_b,
            "competing_consequences": list(result.scorecard.tradeoff.competing_consequences),
        }
    return payload


def retain_persisted_scorecards(
    record: contracts.Assay,
    *,
    baseline: str,
    candidate: str,
    rungs: Iterable[Rung],
    results: Iterable[CompareResult],
) -> None:
    """Attach only the scorecards the Compare engines actually returned.

    The record already carries the result entries when this is called.  This
    ordering is what lets the projection bind each row set to its final,
    persisted evidence pointer and settled subject identity.
    """
    scorecards = tuple(
        _scorecard_payload(record=record, rung=rung, result=result)
        for rung, result in zip(rungs, results, strict=True)
    )
    record.extensions = dict(record.extensions or {})
    record.extensions[SCORECARD_EXTENSION] = {
        "version": _FORMAT_VERSION,
        "baseline": baseline,
        "candidate": candidate,
        "subject_ref": record.subject.version.digest,
        "provenance": {
            "offline": record.provenance.offline,
            "seed": record.provenance.seed,
            "reproduce": list(record.provenance.reproduce),
        },
        "scorecards": list(scorecards),
    }


def _evidence_is_current(record: contracts.Assay, evidence: Any) -> None:
    if not isinstance(evidence, list) or not evidence:
        _fail("each scorecard must point to at least one persisted entry")
    for item in evidence:
        pointer = _mapping(item, field="evidence item")
        _require_keys(
            pointer,
            field="evidence item",
            keys={"entry_id", "section", "pointer", "subject_ref"},
        )
        section = pointer["section"]
        entry_id = pointer["entry_id"]
        location = pointer["pointer"]
        subject_ref = pointer["subject_ref"]
        if section not in contracts.Measurement.model_fields or not isinstance(entry_id, str):
            _fail("evidence item names an invalid section or entry id")
        entries = getattr(record.measurement, section)
        expected_pointer = None
        for index, entry in enumerate(entries):
            if entry.entry_id == entry_id:
                expected_pointer = f"/measurement/{section}/{index}"
                if entry.subject_ref != subject_ref:
                    _fail(f"evidence item {entry_id!r} has a changed subject identity")
                break
        if expected_pointer is None or location != expected_pointer:
            _fail(f"evidence item {entry_id!r} does not point to this saved record")


def _read_scorecard(payload: Any, *, record: contracts.Assay) -> Scorecard:
    encoded = _mapping(payload, field="scorecard")
    required = {"rung", "scope", "rows", "evidence"}
    allowed = required | {"tradeoff"}
    if set(encoded) - allowed or not required <= set(encoded):
        _fail("scorecard has unsupported or missing fields")
    try:
        rung = Rung(encoded["rung"])
    except (TypeError, ValueError) as error:
        raise PersistedScorecardError(
            "invalid persisted compare scorecard: scorecard rung is invalid"
        ) from error
    scope = encoded["scope"]
    if scope not in {
        "evaluator_comparison",
        "selection_stress",
        "reconstructed_training_pressure",
        "controlled_update_response",
        "short_fork_behavior",
        "held_out_transfer",
    }:
        _fail("scorecard scope is invalid")
    rows = encoded["rows"]
    if not isinstance(rows, list) or len(rows) != len(ROWS):
        _fail("scorecard must carry exactly the six fixed rows")
    decoded_rows = []
    for index, item in enumerate(rows):
        row = _mapping(item, field=f"rows[{index}]")
        _require_keys(row, field=f"rows[{index}]", keys={"id", "label", "scope", "value"})
        if row["scope"] != scope:
            _fail(f"rows[{index}] has a different scope from its scorecard")
        decoded_rows.append(
            ScorecardRow(
                id=row["id"],
                label=row["label"],
                value=_decode_value(row["value"], field=f"rows[{index}].value"),
                scope=scope,
            )
        )
    tradeoff = None
    if "tradeoff" in encoded:
        item = _mapping(encoded["tradeoff"], field="tradeoff")
        _require_keys(
            item,
            field="tradeoff",
            keys={"option_a", "option_b", "competing_consequences"},
        )
        consequences = item["competing_consequences"]
        if not isinstance(consequences, list):
            _fail("tradeoff.competing_consequences must be a list")
        tradeoff = Tradeoff(
            option_a=item["option_a"],
            option_b=item["option_b"],
            competing_consequences=tuple(consequences),
        )
    _evidence_is_current(record, encoded["evidence"])
    return Scorecard.render(decoded_rows, rung=rung, tradeoff=tradeoff)


def read_persisted_scorecards(record: contracts.Assay) -> tuple[Scorecard, ...]:
    """Return the exact scorecards sealed into a reopened Compare record.

    This accepts no ordinary Assay that merely looks similar.  The extension is
    bound to the record's current subject, provenance, evidence locations, and
    canonical record digest before any row is reconstructed as a typed value.
    """
    if not isinstance(record, contracts.Assay):
        raise TypeError("read_persisted_scorecards requires a validated Assay")
    if record.assay_id != contracts.digest(record.to_dict()):
        _fail("record digest does not match its saved content")
    extension = (record.extensions or {}).get(SCORECARD_EXTENSION)
    if extension is None:
        _fail("record has no persisted compare scorecard")
    document = _mapping(extension, field="scorecard extension")
    required = {"version", "baseline", "candidate", "subject_ref", "provenance", "scorecards"}
    if set(document) != required:
        _fail("scorecard extension has unsupported or missing fields")
    if document["version"] != _FORMAT_VERSION:
        _fail("scorecard extension version is unsupported")
    if not all(
        isinstance(document[name], str) and document[name]
        for name in ("baseline", "candidate", "subject_ref")
    ):
        _fail("scorecard extension baseline, candidate, and subject_ref must be non-empty strings")
    if document["subject_ref"] != record.subject.version.digest:
        _fail("scorecard extension names a foreign subject")
    provenance = _mapping(document["provenance"], field="scorecard provenance")
    _require_keys(provenance, field="scorecard provenance", keys={"offline", "seed", "reproduce"})
    if provenance != {
        "offline": record.provenance.offline,
        "seed": record.provenance.seed,
        "reproduce": list(record.provenance.reproduce),
    }:
        _fail("scorecard provenance does not match its saved record")
    scorecards = document["scorecards"]
    if not isinstance(scorecards, list) or not scorecards:
        _fail("scorecard extension must carry one or more rung scorecards")
    decoded = tuple(_read_scorecard(item, record=record) for item in scorecards)
    rungs = [scorecard.rung for scorecard in decoded]
    if len(rungs) != len(set(rungs)):
        _fail("scorecard extension repeats a rung")
    return decoded
