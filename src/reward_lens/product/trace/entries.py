"""A `TraceRecord` as `contracts.Entry` objects: the record-level half of the wave-2 join.

Every entry carries `subject_ref`, the digest of the audited version, and `depends_on`, the digest
classes the reading rests on. P-HELDOUT-2 asserts that identity, and every wave-2 join is made on
it, so an entry that named the reader instead of the subject would make the join meaningless while
still validating.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from reward_lens import contracts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .record import TraceRecord

__all__ = ["DEPENDS_ON", "trace_entries"]

#: What a reconstruction of training pressure rests on, from the vocabulary the schema admits.
DEPENDS_ON: tuple[str, ...] = (
    "digest:training_semantics",
    "digest:samples",
    "digest:policy",
)

_VERSION = "1.0.0"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def trace_entries(record: "TraceRecord") -> tuple[contracts.Entry, ...]:
    """Verified trace measurements plus explicit holes for unavailable higher measurements."""
    provenance = contracts.EntryProvenance(
        started=_now(), duration_s=0.0, sandbox_tier="T0", offline=True
    )
    depends = list(record.depends_on or DEPENDS_ON)
    limitations = list(record.limitations)
    failed = tuple(one for one in record.integrity if not one.passed)
    unavailable = [name for name, series in record.series.items() if not series.present]
    supported = [name for name, series in record.series.items() if series.present]

    reconstruction = contracts.Entry(
        entry_id="trace.reconstruction",
        section="trace",
        kind="witness",
        measurand="what the optimiser saw, as four named series over the rows the record holds",
        method=contracts.Method(
            id="trace.reconstruction",
            version=_VERSION,
            params_digest=contracts.digest(
                {"reader": record.reader, "policy": record.policy, "series": list(record.series)}
            ),
            procedure=(
                "the artifact is read by the named reader, the native component scores are kept "
                "as separate named series, and the aggregates are emitted as distinct series "
                "with the weight vector beside each"
            ),
        ),
        scope="reconstructed_training_pressure",
        subject_ref=record.subject_ref,
        depends_on=depends,
        state="complete" if record.complete and not unavailable else "partial",
        provenance=provenance,
        limitations=limitations,
        result={
            "rows": len(record.rows),
            "series": list(record.series),
            "component_series": sorted(record.components),
            "reader": record.reader,
            "framework": record.framework,
            "framework_version": record.framework_version,
            "complete": record.complete,
            "supported_series": supported,
            "unavailable_series": unavailable,
            "component_provenance": {
                name: dict(series.provenance) for name, series in record.components.items()
            },
            "convenience_aggregates": {
                name: {
                    "components": list(series.components),
                    "weights": list(series.weights or ()),
                    "provenance": dict(series.provenance),
                }
                for name, series in record.convenience.items()
            },
        },
        witness=contracts.Witness(
            inputs={
                "reader": record.reader,
                "source": record.source,
                "policy": record.policy,
                "framework": record.framework,
                "framework_version": record.framework_version,
            },
            procedure=f"read {record.source} with the {record.reader} reader",
            observed=(
                f"{len(record.rows)} rows, supported series {', '.join(supported) or 'none'}, "
                f"unavailable series {', '.join(unavailable) or 'none'}, and "
                f"{len(record.components)} native component score series"
            ),
            path=record.source,
        ),
    )

    integrity = contracts.Entry(
        entry_id="trace.integrity",
        section="trace",
        kind="check",
        measurand="whether every named integrity rule of section 7.4 holds on these rows",
        method=contracts.Method(
            id="trace.integrity",
            version=_VERSION,
            params_digest=contracts.digest({"rules": [one.rule for one in record.integrity]}),
            procedure=(
                "each named rule is run over the rows; no rule reads a row's neighbours, so no "
                "group is ever reconstructed from adjacency"
            ),
        ),
        scope="reconstructed_training_pressure",
        subject_ref=record.subject_ref,
        depends_on=depends,
        state="complete",
        provenance=provenance,
        limitations=limitations,
        result={
            "rules": [one.rule for one in record.integrity],
            "failed": [one.rule for one in failed],
            "rows": len(record.rows),
        },
        check=contracts.Check(
            predicate="every named integrity rule passes on every row",
            passed=not failed,
            scope_tested=(
                f"{len(record.integrity)} rules over {len(record.rows)} rows"
                if record.integrity
                else "no rows were read"
            ),
        ),
    )

    access = contracts.Entry(
        entry_id="trace.access_level",
        section="trace",
        kind="witness",
        measurand="the access level this input reached, which decides what it can support",
        method=contracts.Method(
            id="trace.access_level",
            version=_VERSION,
            params_digest=contracts.digest({"access_level": record.access_level}),
            procedure=(
                "the access matrix the input declares is reduced to the strongest rung every "
                "component holds, and that rung is stated rather than assumed"
            ),
        ),
        scope="reconstructed_training_pressure",
        subject_ref=record.subject_ref,
        depends_on=depends,
        state="complete",
        provenance=provenance,
        limitations=limitations,
        result={
            "access_level": record.access_level,
            "complete": record.complete,
            "statement": record.access_statement,
            "measurement_boundary": "observed_loss_level",
            "gradient_observed": False,
            "optimizer_transform_observed": False,
            "parameter_update_observed": False,
        },
        witness=contracts.Witness(
            inputs={"access_level": record.access_level, "complete": record.complete},
            procedure="reduce the declared access matrix to the strongest common rung",
            observed=(
                f"{record.access_statement} This record reaches the observed loss-level "
                "boundary only; it does not observe gradients, optimizer transforms, or "
                "parameter updates."
            ),
        ),
    )

    absences = tuple(
        contracts.absence(
            section="trace",
            entry_id=_absence_id(name),
            measurand=f"the {name} measurement for this trace",
            missing_access=record.series[name].absent_reason
            or f"the source does not establish {name}",
            remedy="record the missing measurement at the trainer before the run closes",
            affected_claims=[name],
            subject_ref=record.subject_ref,
            provenance=provenance,
            depends_on=depends,
            state="COULD_NOT_CHECK",
            scope="reconstructed_training_pressure",
            limitations=limitations,
        )
        for name in unavailable
    )
    rows = tuple(
        _row_evidence(record, row, provenance, depends, limitations) for row in record.rows
    )
    raw_reward = tuple(_raw_reward_finding(record, provenance, depends, limitations))
    return (reconstruction, integrity, access, *rows, *raw_reward, *absences)


def _row_evidence(
    record: "TraceRecord",
    row,
    provenance: contracts.EntryProvenance,
    depends: list[str],
    limitations: list[str],
) -> contracts.Entry:
    """One joinable row, or an explicit absence when recorded identity is incomplete."""
    entry_id = f"trace.row.{row.index}"
    missing = []
    if row.task_id is None:
        missing.append("task identity")
    if row.response_digest is None or row.response_codec is None:
        missing.append("recorded response identity")
    if row.grader_source is None:
        missing.append("recorded grader source")
    if not record.run_identity:
        missing.append("recorded run identity")
    if missing:
        return contracts.absence(
            section="trace",
            entry_id=entry_id,
            measurand="the canonical task/response/source binding for one trace row",
            missing_access=", ".join(missing),
            remedy="record the missing task, response, and grader-source identity before the run closes",
            affected_claims=["trace_row_join"],
            subject_ref=record.subject_ref,
            provenance=provenance,
            depends_on=depends,
            state="COULD_NOT_CHECK",
            scope="reconstructed_training_pressure",
            limitations=limitations,
        )
    result = {
        "row_id": row.row_id,
        "run_identity": record.run_identity,
        "task_id": row.task_id,
        "response_digest": row.response_digest,
        "response_codec": row.response_codec,
        "response_provenance": (
            None if row.response_provenance is None else dict(row.response_provenance)
        ),
        "grader_source": dict(row.grader_source),
        "components": dict(row.components),
        "reward_recorded": row.reward_recorded,
        "reward_applied": row.reward_applied,
        "advantage": row.advantage,
        "applied_weight": row.applied_weight,
        "applied_weight_source": row.applied_weight_source,
        "selection": row.selection,
    }
    return contracts.Entry(
        entry_id=entry_id,
        section="trace",
        kind="witness",
        measurand="one recorded task/response and grader-source binding with its trace quantities",
        method=contracts.Method(
            id="trace.row_evidence",
            version=_VERSION,
            params_digest=contracts.digest({"codec": row.response_codec, "reader": record.reader}),
            procedure="preserve recorded task, response, source, score, advantage, and loss fields without substitution",
        ),
        scope="reconstructed_training_pressure",
        subject_ref=record.subject_ref,
        depends_on=depends,
        state="complete",
        provenance=provenance,
        limitations=limitations,
        result=result,
        witness=contracts.Witness(
            inputs={
                "task_id": row.task_id,
                "response_digest": row.response_digest,
                "run_identity": record.run_identity,
            },
            procedure="read canonical row identity from the recorded run",
            observed=f"row {row.row_id}",
        ),
    )


def _raw_reward_finding(
    record: "TraceRecord",
    provenance: contracts.EntryProvenance,
    depends: list[str],
    limitations: list[str],
) -> tuple[contracts.Entry, ...]:
    series = record.convenience.get("recorded_raw_reward")
    if series is None:
        return ()
    return (
        contracts.Entry(
            entry_id="trace.raw_reward_convenience",
            section="trace",
            kind="witness",
            measurand="a raw imported reward column retained only as a convenience aggregate",
            method=contracts.Method(
                id="trace.raw_reward_convenience",
                version=_VERSION,
                params_digest=contracts.digest(dict(series.provenance)),
                procedure="retain the source column outside canonical optimisation series",
            ),
            scope="reconstructed_training_pressure",
            subject_ref=record.subject_ref,
            depends_on=depends,
            state="complete",
            provenance=provenance,
            limitations=limitations,
            result={
                "finding": "raw reward is a convenience aggregate, not an optimised composite",
                "provenance": dict(series.provenance),
            },
            witness=contracts.Witness(
                inputs={"series": series.name},
                procedure="preserve raw input column as convenience only",
                observed="not used as reward_applied",
            ),
        ),
    )


def _absence_id(series: str) -> str:
    """Stable public IDs for the measurements trace knows how to leave incomplete."""
    names = {
        "reward_recorded": "trace.displayed_aggregate",
        "reward_applied": "trace.optimized_composite",
        "advantage": "trace.advantage",
        "selection": "trace.selection",
    }
    return names[series]
