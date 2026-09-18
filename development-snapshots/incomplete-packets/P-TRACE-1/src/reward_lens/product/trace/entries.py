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
    """Three entries: what was reconstructed, what the checks found, and what access was reached."""
    provenance = contracts.EntryProvenance(
        started=_now(), duration_s=0.0, sandbox_tier="T0", offline=True
    )
    depends = list(record.depends_on or DEPENDS_ON)
    limitations = list(record.limitations)
    named = ", ".join(record.series)
    failed = tuple(one for one in record.integrity if not one.passed)

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
        state="complete" if record.complete else "partial",
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
                f"{len(record.rows)} rows, series {named}, "
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
        },
        witness=contracts.Witness(
            inputs={"access_level": record.access_level, "complete": record.complete},
            procedure="reduce the declared access matrix to the strongest common rung",
            observed=record.access_statement,
        ),
    )

    return (reconstruction, integrity, access)
