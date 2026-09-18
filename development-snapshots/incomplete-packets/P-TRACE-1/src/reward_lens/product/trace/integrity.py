"""The integrity checks of section 7.4, each one a named rule that fails on its own fixture.

Seven rules, and the seventh is the one that matters most. `trace.group_not_declared` is the
adjacency rule stated so that it can fail: a row that carries a group-relative advantage and no
declared group is a violation. The alternative, which every importer is tempted into, is to chunk
consecutive rows into groups of `num_generations` and call the result a grouping. That produces
advantages that replay perfectly against a grouping nobody recorded, which is worse than refusing.
So nothing here reads a row's neighbours, and `trace.incomplete_group` has nothing at all to say
about rows whose group is undeclared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from .errors import RecordIncomplete, RecordIntegrity
from .record import SERIES_NAMES, Series, TraceRow

__all__ = [
    "RULES",
    "CheckInput",
    "IntegrityRule",
    "RuleResult",
    "Violation",
    "check",
    "enforce",
    "require_series",
    "rows_digest",
]


@dataclass(frozen=True)
class Violation:
    rule: str
    row: str | None
    message: str


@dataclass(frozen=True)
class RuleResult:
    rule: str
    passed: bool
    violations: tuple[Violation, ...] = ()


@dataclass(frozen=True)
class CheckInput:
    """Everything the rules are allowed to look at, and nothing else."""

    rows: tuple[TraceRow, ...]
    series: Mapping[str, Series] = field(default_factory=dict)
    components: Mapping[str, Series] = field(default_factory=dict)
    declared_rows: int | None = None
    declared_digest: str | None = None
    declared_group_size: int | None = None
    group_relative_advantage: bool = True
    source: str = ""


@dataclass(frozen=True)
class IntegrityRule:
    id: str
    name: str
    description: str
    check: Callable[[CheckInput], tuple[Violation, ...]]


def rows_digest(rows: tuple[TraceRow, ...]) -> str:
    """The digest a checksum claim is compared against, over the fields a row is identified by."""
    from reward_lens.contracts import digest

    return digest(
        {
            "rows": [
                {
                    "index": row.index,
                    "step": row.step,
                    "group_id": row.group_id,
                    "row_id": row.row_id,
                }
                for row in rows
            ]
        }
    )


def _duplicate_row_id(data: CheckInput) -> tuple[Violation, ...]:
    seen: set[str] = set()
    found: list[Violation] = []
    for row in data.rows:
        if row.row_id in seen:
            found.append(
                Violation(
                    "trace.duplicate_row_id",
                    row.row_id,
                    f"row {row.row_id!r} appears more than once; a row identity is unique or it "
                    f"is not an identity",
                )
            )
        seen.add(row.row_id)
    return tuple(found)


def _incomplete_group(data: CheckInput) -> tuple[Violation, ...]:
    if data.declared_group_size is None:
        return ()
    sizes: dict[str, list[TraceRow]] = {}
    for row in data.rows:
        # A row whose group was never declared is not evidence about any group's size. Counting it
        # would be inferring the group, which is the one thing this file refuses to do.
        if row.group_id is None:
            continue
        sizes.setdefault(row.group_id, []).append(row)
    found: list[Violation] = []
    for group, members in sizes.items():
        if len(members) != data.declared_group_size:
            found.append(
                Violation(
                    "trace.incomplete_group",
                    members[0].row_id,
                    f"group {group!r} holds {len(members)} rows where the record declares "
                    f"{data.declared_group_size}",
                )
            )
    return tuple(found)


def _out_of_order_steps(data: CheckInput) -> tuple[Violation, ...]:
    found: list[Violation] = []
    previous: int | None = None
    for row in data.rows:
        if previous is not None and row.step < previous:
            found.append(
                Violation(
                    "trace.out_of_order_steps",
                    row.row_id,
                    f"row {row.row_id!r} is at step {row.step} after step {previous}",
                )
            )
        previous = row.step
    return tuple(found)


def _length_mismatch(kind: str, rule: str) -> Callable[[CheckInput], tuple[Violation, ...]]:
    def rule_body(data: CheckInput) -> tuple[Violation, ...]:
        mapping: Mapping[str, Series] = getattr(data, kind)
        found: list[Violation] = []
        for name, series in mapping.items():
            if not series.present:
                continue
            if len(series.values) != len(data.rows):
                boundary = min(len(series.values), max(len(data.rows) - 1, 0))
                row = data.rows[boundary].row_id if data.rows else None
                found.append(
                    Violation(
                        rule,
                        row,
                        f"series {name!r} carries {len(series.values)} values for "
                        f"{len(data.rows)} rows; the first row it cannot cover is {row!r}",
                    )
                )
        return tuple(found)

    return rule_body


def _checksum_drift(data: CheckInput) -> tuple[Violation, ...]:
    found: list[Violation] = []
    if data.declared_rows is not None and data.declared_rows != len(data.rows):
        found.append(
            Violation(
                "trace.checksum_drift",
                data.rows[0].row_id if data.rows else None,
                f"the record declares {data.declared_rows} rows and holds {len(data.rows)}",
            )
        )
    if data.declared_digest is not None:
        actual = rows_digest(data.rows)
        if actual != data.declared_digest:
            found.append(
                Violation(
                    "trace.checksum_drift",
                    data.rows[0].row_id if data.rows else None,
                    f"the record declares {data.declared_digest} and the rows hash to {actual}",
                )
            )
    return tuple(found)


def _group_not_declared(data: CheckInput) -> tuple[Violation, ...]:
    if not data.group_relative_advantage:
        return ()
    found: list[Violation] = []
    for row in data.rows:
        if row.group_id is None and row.advantage is not None:
            found.append(
                Violation(
                    "trace.group_not_declared",
                    row.row_id,
                    f"row {row.row_id!r} carries a group-relative advantage and no declared "
                    f"group; a group is never reconstructed from row adjacency",
                )
            )
    return tuple(found)


RULES: tuple[IntegrityRule, ...] = (
    IntegrityRule(
        "trace.duplicate_row_id",
        "duplicate row identity",
        "two rows share an identity, so nothing can be joined onto either of them",
        _duplicate_row_id,
    ),
    IntegrityRule(
        "trace.incomplete_group",
        "incomplete group",
        "a declared group holds fewer or more rows than the record says it should",
        _incomplete_group,
    ),
    IntegrityRule(
        "trace.out_of_order_steps",
        "out-of-order steps",
        "a row is at an earlier step than the row before it",
        _out_of_order_steps,
    ),
    IntegrityRule(
        "trace.series_length_mismatch",
        "series length mismatch",
        "a named series carries a different number of values than the record has rows",
        _length_mismatch("series", "trace.series_length_mismatch"),
    ),
    IntegrityRule(
        "trace.component_length_mismatch",
        "component length mismatch",
        "a native component score carries a different number of values than there are rows",
        _length_mismatch("components", "trace.component_length_mismatch"),
    ),
    IntegrityRule(
        "trace.checksum_drift",
        "checksum drift",
        "the rows do not hash to what the record declares, or there are not as many as declared",
        _checksum_drift,
    ),
    IntegrityRule(
        "trace.group_not_declared",
        "group not declared",
        "a row carries a group-relative advantage and no declared group, and adjacency is not a "
        "grouping",
        _group_not_declared,
    ),
)


def check(data: CheckInput) -> tuple[RuleResult, ...]:
    """Every rule against one input, in rule order. Never raises."""
    results: list[RuleResult] = []
    for rule in RULES:
        violations = rule.check(data)
        results.append(RuleResult(rule.id, not violations, violations))
    return tuple(results)


def enforce(data: CheckInput, *, policy: str = "strict") -> tuple[RuleResult, ...]:
    """Run the rules and, under `strict`, refuse on the first violation, naming the row."""
    results = check(data)
    if policy == "permissive":
        return results
    for result in results:
        if result.passed:
            continue
        first = result.violations[0]
        rule = next(one for one in RULES if one.id == result.rule)
        raise RecordIntegrity(
            message=f"{rule.name}: {first.message}",
            remediation=(
                "record the missing structure at the source, or read this artifact with "
                'policy="permissive" to carry the violation in the record instead of refusing'
            ),
            context={
                "rule": result.rule,
                "row": first.row,
                "source": data.source,
                "violations": len(result.violations),
            },
        )
    return results


def require_series(series: Mapping[str, Series], *, source: str) -> None:
    """Refuse a record that cannot name all four series, naming the ones it cannot."""
    missing = [name for name in SERIES_NAMES if name not in series]
    if not missing:
        return
    raise RecordIncomplete(
        message=(
            f"{source} cannot supply the series {', '.join(missing)}: a trace record names all "
            f"four of {', '.join(SERIES_NAMES)} or it names which it could not build"
        ),
        remediation=(
            "record the missing series at the source, or declare it absent with the reason, "
            "which is what a reader does when the artifact genuinely does not hold it"
        ),
        context={"series": missing, "source": source},
    )
