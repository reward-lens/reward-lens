"""The six fixed comparison rows, including the measured capability floor.

The scorecard is deliberately a small value layer.  Rung implementations compute
their measurements; this module preserves their order, uncertainty, scope, and
the declared held-out capability check that policy consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Iterable

from reward_lens import contracts
from reward_lens.contracts import ProjectConfig, Scope
from reward_lens.outcome import Outcome, OutcomeState, QualifiedProtocol
from reward_lens.partitions import Partition, PartitionKind

__all__ = [
    "ROWS",
    "Absence",
    "CapabilityFloor",
    "Estimate",
    "Scorecard",
    "ScorecardRow",
    "Tradeoff",
    "capability_floor_assay_entry",
]


ROWS: tuple[str, ...] = (
    "intended_task_success",
    "legitimate_solutions_kept",
    "solution_diversity",
    "exploit_pressure",
    "signal",
    "cost",
)

_SCALES = frozenset(
    {
        "evaluator_comparison",
        "selection_stress",
        "reconstructed_training_pressure",
        "controlled_update_response",
        "short_fork_behavior",
        "held_out_transfer",
    }
)
_QUANTUM = Decimal("0.000001")
_CAPABILITY_FLOOR_ENTRY_ID = "soundness.capability_floor"
_CAPABILITY_FLOOR_METHOD_VERSION = "1.0.0"
_CAPABILITY_FLOOR_CONFIDENCE = Decimal("0.950")
_CAPABILITY_FLOOR_SIDEDNESS = "two-sided"


@dataclass(frozen=True)
class Estimate:
    """A measurement with its interval on the same row that reports it."""

    value: Decimal
    interval: tuple[Decimal, Decimal] | None
    n: int
    n_eff: Decimal | None = None
    unit: str = "proportion"
    samples: int | None = None

    def __post_init__(self) -> None:
        if type(self.value) is not Decimal:
            raise TypeError("estimate value must be a Decimal")
        if self.interval is None:
            raise ValueError("every measured scorecard value carries an interval on its row")
        if len(self.interval) != 2 or any(type(bound) is not Decimal for bound in self.interval):
            raise TypeError("estimate interval must contain exactly two Decimal endpoints")
        low, high = self.interval
        if low > high or not low <= self.value <= high:
            raise ValueError("estimate value must lie inside its interval")
        if type(self.n) is not int or self.n <= 0:
            raise TypeError("estimate n must be a positive integer")
        if self.n_eff is not None and (type(self.n_eff) is not Decimal or self.n_eff <= 0):
            raise TypeError("estimate n_eff must be a positive Decimal when stated")
        if not isinstance(self.unit, str) or not self.unit:
            raise TypeError("estimate unit must be a non-empty string")
        if self.samples is not None and (type(self.samples) is not int or self.samples < 0):
            raise TypeError("estimate samples must be a non-negative integer when stated")


@dataclass(frozen=True)
class Absence:
    """A named unavailable measurement, with its remedy instead of a fabricated value."""

    reason: str
    remedy: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise TypeError("absence reason must be a non-empty string")
        if not isinstance(self.remedy, str) or not self.remedy:
            raise TypeError("absence remedy must be a non-empty string")


@dataclass(frozen=True)
class CapabilityFloor(Estimate):
    """The legitimate-solution row with its declared held-out set and threshold."""

    partition_id: str = ""
    estimand: str = ""
    floor: Decimal = Decimal("0")
    measured: Estimate | None = None
    held: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.partition_id, str) or not self.partition_id:
            raise TypeError("capability floor partition_id must be a non-empty string")
        if not isinstance(self.estimand, str) or not self.estimand:
            raise TypeError("capability floor estimand must be a non-empty string")
        if type(self.floor) is not Decimal:
            raise TypeError("capability floor must be a Decimal")
        if type(self.measured) is not Estimate or isinstance(self.measured, CapabilityFloor):
            raise TypeError("capability floor measured must be an Estimate")
        if (
            self.value != self.measured.value
            or self.interval != self.measured.interval
            or self.n != self.measured.n
            or self.n_eff != self.measured.n_eff
        ):
            raise ValueError("capability floor must preserve its measured estimate exactly")
        if type(self.held) is not bool or self.held is not (self.measured.value >= self.floor):
            raise ValueError("capability floor held must be computed from measured value and floor")

    @classmethod
    def from_outcomes(
        cls,
        *,
        project: ProjectConfig,
        project_root: Path | str,
        partition: Partition | None,
        outcomes: Iterable[Outcome],
        floor: Decimal,
    ) -> CapabilityFloor | Absence:
        """Measure the declared held-out partition from real item-level outcomes.

        The capability declaration is intentionally distinct from `OutcomeSpec.path`, which names
        the protected outcome check. A caller supplies the project root and a real
        `HELD_OUT_TASKS` partition. This function resolves only the explicit declaration and
        never selects files or substitutes the protected suite.
        """
        if type(project) is not ProjectConfig:
            raise TypeError("project must be the validated ProjectConfig from rewardlens.yaml")
        if type(floor) is not Decimal:
            raise TypeError("capability floor must be a Decimal")
        outcome = project.outcome
        declaration = None if outcome is None else outcome.held_out_capability
        if declaration is None or partition is None:
            return Absence(
                "not measured: no held-out capability set declared",
                "declare the held-out capability partition under outcome in rewardlens.yaml",
            )
        if (
            not isinstance(partition, Partition)
            or partition.kind is not PartitionKind.HELD_OUT_TASKS
        ):
            raise TypeError("capability floor requires a HELD_OUT_TASKS Partition")
        if not isinstance(project_root, (Path, str)):
            raise TypeError("project_root must be the project directory containing rewardlens.yaml")
        declared_root = (Path(project_root) / declaration.path).resolve()
        if partition.id != declaration.partition_id:
            return Absence(
                "not measured: declared held-out capability id does not identify this partition",
                "declare the supplied HELD_OUT_TASKS partition id under outcome.held_out_capability",
            )
        if partition.root.resolve() != declared_root:
            return Absence(
                "not measured: declared held-out capability path does not resolve to supplied partition",
                "declare the supplied HELD_OUT_TASKS partition path under outcome.held_out_capability",
            )
        current_partition = Partition.from_dir(
            declaration.partition_id,
            PartitionKind.HELD_OUT_TASKS,
            declared_root,
            log_path=partition.access_log.path,
            public_root=partition.public_root,
        )
        manifest = dict(current_partition.manifest())
        if not manifest:
            return Absence(
                "not measured: declared held-out capability set is empty",
                "declare active held-out capability items before measuring the capability floor",
            )
        rows = tuple(outcomes)
        if any(type(row) is not Outcome for row in rows):
            raise TypeError("capability floor outcomes must be item-level Outcome values")
        item_ids = tuple(row.item_id for row in rows)
        if len(item_ids) != len(set(item_ids)):
            return Absence(
                "not measured: held-out outcome coverage contains duplicate item ids",
                "evaluate each active held-out item exactly once",
            )
        item_set = set(item_ids)
        active_set = set(manifest)
        if item_set - active_set:
            return Absence(
                "not measured: held-out outcome coverage contains foreign item ids",
                "evaluate only the active items in the declared held-out capability partition",
            )
        if active_set - item_set:
            return Absence(
                "not measured: held-out outcome coverage is incomplete",
                "evaluate every active item in the declared held-out capability partition",
            )
        protocol_digests = {row.protocol_digest for row in rows}
        scopes = {row.scope for row in rows}
        if len(protocol_digests) != 1 or None in protocol_digests:
            return Absence(
                "not measured: held-out outcomes do not share one qualified protocol",
                "evaluate every declared held-out item under one qualified outcome protocol",
            )
        if len(scopes) != 1:
            return Absence(
                "not measured: held-out outcomes do not share one scope",
                "evaluate every declared held-out item in one declared comparison scope",
            )
        if any(
            value is None
            for row in rows
            for value in (
                row.evaluated_on_partition_id,
                row.evaluated_on_partition_digest,
                row.evaluated_item_digest,
            )
        ):
            return Absence(
                "not measured: held-out outcome lacks an evaluated-on partition witness",
                "record an evaluated-on partition and item witness for every held-out outcome",
            )
        if {row.evaluated_on_partition_id for row in rows} != {current_partition.id}:
            return Absence(
                "not measured: evaluated-on partition id does not match declared target",
                "evaluate every held-out outcome on the declared capability partition",
            )
        if {row.evaluated_on_partition_digest for row in rows} != {current_partition.digest}:
            return Absence(
                "not measured: evaluated-on partition digest does not match current declared target",
                "rerun the outcome protocol against the current declared held-out capability files",
            )
        if any(row.evaluated_item_digest != manifest[row.item_id] for row in rows):
            return Absence(
                "not measured: evaluated-on item digest does not match current active membership",
                "rerun the outcome protocol against every current active held-out item",
            )
        resolved = tuple(row for row in rows if row.state in (OutcomeState.PASS, OutcomeState.FAIL))
        if not resolved:
            return Absence(
                "not measured: no resolved held-out outcomes",
                "run the independent outcome protocol on the declared held-out capability set",
            )
        passed = sum(row.state is OutcomeState.PASS for row in resolved)
        measured = Estimate(
            value=(Decimal(passed) / Decimal(len(resolved))).quantize(_QUANTUM),
            interval=_wilson_interval(passed, len(resolved)),
            n=len(resolved),
        )
        return cls(
            value=measured.value,
            interval=measured.interval,
            n=measured.n,
            n_eff=measured.n_eff,
            unit=measured.unit,
            samples=measured.samples,
            partition_id=partition.id,
            estimand="held-out legitimate-solution success",
            floor=floor,
            measured=measured,
            held=measured.value >= floor,
        )


def _canonical_capability_floor_entry(entry: contracts.Entry) -> contracts.Entry:
    """Reopen the one producer result through canonical contract bytes.

    Compare/Audit own the later act of placing this entry in an Assay and persisting it.  This
    boundary makes the producer's hand-off a normal frozen contract object, rather than a local
    scorecard value that policy would have to interpret specially.
    """
    if type(entry) is not contracts.Entry:
        raise TypeError("capability-floor producer must emit a contracts.Entry")
    if entry.entry_id != _CAPABILITY_FLOOR_ENTRY_ID or entry.section != "soundness":
        raise ValueError("capability-floor producer emitted the wrong canonical entry identity")
    return contracts.Entry.model_validate_json(contracts.canonical_bytes(entry))


def _capability_floor_absence_entry(
    absence: Absence,
    *,
    subject_ref: str,
    provenance: contracts.EntryProvenance,
    scope: Scope,
) -> contracts.Entry:
    """Represent an unavailable floor as a contract absence, never an unheld measurement."""
    return _canonical_capability_floor_entry(
        contracts.absence(
            "soundness",
            _CAPABILITY_FLOOR_ENTRY_ID,
            "whether the declared held-out legitimate-solution success floor is met",
            absence.reason,
            absence.remedy,
            ("held-out capability floor",),
            subject_ref=subject_ref,
            provenance=provenance,
            scope=scope,
            limitations=(
                "an absence means no capability-floor decision is supported; it is not a failed floor",
            ),
        )
    )


def capability_floor_assay_entry(
    *,
    project: ProjectConfig,
    project_root: Path | str,
    partition: Partition | None,
    outcomes: Iterable[Outcome],
    protocol: QualifiedProtocol | None,
    floor: Decimal,
    subject_ref: str,
    provenance: contracts.EntryProvenance,
    scope: Scope,
) -> contracts.Entry:
    """Build the canonical policy-consumable capability-floor entry from qualified target rows.

    The target partition is the explicitly declared held-out capability partition.  The protocol
    is the separately qualified evaluator-control witness.  A favorable entry therefore binds
    both identities, their digests, every active target item, the actual provenance, and the
    subject it evaluated.  Unknown, incomplete, stale, or unqualified target evidence is emitted
    as an honest absence, leaving policy unresolved rather than fabricating an unheld result.
    """
    if type(provenance) is not contracts.EntryProvenance:
        raise TypeError("capability-floor provenance must be the actual EntryProvenance")
    if type(subject_ref) is not str:
        raise TypeError("capability-floor subject_ref must be the evaluated subject digest")
    rows = tuple(outcomes)
    reading = CapabilityFloor.from_outcomes(
        project=project,
        project_root=project_root,
        partition=partition,
        outcomes=rows,
        floor=floor,
    )
    if isinstance(reading, Absence):
        return _capability_floor_absence_entry(
            reading, subject_ref=subject_ref, provenance=provenance, scope=scope
        )

    if type(protocol) is not QualifiedProtocol:
        return _capability_floor_absence_entry(
            Absence(
                "not measured: held-out outcomes lack the supplied qualified protocol",
                "supply the QualifiedProtocol that resolved the declared held-out outcomes",
            ),
            subject_ref=subject_ref,
            provenance=provenance,
            scope=scope,
        )
    if {row.protocol_digest for row in rows} != {protocol.protocol_digest}:
        return _capability_floor_absence_entry(
            Absence(
                "not measured: held-out outcomes do not match the supplied qualified protocol",
                "supply the exact QualifiedProtocol that resolved every declared held-out item",
            ),
            subject_ref=subject_ref,
            provenance=provenance,
            scope=scope,
        )
    if {row.scope for row in rows} != {scope} or protocol.scope != scope:
        return _capability_floor_absence_entry(
            Absence(
                "not measured: held-out outcome scope does not match the supplied qualified protocol",
                "preserve one declared scope from outcome qualification through target evaluation",
            ),
            subject_ref=subject_ref,
            provenance=provenance,
            scope=scope,
        )
    if any(row.provenance != provenance for row in rows) or protocol.provenance != provenance:
        return _capability_floor_absence_entry(
            Absence(
                "not measured: held-out outcome provenance does not match the supplied evaluation attempt",
                "use the actual QualifiedProtocol provenance and target outcomes from one evaluation attempt",
            ),
            subject_ref=subject_ref,
            provenance=provenance,
            scope=scope,
        )

    target_digest = rows[0].evaluated_on_partition_digest
    assert target_digest is not None  # guaranteed by CapabilityFloor.from_outcomes above
    if (
        protocol.qualified_on_partition_id == reading.partition_id
        or protocol.qualified_on_partition_digest == target_digest
    ):
        return _capability_floor_absence_entry(
            Absence(
                "not measured: evaluator qualification controls are not distinct from the held-out target",
                "qualify the evaluator on a distinct protected control partition before target evaluation",
            ),
            subject_ref=subject_ref,
            provenance=provenance,
            scope=scope,
        )

    target_items = {
        row.item_id: row.evaluated_item_digest for row in sorted(rows, key=lambda row: row.item_id)
    }
    assert all(digest is not None for digest in target_items.values())
    passed = sum(row.state is OutcomeState.PASS for row in rows)
    failed = sum(row.state is OutcomeState.FAIL for row in rows)
    unresolved = sum(row.state is OutcomeState.UNQUALIFIED for row in rows)
    assert reading.interval is not None
    uncertainty = contracts.Uncertainty(
        method="wilson",
        level=float(_CAPABILITY_FLOOR_CONFIDENCE),
        interval=[float(reading.interval[0]), float(reading.interval[1])],
        clusters=reading.n,
        cluster_unit="declared held-out capability item",
    )
    result: dict[str, object] = {
        "type": "held_out_capability_floor/v1",
        "target": {
            "partition_id": reading.partition_id,
            "partition_digest": target_digest,
            "active_item_digests": target_items,
        },
        "qualification": {
            "protocol_digest": protocol.protocol_digest,
            "control_partition_id": protocol.qualified_on_partition_id,
            "control_partition_digest": protocol.qualified_on_partition_digest,
            "control_distinct_from_target": True,
        },
        "estimate": {
            "value": str(reading.value),
            "interval": [str(reading.interval[0]), str(reading.interval[1])],
            "n": reading.n,
            "pass_count": passed,
            "fail_count": failed,
            "unqualified_count": unresolved,
            "confidence": str(_CAPABILITY_FLOOR_CONFIDENCE),
            "sidedness": _CAPABILITY_FLOOR_SIDEDNESS,
        },
        "threshold": {"operator": ">=", "floor": str(reading.floor)},
        "held": reading.held,
    }
    method = contracts.Method(
        id=_CAPABILITY_FLOOR_ENTRY_ID,
        version=_CAPABILITY_FLOOR_METHOD_VERSION,
        params_digest=contracts.digest(
            {
                "target": result["target"],
                "qualification": result["qualification"],
                "threshold": result["threshold"],
                "confidence": result["estimate"]["confidence"],  # type: ignore[index]
                "sidedness": _CAPABILITY_FLOOR_SIDEDNESS,
            }
        ),
        procedure=(
            "reload the declared held-out capability partition, require complete immutable item "
            "witnesses resolved under one separately qualified evaluator protocol, compute the "
            "Wilson interval over resolved target items, then compare the observed success share "
            "with the declared floor"
        ),
    )
    entry = contracts.estimate(
        "soundness",
        _CAPABILITY_FLOOR_ENTRY_ID,
        "whether the declared held-out legitimate-solution success floor is met",
        float(reading.value),
        "fraction",
        reading.n,
        "declared held-out capability item",
        uncertainty,
        subject_ref=subject_ref,
        provenance=provenance,
        method=method,
        result=result,
        scope=scope,
        denominator="resolved declared held-out capability items",
        exclusions=(
            "unqualified target outcomes are excluded from the resolved denominator but retained in coverage",
        ),
        limitations=(
            "the floor is only a target-specific capability check and does not establish a learning or selection decision",
        ),
    ).model_copy(
        update={
            "check": contracts.Check(
                predicate=f"held-out capability success >= {reading.floor}",
                passed=reading.held,
                scope_tested=(
                    f"declared held-out target {reading.partition_id} under qualified control "
                    f"protocol {protocol.protocol_digest}"
                ),
            )
        }
    )
    return _canonical_capability_floor_entry(entry)


@dataclass(frozen=True)
class ScorecardRow:
    """One fixed scorecard row; no rung contributes an extra column."""

    id: str
    label: str
    value: Estimate | Absence
    scope: Scope

    def __post_init__(self) -> None:
        if self.id not in ROWS:
            raise ValueError(f"{self.id!r} is not a scorecard row")
        if not isinstance(self.label, str) or not self.label:
            raise TypeError("scorecard row label must be a non-empty string")
        if not isinstance(self.value, (Estimate, Absence)):
            raise TypeError("scorecard row value must be an Estimate or Absence")
        if self.scope not in _SCALES:
            raise ValueError("scorecard row scope must be one of the six rung scopes")
        if self.id == "legitimate_solutions_kept" and not isinstance(
            self.value, (CapabilityFloor, Absence)
        ):
            raise TypeError("legitimate_solutions_kept must carry CapabilityFloor or Absence")
        if self.id == "signal" and isinstance(self.value, Estimate) and self.value.n_eff is None:
            raise ValueError(
                "signal carries its effective sample size beside the contrast fraction"
            )
        if self.id == "cost" and isinstance(self.value, Estimate):
            if self.value.samples is None or self.value.unit != "USD":
                raise ValueError("cost carries money in USD and its sample count")


@dataclass(frozen=True)
class Tradeoff:
    """Two options and their competing consequences, never a scalar score."""

    option_a: str
    option_b: str
    competing_consequences: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(option, str) and option for option in (self.option_a, self.option_b)):
            raise TypeError("tradeoff options must be non-empty strings")
        if len(self.competing_consequences) < 2 or any(
            not isinstance(item, str) or not item for item in self.competing_consequences
        ):
            raise ValueError("tradeoff names at least two non-empty competing consequences")


@dataclass(frozen=True)
class Scorecard:
    """A comparison result at one rung, retaining a clean result or an explicit tradeoff."""

    rows: tuple[ScorecardRow, ...]
    rung: object
    tradeoff: Tradeoff | None = None

    @classmethod
    def render(
        cls, rows: Iterable[ScorecardRow], *, rung: object, tradeoff: Tradeoff | None = None
    ) -> Scorecard:
        rendered = tuple(rows)
        if tuple(row.id for row in rendered) != ROWS:
            raise ValueError(
                "scorecard rows must be exactly the fixed ROWS in their declared order"
            )
        if len({row.scope for row in rendered}) != 1:
            raise ValueError("all scorecard rows must carry the same comparison scope")
        if not isinstance(rendered[1].value, (CapabilityFloor, Absence)):
            raise TypeError("the second row must carry the capability floor measurement or absence")
        if tradeoff is not None and not isinstance(tradeoff, Tradeoff):
            raise TypeError("tradeoff must be a Tradeoff when stated")
        return cls(rows=rendered, rung=rung, tradeoff=tradeoff)

    @property
    def capability_floor(self) -> CapabilityFloor | Absence:
        """The floor policy consumes, directly from the second fixed row."""
        value = self.rows[1].value
        assert isinstance(value, (CapabilityFloor, Absence))
        return value

    @property
    def is_result(self) -> bool:
        """A clean comparison is still a result; only competing consequences form a tradeoff."""
        return self.tradeoff is None


def _wilson_interval(passed: int, total: int) -> tuple[Decimal, Decimal]:
    """Two-sided 95% Wilson interval for a resolved held-out capability rate."""
    z = Decimal("1.959964")
    p = Decimal(passed) / Decimal(total)
    denominator = Decimal(1) + z * z / Decimal(total)
    centre = (p + z * z / (Decimal(2) * Decimal(total))) / denominator
    half = (
        z
        * (
            p * (Decimal(1) - p) / Decimal(total) + z * z / (Decimal(4) * Decimal(total) ** 2)
        ).sqrt()
        / denominator
    )
    return (
        max(Decimal(0), centre - half).quantize(_QUANTUM, rounding=ROUND_HALF_UP),
        min(Decimal(1), centre + half).quantize(_QUANTUM, rounding=ROUND_HALF_UP),
    )
