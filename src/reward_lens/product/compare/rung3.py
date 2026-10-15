"""Rung 3: replay recorded trainer advantages under two explicit reward versions.

This module reconstructs the score-to-advantage pressure a closed run record
describes.  It does not observe a loss, gradient, optimizer transform, or update.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from reward_lens import contracts
from reward_lens.contracts import UsageError
from reward_lens.product.trace.record import TraceRecord
from reward_lens.record.schema import EstimatorSpec

from .registry import CompareInputs, CompareResult, Rung, register
from .scorecard import ROWS, Absence, Estimate, Scorecard, ScorecardRow

_SCOPE = "reconstructed_training_pressure"
_QUANTUM = Decimal("0.000001")
_REPLAY_TOLERANCE = Decimal("0.000000000001")
R3_ENTRY_ID = "reward_statistics.advantage_replay"
_CANDIDATE_ENTRY_PREFIX = R3_ENTRY_ID + ".candidate."


@dataclass(frozen=True)
class ReplayVersion:
    """A reward version specified over components actually recorded in one run.

    The explicit full weight vector is intentional.  A missing weight could mean
    zero, one, or a component that was absent from the old reward; none of those
    is recoverable from a version label, so this rung refuses it instead.
    """

    id: str
    weights: Mapping[str, Decimal]
    offset: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise TypeError("replay version id must be a non-empty string")
        if type(self.offset) is not Decimal:
            raise TypeError("replay version offset must be a Decimal")
        frozen = dict(self.weights)
        if any(not isinstance(name, str) or not name for name in frozen):
            raise TypeError("replay version weights must use non-empty component names")
        if any(type(weight) is not Decimal for weight in frozen.values()):
            raise TypeError("replay version weights must be Decimals")
        object.__setattr__(self, "weights", MappingProxyType(frozen))


@dataclass(frozen=True)
class PressureReading:
    """The displayed-score and reconstructed-advantage comparison for one record."""

    displayed_score_delta: Decimal
    advantage_delta: Decimal
    verdict: str
    advantages_a: Mapping[str, Decimal]
    advantages_b: Mapping[str, Decimal]
    estimator_conditions: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "advantages_a", MappingProxyType(dict(self.advantages_a)))
        object.__setattr__(self, "advantages_b", MappingProxyType(dict(self.advantages_b)))
        object.__setattr__(
            self,
            "estimator_conditions",
            MappingProxyType(
                {
                    key: MappingProxyType(dict(value))
                    for key, value in self.estimator_conditions.items()
                }
            ),
        )


@dataclass(frozen=True)
class CandidateReplay:
    """One actual local replay of one declared reward version."""

    version: ReplayVersion
    displayed_scores: Mapping[str, Decimal]
    advantages: Mapping[str, Decimal]
    provenance: contracts.EntryProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "displayed_scores", MappingProxyType(dict(self.displayed_scores)))
        object.__setattr__(self, "advantages", MappingProxyType(dict(self.advantages)))


def _input_error(message: str, *, field: str, **context: object) -> UsageError:
    """Use the reserved Rung-3 input-contract code for non-replayable records."""
    return UsageError(
        code="RL0131",
        message=message,
        remediation=(
            "supply a closed run-record trace with declared group ids, canonical group estimators, "
            "recorded component values, and an explicit full weight vector for each replay version"
        ),
        context={"field": field, **context},
    )


def _require_record(run_record: TraceRecord | None) -> TraceRecord:
    if run_record is None:
        raise _input_error(
            "compare rung 3 requires a run record carrying declared groups, canonical estimators, and recorded components",
            field="run_record",
        )
    if not isinstance(run_record, TraceRecord):
        raise _input_error("compare rung 3 requires a TraceRecord run record", field="run_record")
    if run_record.reader != "run_record" or not run_record.complete:
        raise _input_error(
            "compare rung 3 requires a closed run record read by the run_record reader",
            field="run_record.reader",
            reader=run_record.reader,
            complete=run_record.complete,
        )
    if not run_record.rows:
        raise _input_error("compare rung 3 requires run-record rows", field="run_record.rows")
    if not run_record.components:
        raise _input_error(
            "compare rung 3 requires recorded reward components", field="run_record.components"
        )
    return run_record


def _require_weights(version: ReplayVersion, components: tuple[str, ...]) -> None:
    expected = set(components)
    actual = set(version.weights)
    if actual != expected:
        raise _input_error(
            f"replay version {version.id!r} weights do not cover exactly the recorded components",
            field="version.weights",
            version=version.id,
            missing=sorted(expected - actual),
            unexpected=sorted(actual - expected),
        )


def _score(row, version: ReplayVersion, components: tuple[str, ...]) -> Decimal | None:
    total = version.offset
    for component in components:
        if component not in row.components:
            raise _input_error(
                f"run record row {row.row_id!r} is missing recorded component {component!r}",
                field=f"rows[{row.row_id}].components.{component}",
            )
        value = row.components[component]
        # A recorded None is the source's missing/abstention fact.  Carry it as a NaN to the
        # canonical replay kernel; never replace it with a score.
        if value is None:
            return None
        total += Decimal(str(value)) * version.weights[component]
    return total


def _estimator(record: TraceRecord, group_id: str) -> EstimatorSpec:
    canonical = record.metadata.group_estimators.get(group_id)
    if canonical is None:
        raise _input_error(
            f"run record has no canonical estimator for declared group {group_id!r}",
            field="metadata.group_estimators",
            group_id=group_id,
        )
    try:
        return EstimatorSpec.from_canonical(canonical)
    except (TypeError, ValueError) as error:
        raise _input_error(
            f"run record has an invalid canonical estimator for declared group {group_id!r}: {error}",
            field="metadata.group_estimators",
            group_id=group_id,
        ) from error


def _replay_group(
    scores: list[Decimal | None], spec: EstimatorSpec, *, group_id: str
) -> tuple[float, ...]:
    """Replay the accepted group-relative estimator math without an optional NumPy extra.

    `record.scores.replay_advantages` owns the same canonical transform, but its
    NumPy implementation is deliberately optional in the base product closure.
    This scalar path preserves its stated semantics: source absences remain NaN,
    the declared ``std_ddof`` and ``std_epsilon`` stay in the denominator, and
    ratio clipping is never confused with advantage clipping.
    """
    values = [math.nan if value is None else float(value) for value in scores]
    live = [value for value in values if math.isfinite(value)]
    if not spec.group_centred:
        raise _input_error(
            f"canonical estimator for group {group_id!r} is not group-relative, so advantages cannot be reconstructed from group scores",
            field="metadata.group_estimators",
            group_id=group_id,
        )
    if spec.std_normalised and spec.std_epsilon is None:
        raise _input_error(
            f"canonical estimator for group {group_id!r} omits std_epsilon from its normalization denominator",
            field="metadata.group_estimators",
            group_id=group_id,
        )
    if spec.std_normalised and spec.std_ddof is None:
        raise _input_error(
            f"canonical estimator for group {group_id!r} omits std_ddof from its normalization denominator",
            field="metadata.group_estimators",
            group_id=group_id,
        )
    if not live:
        return tuple(math.nan for _ in values)

    mean = sum(live) / len(live)
    ddof = int(spec.std_ddof or 0) if len(live) > 1 else 0
    divisor = len(live) - ddof
    std = math.sqrt(sum((value - mean) ** 2 for value in live) / divisor) if divisor else 0.0
    epsilon = float(spec.std_epsilon or 0.0)
    degenerate = std <= epsilon
    if degenerate and spec.degenerate_policy == "unknown":
        raise _input_error(
            f"canonical estimator for group {group_id!r} omits its degenerate-group policy",
            field="metadata.group_estimators",
            group_id=group_id,
        )

    multiplier = 1.0
    if spec.family.lower() in ("rloo", "loo"):
        multiplier = math.nan if len(live) < 2 else len(live) / (len(live) - 1)
    replayed = []
    for value in values:
        if not math.isfinite(value):
            replayed.append(math.nan)
            continue
        advantage = (value - mean) * multiplier
        if spec.std_normalised:
            advantage /= std + epsilon
        if degenerate and spec.degenerate_policy == "zero":
            advantage = 0.0
        elif degenerate and spec.degenerate_policy == "skip":
            advantage = math.nan
        replayed.append(advantage)
    return tuple(replayed)


def _groups(record: TraceRecord) -> dict[str, list[object]]:
    groups: dict[str, list[object]] = {}
    for row in record.rows:
        if row.group_id is None:
            raise _input_error(
                f"run record row {row.row_id!r} has no declared group_id; rung 3 never reconstructs groups from row adjacency",
                field="rows.group_id",
                row_id=row.row_id,
            )
        groups.setdefault(row.group_id, []).append(row)
    return groups


def _candidate_replay(
    groups: Mapping[str, list[object]],
    estimators: Mapping[str, EstimatorSpec],
    components: tuple[str, ...],
    version: ReplayVersion,
) -> CandidateReplay:
    """Replay one declared version once, retaining only measured local counters."""
    started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    displayed_scores: dict[str, Decimal] = {}
    advantages: dict[str, Decimal] = {}
    for group_id, rows in groups.items():
        scores = [_score(row, version, components) for row in rows]
        replayed = _replay_group(scores, estimators[group_id], group_id=group_id)
        for row, score, advantage in zip(rows, scores, replayed, strict=True):
            if score is not None:
                displayed_scores[row.row_id] = score
            if math.isfinite(advantage):
                advantages[row.row_id] = Decimal(str(advantage))
    wall_ns = time.perf_counter_ns() - wall_start
    cpu_ns = time.process_time_ns() - cpu_start
    return CandidateReplay(
        version=version,
        displayed_scores=displayed_scores,
        advantages=advantages,
        provenance=contracts.EntryProvenance(
            started=started,
            duration_s=Decimal(wall_ns) / Decimal(1_000_000_000),
            sandbox_tier="T0",
            offline=True,
            counters=contracts.Counters(
                calls=1,
                wall_s=Decimal(wall_ns) / Decimal(1_000_000_000),
                cpu_s=Decimal(cpu_ns) / Decimal(1_000_000_000),
                api_calls=0,
                usd="0.00",
            ),
            budget_usd="0.00",
        ),
    )


def _replay(
    record: TraceRecord, version_a: ReplayVersion, version_b: ReplayVersion
) -> tuple[PressureReading, tuple[CandidateReplay, CandidateReplay]]:
    components = tuple(record.components)
    _require_weights(version_a, components)
    _require_weights(version_b, components)
    groups = _groups(record)
    estimators = {group_id: _estimator(record, group_id) for group_id in groups}
    conditions = {group_id: estimator.__canonical__() for group_id, estimator in estimators.items()}
    replay_a = _candidate_replay(groups, estimators, components, version_a)
    replay_b = _candidate_replay(groups, estimators, components, version_b)
    displayed_deltas = [
        replay_b.displayed_scores[row_id] - replay_a.displayed_scores[row_id]
        for row_id in replay_a.displayed_scores
        if row_id in replay_b.displayed_scores
    ]
    shared_rows = tuple(key for key in replay_a.advantages if key in replay_b.advantages)

    if not displayed_deltas:
        raise _input_error(
            "run record carries no paired displayed scores for the two replay versions",
            field="rows.components",
        )
    if not shared_rows:
        raise _input_error(
            "run record carries no paired replayable advantages for the two replay versions",
            field="metadata.group_estimators",
        )
    displayed_delta = sum(displayed_deltas, Decimal("0")) / Decimal(len(displayed_deltas))
    maximum_delta = max(
        abs(replay_b.advantages[row_id] - replay_a.advantages[row_id]) for row_id in shared_rows
    )
    # The same group arithmetic reaches a different floating-point evaluation order after an
    # additive offset.  Treat only sub-picounit residue as zero; the displayed-score difference
    # remains exact Decimal arithmetic and any material pressure change remains visible.
    if maximum_delta <= _REPLAY_TOLERANCE:
        maximum_delta = Decimal("0")
    return (
        PressureReading(
            displayed_score_delta=displayed_delta,
            advantage_delta=maximum_delta,
            verdict="unchanged pressure"
            if maximum_delta == 0
            else "changed reconstructed pressure",
            advantages_a=replay_a.advantages,
            advantages_b=replay_b.advantages,
            estimator_conditions=conditions,
        ),
        (replay_a, replay_b),
    )


def _scorecard(reading: PressureReading) -> Scorecard:
    measured = Estimate(
        value=reading.advantage_delta,
        interval=(reading.advantage_delta, reading.advantage_delta),
        n=len(reading.advantages_a),
        unit="maximum absolute replayed advantage delta",
    )
    rows = []
    for row_id in ROWS:
        value = (
            measured
            if row_id == "exploit_pressure"
            else Absence(
                "not measured: rung 3 reconstructs recorded training pressure only",
                "run the rung that measures this scorecard row",
            )
        )
        rows.append(ScorecardRow(row_id, row_id.replace("_", " "), value, _SCOPE))
    return Scorecard.render(rows, rung=Rung.ADVANTAGE_REPLAY)


def _version_identity(version: ReplayVersion) -> dict[str, object]:
    """The declared version values the replay actually consumed."""
    return {
        "id": version.id,
        "weights": {name: str(weight) for name, weight in version.weights.items()},
        "offset": str(version.offset),
    }


def _candidate_entry(record: TraceRecord, replay: CandidateReplay) -> contracts.Entry:
    """Bind one observed local replay cost to its declared reward-version inputs."""
    identity = _version_identity(replay.version)
    identity_digest = contracts.digest(identity)
    return contracts.Entry(
        entry_id=_CANDIDATE_ENTRY_PREFIX + identity_digest.removeprefix("sha256:")[:16],
        section="reward_statistics",
        kind="witness",
        measurand="local CPU and wall time of one explicit reward-version replay",
        method=contracts.Method(
            id=R3_ENTRY_ID,
            version="1.0.0",
            params_digest=contracts.digest(
                {
                    "measurement_stage": "candidate_replay",
                    "candidate": identity,
                }
            ),
            procedure=(
                "recompose the closed record's group scores under this explicit component-weight vector, "
                "then replay the recorded canonical estimator once while observing local process and wall clocks"
            ),
        ),
        scope=_SCOPE,
        subject_ref=record.subject_ref,
        depends_on=list(record.depends_on),
        state="complete",
        provenance=replay.provenance,
        limitations=[
            "local replay execution only; this is not behavioural capability, full training, or provider-cost evidence",
            "unobserved execution dimensions are absent rather than inferred from the local replay counters",
        ],
        result={
            "candidate_id": replay.version.id,
            "candidate_identity": identity_digest,
        },
        witness=contracts.Witness(
            inputs={"candidate": identity},
            procedure=(
                "one direct local invocation of the recorded score and estimator replay for the declared candidate version"
            ),
            observed="per-version process CPU and wall-clock observations for that direct local replay",
            artifact_identity=identity_digest,
        ),
    )


def _entry(
    record: TraceRecord,
    version_a: ReplayVersion,
    version_b: ReplayVersion,
    reading: PressureReading,
) -> contracts.Entry:
    return contracts.Entry(
        entry_id=R3_ENTRY_ID,
        section="reward_statistics",
        kind="estimate",
        measurand="maximum absolute change in replayed group-relative advantages",
        method=contracts.Method(
            id=R3_ENTRY_ID,
            version="1.0.0",
            params_digest=contracts.digest(
                {
                    "versions": [version_a.id, version_b.id],
                    "weights_a": {name: str(weight) for name, weight in version_a.weights.items()},
                    "weights_b": {name: str(weight) for name, weight in version_b.weights.items()},
                    "offset_a": str(version_a.offset),
                    "offset_b": str(version_b.offset),
                }
            ),
            procedure=(
                "recompose each recorded group's scores under both explicit component-weight vectors, "
                "then replay that group's recorded canonical estimator including centring, standard "
                "deviation denominator, epsilon, degenerate policy, and recorded missing semantics"
            ),
        ),
        scope=_SCOPE,
        subject_ref=record.subject_ref,
        depends_on=list(record.depends_on),
        state="complete",
        provenance=contracts.EntryProvenance(
            started="2026-09-26T00:00:00Z", duration_s=0.0, sandbox_tier="T0", offline=True
        ),
        limitations=[
            "this reconstructs score-to-advantage training pressure from a record; it does not observe scalar loss, gradients, optimizer state, or applied updates",
            "score order alone is not used as evidence of unchanged pressure",
            "standardized replay is invariant to positive rescaling only when the recorded denominator and epsilon permit it; policy-ratio clipping is not an advantage clip",
        ],
        result={
            "version_a": version_a.id,
            "version_b": version_b.id,
            "displayed_score_delta": str(reading.displayed_score_delta),
            "advantage_delta": str(reading.advantage_delta),
            "verdict": reading.verdict,
        },
        value=float(reading.advantage_delta),
        unit="maximum absolute replayed advantage delta",
        n=len(reading.advantages_a),
        sampling_unit="recorded trajectory",
        uncertainty=contracts.Uncertainty(
            method=contracts.NO_INTERVAL,
            level=0.95,
            reason="deterministic replay of one closed run record",
            clusters=len(reading.estimator_conditions),
        ),
    )


def advantage_replay(
    run_record: TraceRecord | None, version_a: ReplayVersion, version_b: ReplayVersion
) -> CompareResult:
    """Replay two explicit reward versions through the recorded trainer transform."""
    record = _require_record(run_record)
    if not isinstance(version_a, ReplayVersion) or not isinstance(version_b, ReplayVersion):
        raise _input_error(
            "compare rung 3 requires explicit ReplayVersion values with complete recorded-component weights",
            field="version_a/version_b",
        )
    reading, candidates = _replay(record, version_a, version_b)
    return CompareResult(
        _SCOPE,
        _scorecard(reading),
        [
            _entry(record, version_a, version_b, reading),
            *(_candidate_entry(record, candidate) for candidate in candidates),
        ],
        "a deterministic reconstruction of score-to-advantage pressure for the closed recorded run",
        "observed training, scalar loss, gradients, optimizer transforms, applied updates, or a pressure conclusion from score rank alone",
        reading,
    )


def run(inputs: CompareInputs) -> CompareResult:
    """Registry adapter; P-RUNG1 wires the typed public request into this real engine."""
    return advantage_replay(inputs.run_record, inputs.version_a, inputs.version_b)  # type: ignore[arg-type]


register(Rung.ADVANTAGE_REPLAY, run)


__all__ = ["PressureReading", "R3_ENTRY_ID", "ReplayVersion", "advantage_replay", "run"]
