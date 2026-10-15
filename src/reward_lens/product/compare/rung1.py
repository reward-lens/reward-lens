"""Rung 1: paired re-scoring of one finite response bank."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Mapping

from reward_lens import contracts
from reward_lens.bank import ResponseBank
from reward_lens.bank.select import require_bank

from .registry import CompareInputs, CompareResult, Rung, register
from .scorecard import ROWS, Absence, Estimate, Scorecard, ScorecardRow

_SCOPE = "evaluator_comparison"
_QUANTUM = Decimal("0.000001")
_Z_95 = Decimal("1.959964")
_MIN_INTERVAL_CLUSTERS = 15


@dataclass(frozen=True)
class PairedRescore:
    version_a: str
    version_b: str
    cluster_unit: str
    task_differences: Mapping[str, Decimal]
    mean_difference: Decimal
    standard_error: Decimal
    interval: tuple[Decimal, Decimal] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_differences", MappingProxyType(dict(self.task_differences)))


def _rounded(value: Decimal) -> Decimal:
    return value.quantize(_QUANTUM, rounding=ROUND_HALF_UP)


def _response_score(response, version: str) -> Decimal:
    scores = response.recorded_scores
    if scores is None or version not in scores:
        raise ValueError(
            f"response {response.digest} has no recorded score for evaluator version {version!r}"
        )
    return Decimal(scores[version])


def _task_difference(bank: ResponseBank, task, version_a: str, version_b: str) -> Decimal:
    responses = bank.responses_for(task)
    differences = tuple(
        _response_score(response, version_b) - _response_score(response, version_a)
        for response in responses
    )
    return sum(differences, Decimal(0)) / Decimal(len(differences))


def _reading(
    bank: ResponseBank, version_a: str, version_b: str, cluster_unit: str
) -> PairedRescore:
    if cluster_unit != "task":
        raise ValueError("rung 1 clusters paired evaluator differences at the task unit")
    by_task = {str(task): _task_difference(bank, task, version_a, version_b) for task in bank.tasks}
    values = tuple(by_task.values())
    mean = sum(values, Decimal(0)) / Decimal(len(values))
    if len(values) == 1:
        standard_error = Decimal(0)
    else:
        sample_variance = sum((value - mean) ** 2 for value in values) / Decimal(len(values) - 1)
        standard_error = (sample_variance / Decimal(len(values))).sqrt()
    interval = None
    if len(values) >= _MIN_INTERVAL_CLUSTERS:
        interval = (
            _rounded(mean - _Z_95 * standard_error),
            _rounded(mean + _Z_95 * standard_error),
        )
    return PairedRescore(
        version_a,
        version_b,
        cluster_unit,
        by_task,
        _rounded(mean),
        _rounded(standard_error),
        interval,
    )


def _scorecard(reading: PairedRescore) -> Scorecard:
    if reading.interval is None:
        measured = Absence(
            "not measured: fewer than 15 task clusters cannot support a rung-1 interval",
            "re-score at least 15 independent tasks under both evaluator versions",
        )
    else:
        measured = Estimate(
            reading.mean_difference,
            reading.interval,
            n=len(reading.task_differences),
            unit="score difference",
        )
    rows = []
    for row_id in ROWS:
        value = (
            measured
            if row_id == "intended_task_success"
            else Absence(
                "not measured: rung 1 only re-scores the existing response bank",
                "run the rung that measures this scorecard row",
            )
        )
        rows.append(ScorecardRow(row_id, row_id.replace("_", " "), value, _SCOPE))
    return Scorecard.render(rows, rung=Rung.RE_SCORE)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _entry(bank: ResponseBank, reading: PairedRescore) -> contracts.Entry:
    clusters = len(reading.task_differences)
    if reading.interval is None:
        uncertainty = contracts.Uncertainty(
            method=contracts.NO_INTERVAL,
            level=0.95,
            reason="fewer than 15 independent task clusters",
            clusters=clusters,
        )
    else:
        uncertainty = contracts.Uncertainty(
            method="paired_difference",
            level=0.95,
            interval=[float(reading.interval[0]), float(reading.interval[1])],
            standard=float(reading.standard_error),
            cluster_unit="task",
            clusters=clusters,
        )
    return contracts.Entry(
        entry_id="reward_statistics.evaluator_rescore",
        section="reward_statistics",
        kind="estimate",
        measurand="mean paired difference between two recorded evaluator versions",
        method=contracts.Method(
            id="reward_statistics.evaluator_rescore",
            version="1.0.0",
            params_digest=contracts.digest(
                {"cluster_unit": "task", "versions": [reading.version_a, reading.version_b]}
            ),
            procedure="each response is read under both recorded evaluator versions; response differences are averaged within task and task means define the clustered standard error",
        ),
        scope=_SCOPE,
        subject_ref=_digest(bank.digest),
        depends_on=[],
        state="complete",
        provenance=contracts.EntryProvenance(
            started="2026-09-26T00:00:00Z", duration_s=0.0, sandbox_tier="T0", offline=True
        ),
        limitations=[
            "this establishes an evaluator-score difference on this finite bank, not behavior after training"
        ],
        result={"version_a": reading.version_a, "version_b": reading.version_b},
        value=float(reading.mean_difference),
        unit="score difference",
        n=clusters,
        sampling_unit="task",
        uncertainty=uncertainty,
    )


def rescore(
    bank: ResponseBank | None, version_a: str, version_b: str, *, cluster_unit: str = "task"
) -> CompareResult:
    """Re-score the same finite responses under two named evaluator versions."""
    finite_bank = require_bank(bank)
    reading = _reading(finite_bank, version_a, version_b, cluster_unit)
    return CompareResult(
        _SCOPE,
        _scorecard(reading),
        [_entry(finite_bank, reading)],
        "a paired evaluator-score difference on the supplied finite response bank",
        "behavior after training, selection pressure, or a causal effect of an update",
        reading,
    )


def run(inputs: CompareInputs) -> CompareResult:
    return rescore(inputs.bank, inputs.version_a, inputs.version_b)


register(Rung.RE_SCORE, run)

__all__ = ["PairedRescore", "rescore", "run"]
