"""Sealed issue/resolve records and T1-only sequential evidence.

The types in this module deliberately use no predictor implementation.  A
predictor supplies an already checkpoint-bounded distribution and model digest;
this boundary seals it, scores it later, and rejects any attempt to change the
past or use information from the future.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable, Mapping

import rfc8785

from reward_lens.forecast.scoring import REGISTERED_QUANTILES, score_fraction, score_t1

BASELINES: tuple[str, ...] = ("persistence", "climatology", "drift", "covariance")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _require_sha256(value: str, *, field: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be sha256: followed by 64 lowercase hexadecimal digits")


class ForecastRefusal(ValueError):
    """A named forecast refusal whose message carries the checkable field."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    """Stable, finite UTF-8 bytes for an issued payload.

    The payload uses strings for Decimal scores and primitive JSON values only;
    rejecting NaN/Infinity here protects the bytes which the resolver verifies.
    """
    try:
        return rfc8785.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("issued forecast contains non-canonical JSON data") from exc


@dataclass(frozen=True)
class ForecastSpec:
    targets: tuple[str, ...] = ("T1", "T2", "T3")
    threshold: Decimal = Decimal("0.5")
    cadence: int = 20
    horizon: int = 20
    baselines: tuple[str, ...] = BASELINES
    quantiles: tuple[Decimal, ...] = REGISTERED_QUANTILES
    n_min: int = 20
    alpha: Decimal = Decimal("0.05")
    null_seed_set: tuple[int, ...] = tuple(range(1000))
    null_constructions: tuple[str, ...] = BASELINES
    null_multiplicity: int = 4

    def __post_init__(self) -> None:
        if self.targets != ("T1", "T2", "T3"):
            raise ValueError("ForecastSpec targets must be exactly T1, T2, T3")
        if self.cadence <= 0 or self.horizon <= 0 or self.n_min <= 0:
            raise ValueError("cadence, horizon and n_min must be positive")
        if self.baselines != BASELINES:
            raise ValueError(
                "ForecastSpec requires D70 baselines persistence, climatology, drift, covariance"
            )
        if self.quantiles != REGISTERED_QUANTILES:
            raise ValueError("ForecastSpec quantiles must be the registered WIS quantiles")
        if not Decimal(0) < self.threshold < Decimal(1):
            raise ValueError("onset threshold must be in (0, 1)")
        if not Decimal(0) < self.alpha < Decimal(1):
            raise ValueError("alpha must be in (0, 1)")
        if len(set(self.null_seed_set)) != len(self.null_seed_set):
            raise ValueError("null_seed_set contains duplicate registered seeds")
        if self.null_constructions != BASELINES:
            raise ValueError("null constructions must register the four fixed baseline processes")
        if self.null_multiplicity != len(self.null_constructions):
            raise ValueError("null_multiplicity must equal the registered fixed constructions")

    @classmethod
    def default(cls) -> "ForecastSpec":
        return cls()

    def canonical(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "threshold": str(self.threshold),
            "cadence": self.cadence,
            "horizon": self.horizon,
            "baselines": list(self.baselines),
            "quantiles": [str(q) for q in self.quantiles],
            "n_min": self.n_min,
            "alpha": str(self.alpha),
            "null_seed_set": list(self.null_seed_set),
            "null_constructions": list(self.null_constructions),
            "null_multiplicity": self.null_multiplicity,
        }

    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical(self.canonical())).hexdigest()


@dataclass(frozen=True)
class IssuedForecast:
    target: str
    checkpoint: int
    source_name: str
    source_cutoff: int
    forecast: Any
    baselines: Mapping[str, Any]
    model_digest: str
    spec_digest: str
    feature_cutoff: int
    model_selection_cutoff: int | None
    issued_bytes: bytes
    sealed_digest: str
    baseline_provenance: Mapping[str, Mapping[str, Any]]
    claim: str = ""

    def canonical(self) -> dict[str, Any]:
        return json.loads(self.issued_bytes.decode("utf-8"))

    @classmethod
    def from_issued_bytes(cls, issued_bytes: bytes) -> "IssuedForecast":
        """Reconstruct a sealed issue without recomputing a prediction or baseline."""
        payload = json.loads(issued_bytes.decode("utf-8"))
        digest = "sha256:" + hashlib.sha256(issued_bytes).hexdigest()
        return cls(
            target=str(payload["target"]),
            checkpoint=int(payload["checkpoint"]),
            source_name=str(payload["source_name"]),
            source_cutoff=int(payload["source_cutoff"]),
            forecast=payload["forecast"],
            baselines=dict(payload["baselines"]),
            model_digest=str(payload["model_digest"]),
            spec_digest=str(payload["spec_digest"]),
            feature_cutoff=int(payload["feature_cutoff"]),
            model_selection_cutoff=(
                None
                if payload.get("model_selection_cutoff") is None
                else int(payload["model_selection_cutoff"])
            ),
            issued_bytes=issued_bytes,
            sealed_digest=digest,
            baseline_provenance=dict(payload.get("baseline_provenance", {})),
            claim=str(payload.get("claim", "")),
        )


@dataclass(frozen=True)
class ResolvedForecast:
    issued: IssuedForecast
    outcome: bool | Decimal
    score: Decimal
    baseline_scores: Mapping[str, Decimal]
    hardest_difference: Decimal


@dataclass(frozen=True)
class BaselineProvenance:
    """The pre-checkpoint fit evidence sealed for one fixed baseline."""

    cutoff: int
    fit_digest: str

    def __post_init__(self) -> None:
        _require_sha256(self.fit_digest, field="baseline fit_digest")

    def canonical(self) -> dict[str, Any]:
        return {"cutoff": self.cutoff, "fit_digest": self.fit_digest}


def _check_cutoff(*, checkpoint: int, cutoff: int, name: str, refusal: str) -> None:
    if cutoff > checkpoint:
        raise ForecastRefusal(
            f"{refusal} checkpoint={checkpoint} source={name} cutoff={cutoff}; "
            "issue inputs, scalers, and model selection must end at the checkpoint"
        )


def issue(
    *,
    checkpoint: int,
    spec: ForecastSpec,
    target: str,
    source_name: str,
    source_cutoff: int,
    forecast: Any,
    baselines: Mapping[str, Any],
    model_digest: str,
    feature_cutoff: int | None = None,
    model_selection_cutoff: int | None = None,
    baseline_provenance: Mapping[str, BaselineProvenance] | None = None,
    claim: str = "",
) -> IssuedForecast:
    """Seal a checkpoint-bounded forecast and its four comparators into immutable bytes."""
    if checkpoint < 0:
        raise ValueError("checkpoint must be non-negative")
    if target not in spec.targets:
        raise ValueError(f"target {target!r} is not registered in this ForecastSpec")
    _check_cutoff(
        checkpoint=checkpoint,
        cutoff=source_cutoff,
        name=source_name,
        refusal="FORECAST_FUTURE_INPUT",
    )
    feature_cutoff = source_cutoff if feature_cutoff is None else feature_cutoff
    _check_cutoff(
        checkpoint=checkpoint,
        cutoff=feature_cutoff,
        name="feature_cutoff",
        refusal="FORECAST_FUTURE_INPUT",
    )
    if model_selection_cutoff is not None:
        _check_cutoff(
            checkpoint=checkpoint,
            cutoff=model_selection_cutoff,
            name="model_selection",
            refusal="FORECAST_FUTURE_INPUT",
        )
    _require_sha256(model_digest, field="model_digest")
    if set(baselines) != set(BASELINES) or len(baselines) != len(BASELINES):
        raise ValueError("issued forecast must include exactly four D70 baselines")
    if baseline_provenance is None or set(baseline_provenance) != set(BASELINES):
        raise ForecastRefusal(
            "FORECAST_BASELINE_LEAKAGE requires pre-checkpoint provenance for persistence, "
            "climatology, drift, covariance"
        )
    for name, provenance in baseline_provenance.items():
        _check_cutoff(
            checkpoint=checkpoint,
            cutoff=provenance.cutoff,
            name=name,
            refusal="FORECAST_BASELINE_LEAKAGE",
        )
    if target == "T1":
        score_t1(forecast, False)
        for name, value in baselines.items():
            score_t1(value, False)
    else:
        if not isinstance(forecast, Mapping):
            raise ValueError("T2/T3 issue requires a quantile distribution mapping")
        values = tuple(forecast.get("values", ()))
        score_fraction(spec.quantiles, values, Decimal("0.5"))
        for name, value in baselines.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"T2/T3 baseline {name} requires a quantile distribution mapping")
            score_fraction(spec.quantiles, tuple(value.get("values", ())), Decimal("0.5"))
    payload = {
        "target": target,
        "checkpoint": checkpoint,
        "source_name": source_name,
        "source_cutoff": source_cutoff,
        "feature_cutoff": feature_cutoff,
        "model_selection_cutoff": model_selection_cutoff,
        "forecast": forecast,
        "baselines": dict(baselines),
        "baseline_provenance": {
            name: provenance.canonical() for name, provenance in baseline_provenance.items()
        },
        "model_digest": model_digest,
        "spec_digest": spec.digest(),
        "claim": claim,
    }
    sealed = _canonical(payload)
    digest = "sha256:" + hashlib.sha256(sealed).hexdigest()
    return IssuedForecast(
        target,
        checkpoint,
        source_name,
        source_cutoff,
        forecast,
        dict(baselines),
        model_digest,
        spec.digest(),
        feature_cutoff,
        model_selection_cutoff,
        sealed,
        digest,
        {name: provenance.canonical() for name, provenance in baseline_provenance.items()},
        claim,
    )


def resolve(
    issued: IssuedForecast, *, outcome: bool | float | Decimal, issued_bytes: bytes | None = None
) -> ResolvedForecast:
    """Score an outcome against the frozen prediction; no resolution can alter it."""
    proposed = issued.issued_bytes if issued_bytes is None else issued_bytes
    digest = "sha256:" + hashlib.sha256(proposed).hexdigest()
    if proposed != issued.issued_bytes or digest != issued.sealed_digest:
        raise ForecastRefusal(
            f"FORECAST_RESOLUTION_MISMATCH sealed_digest={issued.sealed_digest} supplied_digest={digest}"
        )
    if issued.target == "T1":
        if not isinstance(outcome, bool):
            raise ValueError("T1 resolution outcome must be bool")
        own = score_t1(issued.forecast, outcome)
        scores = {name: score_t1(value, outcome) for name, value in issued.baselines.items()}
    else:
        value = Decimal(str(outcome))
        own = score_fraction(REGISTERED_QUANTILES, tuple(issued.forecast["values"]), value)
        scores = {
            name: score_fraction(REGISTERED_QUANTILES, tuple(row["values"]), value)
            for name, row in issued.baselines.items()
        }
    hardest = min(scores.values())
    return ResolvedForecast(
        issued,
        outcome if isinstance(outcome, bool) else Decimal(str(outcome)),
        own,
        scores,
        hardest - own,
    )


@dataclass
class T1EProcess:
    """Predictable hedged-capital e-process for one fixed T1 baseline."""

    wealth: Decimal = Decimal(1)
    _past_x: list[Decimal] = field(default_factory=list)
    bets: list[Decimal] = field(default_factory=list)
    wealth_history: list[Decimal] = field(default_factory=list)
    first_crossing_index: int | None = None
    first_crossing_time: str | None = None

    def next_bet(self) -> Decimal:
        """A [0, 1/2] bet determined only by previously resolved score differences."""
        if not self._past_x:
            return Decimal(0)
        edge = sum(self._past_x) / Decimal(len(self._past_x)) - Decimal("0.5")
        return min(Decimal("0.5"), max(Decimal(0), Decimal(2) * edge))

    def update(
        self,
        *,
        forecast_brier: object,
        baseline_brier: object,
        bet: Decimal | None = None,
        resolution_index: int | None = None,
        resolved_at: str | None = None,
    ) -> Decimal:
        own = Decimal(str(forecast_brier))
        base = Decimal(str(baseline_brier))
        if not (own.is_finite() and base.is_finite()) or own < 0 or own > 1 or base < 0 or base > 1:
            raise ValueError("T1 e-process requires finite Brier terms in [0, 1]")
        expected_bet = self.next_bet()
        if bet is None:
            bet = expected_bet
        elif Decimal(str(bet)) != expected_bet:
            raise ForecastRefusal(
                "FORECAST_FUTURE_BET sealed stake does not match its pre-resolution state"
            )
        x = (base - own + Decimal(1)) / Decimal(2)
        self.wealth *= Decimal(1) + bet * (x - Decimal("0.5"))
        self.bets.append(Decimal(str(bet)))
        self._past_x.append(x)
        self.wealth_history.append(self.wealth)
        if self.first_crossing_index is None and self.wealth >= Decimal(20):
            self.first_crossing_index = (
                len(self.wealth_history) - 1 if resolution_index is None else resolution_index
            )
            self.first_crossing_time = resolved_at
        return self.wealth


def future_informed_bet(history: Iterable[Decimal], current_x: Decimal) -> Decimal:
    """Deliberately invalid bettor: its stake changes with the current outcome-derived X."""
    del history
    return Decimal("0.5") if current_x > Decimal("0.5") else Decimal(0)


def validate_predictable_bettor(bettor: Any) -> None:
    """Reject a bettor if a fixed past admits different stakes for the current X."""
    history = (Decimal("0.5"),)
    left = Decimal(str(bettor(history, Decimal(0))))
    right = Decimal(str(bettor(history, Decimal(1))))
    if left != right:
        raise ForecastRefusal(
            "FORECAST_FUTURE_BET history=(0.5) current_x changes the stake; bets must be predictable"
        )


def reject_future_informed_bettor() -> None:
    """Run the concrete invalid bettor and assert the predictability gate catches it."""
    try:
        validate_predictable_bettor(future_informed_bet)
    except ForecastRefusal:
        return
    raise AssertionError("future-informed negative control was not rejected")


@dataclass(frozen=True)
class T1Evidence:
    """One predictable e-process per fixed baseline; skill requires all four crossings."""

    processes: Mapping[str, T1EProcess]

    @property
    def all_crossed(self) -> bool:
        return bool(self.headline_state["qualified"])

    @property
    def headline_state(self) -> dict[str, Any]:
        return t1_headline_state(self.processes)


def t1_headline_state(processes: Mapping[str, T1EProcess]) -> dict[str, Any]:
    """Machine-readable history and diagnostic state for four fixed T1 processes."""
    if set(processes) != set(BASELINES):
        raise ValueError("T1 evidence requires the four fixed named baseline processes")
    current_wealth = {name: str(processes[name].wealth) for name in BASELINES}
    first_crossing_index = {name: processes[name].first_crossing_index for name in BASELINES}
    first_crossing_time = {name: processes[name].first_crossing_time for name in BASELINES}
    crossed = tuple(name for name in BASELINES if first_crossing_index[name] is not None)
    return {
        "criterion": "all_four_fixed_processes_ever_crossed_20",
        "threshold": "20",
        "crossing_time_basis": "local_tool_recorded_at_append_not_externally_authenticated",
        "qualified": len(crossed) == len(BASELINES),
        "crossed_baselines": list(crossed),
        "first_crossing_index": first_crossing_index,
        "first_crossing_time": first_crossing_time,
        "current_wealth": current_wealth,
        "wealth_history": {
            name: [str(value) for value in processes[name].wealth_history] for name in BASELINES
        },
        "min_current_log10_wealth": str(
            min(process.wealth for process in processes.values()).log10()
        ),
    }


def sequential_evidence(target: str, resolved: Iterable[ResolvedForecast]) -> T1Evidence:
    if target in {"T2", "T3"}:
        raise ValueError("T2/T3 expose prospective proper scores but no sequential skill claim")
    if target != "T1":
        raise ValueError(f"unknown forecast target {target!r}")
    processes = {name: T1EProcess() for name in BASELINES}
    for index, row in enumerate(resolved):
        if row.issued.target != "T1":
            raise ValueError("T1 sequential evidence cannot include fraction forecasts")
        if set(row.baseline_scores) != set(BASELINES):
            raise ValueError("T1 sequential evidence requires all four fixed baseline scores")
        for name, process in processes.items():
            process.update(
                forecast_brier=row.score,
                baseline_brier=row.baseline_scores[name],
                resolution_index=index,
            )
    return T1Evidence(processes)


def exact_binomial_excess(crossings: int, streams: int, *, alpha: float = 0.05) -> float:
    """Exact one-sided probability of observing at least ``crossings`` null crossings."""
    if not 0 <= crossings <= streams or not 0 < alpha < 1:
        raise ValueError("invalid binomial excess arguments")
    return sum(
        math.comb(streams, count) * alpha**count * (1 - alpha) ** (streams - count)
        for count in range(crossings, streams + 1)
    )


def null_stream_regression(
    spec: ForecastSpec, *, length: int = 100, threshold: Decimal = Decimal(20)
) -> Mapping[str, tuple[int, float]]:
    """Registered 1,000-stream null check with an exact alpha=.01 excess test.

    Each stream has conditional mean X=1/2. The result is a regression guard;
    it does not replace the predictable-bet proof in :class:`T1EProcess`. The
    registered multiplicity controls the per-construction crossing probability.
    """
    if len(spec.null_seed_set) != 1000:
        raise ValueError("the 3.1.0 null regression requires exactly 1,000 registered seeds")
    crossings = {name: 0 for name in BASELINES}
    for seed in spec.null_seed_set:
        rng = random.Random(seed)
        processes = {name: T1EProcess() for name in BASELINES}
        for _ in range(length):
            # Brier values encode X in [0, 1] while preserving E[X | past] = 1/2.
            x = Decimal(1) if rng.random() < 0.5 else Decimal(0)
            for process in processes.values():
                process.update(forecast_brier=Decimal(1) - x, baseline_brier=x)
        for name, process in processes.items():
            # The ledger is continuously inspectable: count an *ever* crossing,
            # including a process whose wealth later fell below the threshold.
            crossings[name] += int(any(wealth >= threshold for wealth in process.wealth_history))
    results = {
        name: (
            count,
            exact_binomial_excess(
                count, len(spec.null_seed_set), alpha=float(spec.alpha) / spec.null_multiplicity
            ),
        )
        for name, count in crossings.items()
    }
    if any(p_value < 0.01 for _, p_value in results.values()):
        raise AssertionError(f"null crossing excess by fixed baseline: {results}")
    reject_future_informed_bettor()
    return results


__all__ = [
    "BASELINES",
    "BaselineProvenance",
    "ForecastRefusal",
    "ForecastSpec",
    "IssuedForecast",
    "ResolvedForecast",
    "T1EProcess",
    "T1Evidence",
    "exact_binomial_excess",
    "issue",
    "future_informed_bet",
    "null_stream_regression",
    "resolve",
    "reject_future_informed_bettor",
    "sequential_evidence",
    "t1_headline_state",
    "validate_predictable_bettor",
]
