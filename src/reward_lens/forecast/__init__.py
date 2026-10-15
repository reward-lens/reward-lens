"""Forecasting: the layer that makes every other measurement worth something (Part 7).

A measurement that cannot support a claim about the future cannot support a decision, so this is
not a section of the science layer. It is what the rest of the library is for.

The competitive position is narrow and it is real. Prospective monitoring of reinforcement-learning
internals is a small live subfield with at least four methods, so "we watch internals early" is not
a differentiator. But across every method read in the scan, **not one fits its predictor strictly
before the outcome it predicts.** So the position is temporal validity, enforced by type, with
baselines and a public score.

Five things, in the order you meet them:

`issue` refuses to build a forecast whose inputs, transitively, postdate it. No override.

Four baselines are mandatory and a forecast without them cannot be constructed: climatology,
persistence, the dumb statistic, and the scaffolded black-box method, which currently means
contrastive belief-flipping and is the first thing a reviewer will raise.

`RunCorpus` splits over runs and time, never over items, because splitting over items is what makes
a temporal claim mean nothing.

`resolve` returns `Resolved` or `Void`, and an expired or unresolvable forecast is void with a
reason, never a miss.

`CalibrationLedger` is append-only and prints its own worst result at the top.

    from reward_lens.forecast import issue, resolve, CalibrationLedger, entry_from

    forecast = issue(target, subject, rule, distribution, inputs, at=now, store=store, ...)
    outcome = resolve(forecast, metrics, at=later)
    ledger.append(entry_from(forecast, outcome))
    print(ledger.render())
"""

import importlib
import sys
from typing import Any

# The historical forecast surface includes barrier, corpus and calibration types that transitively
# load the evidence store.  The product issue route only needs the new sealed-ledger modules.  A
# normal ``from reward_lens.forecast import Forecast`` still returns the same object; it is simply
# imported when requested instead of while a pure scoring module is being imported.
_EXPORT_MODULES: dict[str, tuple[str, str]] = {}


def _exports(module: str, *names: str) -> None:
    _EXPORT_MODULES.update({name: (module, name) for name in names})


_exports(
    "reward_lens.forecast.barrier",
    "ForecastLeakageError",
    "ancestry",
    "check_barrier",
    "information_time_of",
    "issue",
)
_exports(
    "reward_lens.forecast.baselines",
    "BELIEF_FLIP_VERSION",
    "BeliefFlipReading",
    "RecordsTest",
    "belief_flip_hash",
    "climatology",
    "climatology_interval",
    "contrastive_belief_flip",
    "dumb_statistic",
    "dumb_statistic_from_gradnorm",
    "harmonic",
    "persistence",
    "persistence_rate",
    "records_null_forecast",
    "records_test",
    "render_belief_flip",
    "d70_climatology",
    "d70_covariance",
    "d70_drift",
    "d70_persistence",
)
_exports(
    "reward_lens.forecast.evidence",
    "ForecastRefusal",
    "ForecastSpec",
    "IssuedForecast",
    "ResolvedForecast",
    "T1EProcess",
    "exact_binomial_excess",
    "null_stream_regression",
    "sequential_evidence",
)
_EXPORT_MODULES["D70_BASELINES"] = ("reward_lens.forecast.evidence", "BASELINES")
_exports("reward_lens.forecast.scoring", "REGISTERED_QUANTILES", "score_fraction", "score_t1")
_exports(
    "reward_lens.forecast.corpus",
    "AISI_TRAPS",
    "Fold",
    "RunCorpus",
    "SplitKind",
    "corpus_from_runs",
    "iter_folds",
)
_exports(
    "reward_lens.forecast.ledger",
    "FORECAST_BRIER_QUANTITY",
    "PUBLISHED_COIN_BRIER",
    "PUBLISHED_DIRECTIONAL_BRIER",
    "PUBLISHED_INTERVAL_COVERAGE",
    "REGISTERED_NOMINAL_COVERAGE",
    "CalibrationLedger",
    "ForecastCalibration",
    "LedgerEntry",
    "LedgerScore",
    "entry_from",
)
_exports(
    "reward_lens.forecast.resolve",
    "Resolution",
    "Resolved",
    "Void",
    "VoidReason",
    "partition",
    "resolve",
)
_exports(
    "reward_lens.forecast.schema",
    "MANDATORY_BASELINE_KINDS",
    "BaselineForecast",
    "BaselineKind",
    "BinaryProbability",
    "Comparator",
    "DecisionSpec",
    "Distribution",
    "Forecast",
    "ForecastError",
    "ForecastID",
    "HorizonSpec",
    "InformationTime",
    "IntervalForecast",
    "QuantileForecast",
    "ReferenceClass",
    "ReferenceClassID",
    "ResolutionRule",
    "forecast_id",
)
_exports(
    "reward_lens.forecast.score",
    "CoverageScore",
    "DecisionValue",
    "MurphyDecomposition",
    "ReliabilityDiagram",
    "SkillScore",
    "brier",
    "brier_terms",
    "coverage_score",
    "decision_value",
    "forecast_lead_time",
    "log_score",
    "murphy_decomposition",
    "recalibrate",
    "reliability_diagram",
    "skill_score",
    "transition_fit",
)
_exports("reward_lens.measure.rate.transition", "LeadTime", "TransitionFit")


def __getattr__(name: str) -> Any:
    """Load a retained public export on first use without changing its identity."""
    try:
        module_name, attribute = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    # A lazy sibling import can load forecast.resolve and assign its module on
    # this package. Restore the documented callable before returning any export.
    resolved_module = sys.modules.get("reward_lens.forecast.resolve")
    if resolved_module is not None:
        globals()["resolve"] = getattr(resolved_module, "resolve")
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))


__all__ = [
    "AISI_TRAPS",
    "BELIEF_FLIP_VERSION",
    "FORECAST_BRIER_QUANTITY",
    "MANDATORY_BASELINE_KINDS",
    "PUBLISHED_COIN_BRIER",
    "PUBLISHED_DIRECTIONAL_BRIER",
    "PUBLISHED_INTERVAL_COVERAGE",
    "REGISTERED_NOMINAL_COVERAGE",
    "BaselineForecast",
    "BaselineKind",
    "BeliefFlipReading",
    "BinaryProbability",
    "CalibrationLedger",
    "D70_BASELINES",
    "Comparator",
    "CoverageScore",
    "DecisionSpec",
    "DecisionValue",
    "Distribution",
    "Fold",
    "Forecast",
    "ForecastCalibration",
    "ForecastError",
    "ForecastID",
    "ForecastLeakageError",
    "ForecastRefusal",
    "ForecastSpec",
    "HorizonSpec",
    "InformationTime",
    "IssuedForecast",
    "IntervalForecast",
    "LeadTime",
    "LedgerEntry",
    "LedgerScore",
    "MurphyDecomposition",
    "QuantileForecast",
    "RecordsTest",
    "ReferenceClass",
    "ReferenceClassID",
    "ReliabilityDiagram",
    "Resolution",
    "ResolutionRule",
    "Resolved",
    "ResolvedForecast",
    "RunCorpus",
    "SkillScore",
    "SplitKind",
    "TransitionFit",
    "T1EProcess",
    "Void",
    "VoidReason",
    "ancestry",
    "belief_flip_hash",
    "brier",
    "brier_terms",
    "check_barrier",
    "climatology",
    "climatology_interval",
    "contrastive_belief_flip",
    "corpus_from_runs",
    "coverage_score",
    "d70_climatology",
    "d70_covariance",
    "d70_drift",
    "d70_persistence",
    "decision_value",
    "dumb_statistic",
    "dumb_statistic_from_gradnorm",
    "entry_from",
    "exact_binomial_excess",
    "forecast_id",
    "harmonic",
    "information_time_of",
    "issue",
    "iter_folds",
    "forecast_lead_time",
    "transition_fit",
    "log_score",
    "murphy_decomposition",
    "null_stream_regression",
    "partition",
    "persistence",
    "persistence_rate",
    "records_null_forecast",
    "records_test",
    "recalibrate",
    "REGISTERED_QUANTILES",
    "reliability_diagram",
    "render_belief_flip",
    "resolve",
    "score_fraction",
    "score_t1",
    "sequential_evidence",
    "skill_score",
]
