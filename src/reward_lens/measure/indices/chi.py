"""A12 χ: the susceptibility spectrum ``χ_i = Cov_0(f_i, r)`` (Appendix A12).

Formal definition: Appendix A12. For a feature bank ``{f_i}`` evaluated on base-policy samples,
``χ_i = Cov_0(f_i, r)`` is the covariance of feature ``i`` with the reward under the base policy
``π_0``. This is the fluctuation-dissipation identity for the exponential tilt family
``π_λ ∝ π_0 exp(λ r)``: to first order, ``d E_λ[f_i]/dλ |_{λ=0} = Cov_0(f_i, r)``, so ``χ_i`` is the
predicted initial drift of feature ``i`` once the policy starts optimizing against the reward. A
feature with ``χ_i > 0`` will be pushed up early in optimization.

The predicted hack modes are the features the reward rewards but the gold objective does not:
``χ_i > 0`` with ``Cov_0(f_i, gold) ≤ 0``. Those are the directions optimization will inflate while
the true objective is flat or falling, which is the operational definition of a reward hack this
index forecasts before any optimization is run.

Deviations from A12: the second-order term ``d²E_λ[f]/dλ² = κ_3(f, r, r)`` that A12 also names is not
computed here; χ is the first-order (zeroth-order-in-λ covariance) susceptibility, which is the drift
predictor the cards use. The features come from a ``FeatureBank`` (the concept layer's contract); with
the synthetic ``LinearFeatureBank`` the planted ``Cov(feature, reward)`` is recovered exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import (
    FeatureBank,
    final_activations,
    load_default_bank,
    reward_scores,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def susceptibility(features: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """The susceptibility spectrum ``χ_i = Cov_0(f_i, r)`` (Appendix A12).

    ``features`` is ``(n, k)`` and ``reward`` is ``(n,)``; returns the ``(k,)`` vector of covariances
    between each feature and the reward, using the population (biased) covariance so a planted
    ``Cov(f_i, r)`` is recovered on the nose as ``n`` grows. Centering both sides makes χ invariant to
    the arbitrary additive origins of the features and the reward.
    """
    f = np.asarray(features, dtype=np.float64)
    r = np.asarray(reward, dtype=np.float64).ravel()
    if f.ndim == 1:
        f = f[:, None]
    fc = f - f.mean(axis=0, keepdims=True)
    rc = r - r.mean()
    return (fc * rc[:, None]).mean(axis=0)


def predicted_hack_modes(
    chi: np.ndarray,
    chi_gold: np.ndarray,
    *,
    tol: float = 0.0,
) -> np.ndarray:
    """Boolean mask of predicted hack features: ``χ_i > tol`` and ``Cov_0(f_i, gold) ≤ tol`` (A12).

    A hack mode is a feature the reward pulls up (``χ_i`` positive) while the gold objective does not
    reward it (``χ_gold_i`` non-positive). ``tol`` sets a dead band around zero so noise near the
    boundary does not flip the flag; the default ``0.0`` is the literal definition.
    """
    chi = np.asarray(chi, dtype=np.float64).ravel()
    chi_gold = np.asarray(chi_gold, dtype=np.float64).ravel()
    return (chi > tol) & (chi_gold <= tol)


def _chi_shuffle_null(features: np.ndarray, reward: np.ndarray, seed: int, n: int) -> dict:
    """Per-feature label-permutation null for χ (the noise floor for "feature carries reward").

    Uses ``stats.nulls.shuffle_null`` per feature, permuting the reward against the fixed feature
    column so the null spectrum is the χ a decoupled feature/reward pairing would show. Returns the
    per-feature p-values and the shared null mean magnitude.
    """
    from reward_lens.stats.nulls import shuffle_null

    f = np.asarray(features, dtype=np.float64)
    r = np.asarray(reward, dtype=np.float64).ravel()
    if f.ndim == 1:
        f = f[:, None]

    def _cov(vals: np.ndarray, labels: np.ndarray) -> float:
        v = vals - vals.mean()
        lab = labels - labels.mean()
        return float((v * lab).mean())

    p_values: list[float] = []
    null_means: list[float] = []
    for i in range(f.shape[1]):
        res = shuffle_null(f[:, i], r, _cov, n=n, seed=seed + i)
        p_values.append(res["p_value"])
        null_means.append(abs(res["null_mean"]))
    return {
        "p_values": p_values,
        "null_mean_abs": float(np.mean(null_means)) if null_means else 0.0,
    }


class Chi(BaseObservable):
    """A12 susceptibility spectrum ``χ_i = Cov_0(f_i, r)`` over a feature bank on base-policy samples.

    Requires activation capture, scores, and a linear readout. Captures the base-policy activations,
    turns them into feature values through the feature bank (injected, else the concept layer's default,
    else a graceful no-bank report), scores the same samples, and reports the χ spectrum with a
    per-feature shuffle null. When a ``gold`` signal is supplied it also scores the gold objective and
    flags the predicted hack modes (``χ_i > 0`` with ``χ_gold_i ≤ 0``).

    Gauge is INVARIANT: χ is a within-signal spectrum relative to a fixed feature bank. Comparing
    individual ``χ_i`` across signals requires the same bank read in a shared frame; that is a
    cross-signal comparison the caller must set up, and is noted as a deviation.
    """

    name = "Chi"
    version = "1.0"
    requires = Capability.ACTIVATIONS | Capability.SCORES | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A12"
    deviations = (
        "first-order susceptibility only; the second-order term kappa_3(f, r, r) is not computed",
        "spectrum is relative to the supplied feature bank; cross-signal comparison of individual "
        "chi_i needs the same bank read in a shared frame",
    )

    def __init__(
        self,
        feature_bank: FeatureBank | None = None,
        *,
        gold: Any = None,
        null_draws: int = 2000,
        seed: int = 0,
    ) -> None:
        self.feature_bank = feature_bank
        self.gold = gold
        self.null_draws = int(null_draws)
        self.seed = int(seed)

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        bank = self.feature_bank or load_default_bank(signal)
        acts = final_activations(signal, ctx.view, readout=ctx.readout)
        reward = reward_scores(signal, ctx.view, ctx.readout)

        if bank is None:
            return ctx.emit(
                {
                    "chi": [],
                    "feature_names": [],
                    "note": "no feature bank available (concepts absent and none injected)",
                    "n_items": int(acts.shape[0]),
                },
                uncertainty=Uncertainty(n=int(acts.shape[0]), method="none"),
            )

        features = bank.featurize(acts)
        chi = susceptibility(features, reward)
        null = _chi_shuffle_null(features, reward, self.seed, self.null_draws)

        payload: dict[str, Any] = {
            "chi": chi.tolist(),
            "feature_names": list(getattr(bank, "names", tuple())),
            "null_p_values": null["p_values"],
            "null_mean_abs": null["null_mean_abs"],
            "n_items": int(acts.shape[0]),
            "n_features": int(chi.size),
        }

        if self.gold is not None:
            gold_reward = reward_scores(self.gold, ctx.view, ctx.readout)
            chi_gold = susceptibility(features, gold_reward)
            hacks = predicted_hack_modes(chi, chi_gold)
            payload["chi_gold"] = chi_gold.tolist()
            payload["predicted_hack_modes"] = hacks.tolist()
            payload["n_predicted_hacks"] = int(hacks.sum())

        return ctx.emit(payload, uncertainty=Uncertainty(n=int(acts.shape[0]), method="none"))


__all__ = ["susceptibility", "predicted_hack_modes", "Chi"]
