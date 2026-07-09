"""A10 Dark reward: the fraction of reward variance through no named channel (Appendix A10).

Formal definition: Appendix A10. Dark reward is the fraction of ``Var(r)`` not causally mediated by any
named channel (criterion or feature). It is a card statistic: the reward variance a full accounting of
the intended criteria still cannot explain, the leakage capacity theory (S5) predicts grows with
``K/d_eff`` as the reward tries to carry more criteria than its effective dimension supports (A9).

Deviation from A10: the pure function measures the variance of ``r`` not linearly explained by the
named-channel contributions (``1 − R²`` of ``r`` on the channels), which is the observational reading;
the causal-mediation reading substitutes steering-measured channel contributions and is the production
path. The synthetic test plants ``r`` as named channels plus a known dark component and recovers its
fraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import reward_scores

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def dark_reward(reward: np.ndarray, named_contributions: np.ndarray) -> float:
    """Fraction of ``Var(r)`` not explained by the named channels: ``1 − R²`` (Appendix A10).

    Regresses the reward on the named-channel contributions (``n × K``) and returns one minus the
    fraction of variance explained. All variance captured by the channels gives ``0`` (nothing dark);
    a reward orthogonal to every channel gives ``1`` (entirely dark). ``reward`` is ``(n,)``. A
    constant term is included so the channels are not charged for the reward's mean.
    """
    r = np.asarray(reward, dtype=np.float64).ravel()
    c = np.asarray(named_contributions, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    var_r = float(np.var(r, ddof=0))
    if var_r == 0:
        return float("nan")
    design = np.column_stack([np.ones(c.shape[0]), c])
    coef, *_ = np.linalg.lstsq(design, r, rcond=None)
    resid = r - design @ coef
    return float(np.var(resid, ddof=0) / var_r)


class DarkReward(BaseObservable):
    """A10 fraction of reward variance mediated by no named channel.

    Requires scores. The named-channel contributions are injected (the concept/criterion layer supplies
    them in production; the causal reading uses steering-measured contributions). Reports the dark
    fraction. Gauge is INVARIANT: a variance fraction is scale-free.
    """

    name = "DarkReward"
    version = "1.0"
    requires = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A10"
    deviations = (
        "observational reading (1 - R^2 of r on the named channels); the causal-mediation reading "
        "uses steering-measured channel contributions and is the production path",
    )

    def __init__(self, named_contributions: np.ndarray | None = None) -> None:
        self.named_contributions = named_contributions

    def measure(self, ctx: Context) -> "Evidence":
        if self.named_contributions is None:
            return ctx.emit(
                {"note": "dark_reward needs named-channel contributions; none injected"},
                uncertainty=Uncertainty(method="none"),
            )
        reward = reward_scores(ctx.signal, ctx.view, ctx.readout)
        dark = dark_reward(reward, self.named_contributions)
        n_channels = int(np.atleast_2d(np.asarray(self.named_contributions).T).shape[0])
        return ctx.emit(
            {"dark_reward": dark, "explained_fraction": 1.0 - dark, "n_channels": n_channels},
            uncertainty=Uncertainty(n=int(reward.size), method="none"),
        )


__all__ = ["dark_reward", "DarkReward"]
