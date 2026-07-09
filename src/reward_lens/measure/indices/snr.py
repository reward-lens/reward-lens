"""Robustness SNR: reward signal against perturbation noise (Appendix A, robustness diagnostic).

The robustness signal-to-noise ratio asks whether a reward's differences between conditions survive the
noise it shows under meaning-preserving perturbation. It is v1's ``PromptSNR`` as an index: cluster the
samples so that within a cluster the inputs are paraphrases (the perturbation the reward should ignore)
and across clusters they are genuinely different (the signal the reward should track). Then

    ``SNR = Var(cluster means) / mean(within-cluster variance)``,

the between-group reward variance the reward means carry over the within-group variance a paraphrase
induces. A high SNR means the reward's between-condition ordering is stable under perturbation; an SNR
near or below one means paraphrase noise swamps the signal, and any ranking read off the reward is
fragile.

This module has no single Appendix A letter; it is the robustness statistic the cards and the
adversarial-robustness science (S13) consume. Deviation: the pure function is the variance-ratio
arithmetic on supplied grouped scores; the production path scores paraphrase clusters through the
signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import reward_scores

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def robustness_snr(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Between-group over within-group reward variance (the robustness SNR).

    ``values`` are the per-sample rewards; ``groups`` labels each sample's paraphrase cluster. Returns
    the between-group variance (of the cluster means), the mean within-group variance (the paraphrase
    noise), and their ratio ``snr``. Clusters of size one contribute no within-group variance and are
    excluded from the noise estimate. A zero noise floor yields ``inf`` (perfectly stable), which is the
    honest reading, not a divide-by-zero error.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    labels = np.unique(g)
    means = np.array([v[g == lab].mean() for lab in labels], dtype=np.float64)
    within = [np.var(v[g == lab], ddof=0) for lab in labels if np.sum(g == lab) > 1]
    between_var = float(np.var(means, ddof=0))
    within_var = float(np.mean(within)) if within else 0.0
    if within_var == 0:
        snr = float("inf") if between_var > 0 else float("nan")
    else:
        snr = between_var / within_var
    return {
        "snr": snr,
        "between_var": between_var,
        "within_var": within_var,
        "n_groups": labels.size,
    }


class RobustnessSNR(BaseObservable):
    """Robustness SNR: between-condition reward signal over within-condition paraphrase noise.

    Requires scores. The paraphrase-cluster labels are injected (or read from the view's group ids);
    the production path scores paraphrase clusters through the signal. Gauge is INVARIANT: a
    variance ratio is scale-free.
    """

    name = "RobustnessSNR"
    version = "1.0"
    requires = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (
        "no single Appendix A letter; this is v1's PromptSNR as an index (robustness diagnostic for "
        "S13). The pure function is the variance-ratio arithmetic on grouped scores.",
    )

    def __init__(self, groups: Any = None) -> None:
        self.groups = groups

    def measure(self, ctx: Context) -> "Evidence":
        if self.groups is None:
            return ctx.emit(
                {"note": "robustness_snr needs paraphrase-cluster group labels; none injected"},
                uncertainty=Uncertainty(method="none"),
            )
        reward = reward_scores(ctx.signal, ctx.view, ctx.readout)
        report = robustness_snr(reward, np.asarray(self.groups))
        return ctx.emit(report, uncertainty=Uncertainty(n=int(reward.size), method="none"))


__all__ = ["robustness_snr", "RobustnessSNR"]
