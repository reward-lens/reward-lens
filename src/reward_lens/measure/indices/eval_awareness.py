"""A17 Eval-awareness: does the reward recognize benchmark inputs (Appendix A17).

Formal definition: Appendix A17. The balanced accuracy of a probe discriminating benchmark-style from
organic inputs from the reward model's activations, plus the causal ``Δr`` from steering that direction
(does recognition inflate the score?). A reward that can tell a benchmark item from an organic one has a
handle an optimizer can pull, and if steering the recognition direction moves the reward, the reward is
partly scoring "this looks like a test" rather than the response itself (the grader eval-awareness
program, N5/S16).

Deviation from A17: the probe is a held-out linear discriminant (mean-difference direction fit on a
train split, evaluated on a test split) rather than a full calibrated classifier, so the balanced
accuracy is honest about generalization without a heavy dependency; the causal ``Δr`` from steering is
injected here and is the production path through the interventions subsystem. The label-permutation null
(``stats.nulls.shuffle_null``) gives the chance floor the accuracy must beat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import final_activations

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def _balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Balanced accuracy: the mean of per-class recalls (chance is 0.5 for two classes)."""
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    recalls = []
    for cls in (0, 1):
        mask = true == cls
        if np.any(mask):
            recalls.append(float(np.mean(pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else float("nan")


def eval_awareness_probe(
    activations: np.ndarray,
    is_benchmark: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Held-out balanced accuracy of a linear benchmark-vs-organic probe (Appendix A17).

    Splits the samples in half, fits a mean-difference direction ``μ_bench − μ_organic`` on the train
    half, thresholds projections at the midpoint of the class-mean projections, and reports balanced
    accuracy on the test half. Held-out evaluation is what keeps a random labelling near ``0.5`` instead
    of overfitting to noise. ``activations`` is ``(n, d)``; ``is_benchmark`` is a 0/1 label vector.
    Returns the balanced accuracy and the split sizes.
    """
    a = np.asarray(activations, dtype=np.float64)
    y = np.asarray(is_benchmark).ravel().astype(int)
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    tr, te = perm[:half], perm[half:]
    a_tr, y_tr, a_te, y_te = a[tr], y[tr], a[te], y[te]
    if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
        return {"balanced_accuracy": float("nan"), "n_train": tr.size, "n_test": te.size}
    direction = a_tr[y_tr == 1].mean(axis=0) - a_tr[y_tr == 0].mean(axis=0)
    proj_tr = a_tr @ direction
    threshold = 0.5 * (proj_tr[y_tr == 1].mean() + proj_tr[y_tr == 0].mean())
    pred_te = (a_te @ direction > threshold).astype(int)
    return {
        "balanced_accuracy": _balanced_accuracy(pred_te, y_te),
        "n_train": int(tr.size),
        "n_test": int(te.size),
    }


class EvalAwareness(BaseObservable):
    """A17 balanced accuracy of a benchmark-vs-organic probe on reward activations, with a null.

    Requires activations. Labels marking which inputs are benchmark-style are injected (the data plane
    supplies them in production); the causal ``Δr`` from steering the recognition direction is injected
    and is the production path. Reports the held-out balanced accuracy against a label-permutation null.
    Gauge is INVARIANT: balanced accuracy is a within-signal classification quality.
    """

    name = "EvalAwareness"
    version = "1.0"
    requires = Capability.ACTIVATIONS
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A17"
    deviations = (
        "probe is a held-out mean-difference linear discriminant, not a full calibrated classifier",
        "the causal delta-r from steering the recognition direction is injected and is the production "
        "path (interventions)",
    )

    def __init__(
        self,
        is_benchmark: np.ndarray | None = None,
        *,
        steering_delta_r: float | None = None,
        null_draws: int = 2000,
        seed: int = 0,
    ) -> None:
        self.is_benchmark = is_benchmark
        self.steering_delta_r = steering_delta_r
        self.null_draws = int(null_draws)
        self.seed = int(seed)

    def measure(self, ctx: Context) -> "Evidence":
        if self.is_benchmark is None:
            return ctx.emit(
                {"note": "eval_awareness needs benchmark/organic labels; none injected"},
                uncertainty=Uncertainty(method="none"),
            )
        acts = final_activations(ctx.signal, ctx.view, readout=ctx.readout)
        y = np.asarray(self.is_benchmark).ravel().astype(int)
        probe = eval_awareness_probe(acts, y, seed=self.seed)

        from reward_lens.stats.nulls import shuffle_null

        def _stat(vals: np.ndarray, labels: np.ndarray) -> float:
            return eval_awareness_probe(vals, labels, seed=self.seed)["balanced_accuracy"]

        null = shuffle_null(acts, y, _stat, n=min(self.null_draws, 500), seed=self.seed)

        payload = {
            "balanced_accuracy": probe["balanced_accuracy"],
            "null_mean": null["null_mean"],
            "null_p_value": null["p_value"],
            "steering_delta_r": self.steering_delta_r,
            "n_items": int(acts.shape[0]),
        }
        return ctx.emit(payload, uncertainty=Uncertainty(n=int(acts.shape[0]), method="none"))


__all__ = ["eval_awareness_probe", "EvalAwareness"]
