"""A3 TeacherCompatibility: the induced reward variance ``w_rᵀ Σ_π w_r`` (Appendix A3).

Formal definition: Appendix A3. ``TC(rm, π) = Var_{y∼π}(w_rᵀ h(y)) = w_rᵀ Σ_π w_r``, the variance of
the reward projection over the on-policy activation distribution, decomposable by layer and by
spectral mode. This is Razin's teacher-induced variance (faithful_to Razin 2503.15477): a reward
model whose scores barely move across a policy's samples is a poor teacher for that policy no matter
how accurate its ranking, because RLHF's first-order learning signal is proportional to the reward
variance the policy actually sees. It is also L1's zeroth-order susceptibility and equals the ``f = r``
diagonal of the χ response identity (A12): ``TC = Cov_0(r, r) = Var(r)``.

Deviations from A3: none in the scalar. The layer decomposition reads each captured residual site's
reward projection variance, and the spectral decomposition splits ``w_rᵀ Σ w_r`` over the eigenbasis
of ``Σ`` (an exact, basis-free split of the same total), which Appendix A3 names "decomposable by
layer/feature" without fixing the feature basis; the eigenbasis is the canonical choice and is noted
here as the concrete reading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.types import Capability, GaugeStatus, Site
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import (
    final_activations,
    reward_vector,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def teacher_compatibility(w_r: np.ndarray, activations: np.ndarray) -> float:
    """The induced reward variance ``TC = w_rᵀ Σ_π w_r`` (Appendix A3).

    Equivalently ``Var_y(w_rᵀ h(y))``: project every activation onto the reward direction and take the
    population variance of the resulting scores. The two forms agree exactly because the variance of a
    linear functional is that functional's quadratic form against the covariance, which is the identity
    the test asserts. ``activations`` is ``(n, d)`` and ``w_r`` is ``(d,)``.
    """
    a = np.asarray(activations, dtype=np.float64)
    w = np.asarray(w_r, dtype=np.float64).ravel()
    proj = a @ w
    return float(np.var(proj, ddof=0))


def teacher_compatibility_spectral(
    w_r: np.ndarray, activations: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Split ``w_rᵀ Σ w_r`` over the eigenbasis of ``Σ`` (Appendix A3, the feature decomposition).

    Writes the total as ``Σ_k λ_k (w_r · u_k)²`` for eigenpairs ``(λ_k, u_k)`` of the on-policy
    covariance ``Σ``. Each term is the contribution of one principal direction of the activation
    distribution to the reward variance, and they sum to ``teacher_compatibility`` exactly. For an
    independent-feature (diagonal ``Σ``) distribution the eigenbasis is the coordinate basis and the
    contributions reduce to ``σ_ii · w_i²``, which is the closed form the test checks. Returns
    ``(total, contributions_desc, eigenvalues_desc)`` sorted by descending contribution.
    """
    a = np.asarray(activations, dtype=np.float64)
    w = np.asarray(w_r, dtype=np.float64).ravel()
    cov = np.cov(a, rowvar=False, bias=True)
    cov = np.atleast_2d(cov)
    evals, evecs = np.linalg.eigh(cov)
    loadings = evecs.T @ w
    contribs = evals * loadings**2
    order = np.argsort(contribs)[::-1]
    return float(contribs.sum()), contribs[order], evals[order]


def teacher_compatibility_by_layer(
    w_r: np.ndarray, activations_by_site: dict[Any, np.ndarray]
) -> dict[str, float]:
    """Per-site induced variance (Appendix A3, the layer decomposition).

    Applies ``teacher_compatibility`` at each captured residual site with the same reward direction,
    tracing where across depth the policy's samples spread the reward. The keys are the string forms of
    the sites so the payload is JSON-clean.
    """
    return {
        str(site): teacher_compatibility(w_r, acts) for site, acts in activations_by_site.items()
    }


class TeacherCompatibility(BaseObservable):
    """A3 induced reward variance ``w_rᵀ Σ_π w_r``, layer- and spectrum-decomposable.

    Requires activation capture and a linear readout. Reads ``w_r`` off the readout, captures the
    on-policy final-token activations at the readout site (and at every residual layer when the signal
    reports its depth, for the layer decomposition), and reports the total with its spectral split.
    Gauge is INVARIANT: it is a single-signal functional, not a cross-signal comparison. It carries
    reward-scale² units, so a cross-model magnitude comparison first needs the two rewards on a common
    scale; that caveat is stated as a deviation rather than silently ignored.
    """

    name = "TeacherCompatibility"
    version = "1.0"
    requires = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A3"
    deviations = (
        "carries reward-scale-squared units; cross-model magnitude comparison requires the two "
        "rewards on a common scale (INVARIANT is with respect to representation rotation, not "
        "reward rescaling)",
        "the feature decomposition is over the eigenbasis of the on-policy covariance, the "
        "canonical basis-free reading of A3's 'decomposable by feature'",
    )

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        w_r = reward_vector(signal, ctx.readout)
        site = None  # the readout's own site
        acts = final_activations(signal, ctx.view, site, readout=ctx.readout)

        total = teacher_compatibility(w_r, acts)
        _, contribs, evals = teacher_compatibility_spectral(w_r, acts)

        by_layer: dict[str, float] = {}
        n_layers = getattr(signal.meta, "n_layers", None)
        if n_layers:
            sites = {Site(layer, "resid_post"): None for layer in range(int(n_layers))}
            captured: dict[Any, np.ndarray] = {}
            for s in sites:
                try:
                    captured[s] = final_activations(signal, ctx.view, s, readout=ctx.readout)
                except Exception:  # noqa: BLE001 - a site the runtime cannot read is simply skipped
                    continue
            if captured:
                by_layer = teacher_compatibility_by_layer(w_r, captured)

        payload = {
            "teacher_compatibility": total,
            "spectral_contributions": contribs.tolist(),
            "spectral_eigenvalues": evals.tolist(),
            "by_layer": by_layer,
            "d_model": int(w_r.size),
            "n_items": int(acts.shape[0]),
        }
        return ctx.emit(payload, uncertainty=Uncertainty(n=int(acts.shape[0]), method="none"))


__all__ = [
    "teacher_compatibility",
    "teacher_compatibility_spectral",
    "teacher_compatibility_by_layer",
    "TeacherCompatibility",
]
