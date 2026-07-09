"""``ConceptDoseResponse`` (E08): concept directions, reward alignment, and dose response (section 2.8).

A concept (verbosity, confidence, formality) is a direction in activation space, estimated as the
mean difference between activations that have the concept and activations that do not. Two questions
follow. How aligned is the concept with the reward direction, that is, does having the concept push
reward up or down? And what is the causal dose response, how does the reward move as you steer the
activation along the concept direction? A concept that both aligns with reward and moves it causally
is a reward-hacking lever.

This ports v1's ``ConceptExtractor`` through the canonical concept functions in
:mod:`reward_lens.concepts.vectors`: the direction is the unit-normalized mean difference, the
alignment is its cosine with ``w_r``, and the dose response is the least-squares slope of reward
against steering strength. The concept-pair activations come from the pairs in the view (chosen is the
positive side of the concept, rejected the negative), so the same diagnostic data drives it. The
alignment and dose are raw-coordinate quantities (RAW_ONLY): they depend on the residual-stream basis
and are meaningful within one signal, not across signals without a frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.concepts.vectors import concept_direction, dose_response_slope, reward_alignment
from reward_lens.core.types import Capability, GaugeStatus, Site
from reward_lens.interventions.patch import ResidualAddPatch, run_patched_scores
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import capture_sites, pair_sides, reward_direction

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


class ConceptDoseResponse(BaseObservable):
    """Concept direction, its reward alignment, and its causal dose response (E08).

    Requires activation capture and a linear readout. The concept is read from the pairs in the view
    (chosen positive, rejected negative). Marked RAW_ONLY because the direction, its alignment, and the
    dose are all in raw residual-stream coordinates.
    """

    name = "ConceptDoseResponse"
    version = "1.0"
    requires = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.RAW_ONLY
    faithful_to = "E08 concept dose response"
    deviations = (
        "concept direction is the unit-normalized mean difference over the view's pairs; the dose "
        "response steers the final residual and reads the reward slope; RAW_ONLY (basis-dependent)",
    )

    def measure(self, ctx: Context) -> "Evidence":
        import torch

        signal = ctx.signal
        n_layers = int(signal.meta.n_layers)
        readout = ctx.readout
        w_r = reward_direction(signal, readout)
        concept_site = Site(n_layers - 1, "resid_post")

        chosen, rejected = pair_sides(ctx.view)
        pos = capture_sites(signal, chosen, (concept_site,))[concept_site]
        neg = capture_sites(signal, rejected, (concept_site,))[concept_site]
        direction = concept_direction(pos, neg)  # unit vector (d_model,)
        alignment = reward_alignment(direction, w_r)

        # Dose response: steer the final token's residual along the concept direction and read the
        # reward, matching v1's last-token intervene_on_concept. The baseline (dose 0) recovers the
        # clean reward.
        doses = np.linspace(-2.0, 2.0, 5)
        base_items = chosen
        dir_t = torch.tensor(direction, dtype=torch.float32)
        rewards = []
        for dose in doses:
            reward = self._steered_reward(
                signal, base_items, concept_site, dir_t * float(dose), readout
            )
            rewards.append(float(np.mean(reward)))
        slope = dose_response_slope(doses, np.array(rewards))

        payload = {
            "reward_alignment": alignment,
            "dose_response_slope": slope,
            "doses": doses.tolist(),
            "mean_reward_at_dose": rewards,
            "concept_norm": float(np.linalg.norm(direction)),
            "n_pairs": len(chosen),
        }
        return ctx.emit(payload)

    @staticmethod
    def _steered_reward(signal, items, site: Site, vector, readout: str) -> np.ndarray:
        """Reward after adding ``vector`` at the final token of ``site``'s layer output.

        Left padding aligns the final token at the last column for every row, so a delta that is zero
        everywhere except the last column steers exactly the final-token residual, which is where v1's
        ``intervene_on_concept`` added the concept vector.
        """
        import torch

        tokenized = [signal.tokenize(it) for it in items]
        max_t = max(len(t.input_ids) for t in tokenized)
        delta = torch.zeros((1, max_t, vector.shape[-1]), dtype=torch.float32)
        delta[0, -1, :] = vector
        patch = ResidualAddPatch(site=site, delta=delta)
        return run_patched_scores(signal, patch.compile(signal), items, readout)


__all__ = ["ConceptDoseResponse"]
