"""``ConflictMatrix`` (E09): the geometry of competing reward terms (section 2.8).

Different quality axes (helpfulness, verbosity, formatting, ...) each define a direction in activation
space, estimated as the mean chosen-minus-rejected difference for that axis. Their pairwise cosines
say whether two axes cooperate (aligned), are independent (orthogonal), or pull against each other
(conflict). Conflicting reward terms are where monitorability degrades, because optimizing one term
degrades another the grader also cares about.

This ports v1's ``RewardConflictAnalyzer``: learn a direction per axis by mean difference, then read
the cosine matrix and classify each pair. The cosines are raw-coordinate (RAW_ONLY): they depend on
the residual-stream basis and are a single-model internal geometry. The view is grouped by each pair's
``axis``, so a diagnostic set with several axes yields the full inter-axis conflict matrix.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.errors import CapabilityError
from reward_lens.core.types import Capability, GaugeStatus, Site
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import capture_sites
from reward_lens.measure.battery.geometry import cosine_matrix

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


class ConflictMatrix(BaseObservable):
    """Inter-axis reward-term conflict geometry (E09).

    Requires activation capture and a linear readout. The view must span at least two axes. Marked
    RAW_ONLY because the term cosines are in raw residual-stream coordinates.
    """

    name = "ConflictMatrix"
    version = "1.0"
    requires = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.RAW_ONLY
    faithful_to = "E09 reward-term conflict geometry"
    deviations = (
        "term directions are the mean chosen-minus-rejected difference per axis (unnormalized, as "
        "v1 learned them); cosines are RAW_ONLY (basis-dependent), meaningful within one model",
    )

    def measure(self, ctx: Context) -> "Evidence":
        import torch

        signal = ctx.signal
        n_layers = int(signal.meta.n_layers)
        site = Site(n_layers - 1, "resid_post")

        by_axis: dict[str, list] = defaultdict(list)
        for pair in ctx.view:
            axis = getattr(pair, "axis", "default")
            by_axis[axis].append(pair)
        axes = sorted(by_axis)
        if len(axes) < 2:
            raise CapabilityError(
                f"ConflictMatrix needs pairs spanning >=2 axes; the view has {len(axes)}. "
                f"Pass a multi-axis diagnostic view."
            )

        directions = []
        for axis in axes:
            pairs = by_axis[axis]
            chosen = [(p.prompt_text, p.chosen.text) for p in pairs]
            rejected = [(p.prompt_text, p.rejected.text) for p in pairs]
            hc = capture_sites(signal, chosen, (site,))[site].to(torch.float32)
            hr = capture_sites(signal, rejected, (site,))[site].to(torch.float32)
            directions.append((hc - hr).mean(dim=0).cpu().numpy())
        vectors = np.stack(directions, axis=0)
        cosines = cosine_matrix(vectors)
        off = cosines[~np.eye(len(axes), dtype=bool)]

        payload = {
            "axes": axes,
            "cosine_matrix": cosines.tolist(),
            "mean_offdiagonal_cosine": float(np.mean(off)),
            "min_cosine": float(np.min(off)),
            "n_conflicting_pairs": int(np.sum(off < -0.3) // 2),
            "n_axes": len(axes),
        }
        return ctx.emit(payload)


__all__ = ["ConflictMatrix"]
