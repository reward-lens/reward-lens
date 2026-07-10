"""Atlas meta-studies: the two population-scale laws (universality T13, performativity T11).

These are meta-studies over the Atlas rather than single-model sciences. `universality` computes the
value-convergence-excess (VCE) that scoreboard row T13 tracks; `performative` measures the audit
half-life that row T11 tracks. Each is a frozen study with a calibrated arm on synthetic/organism
constructions and an explicitly gated real-population follow-on.
"""

from studies.atlas_meta.performative import build_spec as build_performative_spec
from studies.atlas_meta.universality import build_spec as build_universality_spec

__all__ = ["build_universality_spec", "build_performative_spec"]
