"""M10 as an instrument: the power plan, the MDE and the resolution ratio, before the run.

The arithmetic lives in `stats.power`. This is the part that plugs into a preflight, so the
question "can this experiment answer the question" is asked at the point where the answer is still
free to act on, and the answer arrives as Evidence with the five standard calculators beside it as
baselines.

The division of labour between M10 and M5 is worth stating, because both of them are about power
and only one of them refuses. M10 computes what a design can see, before anything runs, and
reports it: a design that cannot see the effect is a fact about the design and Evidence is the
right carrier. M5 adjudicates a null after the fact, and there a missing control is a refusal
because the alternative is publishing a result that cannot be interpreted. Plan, then gate.

`resolve_row` is the leaderboard case. When `q = N/N* < 1` it returns a refusal rather than a
ranking, because a rank produced by a comparison that could not separate the two systems is a coin
flip with a decimal point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.controls._base import ControlInstrument
from reward_lens.stats.power import (
    CALCULATORS,
    TARGET_POWER,
    PairedBinaryDesign,
    PowerPlan,
    Resolution,
    plan,
)

#: The five standard calculators, as baseline ids. All five are computed and reported, because the
#: finding this instrument carries is about the gap between them and the simulation.
CALCULATOR_BASELINES: tuple[BaselineID, ...] = tuple(f"baseline.{name}" for name in CALCULATORS)


POWER_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a power calculation is a statement about a design and it is made before any run exists, "
        "so no regime of a run can violate it. What can be wrong is the design's own assumption "
        "about how much its n is worth, and that enters as the `ess` argument rather than as a "
        "precondition nobody checked."
    ),
)


# ---------------------------------------------------------------------------
# The estimands this simulator does not cover, as objects rather than as a sentence
# ---------------------------------------------------------------------------
#
# `deviations` said, and had said since the instrument shipped, that a continuous or ordinal
# outcome "needs its own simulator and this instrument declines rather than approximating one".
# Nothing declined. `deviations` is a declaration string read by the documentation catalogue, and
# the only thing standing between a caller and an answer computed by the wrong simulator was the
# `PairedBinaryDesign | None` annotation on `__init__`, which stops a type checker and does not stop
# a call. This project's own primary estimand is continuous, so the caller most likely to walk into
# it was this project.
#
# Two things were needed and neither existed: a way for a caller to *state* a continuous or ordinal
# estimand, and a refusal that fires on it. The design objects below are the first. They deliberately
# hold only what a power calculation for such an outcome would need, so that the day someone writes
# the simulator the input type is already registered and already carries the fields; today every one
# of them refuses.


@dataclass(frozen=True)
class ContinuousDesign:
    """Two arms compared on an outcome measured on a continuous scale.

    `sd` is the within-arm standard deviation on that scale and `rho` the per-item correlation
    between the arms, the same role it plays in `PairedBinaryDesign`. `units` is carried because a
    continuous effect with no unit is the case `UNIT_MISMATCH` exists for, and a design that cannot
    say what its numbers are in should not be silently powered.

    No power number comes out of this object today. `PowerAndMDE` refuses it, by design and not by
    omission: the simulator underneath is exact for the paired binary test and approximating a
    continuous outcome with it would return a confident number computed against the wrong null.
    """

    n: int
    mean_a: float
    mean_b: float
    sd: float
    rho: float = 0.0
    units: str = ""

    #: What kind of estimand this is, for the refusal to name. Not a free-text field.
    estimand_kind: str = "continuous"

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1; got {self.n}")
        if not self.sd > 0.0:
            raise ValueError(f"sd must be positive; got {self.sd}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must lie in [-1, 1]; got {self.rho}")

    @property
    def delta(self) -> float:
        """The effect on the outcome's own scale."""
        return self.mean_b - self.mean_a


@dataclass(frozen=True)
class OrdinalDesign:
    """Two arms compared on an outcome with `n_levels` ordered categories.

    Separated from `ContinuousDesign` because the test that would be run is a different one, a rank
    test rather than a mean difference, and a simulator for one is not a simulator for the other.
    Refused today for the same reason and by the same path.
    """

    n: int
    n_levels: int
    mean_shift: float
    rho: float = 0.0

    #: What kind of estimand this is, for the refusal to name.
    estimand_kind: str = "ordinal"

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1; got {self.n}")
        if self.n_levels < 2:
            raise ValueError(f"an ordinal outcome needs at least two levels; got {self.n_levels}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must lie in [-1, 1]; got {self.rho}")


#: Any design object this instrument accepts. The union is documentation; the runtime check below
#: is deliberately not written against it, because a whitelist of the three types somebody thought
#: of is how the fourth one reaches the simulator.
SupportedDesign = PairedBinaryDesign | ContinuousDesign | OrdinalDesign

#: The reason the refusal below carries, hoisted so that ratifying a different one is a single edit
#: rather than a sweep.
#:
#: **This is an open decision and it belongs to the maintainer, not to this module.** The condition
#: is "you handed this instrument an estimand its simulator does not compute in", and none of the
#: seventeen reasons was written for it. `ACCESS_INSUFFICIENT` is used here because it is the one
#: whose test the condition passes: its remedy is answerable where the reader is standing, by
#: bringing a different instrument or by stating a dichotomised outcome and saying so. There is
#: precedent in this same file, where `design=None` refuses under the same reason and is likewise
#: not an access-matrix failure; `requires` on this instrument is empty.
#:
#: The two runners-up and why they lost. `ENVELOPE_VIOLATED` contradicts `POWER_ENVELOPE`, which is
#: `unconditional=True` and says in its own justification that no regime can violate it.
#: `QUANTITY_UNDEFINED` is for the question that applies nowhere, and the power of a continuous
#: comparison is perfectly well defined; it simply is not defined *here*.
#:
#: An eighteenth member would fit better than any of the three. It is not taken here: the enum's own
#: docstring records that both previous amendments were "ratified by the maintainer rather than
#: taken by a builder or an integrator", and the count is pinned at seventeen by
#: `tests/acceptance/test_w1_kernel.py` and `tests/acceptance/test_w5_docs_catalogue.py`. See
#: `chain/repair/proofs/BLK-031/ERRATUM-eighteenth-refusal-reason.md`.
CONTINUOUS_REFUSAL_REASON = RefusalReason.ACCESS_INSUFFICIENT


def resolve_row(
    instrument: str,
    resolution: Resolution,
    *,
    observed_gap: float | None = None,
    mde: float | None = None,
) -> Any:
    """A leaderboard row's verdict, or a refusal saying it is not resolved.

    Returns the `Resolution` when `q >= 1`. Below 1 it returns a `Refusal`, because the row has
    not been measured: the sample could not have separated the two systems at the target power, so
    whichever one came out ahead came out ahead of nothing.

    The reason is `BELOW_LOD`. §4.7's limit of detection is `3.3 sigma_blank / S` and the minimum
    detectable effect is the same construction with the sampling standard deviation in place of
    the blank's, so an unresolved row is an effect below the design's own detection limit. That
    reuse is a judgement rather than a reading of the specification, and it is recorded in this
    package's report; the alternative would be a sixteenth refusal reason.
    """
    if resolution.resolved:
        return resolution
    detail = (
        f"q = N/N* = {resolution.q:.3f} at {resolution.target_power:.0%} power "
        f"({resolution.n:,.0f} {resolution.basis} against {resolution.n_star:,.0f} needed), so "
        f"this row is not resolved"
    )
    if observed_gap is not None:
        detail += f". The observed gap is {observed_gap:+.4g}"
        if mde is not None and np.isfinite(mde):
            detail += f" and the smallest detectable one is {mde:.4g}"
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.BELOW_LOD,
        detail=detail,
        remedy=(
            f"raise the sample to {resolution.n_star:,.0f} {resolution.basis} and re-run, or "
            f"publish the row as unresolved. Reporting an ordering here reports the noise. "
            f"`stats.power.plan` gives the n; `stats.power.resolution_from_lineage` gives the "
            f"effective n when the items are expansions of a smaller seed set."
        ),
        statistics={
            "q": resolution.q,
            "n": resolution.n,
            "n_star": resolution.n_star,
            "target_power": resolution.target_power,
            "observed_gap": observed_gap,
            "mde": mde,
        },
    )


class PowerAndMDE(ControlInstrument):
    """M10. Simulated power, the minimum detectable effect and `q = N/N*` for one design.

    Every number is simulated against the test that will actually be run. The five standard
    calculators are computed too and reported as this instrument's baselines, which is the honest
    place for them: they are the comparators, and on close paired comparisons three of them come
    out roughly 2x wrong in the expensive direction.
    """

    name = "PowerAndMDE"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "M10"
    deviations = (
        "the simulator covers the paired binary design (two systems scored right or wrong on the "
        "same items). A continuous or ordinal outcome needs its own simulator, and `compute()` "
        "returns a Refusal naming the estimand's kind rather than approximating one. State the "
        "outcome as a `ContinuousDesign` or an `OrdinalDesign` and the refusal is what comes back; "
        "the sentence and the runtime behaviour are the same claim",
        "`rho` is an input. It is measurable on a pilot for nothing, and planning at rho = 0 is a "
        "different experiment rather than a conservative version of this one",
    )

    quantity = "study.power"
    #: A power calculation before the run needs access to nothing, which is the entire argument for
    #: doing it first. The declaration is empty on purpose rather than by omission, and the field is
    #: `requires` because that is the name section 4.2 gives the access matrix. An empty matrix and a
    #: matrix under the wrong name read the same from `declared_access` and mean opposite things.
    requires: dict[Component, Access] = {}
    substrates = frozenset(
        {
            Substrate.NEURAL_SCALAR,
            Substrate.NEURAL_GEN,
            Substrate.PROGRAM,
            Substrate.PROCEDURAL,
            Substrate.HUMAN,
            Substrate.COMPOSITE,
        }
    )
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = POWER_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = CALCULATOR_BASELINES
    rung = 2

    def __init__(
        self,
        design: SupportedDesign | None = None,
        *,
        target_power: float = TARGET_POWER,
        replicates: int = 8_000,
        seed: int = 0,
        ess: float | None = None,
        with_calculators: bool = True,
    ) -> None:
        self.design = design
        self.target_power = float(target_power)
        self.replicates = int(replicates)
        self.seed = int(seed)
        self.ess = ess
        self.with_calculators = bool(with_calculators)

    def compute(self) -> Any:
        if self.design is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no design was supplied, so there is nothing to power",
                remedy=(
                    "pass `design=PairedBinaryDesign(n=..., accuracy_a=..., accuracy_b=..., "
                    "rho=...)`. A pilot of fifty items gives you all four."
                ),
            )
        if not isinstance(self.design, PairedBinaryDesign):
            return self._refuse_estimand(self.design)
        return plan(
            self.design,
            target_power=self.target_power,
            replicates=self.replicates,
            seed=self.seed,
            with_calculators=self.with_calculators,
            ess=self.ess,
        )

    def _refuse_estimand(self, design: Any) -> Refusal:
        """Refuse a design this simulator does not cover, naming the estimand's kind.

        Written against `not isinstance(design, PairedBinaryDesign)` at the call site rather than
        against a tuple of the unsupported types, so a design type nobody has written yet refuses
        by default instead of falling through to a simulator that cannot represent it. That is the
        direction the failure has to point: the cost of refusing a design that would have worked is
        a message, and the cost of powering one that does not is a number nobody can tell is wrong.
        """
        kind = str(getattr(design, "estimand_kind", "") or "unrecognised")
        type_name = type(design).__name__
        return Refusal(
            instrument=self.name,
            reason=CONTINUOUS_REFUSAL_REASON,
            detail=(
                f"this instrument simulates the paired binary test, and the design supplied is "
                f"{type_name}, whose estimand is {kind}. A {kind} outcome is powered against a "
                f"different test with a different null, so approximating it here would return a "
                f"confident number computed against the wrong one. Declining is the deviation this "
                f"instrument has always declared; this is where it happens."
            ),
            remedy=(
                f"either supply a `PairedBinaryDesign`, stating the dichotomisation you applied "
                f"and why, or power the {kind} estimand with an instrument built for it. There is "
                f"no such instrument in this package today: `stats.power` is exact for the paired "
                f"binary test and has no {kind} simulator, which is a gap rather than an oversight "
                f"and is registered as one."
            ),
            statistics={
                "estimand_kind": kind,
                "design_type": type_name,
                "n": float(getattr(design, "n", float("nan"))),
            },
        )

    def payload(self, computed: PowerPlan) -> dict[str, Any]:
        return {
            "power": computed.power,
            "power_mc_se": computed.power_mc_se,
            "n_star": computed.n_star,
            "mde": computed.mde,
            "q": computed.resolution.q,
            "resolved": computed.resolution.resolved,
            "target_power": computed.target_power,
            "replicates": computed.replicates,
            "validated_against": computed.validated_against,
            "baselines": {
                f"baseline.{name}": float(check.n_star)
                for name, check in computed.calculators.items()
            },
        }


__all__ = [
    "CALCULATOR_BASELINES",
    "CONTINUOUS_REFUSAL_REASON",
    "POWER_ENVELOPE",
    "ContinuousDesign",
    "OrdinalDesign",
    "PowerAndMDE",
    "SupportedDesign",
    "resolve_row",
]
