"""BLK-031 — the power planner declined continuous estimands in prose and not at runtime.

`PowerAndMDE.deviations` carried the sentence "A continuous or ordinal outcome needs its own
simulator and this instrument declines rather than approximating one". A declaration string is read
by a documentation catalogue and by nothing else. `compute()` could return exactly two refusals,
`ACCESS_INSUFFICIENT` for a missing design and `BELOW_LOD` from `resolve_row`, and neither of them
fires on the estimand's type. The practical block was a type annotation, which stops mypy and does
not stop a caller.

This design's primary estimand is continuous, so the caller most likely to hit the declining
sentence is this project.

Every test here asserts the **raise**, in the sense G16 gives it: the refusal is the result, and no
test in this file reads a power number out of an object that should have refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.reading import Refusal
from reward_lens.measure.controls.power import (
    CONTINUOUS_REFUSAL_REASON,
    ContinuousDesign,
    OrdinalDesign,
    PowerAndMDE,
)
from reward_lens.stats.power import PairedBinaryDesign


def _continuous() -> ContinuousDesign:
    """The shape of this design's own primary estimand: two arms on a continuous scale."""
    return ContinuousDesign(n=200, mean_a=0.0, mean_b=0.15, sd=1.0, rho=0.6, units="nats/token")


# ---------------------------------------------------------------------------
# The closure proof: a continuous estimand returns a Refusal naming its type
# ---------------------------------------------------------------------------


def test_a_continuous_estimand_returns_a_refusal_rather_than_a_number() -> None:
    out = PowerAndMDE(design=_continuous()).compute()
    assert isinstance(out, Refusal), f"expected a Refusal, got {type(out).__name__}: {out!r}"
    assert not hasattr(out, "power"), "a refusal must not carry a power number"


def test_the_refusal_names_the_estimand_type() -> None:
    """The closure proof's own wording: the refusal names the estimand type."""
    out = PowerAndMDE(design=_continuous()).compute()
    assert isinstance(out, Refusal)
    assert "continuous" in out.detail.lower(), out.detail
    assert "ContinuousDesign" in out.detail, out.detail
    assert out.statistics is not None
    assert out.statistics["estimand_kind"] == "continuous"


def test_the_refusal_carries_a_remedy_that_is_actionable() -> None:
    """A refusal with no remedy is a failure message. The remedy has to name what would work."""
    out = PowerAndMDE(design=_continuous()).compute()
    assert isinstance(out, Refusal)
    assert out.remedy
    assert "PairedBinaryDesign" in out.remedy, out.remedy


def test_an_ordinal_estimand_refuses_too() -> None:
    """`deviations` declined ordinal outcomes in the same sentence, so both are covered."""
    out = PowerAndMDE(design=OrdinalDesign(n=200, n_levels=5, mean_shift=0.2, rho=0.5)).compute()
    assert isinstance(out, Refusal)
    assert out.statistics is not None
    assert out.statistics["estimand_kind"] == "ordinal"
    assert "ordinal" in out.detail.lower(), out.detail


def test_the_refusal_is_not_a_whitelist_of_two_new_classes() -> None:
    """Anything that is not the paired binary design refuses, including an object nobody foresaw.

    A check written as `isinstance(design, (ContinuousDesign, OrdinalDesign))` would pass every test
    above and let a fourth design type through to a simulator that cannot represent it, which is the
    defect this row is about wearing a different hat.
    """

    class SomeoneElsesDesign:
        n = 100

    out = PowerAndMDE(design=SomeoneElsesDesign()).compute()  # type: ignore[arg-type]
    assert isinstance(out, Refusal)
    assert "SomeoneElsesDesign" in out.detail, out.detail


def test_the_refusal_reason_is_the_one_the_module_declares() -> None:
    """The reason is read off the module constant, so a ratified change is one edit, not a sweep."""
    out = PowerAndMDE(design=_continuous()).compute()
    assert isinstance(out, Refusal)
    assert out.reason is CONTINUOUS_REFUSAL_REASON


# ---------------------------------------------------------------------------
# The refusal must not have eaten the working path
# ---------------------------------------------------------------------------


def test_the_binary_design_still_plans_and_still_returns_a_number() -> None:
    """A refusal that fires on everything closes the row and breaks the instrument."""
    out = PowerAndMDE(
        design=PairedBinaryDesign(n=400, accuracy_a=0.50, accuracy_b=0.60, rho=0.5),
        replicates=400,
        with_calculators=False,
    ).compute()
    assert not isinstance(out, Refusal), f"the working path now refuses: {out!r}"
    assert 0.0 <= out.power <= 1.0
    assert np.isfinite(out.n_star)


def test_a_missing_design_keeps_its_own_refusal() -> None:
    """The pre-existing `ACCESS_INSUFFICIENT` for `design=None` is untouched."""
    out = PowerAndMDE(design=None).compute()
    assert isinstance(out, Refusal)
    assert "no design was supplied" in out.detail


# ---------------------------------------------------------------------------
# The declared deviation and the runtime behaviour have to agree
# ---------------------------------------------------------------------------


def test_the_declared_deviation_points_at_a_refusal_that_exists() -> None:
    """The sentence that used to stand alone now names the mechanism, and the mechanism runs.

    Documented is not implemented. This asserts the two are the same claim by requiring the prose to
    name the refusal and requiring the refusal to fire in the same test file.
    """
    declared = " ".join(PowerAndMDE.deviations).lower()
    assert "continuous" in declared and "ordinal" in declared
    assert "refus" in declared, "the deviation still describes a decline with no runtime mechanism"


def test_the_new_designs_validate_their_own_arguments() -> None:
    """A design object that accepts a negative spread would refuse for the wrong reason."""
    with pytest.raises(ValueError):
        ContinuousDesign(n=100, mean_a=0.0, mean_b=0.1, sd=-1.0)
    with pytest.raises(ValueError):
        ContinuousDesign(n=0, mean_a=0.0, mean_b=0.1, sd=1.0)
    with pytest.raises(ValueError):
        OrdinalDesign(n=100, n_levels=1, mean_shift=0.2)
