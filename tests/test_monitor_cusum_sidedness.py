"""BLK-068-a: the CUSUM runner runs the chart its design registers.

`CusumDesign` carries `sides`, `design_cusum` solves ``h`` for that sidedness, and `render` prints
it. The runner alarmed on either accumulator whatever the design said, so a one-sided design was
executed as a two-sided chart at a threshold solved for the one-sided rule, and delivered half the
in-control run length it was designed for. Nothing in the output could disagree: `sides` read 1,
`render()` printed one-sided, ``h`` was the one-sided threshold, and the run record had no field
that recorded what the runner had actually done.

Three proofs, in increasing cost:

- a downward excursion, which a one-sided upper chart must ignore and a two-sided chart must catch;
- the run record naming the rule that ran;
- the in-control average run length by simulation, against the design's own target.

The tolerance on the last one is four Monte Carlo standard errors **of the sample the test itself
draws**, so it is a property of the estimator rather than a number chosen once the answer was known.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.monitor.arl import arl_siegmund, design_cusum
from reward_lens.monitor.cusum import run_cusum

# In-control simulation geometry. `MAX_STEPS` is ten times the one-sided target, so the chance a
# stream survives it uncensored is about exp(-10) and censoring cannot account for a factor of two.
N_STREAMS = 2000
MAX_STEPS = 2400
SIGMA_BAND = 4.0


def _measure_arl0(design, *, seed: int = 20260820) -> tuple[float, float]:
    """In-control average run length of the shipped runner, and its standard error.

    Streams are standard normal, which is what "in control" means once `standardize` has done its
    work, and they are passed with ``standardized=True`` so the measurement is of the alarm rule and
    not of the scaling. A stream that never alarms contributes ``MAX_STEPS``, which biases the
    estimate **downward**; a chart that measures long is therefore measuring at least that long.
    """
    rng = np.random.default_rng(seed)
    lengths = np.empty(N_STREAMS, dtype=np.float64)
    for i in range(N_STREAMS):
        run = run_cusum(rng.standard_normal(MAX_STEPS), design, standardized=True)
        lengths[i] = MAX_STEPS if run.alarm_at is None else run.alarm_at + 1
    mean = float(lengths.mean())
    return mean, float(lengths.std(ddof=1) / np.sqrt(N_STREAMS))


# ---------------------------------------------------------------------------
# The mechanism, deterministically
# ---------------------------------------------------------------------------


def test_a_one_sided_chart_ignores_a_downward_excursion():
    """The registered rule watches for an increase, so a fall is not its alarm to raise.

    A run of -3 standard deviations drives the lower accumulator far past any threshold and leaves
    the upper one at zero. A one-sided upper chart must not fire on it. This is the mechanism the
    average run length measures statistically, isolated so it can be read at a glance.
    """
    design = design_cusum(1.0, 240.0, sides=1)
    falling = np.full(40, -3.0)

    run = run_cusum(falling, design, standardized=True)

    assert run.upper.max() == pytest.approx(0.0)
    assert run.lower.max() > design.h, "the fixture does not exercise the lower accumulator"
    assert run.alarm_at is None, (
        "a one-sided upper chart alarmed on a purely downward series, so it is running two-sided"
    )


def test_a_two_sided_chart_still_catches_the_same_downward_excursion():
    """The guard on the test above: the fixture is a real alarm for the chart that should fire.

    Without this, `test_a_one_sided_chart_ignores_a_downward_excursion` would also pass against a
    runner that had simply stopped alarming, and the repair would be a regression dressed as a fix.
    """
    design = design_cusum(1.0, 240.0, sides=2)
    falling = np.full(40, -3.0)

    run = run_cusum(falling, design, standardized=True)

    assert run.alarm_at is not None
    assert run.lower[run.alarm_at] > design.h


def test_the_run_record_says_which_rule_ran():
    """The record carries the sidedness the runner used, not only the one the design asked for.

    A design and a runner that disagree is the defect. A record that reports what the runner did is
    what makes the disagreement visible next time instead of invisible.
    """
    z = np.full(40, 3.0)

    one = run_cusum(z, design_cusum(1.0, 240.0, sides=1), standardized=True)
    two = run_cusum(z, design_cusum(1.0, 240.0, sides=2), standardized=True)

    assert one.sides == 1
    assert two.sides == 2


# ---------------------------------------------------------------------------
# The false-alarm interval the design was solved for
# ---------------------------------------------------------------------------


def test_the_one_sided_chart_delivers_the_in_control_run_length_it_was_designed_for():
    """At the one-sided threshold the measured ARL_0 is the one-sided target, not half of it.

    `design_cusum(1.0, 240.0, sides=1)` solves ``h = 3.6690`` at ``k = 0.5`` for an in-control run
    length of 240 steps. Running that threshold under a two-sided alarm rule halves it, because the
    two arms are symmetric under the null and their rates add.
    """
    design = design_cusum(1.0, 240.0, sides=1)
    assert design.k == pytest.approx(0.5)
    assert design.h == pytest.approx(3.6690, abs=5e-4)

    measured, standard_error = _measure_arl0(design)

    target = design.arl0_siegmund
    two_sided_at_the_same_h = arl_siegmund(design.h, design.k, 0.0, 2)
    assert abs(measured - target) < SIGMA_BAND * standard_error, (
        f"measured ARL_0 {measured:.1f} +- {standard_error:.1f} against a registered {target:.1f}; "
        f"the two-sided rule at this same threshold gives {two_sided_at_the_same_h:.1f}"
    )


def test_the_two_sided_chart_is_unchanged():
    """The default chart is what it was, measured the same way.

    Every caller in the library designs at the default `sides=2`, so this is the regression guard
    that the repair moved only the case the design registered and nobody was running.
    """
    design = design_cusum(1.0, 240.0, sides=2)

    measured, standard_error = _measure_arl0(design)

    assert abs(measured - design.arl0_siegmund) < SIGMA_BAND * standard_error, (
        f"measured ARL_0 {measured:.1f} +- {standard_error:.1f} against {design.arl0_siegmund:.1f}"
    )
