"""BLK-021: the recorded group standard deviation says which convention it was computed under.

`record/schema.py` documents `EstimatorSpec.std_ddof` at length and says of it, verbatim:

    Every framework in scope uses 1: TRL's `nanstd` multiplies the variance by `count / (count - 1)`
    explicitly, and veRL's `compute_grpo_outcome_advantage` calls `torch.std`, whose default is
    `correction=1`. That makes 1 the near-certain answer and not a safe default, because a
    near-certain assumption about a denominator is exactly the shape of confident wrong number this
    record exists to prevent.

Fifty lines further down the same file computed the group's standard deviation as `arr.std()`, which
is numpy's `ddof=0`, and recorded the result with nothing beside it saying so. The two conventions
differ by `sqrt(K / (K - 1))`, which is 1.069 at the `K = 8` this design runs, so the recorded
standard deviation was 6.9% away from the one the trainer divided by, on every group, with no field
in the record able to disagree.

The repair is the second of the two the row offers, taken with the first: `from_scores` computes at
the convention it is given, and the convention it used is recorded beside the value either way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reward_lens.record.schema import GroupStats

K = 8
# Eight scores with no coincidental structure, so the two conventions are separated by the factor
# and by nothing else.
SCORES = [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0, 1.0]


def _population(values) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def _sample(values) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


# ---------------------------------------------------------------------------
# The two conventions really are different on this fixture
# ---------------------------------------------------------------------------


def test_the_two_conventions_differ_by_the_registered_factor_at_k_eight():
    """`sqrt(K / (K - 1))` = 1.069 at K = 8, which is the number the row states.

    The guard for everything below. If the fixture ever stopped separating the two denominators,
    every assertion in this file would pass under either convention and prove nothing.
    """
    ratio = _sample(SCORES) / _population(SCORES)
    assert ratio == pytest.approx(math.sqrt(K / (K - 1)), rel=1e-12)
    assert ratio == pytest.approx(1.069, abs=5e-4)


# ---------------------------------------------------------------------------
# The value is computed at the convention it was asked for
# ---------------------------------------------------------------------------


def test_the_group_std_is_computed_at_the_recorded_convention():
    """The closure proof: matches the recorded `std_ddof` to 1e-12, and fails against ddof=0."""
    stats = GroupStats.from_scores(SCORES, std_epsilon=1e-8, std_ddof=1)

    assert stats.std == pytest.approx(_sample(SCORES), abs=1e-12)
    assert stats.std != pytest.approx(_population(SCORES), abs=1e-12)
    assert stats.std_ddof == 1


def test_the_population_convention_is_still_available_and_still_says_so():
    """Asking for ddof=0 is legitimate; leaving the reader to guess which one ran is not."""
    stats = GroupStats.from_scores(SCORES, std_epsilon=1e-8, std_ddof=0)

    assert stats.std == pytest.approx(_population(SCORES), abs=1e-12)
    assert stats.std_ddof == 0


# ---------------------------------------------------------------------------
# The convention travels with the value even when nobody stated one
# ---------------------------------------------------------------------------


def test_a_caller_that_states_no_convention_still_gets_one_recorded():
    """ "Record the convention actually used alongside the value" is the row's second option.

    An unstated convention is not an absent one: some denominator was used. The record says which,
    so a reader can tell a population standard deviation from a sample one without knowing what the
    call site looked like.
    """
    stats = GroupStats.from_scores(SCORES, std_epsilon=1e-8)

    assert stats.std_ddof is not None, "the stored std must name its own denominator"
    assert stats.std == pytest.approx(
        float(np.std(np.asarray(SCORES, dtype=float), ddof=stats.std_ddof)), abs=1e-12
    )


def test_a_group_with_no_std_records_no_convention():
    """All-abstained: there is no standard deviation, so there is no denominator to name."""
    stats = GroupStats.from_scores([None, None, None], std_epsilon=1e-8)

    assert stats.std is None
    assert stats.std_ddof is None


# ---------------------------------------------------------------------------
# It survives the record
# ---------------------------------------------------------------------------


def test_the_convention_round_trips_through_the_canonical_form():
    """A field that does not reach disk cannot settle an argument about a record on disk."""
    stats = GroupStats.from_scores(SCORES, std_epsilon=1e-8, std_ddof=1)

    payload = stats.__canonical__()
    assert payload["std_ddof"] == 1

    restored = GroupStats.from_canonical(payload)
    assert restored.std_ddof == 1
    assert restored.std == pytest.approx(_sample(SCORES), abs=1e-12)


def test_a_record_written_before_this_field_reads_back_as_not_recorded():
    """An older payload has no `std_ddof`, and "not recorded" is what is true of it.

    Defaulting it to 0 would turn a gap into a claim, which is the failure this row is an instance
    of rather than a fix for.
    """
    payload = GroupStats.from_scores(SCORES, std_epsilon=1e-8, std_ddof=1).__canonical__()
    del payload["std_ddof"]

    restored = GroupStats.from_canonical(payload)
    assert restored.std_ddof is None


# ---------------------------------------------------------------------------
# The degeneracy flag is read against the same number
# ---------------------------------------------------------------------------


def test_the_degeneracy_flag_is_decided_by_the_std_that_was_recorded():
    """`degenerate` is `std <= std_epsilon`, so it has to use the same denominator as `std`.

    A flag computed off one convention and stored beside a value computed off another is the same
    defect one field over.
    """
    near = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 + 3e-4]
    epsilon = 1.05e-4

    population = GroupStats.from_scores(near, std_epsilon=epsilon, std_ddof=0)
    sample = GroupStats.from_scores(near, std_epsilon=epsilon, std_ddof=1)

    # The fixture is chosen so the two conventions land either side of the epsilon, which is the
    # only case where the flag can disagree with itself.
    assert population.std < epsilon < sample.std
    assert population.degenerate is True
    assert sample.degenerate is False
