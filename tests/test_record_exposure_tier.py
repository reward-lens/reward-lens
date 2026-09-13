"""BLK-008: the exposure tier is a field, it is ordered, and it cannot be raised after the fact.

`e1_analysis/PLAN.md` section 7 states the contract in four sentences:

    Three tiers, assigned before a quantity is computed, stored on the evidence record, and printed
    beside the number wherever it appears. [...] A tier is assigned once and never raised. Exposure
    is transitive through determinants and not through names. No aggregate over tiers is itself
    preregistered.

At the baseline pin none of that had a referent a script could evaluate: `exposure_tier` returned
zero hits across the whole checkout, the three tiers were never ordered so "never raised" was not a
computable predicate, and nothing made the field immutable after its first write.

What is proven here:

- the three tiers exist and are ordered, so "never raised" is a comparison rather than a phrase;
- a quantity whose tier was never assigned **raises** when somebody tries to write a value for it,
  and the test asserts the raise rather than reading a number out of the refusal;
- a second assignment at a different tier **raises**, in both directions, and the message says which
  direction it was, because a raise is the forbidden move and a lowering is a demotion somebody has
  to authorise;
- an identical reassignment is accepted, because a pipeline that assigns the same tier twice is not
  an error and refusing it would make the ledger unusable from two call sites;
- the key is content-addressed, so two descriptions of one quantity share a tier and two different
  quantities do not, which is the half that stops "assigned once" being defeated by renaming.
"""

from __future__ import annotations

import pytest

from reward_lens.record.exposure import (
    ExposureLedger,
    ExposureTier,
    MissingExposureTier,
    TierConflict,
    quantity_id,
)

# The link-1 row, described twice: once by the analysis and once by the emitter. Same quantity.
C1 = dict(
    name="C1",
    estimator="advantages_from_rewards",
    window="every step",
    unit="clone-adjusted prompt group",
    population="mixed groups",
)


# ---------------------------------------------------------------------------
# The ordering, which is what "never raised" needs in order to mean anything
# ---------------------------------------------------------------------------


def test_the_three_tiers_are_ordered():
    """DESCRIPTIVE < PRE_DECLARED < PREDICTED, as a comparison a script can run."""
    assert ExposureTier.DESCRIPTIVE < ExposureTier.PRE_DECLARED < ExposureTier.PREDICTED
    assert sorted(ExposureTier) == [
        ExposureTier.DESCRIPTIVE,
        ExposureTier.PRE_DECLARED,
        ExposureTier.PREDICTED,
    ]
    assert len(ExposureTier) == 3


def test_the_tier_names_carry_the_plan_s_definitions():
    """Each tier says what it means, so the field is readable without the plan open beside it."""
    for tier in ExposureTier:
        assert len(tier.meaning) > 40, tier
    assert "before any code existed" in ExposureTier.PREDICTED.meaning
    assert "no value was predicted" in ExposureTier.PRE_DECLARED.meaning
    assert "already been seen" in ExposureTier.DESCRIPTIVE.meaning


# ---------------------------------------------------------------------------
# The refusal when the field is absent
# ---------------------------------------------------------------------------


def test_writing_a_quantity_with_no_assigned_tier_raises():
    """The closure proof's first half: the raise is the assertion, not a reading off a refusal."""
    ledger = ExposureLedger()

    with pytest.raises(MissingExposureTier) as excinfo:
        ledger.require(quantity_id(**C1))

    message = str(excinfo.value)
    assert "no exposure tier" in message
    assert quantity_id(**C1) in message, "the refusal names the quantity it is about"


def test_a_value_cannot_be_written_for_an_untiered_quantity():
    """The writer refuses, so an untiered number cannot reach a record at all."""
    ledger = ExposureLedger()

    with pytest.raises(MissingExposureTier):
        ledger.write(quantity_id(**C1), value=4.0)

    ledger.assign(quantity_id(**C1), ExposureTier.PREDICTED)
    written = ledger.write(quantity_id(**C1), value=4.0)
    assert written.tier is ExposureTier.PREDICTED
    assert written.value == 4.0


# ---------------------------------------------------------------------------
# First write wins, and the second one raises
# ---------------------------------------------------------------------------


def test_a_rewrite_at_a_higher_tier_raises_and_says_it_was_a_raise():
    """The closure proof's second half, in the direction the plan forbids by name."""
    ledger = ExposureLedger()
    qid = quantity_id(**C1)
    ledger.assign(qid, ExposureTier.DESCRIPTIVE)

    with pytest.raises(TierConflict) as excinfo:
        ledger.assign(qid, ExposureTier.PREDICTED)

    message = str(excinfo.value)
    assert "raise" in message
    assert "DESCRIPTIVE" in message and "PREDICTED" in message
    assert ledger.tier_of(qid) is ExposureTier.DESCRIPTIVE, "the first write must still stand"


def test_a_rewrite_at_a_lower_tier_also_raises_and_says_it_was_a_lowering():
    """A demotion is somebody's decision, not a side effect of a second call."""
    ledger = ExposureLedger()
    qid = quantity_id(**C1)
    ledger.assign(qid, ExposureTier.PREDICTED)

    with pytest.raises(TierConflict) as excinfo:
        ledger.assign(qid, ExposureTier.DESCRIPTIVE)

    assert "lowering" in str(excinfo.value)
    assert ledger.tier_of(qid) is ExposureTier.PREDICTED


def test_assigning_the_same_tier_twice_is_accepted():
    """Idempotent, so two call sites describing one quantity do not fight."""
    ledger = ExposureLedger()
    qid = quantity_id(**C1)

    assert ledger.assign(qid, ExposureTier.PRE_DECLARED) is ExposureTier.PRE_DECLARED
    assert ledger.assign(qid, ExposureTier.PRE_DECLARED) is ExposureTier.PRE_DECLARED
    assert ledger.tier_of(qid) is ExposureTier.PRE_DECLARED


# ---------------------------------------------------------------------------
# Content addressing: the half that stops a rename defeating "assigned once"
# ---------------------------------------------------------------------------


def test_the_same_quantity_described_twice_gets_the_same_id():
    """Key order and dict identity do not change a quantity's id."""
    reordered = {k: C1[k] for k in reversed(list(C1))}
    assert quantity_id(**C1) == quantity_id(**reordered)


def test_changing_the_estimator_changes_the_id():
    """A different estimator is a different quantity, so it does not inherit the other's tier.

    This is the input perturbation that makes the content addressing load-bearing rather than a
    hash of a name: the name is identical and the tier does not carry over.
    """
    other = dict(C1, estimator="advantages_from_rewards_v2")
    assert quantity_id(**other) != quantity_id(**C1)

    ledger = ExposureLedger()
    ledger.assign(quantity_id(**C1), ExposureTier.PREDICTED)

    with pytest.raises(MissingExposureTier):
        ledger.require(quantity_id(**other))


def test_renaming_a_quantity_does_not_launder_its_tier():
    """ "Exposure is transitive through determinants and not through names" (PLAN.md section 7)."""
    renamed = dict(C1, name="C1-prime")
    ledger = ExposureLedger()
    ledger.assign(quantity_id(**C1), ExposureTier.DESCRIPTIVE)

    # A new name is a new key, so it starts untiered rather than inheriting PREDICTED by accident.
    with pytest.raises(MissingExposureTier):
        ledger.require(quantity_id(**renamed))

    # And declaring the renamed one PREDICTED does not touch the original's tier.
    ledger.assign(quantity_id(**renamed), ExposureTier.PREDICTED)
    assert ledger.tier_of(quantity_id(**C1)) is ExposureTier.DESCRIPTIVE


# ---------------------------------------------------------------------------
# Transitivity through determinants
# ---------------------------------------------------------------------------


def test_a_quantity_cannot_be_tiered_above_its_determinants():
    """PLAN.md section 7: exposure is transitive through determinants.

    A quantity computed from something already seen has been seen. Without this the tier is
    decoration: the row's own consequence is "a quantity computed after the data existed can be
    written PREDICTED and the presence check passes silently", and the determinant is how the data
    got in.
    """
    ledger = ExposureLedger()
    seen = quantity_id(name="mechanism-first-mover", estimator="reconstruct", window="", unit="")
    ledger.assign(seen, ExposureTier.DESCRIPTIVE)

    derived = quantity_id(name="C7", estimator="rank_order", window="", unit="")
    with pytest.raises(TierConflict) as excinfo:
        ledger.assign(derived, ExposureTier.PREDICTED, determinants=(seen,))

    assert "determinant" in str(excinfo.value)
    assert "DESCRIPTIVE" in str(excinfo.value)


def test_a_quantity_at_or_below_its_determinants_is_accepted():
    """The cap is a cap, not a ban: at or under the weakest determinant is fine."""
    ledger = ExposureLedger()
    a = quantity_id(name="a", estimator="e", window="", unit="")
    b = quantity_id(name="b", estimator="e", window="", unit="")
    ledger.assign(a, ExposureTier.PREDICTED)
    ledger.assign(b, ExposureTier.PRE_DECLARED)

    derived = quantity_id(name="c", estimator="e", window="", unit="")
    assert (
        ledger.assign(derived, ExposureTier.PRE_DECLARED, determinants=(a, b))
        is ExposureTier.PRE_DECLARED
    )


def test_an_unknown_determinant_raises_rather_than_being_treated_as_unconstraining():
    """An unmeasured determinant is not a permissive one. Unavailable is not pass (Law 4)."""
    ledger = ExposureLedger()
    derived = quantity_id(name="c", estimator="e", window="", unit="")
    unknown = quantity_id(name="who", estimator="?", window="", unit="")

    with pytest.raises(MissingExposureTier):
        ledger.assign(derived, ExposureTier.PREDICTED, determinants=(unknown,))


# ---------------------------------------------------------------------------
# Round-trip, so the emitter can persist it
# ---------------------------------------------------------------------------


def test_the_ledger_round_trips_through_a_plain_mapping():
    """BLK-048's emitter has to write this to disk beside the numbers it tiers."""
    ledger = ExposureLedger()
    ledger.assign(quantity_id(**C1), ExposureTier.PREDICTED)
    ledger.assign(
        quantity_id(name="C7", estimator="rank_order", window="", unit=""), ExposureTier.DESCRIPTIVE
    )

    restored = ExposureLedger.from_canonical(ledger.__canonical__())

    assert restored.tier_of(quantity_id(**C1)) is ExposureTier.PREDICTED
    assert restored.__canonical__() == ledger.__canonical__()

    # And the restored ledger is just as immutable as the one it came from.
    with pytest.raises(TierConflict):
        restored.assign(quantity_id(**C1), ExposureTier.PRE_DECLARED)
