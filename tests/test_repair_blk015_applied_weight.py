"""BLK-015 — nothing separated the applied weight from the advantage, so the source string lied.

`selection_differential`'s second argument was annotated `advantages: np.ndarray`: a bare array,
no `NewType`, no `Annotated`, no runtime check. The only things that touched it were a finite check,
a within-group centring and a dot product, none of which can tell one scalar from another. So
`a-tilde`, the per-rollout weight the loss actually applied, could be passed positionally and the
ledger's recorded source would still read `"recorded"`.

That string was never wrong. It answers **where the numbers came from**, and the question nobody
could answer afterwards is **what they were**. `A6_STALENESS` verified the duck typing at the pinned
SHA and the consequence is that the released artifact cannot be audited for which quantity produced
the headline ratio, which is `C4`'s interpretability rather than its value.

The repair is three edits and all three are required: a type with a declared quantity
(`SelectionWeights`), a field on the rollout that a writer populates (`Trajectory.applied_weight`,
carried into `StepSample.applied_weights`), and a refusal that fires when a differential is read as
`S_applied` without one (`Differential.as_applied`).

Raw before-and-after at `chain/repair/proofs/BLK-015/`.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.measure.ledger.price import (
    Differential,
    UnlabelledSelectionWeights,
    selection_differential,
)
from reward_lens.record.schema import SelectionQuantity, SelectionWeights

GROUPS = 12
K = 8
N = GROUPS * K
KNOWN_FACTOR = 2.5


def _fixture():
    """Features, an advantage, and an applied weight that is the advantage times a known factor.

    The factor is what the `dapo` aggregation does to the scalar between the advantage and what the
    loss multiplies the ratio by. Constructed rather than simulated, so the differential's response
    to it is exactly predictable and the test has a closed form.
    """
    rng = np.random.default_rng(15)
    group_ids = np.repeat(np.arange(GROUPS), K)
    features = rng.standard_normal((N, 3))
    advantage = rng.standard_normal(N) + 0.4 * features[:, 0]
    advantage = advantage - np.repeat(
        np.asarray([advantage[group_ids == g].mean() for g in range(GROUPS)]), K
    )
    applied = KNOWN_FACTOR * advantage
    return features, advantage, applied, group_ids


NAMES = ("f0", "f1", "f2")


# ---------------------------------------------------------------------------
# The closure proof: the two quantities differ, and the record says which was used
# ---------------------------------------------------------------------------


def test_the_two_quantities_give_different_differentials() -> None:
    """By the known factor, exactly. The arithmetic cannot tell them apart; the label must."""
    features, advantage, applied, group_ids = _fixture()
    on_advantage = selection_differential(
        features,
        SelectionWeights(advantage, SelectionQuantity.ADVANTAGE),
        group_ids,
        NAMES,
    )
    on_applied = selection_differential(
        features,
        SelectionWeights(applied, SelectionQuantity.APPLIED_WEIGHT),
        group_ids,
        NAMES,
    )
    assert np.allclose(on_applied.value, KNOWN_FACTOR * on_advantage.value, rtol=1e-12)
    assert not np.allclose(on_applied.value, on_advantage.value)


def test_the_recorded_quantity_says_which_was_used() -> None:
    """The field the row asks for. Before the repair there was no such field to read."""
    features, advantage, applied, group_ids = _fixture()
    on_advantage = selection_differential(
        features, SelectionWeights(advantage, SelectionQuantity.ADVANTAGE), group_ids, NAMES
    )
    on_applied = selection_differential(
        features,
        SelectionWeights(applied, SelectionQuantity.APPLIED_WEIGHT, "reconstructed"),
        group_ids,
        NAMES,
    )
    assert on_advantage.quantity is SelectionQuantity.ADVANTAGE
    assert on_advantage.weight_source == "recorded"
    assert on_applied.quantity is SelectionQuantity.APPLIED_WEIGHT
    assert on_applied.weight_source == "reconstructed"
    assert on_advantage.quantity is not on_applied.quantity


# ---------------------------------------------------------------------------
# The refusal, which is the third of the three edits
# ---------------------------------------------------------------------------


def test_reading_an_advantage_differential_as_s_applied_is_refused() -> None:
    """`C4` is a claim about the applied weight. The advantage's number is a different quantity."""
    features, advantage, _, group_ids = _fixture()
    got = selection_differential(
        features, SelectionWeights(advantage, SelectionQuantity.ADVANTAGE), group_ids, NAMES
    )
    with pytest.raises(UnlabelledSelectionWeights) as caught:
        got.as_applied()
    assert caught.value.quantity is SelectionQuantity.ADVANTAGE
    # The same numbers are still readable under their own name.
    assert got.as_dict()["f0"] == pytest.approx(float(got.value[0]))


def test_a_bare_array_is_recorded_as_unlabelled_and_refused_as_s_applied() -> None:
    """The state every in-repository caller was in. Not assumed to be the advantage: unlabelled.

    Defaulting an unlabelled array to `ADVANTAGE` would make the refusal unreachable for exactly
    the caller that caused the defect, which is the one passing `a-tilde` positionally.
    """
    features, _, applied, group_ids = _fixture()
    got = selection_differential(features, applied, group_ids, NAMES)
    assert got.quantity is SelectionQuantity.UNLABELLED
    assert got.weight_source == "unknown"
    with pytest.raises(UnlabelledSelectionWeights):
        got.as_applied()


def test_the_applied_weight_differential_reads_as_s_applied() -> None:
    """The refusal has to be reachable in both directions, or it is not a check."""
    features, _, applied, group_ids = _fixture()
    got = selection_differential(
        features, SelectionWeights(applied, SelectionQuantity.APPLIED_WEIGHT), group_ids, NAMES
    )
    assert got.as_applied() == got.as_dict()


def test_the_refusal_names_the_quantity_it_found() -> None:
    """A refusal that does not say what it got is a refusal nobody can act on."""
    bad = Differential(
        names=NAMES,
        value=np.zeros(3),
        standard_error=np.zeros(3),
        n_scored=0,
        n_groups=0,
        n_degenerate=0,
        quantity=SelectionQuantity.REWARD,
        weight_source="recorded",
    )
    with pytest.raises(UnlabelledSelectionWeights, match="reward"):
        bad.as_applied()


# ---------------------------------------------------------------------------
# The type is checked when the code runs, because that is where the defect lives
# ---------------------------------------------------------------------------


def test_a_string_quantity_is_rejected() -> None:
    """An annotation is erased at runtime and this defect is a positional argument."""
    with pytest.raises(TypeError, match="SelectionQuantity"):
        SelectionWeights(np.zeros(4), "applied_weight")  # type: ignore[arg-type]


def test_an_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="recorded"):
        SelectionWeights(np.zeros(4), SelectionQuantity.APPLIED_WEIGHT, "probably")


def test_a_misaligned_weight_column_is_refused() -> None:
    """One rollout's weight against another's features is a silent pairing error."""
    features, advantage, _, group_ids = _fixture()
    with pytest.raises(ValueError, match="feature rows"):
        selection_differential(
            features,
            SelectionWeights(advantage[:-1], SelectionQuantity.ADVANTAGE),
            group_ids,
            NAMES,
        )


# ---------------------------------------------------------------------------
# The field on the rollout, and the writer that carries it into the estimator
# ---------------------------------------------------------------------------


def test_the_rollout_carries_the_applied_weight_and_its_source() -> None:
    from reward_lens.record.schema import make_trajectory
    from reward_lens.record.turns import Turn

    turn = Turn(index=0, role="assistant", text="hello")
    traj = make_trajectory(
        id="t0",
        task_ref="task-0",
        turns=[turn],
        advantage=0.5,
        applied_weight=1.25,
        applied_weight_source="recorded",
    )
    assert traj.advantage == 0.5
    assert traj.applied_weight == 1.25
    assert traj.applied_weight_source == "recorded"

    absent = make_trajectory(id="t1", task_ref="task-0", turns=[turn], advantage=0.5)
    assert absent.applied_weight is None, "absence must be an absence, not a fallback"


def test_a_step_with_no_applied_weight_falls_back_to_the_advantage_under_its_own_label() -> None:
    """The fallback is allowed. Passing it off as the applied weight is not."""
    from reward_lens.measure.ledger.price import StepSample

    features, advantage, _, group_ids = _fixture()
    sample = StepSample(
        index=0,
        names=NAMES,
        features=features,
        advantages=advantage,
        group_ids=group_ids,
        task_ids=tuple(f"task-{g}" for g in group_ids),
    )
    weights = sample.weights()
    assert weights.quantity is SelectionQuantity.ADVANTAGE
    with pytest.raises(UnlabelledSelectionWeights):
        selection_differential(features, weights, group_ids, NAMES).as_applied()


def test_a_step_carrying_applied_weights_reads_as_s_applied() -> None:
    from reward_lens.measure.ledger.price import StepSample

    features, advantage, applied, group_ids = _fixture()
    sample = StepSample(
        index=0,
        names=NAMES,
        features=features,
        advantages=advantage,
        group_ids=group_ids,
        task_ids=tuple(f"task-{g}" for g in group_ids),
        applied_weights=SelectionWeights(applied, SelectionQuantity.APPLIED_WEIGHT),
    )
    got = selection_differential(features, sample.weights(), group_ids, NAMES)
    assert got.quantity is SelectionQuantity.APPLIED_WEIGHT
    assert got.as_applied()["f0"] == pytest.approx(
        KNOWN_FACTOR
        * float(
            selection_differential(
                features, SelectionWeights(advantage, SelectionQuantity.ADVANTAGE), group_ids, NAMES
            ).value[0]
        )
    )
