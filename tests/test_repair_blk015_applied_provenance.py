"""BLK-015 — the label reaches the emitted payload, or the artifact still cannot be audited.

`tests/test_repair_blk015_applied_weight.py` holds the library to the first three edits: a type
with a declared quantity (`SelectionWeights`), a field on the rollout a writer populates
(`Trajectory.applied_weight`), and a refusal that fires when a differential is read as `S_applied`
without one (`Differential.as_applied`). All three were done and all three stop **inside the
process**.

The residual this file closes is the last hop. `_common_payload` is what `measure` emits, so it is
the released artifact a reader audits, and it carried `advantage_source` and nothing else. That
field answers *where the advantages came from* and reads ``"recorded"`` on a step whose covariance
was taken against `a-tilde`, so the artifact recorded a true sentence about the wrong quantity. The
question `C4` needs answered — *what were these numbers computed on* — had no field at all.

Three edits, in the order the value travels: `StepLedger` carries `quantity` and `weight_source`,
`ledger_between` sets them from the `Differential` that knows, and `_common_payload` emits them.

**Why both fields and not one.** They are not the same field under two names. On a step whose
record carries applied weights, `advantage_source` is still ``"recorded"`` while `quantity` is
`APPLIED_WEIGHT`; on a step without them the two coincide. A test that only exercised the second
case would be unable to tell the repair from a rename, so the case where they disagree is asserted
directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.measure.ledger.price import (
    SelectionResidual,
    SelectionTerm,
    StepSample,
    UnlabelledSelectionWeights,
    _common_payload,
    ledger_series,
    selection_differential,
)
from reward_lens.record.schema import SelectionQuantity, SelectionWeights

GROUPS = 6
K = 8
N = GROUPS * K
NAMES = ("f0", "f1", "f2")
#: What the `dapo` aggregation does to the scalar between the advantage and what the loss
#: multiplies the ratio by. Constructed rather than simulated so the differential's response to it
#: has a closed form.
KNOWN_FACTOR = 2.5


def _fixture(seed: int = 15):
    """Features, a within-group centred advantage, and an applied weight `KNOWN_FACTOR` times it."""
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(GROUPS), K)
    features = rng.standard_normal((N, 3))
    advantage = rng.standard_normal(N) + 0.4 * features[:, 0]
    advantage = advantage - np.repeat(
        np.asarray([advantage[group_ids == g].mean() for g in range(GROUPS)]), K
    )
    return features, advantage, KNOWN_FACTOR * advantage, group_ids


def _samples(*, with_applied: bool, n_steps: int = 2) -> list[StepSample]:
    """Consecutive steps, so `ledger_series` has a pair to report on.

    `advantage_source` is ``"recorded"`` on every one, exactly as `_sample_of` sets it, because
    that is the field that cannot distinguish the two cases and the point of this file is that the
    new ones can.
    """
    return _steps_with([with_applied] * n_steps)


def _steps_with(flags: list[bool]) -> list[StepSample]:
    """One `StepSample` per flag; the flag says whether that step's record carried applied weights."""
    out: list[StepSample] = []
    for index, with_applied in enumerate(flags):
        features, advantage, applied, group_ids = _fixture(seed=15 + index)
        out.append(
            StepSample(
                index=index,
                names=NAMES,
                features=features,
                advantages=advantage,
                group_ids=group_ids,
                task_ids=tuple(f"task-{g}" for g in group_ids),
                advantage_source="recorded",
                applied_weights=(
                    SelectionWeights(applied, SelectionQuantity.APPLIED_WEIGHT, "recorded")
                    if with_applied
                    else None
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Clause 1: the two quantities are different numbers, by a known factor
# ---------------------------------------------------------------------------


def test_the_applied_weight_differential_differs_from_the_advantage_one_by_the_known_factor():
    """The arithmetic is identical for both, which is why the label is the only thing separating
    them. If they came back equal there would be nothing for a provenance field to be about."""
    features, advantage, applied, group_ids = _fixture()
    on_advantage = selection_differential(
        features, SelectionWeights(advantage, SelectionQuantity.ADVANTAGE), group_ids, NAMES
    )
    on_applied = selection_differential(
        features, SelectionWeights(applied, SelectionQuantity.APPLIED_WEIGHT), group_ids, NAMES
    )
    assert np.allclose(on_applied.value, KNOWN_FACTOR * on_advantage.value, rtol=1e-12)
    assert not np.allclose(on_applied.value, on_advantage.value)


# ---------------------------------------------------------------------------
# Clause 2: the refusal, in both directions
# ---------------------------------------------------------------------------


def test_as_applied_raises_on_an_unlabelled_input():
    """A bare array passed positionally is the state every in-repository caller was in."""
    features, _, applied, group_ids = _fixture()
    got = selection_differential(features, applied, group_ids, NAMES)
    assert got.quantity is SelectionQuantity.UNLABELLED
    assert got.weight_source == "unknown"
    with pytest.raises(UnlabelledSelectionWeights):
        got.as_applied()


def test_as_applied_returns_when_the_input_is_labelled_applied_weight():
    """The refusal has to be reachable in both directions or it is a printer."""
    features, _, applied, group_ids = _fixture()
    got = selection_differential(
        features, SelectionWeights(applied, SelectionQuantity.APPLIED_WEIGHT), group_ids, NAMES
    )
    assert got.as_applied() == got.as_dict()


# ---------------------------------------------------------------------------
# Clause 3: the emitted payload names which was used
# ---------------------------------------------------------------------------


def test_the_emitted_payload_carries_quantity_and_weight_source():
    """The keys themselves. Before this repair `_common_payload` had neither."""
    payload = _common_payload(ledger_series(_samples(with_applied=True), eta=1.0))
    assert "quantity" in payload, "the released artifact cannot say what the numbers were"
    assert "weight_source" in payload, "nor where they came from"


def test_the_payload_names_the_applied_weight_when_the_record_carried_one():
    payload = _common_payload(ledger_series(_samples(with_applied=True), eta=1.0))
    assert payload["quantity"] == SelectionQuantity.APPLIED_WEIGHT.value
    assert payload["weight_source"] == "recorded"


def test_the_payload_names_the_advantage_when_the_record_did_not():
    """The fallback is allowed. Passing it off as the applied weight is not."""
    payload = _common_payload(ledger_series(_samples(with_applied=False), eta=1.0))
    assert payload["quantity"] == SelectionQuantity.ADVANTAGE.value
    assert payload["weight_source"] == "recorded"


def test_advantage_source_cannot_tell_the_two_cases_apart_and_quantity_can():
    """The falsifier this row registered: are `advantage_source` and `weight_source` one field
    under two names? On the case where the record carries applied weights they disagree, so no."""
    applied = _common_payload(ledger_series(_samples(with_applied=True), eta=1.0))
    advantage = _common_payload(ledger_series(_samples(with_applied=False), eta=1.0))

    assert applied["advantage_source"] == advantage["advantage_source"] == "recorded"
    assert applied["quantity"] != advantage["quantity"]
    # And the covariances really are two different numbers, so the payloads it labels are not the
    # same artifact with a different string on it.
    applied_rows = ledger_series(_samples(with_applied=True), eta=1.0)[0].rows
    advantage_rows = ledger_series(_samples(with_applied=False), eta=1.0)[0].rows
    assert applied_rows[0].covariance == pytest.approx(
        KNOWN_FACTOR * advantage_rows[0].covariance, rel=1e-12
    )


@pytest.mark.parametrize("instrument", [SelectionTerm, SelectionResidual])
def test_both_registered_instruments_emit_the_label(instrument):
    """`_common_payload` is shared, and both readings are released artifacts."""
    ledgers = ledger_series(_samples(with_applied=True), eta=1.0)
    payload = instrument(None, None).payload(ledgers)  # type: ignore[arg-type]
    assert payload["quantity"] == SelectionQuantity.APPLIED_WEIGHT.value
    assert payload["weight_source"] == "recorded"


# ---------------------------------------------------------------------------
# The label travels on the ledger, not only in the emitted dict
# ---------------------------------------------------------------------------


def test_the_ledger_row_carries_the_label_the_payload_reads():
    """`_common_payload` reads it off the ledger, so the ledger has to have it. Reading it back off
    the payload alone would pass if the payload computed the string itself, which is the defect."""
    ledger = ledger_series(_samples(with_applied=True), eta=1.0)[0]
    assert ledger.quantity is SelectionQuantity.APPLIED_WEIGHT
    assert ledger.weight_source == "recorded"
    assert ledger.advantage_source == "recorded"


def test_a_window_that_mixes_the_two_quantities_is_reported_as_unlabelled():
    """`_sample_of` refuses two provenances inside one step; nothing checked across a window.

    Emitting the first step's label over a mixed window is the same silent substitution one level
    up, so a disagreement is reported as unlabelled, which is what `as_applied` refuses on.
    """
    # The `before` step of each pair is the one whose weights the differential is taken against,
    # so flags [True, False, True] gives two ledgers labelled APPLIED_WEIGHT and ADVANTAGE.
    ledgers = ledger_series(_steps_with([True, False, True]), eta=1.0)
    assert {led.quantity for led in ledgers} == {
        SelectionQuantity.APPLIED_WEIGHT,
        SelectionQuantity.ADVANTAGE,
    }
    payload = _common_payload(ledgers)
    assert payload["quantity"] == SelectionQuantity.UNLABELLED.value
