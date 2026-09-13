"""BLK-005 — an activation capture at an explicit token position was unreachable.

`capture_sites` hard-coded `PositionSpec("final")` and exposed no position parameter across
twenty-two call sites. Both runtimes then collapsed anything that was not `final`: a non-final kind
set `positions=None` and `full_sequence=True`, so an explicit index came back as a whole-sequence
`(B, T, d)` tensor rather than the one position that was asked for. `PositionSpec.resolve` was never
called on the capture path at all: of 86 repository-wide `.resolve(` hits, zero were
`PositionSpec.resolve`, and `PositionSpec("explicit"` returned zero hits repository-wide.

`CaptureMount._store` already gathers one arbitrary index per row with `hidden[batch_idx, pos]`.
The index simply never arrived.

**Why it matters.** The design reads at the **last prompt token**. On a prompt-plus-completion
sequence that is not the final token, so the probe was reading a completion position, which makes
link 2 a measurement of a text description rather than of a pre-generation propensity.

Raw before-and-after at `chain/repair/proofs/BLK-005/`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from reward_lens.core.types import Site  # noqa: E402
from reward_lens.policy.hf import from_pretrained  # noqa: E402
from reward_lens.runtime.backend import CaptureSpec  # noqa: E402
from reward_lens.runtime.hf import resolve_capture_positions  # noqa: E402
from reward_lens.signals.base import PositionSpec  # noqa: E402

TINY_POLICY = "trl-internal-testing/tiny-Qwen3ForCausalLM"

ITEMS = [
    ("count upward from 15", " 16 17 18 19 20 21"),
    ("count upward from 3", " 4 5"),
    ("count upward from 40", " 41 42 43 44"),
]

SITE = Site(layer=1, point="resid_post", head=None)


@pytest.fixture(scope="module")
def policy():
    return from_pretrained(TINY_POLICY, device="cpu")


def _boundaries(policy) -> list[int]:
    """The last prompt token of each item, from the tokenizer's own recorded count."""
    return [int(policy.tokenize(item).meta["n_prompt_tokens"]) - 1 for item in ITEMS]


def _capture(policy, position):
    spec = CaptureSpec(sites=(SITE,), position=position, full_sequence=False, dtype="float32")
    handle = policy.capture(ITEMS, spec)
    capture = next(iter(handle))
    return capture


def _manual_forward(policy):
    """One forward through the same model, kept as a `(B, T, d)` tensor to slice by hand.

    Deliberately not through the capture path: the closure proof compares the capture against a
    forward-and-slice, and comparing the capture against itself would prove nothing.
    """
    tokenized = [policy.tokenize(item) for item in ITEMS]
    batch = policy.runtime.collate(tokenized)
    spec = CaptureSpec(sites=(SITE,), position=None, full_sequence=True, dtype="float32")
    handle = policy.capture(ITEMS, spec)
    whole = next(iter(handle)).tensors[SITE]
    assert whole.ndim == 3, f"the full-sequence read is {tuple(whole.shape)}"
    return whole, batch


# ---------------------------------------------------------------------------
# The closure proof
# ---------------------------------------------------------------------------


def test_a_capture_at_an_explicit_index_matches_a_forward_and_slice(policy) -> None:
    """The row's own tolerance, `1e-6`, against a hand slice of the whole sequence."""
    boundaries = _boundaries(policy)
    whole, _ = _manual_forward(policy)
    by_hand = torch.stack([whole[row, index] for row, index in enumerate(boundaries)])

    captured = _capture(policy, PositionSpec("explicit", detail=boundaries)).tensors[SITE]

    assert captured.ndim == 2, (
        f"an explicit position returned {tuple(captured.shape)}; before the repair a non-final kind "
        f"collapsed to a whole-sequence read"
    )
    assert captured.shape == by_hand.shape
    assert torch.allclose(captured, by_hand, atol=1e-6)


def test_the_boundary_index_is_the_recorded_prompt_token_count_on_every_row(policy) -> None:
    """The second half of the closure proof, asserted per row rather than in aggregate."""
    boundaries = _boundaries(policy)
    capture = _capture(policy, PositionSpec("explicit", detail=boundaries))

    assert len(capture.positions) == len(ITEMS)
    for row, (recorded, item) in enumerate(zip(capture.positions, ITEMS)):
        n_prompt = int(policy.tokenize(item).meta["n_prompt_tokens"])
        assert recorded == [n_prompt - 1], (
            f"row {row}: recorded {recorded}, boundary {n_prompt - 1}"
        )
        assert boundaries[row] == n_prompt - 1


def test_the_last_prompt_token_is_not_the_final_token(policy) -> None:
    """If these two agreed, the coordinate would not have been worth reaching.

    This is the fail case for the whole row: it perturbs a real input, the position, and the two
    vectors have to differ. Before the repair the explicit read did not return a vector at all.
    """
    boundaries = _boundaries(policy)
    at_prompt_end = _capture(policy, PositionSpec("explicit", detail=boundaries)).tensors[SITE]
    at_final = _capture(policy, PositionSpec("final")).tensors[SITE]

    assert at_prompt_end.shape == at_final.shape
    assert not torch.allclose(at_prompt_end, at_final, atol=1e-4), (
        "the last prompt token and the final token returned the same activation, so the fixture "
        "has no completion and this test proves nothing"
    )


def test_one_index_for_the_whole_batch_is_broadcast(policy) -> None:
    captured = _capture(policy, PositionSpec("explicit", detail=2)).tensors[SITE]
    whole, _ = _manual_forward(policy)
    assert captured.ndim == 2
    assert torch.allclose(captured, whole[:, 2, :], atol=1e-6)


def test_the_final_kind_is_unchanged(policy) -> None:
    """Twenty-two call sites pass `final` and none of them may move."""
    at_final = _capture(policy, PositionSpec("final")).tensors[SITE]
    default = _capture(policy, None).tensors[SITE]
    assert torch.allclose(at_final, default, atol=0.0)
    assert at_final.ndim == 2


def test_capture_sites_takes_a_position(policy) -> None:
    """The public path, which is the thing the row says is unreachable.

    `measure.battery._common.capture_sites` is what twenty-two call sites go through and it exposed
    no position parameter at all.
    """
    from reward_lens.measure.battery._common import capture_sites

    boundaries = _boundaries(policy)
    at_prompt_end = capture_sites(
        policy, list(ITEMS), (SITE,), position=PositionSpec("explicit", detail=boundaries)
    )[SITE]
    at_final = capture_sites(policy, list(ITEMS), (SITE,))[SITE]

    assert at_prompt_end.ndim == 2
    assert not torch.allclose(at_prompt_end, at_final, atol=1e-4)


# ---------------------------------------------------------------------------
# The resolver's own contract, without a model
# ---------------------------------------------------------------------------


def _spec(position, full_sequence=False):
    return CaptureSpec(sites=(), position=position, full_sequence=full_sequence, dtype="float32")


def test_the_resolver_keeps_the_final_position_for_final_and_none() -> None:
    final_pos = torch.tensor([7, 5, 9])
    for position in (None, PositionSpec("final")):
        got = resolve_capture_positions(_spec(position), final_pos)
        assert torch.equal(got, final_pos)


def test_the_resolver_leaves_the_genuinely_multi_position_kinds_alone() -> None:
    """`all`, `step_ends`, `span_ends` and `judgment` are not one index and must stay whole."""
    final_pos = torch.tensor([7, 5, 9])
    for kind in ("all", "step_ends", "span_ends", "judgment"):
        assert resolve_capture_positions(_spec(PositionSpec(kind)), final_pos) is None
    assert resolve_capture_positions(_spec(PositionSpec("final"), True), final_pos) is None


def test_a_negative_index_counts_back_from_each_rows_own_last_valid_token() -> None:
    """Padding differs per row, so a negative index cannot be resolved against the tensor width."""
    final_pos = torch.tensor([7, 5, 9])
    got = resolve_capture_positions(_spec(PositionSpec("explicit", -1)), final_pos)
    assert torch.equal(got, final_pos)
    got = resolve_capture_positions(_spec(PositionSpec("explicit", -3)), final_pos)
    assert got.tolist() == [5, 3, 7]


def test_an_index_past_a_rows_valid_span_raises_rather_than_being_clamped() -> None:
    """`CaptureMount._store` clamps, and a clamp is a plausible vector from the wrong token."""
    final_pos = torch.tensor([7, 5, 9])
    with pytest.raises(ValueError, match="outside that row's valid span"):
        resolve_capture_positions(_spec(PositionSpec("explicit", [1, 6, 1])), final_pos)
    with pytest.raises(ValueError, match="outside that row's valid span"):
        resolve_capture_positions(_spec(PositionSpec("explicit", -20)), final_pos)


def test_a_wrong_length_index_list_raises() -> None:
    final_pos = torch.tensor([7, 5, 9])
    with pytest.raises(ValueError, match="for 3 rows"):
        resolve_capture_positions(_spec(PositionSpec("explicit", [1, 2])), final_pos)


def test_an_explicit_kind_with_no_detail_raises() -> None:
    final_pos = torch.tensor([7, 5, 9])
    with pytest.raises(ValueError, match="carries its indices"):
        resolve_capture_positions(_spec(PositionSpec("explicit")), final_pos)
