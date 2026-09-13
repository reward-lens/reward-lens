"""BLK-005 coverage: an explicit index is a *collated* coordinate, and collation left-pads.

`tests/test_repair_blk005_explicit_position.py` is this row's registered acceptance and it passes.
What it never exercises is the step between `n_prompt_tokens` and the index the capture path takes.
Its `_boundaries()` computes `n_prompt_tokens - 1` per item and hands that straight to
`PositionSpec("explicit", detail=...)`, but `runtime.collate` left-pads, so the last prompt token of
row `r` sits at `offsets[r] + n_prompt_tokens[r] - 1` in the collated tensor. The library does that
addition itself in two places, `policy/hf.py` and `measure/frontier/covector.py`, both spelled
`lo = pad + int(tok.meta.get("n_prompt_tokens", 0))`.

On the acceptance fixture the offsets are `[0, 15, 6]`. Row 0 is right by luck. Row 1's index 12
lands three tokens inside its own left padding, where every position carries the same hidden state.
Row 2's index 13 is six tokens early, inside the prompt. The registered acceptance cannot see any of
it: it compares the capture against a hand slice of the same tensor at the same index, so both sides
move together, and it asserts the recorded position equals the index it passed in, which is a
tautology.

That costs the assertion's second clause, "a per-row index list addresses a different token on every
row". With `[13, 12, 13]` it does not: rows 0 and 2 name one index, and row 1's 12 and 13 are
byte-identical padding, so a resolver that ignored the list and broadcast its first entry returns a
byte-identical matrix. With the pad offset the indices are `[13, 27, 19]`, all distinct, and a
broadcast is separable from a per-row read.

Nothing under `src/` changes. This file is the fireable version of that clause, and it is the
coordinate a link 2 caller has to pass.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from reward_lens.core.types import Site  # noqa: E402
from reward_lens.policy.hf import from_pretrained  # noqa: E402
from reward_lens.runtime.backend import CaptureSpec  # noqa: E402
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


@pytest.fixture(scope="module")
def geometry(policy):
    """Everything the collated coordinate is computed from, read once from the real collation."""
    tokenized = [policy.tokenize(item) for item in ITEMS]
    batch = policy.runtime.collate(tokenized)
    assert "offsets" in batch.meta, (
        "the collated batch carries no 'offsets', so the left-pad width cannot be read and no "
        "caller can turn n_prompt_tokens into a capture index"
    )
    offsets = [int(o) for o in batch.meta["offsets"]]
    n_prompt = [int(t.meta["n_prompt_tokens"]) for t in tokenized]
    lengths = [len(t.input_ids) for t in tokenized]
    return offsets, n_prompt, lengths


def _naive(n_prompt: list[int]) -> list[int]:
    """What the registered acceptance passes: a per-item index, with no pad offset."""
    return [n - 1 for n in n_prompt]


def _collated(offsets: list[int], n_prompt: list[int]) -> list[int]:
    """The last prompt token in collated coordinates, the library's own `pad + n_prompt` idiom."""
    return [o + n - 1 for o, n in zip(offsets, n_prompt)]


def _capture(policy, position):
    spec = CaptureSpec(sites=(SITE,), position=position, full_sequence=False, dtype="float32")
    return next(iter(policy.capture(ITEMS, spec)))


def _whole(policy):
    spec = CaptureSpec(sites=(SITE,), position=None, full_sequence=True, dtype="float32")
    whole = next(iter(policy.capture(ITEMS, spec))).tensors[SITE]
    assert whole.ndim == 3
    return whole


def test_the_fixture_pads_and_the_naive_index_lands_in_the_padding(geometry) -> None:
    """The guard on everything below. If collation stopped padding, this fails rather than skips."""
    offsets, n_prompt, _ = geometry
    assert any(offset > 0 for offset in offsets), (
        f"offsets {offsets} say no row was padded, so this file cannot distinguish a per-item "
        f"index from a collated one and proves nothing"
    )
    naive = _naive(n_prompt)
    inside = [index < offset for index, offset in zip(naive, offsets)]
    assert any(inside), (
        f"naive indices {naive} against offsets {offsets}: none of them lands inside the padding, "
        f"so the coordinate confusion this file exists for is not present in the fixture"
    )


def test_the_capture_at_the_collated_boundary_matches_a_forward_and_slice(policy, geometry) -> None:
    offsets, n_prompt, _ = geometry
    boundaries = _collated(offsets, n_prompt)
    whole = _whole(policy)
    by_hand = torch.stack([whole[row, index] for row, index in enumerate(boundaries)])

    captured = _capture(policy, PositionSpec("explicit", detail=boundaries)).tensors[SITE]

    assert captured.ndim == 2
    assert captured.shape == by_hand.shape
    assert torch.allclose(captured, by_hand, atol=1e-6)


def test_a_per_row_list_addresses_a_different_token_on_every_row(policy, geometry) -> None:
    """The registered assertion's second clause, made fireable.

    A resolver that read only the first entry and broadcast it returns the same matrix on the
    acceptance fixture's `[13, 12, 13]`, because rows 0 and 2 share an index and row 1's two
    candidates are both padding. On the collated boundaries it cannot.
    """
    offsets, n_prompt, _ = geometry
    boundaries = _collated(offsets, n_prompt)
    assert len(set(boundaries)) == len(boundaries), (
        f"collated boundaries {boundaries} are not distinct, so a broadcast would be "
        f"indistinguishable from a per-row read on this fixture"
    )

    per_row = _capture(policy, PositionSpec("explicit", detail=boundaries))
    assert per_row.positions == [[index] for index in boundaries]

    for index in boundaries:
        broadcast = _capture(policy, PositionSpec("explicit", detail=index)).tensors[SITE]
        assert not torch.allclose(broadcast, per_row.tensors[SITE], atol=1e-4), (
            f"broadcasting {index} across the batch returned the per-row matrix, so the list was "
            f"never read per row"
        )


def test_the_padding_independent_negative_index_reaches_the_same_token(policy, geometry) -> None:
    """What a caller should actually pass, since it needs no `offsets`.

    `-(n_completion + 1)` counts back from each row's own last valid token, which the resolver
    already resolves per row, so it survives any padding the collation does.
    """
    offsets, n_prompt, lengths = geometry
    boundaries = _collated(offsets, n_prompt)
    negative = [-(length - n + 1) for length, n in zip(lengths, n_prompt)]

    from_negative = _capture(policy, PositionSpec("explicit", detail=negative))
    from_absolute = _capture(policy, PositionSpec("explicit", detail=boundaries))

    assert from_negative.positions == from_absolute.positions == [[i] for i in boundaries]
    assert torch.equal(from_negative.tensors[SITE], from_absolute.tensors[SITE])


def test_the_collated_boundary_is_neither_the_final_token_nor_the_naive_index(
    policy, geometry
) -> None:
    """The row's whole point: a pre-generation read that the final token cannot supply.

    The naive index is included because it is what the registered acceptance reads, and on two of
    three rows it is a different vector.
    """
    offsets, n_prompt, _ = geometry
    boundaries = _collated(offsets, n_prompt)
    naive = _naive(n_prompt)

    at_boundary = _capture(policy, PositionSpec("explicit", detail=boundaries)).tensors[SITE]
    at_final = _capture(policy, PositionSpec("final")).tensors[SITE]
    at_naive = _capture(policy, PositionSpec("explicit", detail=naive)).tensors[SITE]

    assert at_boundary.shape == at_final.shape == at_naive.shape
    assert not torch.allclose(at_boundary, at_final, atol=1e-4)
    assert not torch.allclose(at_boundary, at_naive, atol=1e-4)

    differing = [
        row
        for row in range(len(ITEMS))
        if not torch.allclose(at_boundary[row], at_naive[row], atol=1e-4)
    ]
    assert differing == [row for row in range(len(ITEMS)) if boundaries[row] != naive[row]], (
        f"rows {differing} differ from the naive read; boundaries {boundaries}, naive {naive}"
    )
