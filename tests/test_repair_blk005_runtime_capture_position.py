"""BLK-005, the second consumer: `HFRuntime` honours an explicit position too.

`resolve_capture_positions` has two callers. `policy/hf.py:260` is one and
`tests/test_repair_blk005_explicit_position.py` pins it, because every model-backed test in that
file reaches the runtime through `from_pretrained`, which builds an `HFPolicyRuntime`.
`runtime/hf.py:288` is the other, and it is what `signals.loaders.wrap_hf_model` and `from_tiny`
build. Nothing pinned it.

That was measured rather than assumed. Reverting `runtime/hf.py:288-289` to the pre-repair pair

    single_position = self._is_final_position(spec.position) and not spec.full_sequence
    positions = final_pos if single_position else None

and running the whole library suite leaves it green, so the repaired function had one consumer under
test and one consumer on trust. This file is the second one's test, and it is the same closure proof
the row states: a capture at an explicit index equals a forward and a slice, to `1e-6`.

`from_tiny` rather than the tiny Qwen3 checkpoint, because this path needs no download and the
property is about index arithmetic rather than about a particular architecture. Collation left-pads,
so every row's final valid token is the last column and an explicit index is a different token on
every row: a runtime that had collapsed back to the final position would return the same three
vectors whatever indices it was handed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from reward_lens.core.types import Site  # noqa: E402
from reward_lens.runtime.backend import CaptureSpec  # noqa: E402
from reward_lens.signals.base import PositionSpec  # noqa: E402
from reward_lens.signals.loaders import from_tiny  # noqa: E402

ITEMS = [
    ("Question?", "alpha beta gamma delta epsilon zeta"),
    ("Another question?", "one two"),
    ("A third?", "x y z w"),
]

#: Three indices, none of them the final token, and no two the same. A whole-sequence read that got
#: reported as a single position would have the wrong rank; a collapse back to `final` would have
#: the right rank and the wrong tokens, which is the failure worth catching.
INDICES = [2, 4, 3]


@pytest.fixture(scope="module")
def runtime_and_site():
    signal = from_tiny(seed=5, conformance_quickcheck=False)
    return signal, signal.runtime, Site(signal.meta.n_layers - 1, "resid_post")


def _batch(signal, runtime):
    return runtime.collate([signal.tokenize(item) for item in ITEMS])


def _whole_sequence(runtime, batch, site):
    """One `(B, T, d)` read to slice by hand. The reference the capture is compared against."""
    spec = CaptureSpec(sites=(site,), position=None, full_sequence=True, dtype="float32")
    _, capture = runtime.forward_with_capture(batch, spec)
    whole = capture.tensors[site]
    assert whole.ndim == 3, f"the full-sequence read is {tuple(whole.shape)}"
    return whole


def _explicit(runtime, batch, site, detail):
    spec = CaptureSpec(
        sites=(site,),
        position=PositionSpec("explicit", detail=detail),
        full_sequence=False,
        dtype="float32",
    )
    _, capture = runtime.forward_with_capture(batch, spec)
    return capture


def test_the_fixture_puts_every_final_token_in_the_same_column(runtime_and_site):
    """The premise the rest of the file rests on, checked rather than assumed.

    Collation left-pads, so `final_pos` is the last column for every row and none of `INDICES` is
    it. Without this, a runtime that ignored the indices and read the final token could still agree
    with a hand slice by coincidence.
    """
    signal, runtime, _ = runtime_and_site
    batch = _batch(signal, runtime)
    final_pos = [int(p) for p in runtime._final_positions(batch.attention_mask).tolist()]

    assert len(set(final_pos)) == 1, f"the fixture is not uniformly padded: {final_pos}"
    assert all(index != final_pos[0] for index in INDICES)
    assert len(set(INDICES)) == len(INDICES)


def test_a_capture_at_an_explicit_index_matches_a_forward_and_slice(runtime_and_site):
    """The row's own tolerance, `1e-6`, on the runtime the policy path does not exercise."""
    signal, runtime, site = runtime_and_site
    batch = _batch(signal, runtime)
    whole = _whole_sequence(runtime, batch, site)
    by_hand = torch.stack([whole[row, index] for row, index in enumerate(INDICES)])

    captured = _explicit(runtime, batch, site, INDICES).tensors[site]

    assert captured.ndim == 2, (
        f"an explicit position returned {tuple(captured.shape)}; before the repair a non-final kind "
        f"set `positions=None, full_sequence=True` and collapsed to a whole-sequence read"
    )
    assert captured.shape == by_hand.shape
    assert torch.allclose(captured, by_hand, atol=1e-6)


def test_the_explicit_read_is_not_the_final_position_read(runtime_and_site):
    """The discriminator against a silent collapse back to `final`.

    A runtime that threw the indices away would return the final-token vectors, which have the same
    rank and the same shape as the answer and are not it.
    """
    signal, runtime, site = runtime_and_site
    batch = _batch(signal, runtime)

    final_spec = CaptureSpec(sites=(site,), position=None, full_sequence=False, dtype="float32")
    _, final_capture = runtime.forward_with_capture(batch, final_spec)
    at_final = final_capture.tensors[site]
    at_indices = _explicit(runtime, batch, site, INDICES).tensors[site]

    assert at_final.shape == at_indices.shape
    assert not torch.allclose(at_final, at_indices, atol=1e-6)


def test_the_capture_records_the_indices_it_actually_read(runtime_and_site):
    """`Capture.positions` is the coordinate a caller reads back, so it has to be the resolved one."""
    signal, runtime, site = runtime_and_site
    batch = _batch(signal, runtime)

    capture = _explicit(runtime, batch, site, INDICES)

    assert capture.positions == [[index] for index in INDICES]


def test_one_index_for_the_whole_batch_is_broadcast(runtime_and_site):
    """A single index means the same column on every row, which is what `resolve` promises."""
    signal, runtime, site = runtime_and_site
    batch = _batch(signal, runtime)
    whole = _whole_sequence(runtime, batch, site)

    capture = _explicit(runtime, batch, site, [2])

    assert capture.positions == [[2], [2], [2]]
    assert torch.allclose(capture.tensors[site], whole[:, 2], atol=1e-6)


def test_the_final_kind_is_unchanged(runtime_and_site):
    """The regression guard: every existing caller passes `final` or `None` and must be untouched."""
    signal, runtime, site = runtime_and_site
    batch = _batch(signal, runtime)
    final_pos = [int(p) for p in runtime._final_positions(batch.attention_mask).tolist()]
    whole = _whole_sequence(runtime, batch, site)

    for position in (None, PositionSpec("final")):
        spec = CaptureSpec(sites=(site,), position=position, full_sequence=False, dtype="float32")
        _, capture = runtime.forward_with_capture(batch, spec)
        by_hand = torch.stack([whole[row, index] for row, index in enumerate(final_pos)])
        assert capture.tensors[site].shape == by_hand.shape
        assert torch.allclose(capture.tensors[site], by_hand, atol=1e-6)
        assert capture.positions == [[index] for index in final_pos]


def test_an_index_past_a_rows_valid_span_is_refused_by_this_runtime_too(runtime_and_site):
    """The clamp the resolver refuses to do, refused through this caller rather than in isolation.

    `CaptureMount._store` clamps, and a clamp turns a wrong index into a plausible vector read at
    the wrong token. The resolver's own test asserts the raise; this asserts the raise survives the
    call site, which is the part a revert would take away.
    """
    signal, runtime, site = runtime_and_site
    batch = _batch(signal, runtime)
    past_the_end = int(batch.input_ids.shape[1]) + 5

    with pytest.raises(ValueError, match="outside that row's valid span"):
        _explicit(runtime, batch, site, [past_the_end, 1, 1])
