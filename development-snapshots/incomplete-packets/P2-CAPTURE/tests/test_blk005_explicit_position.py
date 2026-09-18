"""BLK-005: activation capture at an explicit token position, through the public path.

At the baseline the coordinate the design reads at is not expressible. `capture_sites` hard-codes
`PositionSpec("final")` and takes no position argument; `_capture_matrix` takes none either and
`concepts/probes.py` does not mention `PositionSpec` at all; and both runtimes collapse the whole
spec to a boolean, so any kind other than `final` becomes `full_sequence=True`, returns the entire
`(B, T, d)` tensor, and hands back a `Capture` whose `positions` list is **empty** -- which
contradicts that field's own contract in `runtime/backend.py`, "the token indices actually read
per item". `PositionSpec.resolve` has zero callers anywhere in `src`.

`CaptureMount._store` already gathers one arbitrary index per row with `hidden[batch_idx, pos]`,
so the index only has to arrive.

**The `(B, K)` problem, and what is done about it.** `CaptureMount`'s API is one position per
batch row: `positions` is a `(B,)` tensor and the gather is `hidden[batch_idx, pos]`. A
`PositionSpec` can resolve to more than one index per row (`all`, `step_ends`, `span_ends` over a
multi-span item, a list-valued `explicit`), and those counts can differ between rows, so there is
no `(B, K)` tensor to gather and a ragged one is not a tensor at all. Nothing here pretends
otherwise. One index per row is gathered and stored `(B, d)`; more than one on any row keeps the
whole sequence, as it does today, **and records the resolved indices** so the caller can slice
them itself. The change on that path is the position list, not the tensor.

**One off-by-one, named rather than decided.** The row's prose says the design reads "at the last
prompt token"; the row's closure proof says "the boundary index is asserted equal to the recorded
prompt-token count". For a prompt occupying tokens `[0, P)` those are `P - 1` and `P`. `span_ends`
resolves to `span.end`, which is `P` under the half-open convention, so the closure proof's
sentence matches the existing machinery and the prose does not. Both are asserted below, with the
relationship between them spelled out. Which one is the coordinate is a decision for the
integrator, not for this test.
"""

from __future__ import annotations

import pytest
import torch

from reward_lens.core.types import Site
from reward_lens.runtime.backend import CaptureSpec
from reward_lens.signals.base import PositionSpec
from reward_lens.signals.loaders import from_tiny

#: Two items of deliberately different length, so a per-row index is not a batch-wide constant and
#: an implementation that resolves one index and broadcasts it fails.
ITEMS = (
    ("Question one?", "alpha beta gamma"),
    ("A much longer question here about things?", "d e"),
)


@pytest.fixture(scope="module")
def signal():
    return from_tiny(seed=7, conformance_quickcheck=False)


@pytest.fixture(scope="module")
def batch(signal):
    return signal.runtime.collate([signal.tokenize(it) for it in ITEMS])


#: Layer 0's `resid_post` is exactly `output_hidden_states[1]` on this architecture, with no final
#: norm in the way, so the reference below goes through the plain transformers API and shares no
#: code with `CaptureMount`. The last layer would not do: `hidden_states[-1]` is post-final-norm.
SITE = Site(0, "resid_post")

#: An item-local token index the two rows share. **`PositionSpec` indices are in the item's own
#: token coordinates, not in the padded batch's columns**, which is what `PositionSpec.resolve`
#: means by "concrete token indices for a given tokenized input" and what every other kind already
#: does. `collate` left-pads, so the same local index lands in a different column on each row, and
#: that is the off-by-pad error a capture path gets wrong once: it returns a real activation from
#: the wrong token and nothing about the vector says so.
LOCAL_INDEX = 3


def padded(batch, local_index):
    """`local_index` in each row's padded coordinates."""
    return [local_index + int(pad) for pad in batch.meta["offsets"]]


def reference_slice(runtime, batch, index_per_row):
    """A manual forward and slice, through `output_hidden_states` rather than through any hook."""
    with torch.no_grad():
        out = runtime.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
    rows = torch.arange(len(index_per_row))
    cols = torch.tensor(list(index_per_row))
    return out.hidden_states[1][rows, cols].to(torch.float32)


def prompt_span_items():
    """The same two items with an explicit `prompt` character span over the templated text.

    The tiny model has no chat template, so `ClassifierRM._template` renders
    `User: {prompt}\\nAssistant: {response}` and the prompt region is everything before the
    `\\nAssistant:` marker. The span is in characters; `tokenize` maps it into token coordinates
    through the fast tokenizer's own offset mapping, so nothing here counts tokens by hand.
    """
    out = []
    for prompt, response in ITEMS:
        text = f"User: {prompt}\nAssistant: {response}"
        out.append(
            {
                "prompt": prompt,
                "response": response,
                "spans": [(0, text.index("\nAssistant:"), "prompt")],
            }
        )
    return out


# ---------------------------------------------------------------------------
# the fixture's own control: the two rows must not want the same index
# ---------------------------------------------------------------------------


def test_the_two_rows_have_different_prompt_lengths(signal):
    tokenized = [signal.tokenize(item) for item in prompt_span_items()]
    counts = [len([s for s in t.spans if s.kind == "prompt"]) for t in tokenized]
    assert counts == [1, 1], "both rows need exactly one prompt span or the test proves nothing"
    ends = [next(s.end for s in t.spans if s.kind == "prompt") for t in tokenized]
    assert ends[0] != ends[1], "a per-row index that happens to be constant tests broadcasting"


# ---------------------------------------------------------------------------
# the row: an explicit index reaches the gather
# ---------------------------------------------------------------------------


def test_an_explicit_index_is_gathered_rather_than_returning_the_whole_sequence(signal, batch):
    spec = CaptureSpec(
        sites=(SITE,), position=PositionSpec("explicit", detail=LOCAL_INDEX), dtype="float32"
    )
    _raw, capture = signal.runtime.forward_with_capture(batch, spec)
    tensor = capture.tensors[SITE]
    assert tensor.shape == (len(ITEMS), signal.meta.d_model), (
        f"an explicit position returned {tuple(tensor.shape)}; the whole sequence is what the "
        f"collapse to full_sequence=True produced at the baseline"
    )


def test_the_explicit_capture_equals_a_manual_forward_and_slice(signal, batch):
    """The row's closure proof, first half, at its stated tolerance."""
    spec = CaptureSpec(
        sites=(SITE,), position=PositionSpec("explicit", detail=LOCAL_INDEX), dtype="float32"
    )
    _raw, capture = signal.runtime.forward_with_capture(batch, spec)
    reference = reference_slice(signal.runtime, batch, padded(batch, LOCAL_INDEX))
    assert torch.allclose(capture.tensors[SITE], reference, atol=1e-6, rtol=0)


def test_the_captured_positions_are_recorded_rather_than_empty(signal, batch):
    """`Capture.positions` promises "the token indices actually read per item".

    And the two rows land in different columns for one item-local index, because they are padded
    by different amounts. An implementation that treated the spec's index as a batch column would
    return the same column for both and read a pad token on the shorter row.
    """
    spec = CaptureSpec(
        sites=(SITE,), position=PositionSpec("explicit", detail=LOCAL_INDEX), dtype="float32"
    )
    _raw, capture = signal.runtime.forward_with_capture(batch, spec)
    expected = padded(batch, LOCAL_INDEX)
    assert expected[0] != expected[1], "the two rows must be padded differently or this is empty"
    assert capture.positions == [[i] for i in expected]


def test_a_per_row_index_is_not_broadcast_from_the_first_row(signal):
    """Two rows, two different indices, and the vectors have to differ accordingly.

    Resolving one index and reusing it is the shape of mistake this catches: it would pass every
    assertion above, because above the two rows deliberately share an index.
    """
    runtime = signal.runtime
    tokenized = [signal.tokenize(item) for item in prompt_span_items()]
    token_batch = runtime.collate(tokenized)
    spec = CaptureSpec(
        sites=(SITE,), position=PositionSpec("span_ends", detail="prompt"), dtype="float32"
    )
    _raw, capture = runtime.forward_with_capture(token_batch, spec)

    offsets = token_batch.meta["offsets"]
    expected = [
        next(s.end for s in tok.spans if s.kind == "prompt") + pad
        for tok, pad in zip(tokenized, offsets)
    ]
    assert capture.positions == [[i] for i in expected]
    assert expected[0] != expected[1], "the fixture stopped discriminating"
    reference = reference_slice(runtime, token_batch, expected)
    assert torch.allclose(capture.tensors[SITE], reference, atol=1e-6, rtol=0)


def test_the_boundary_index_equals_the_recorded_prompt_token_count(signal):
    """The row's closure proof, second half, on every row.

    And the off-by-one the row leaves open, stated as an assertion rather than as a comment: the
    boundary `span_ends` resolves to is the prompt token *count*, and the last prompt token is one
    before it. Both are reachable; only one of them is the design's coordinate.
    """
    runtime = signal.runtime
    tokenized = [signal.tokenize(item) for item in prompt_span_items()]
    token_batch = runtime.collate(tokenized)
    spec = CaptureSpec(
        sites=(SITE,), position=PositionSpec("span_ends", detail="prompt"), dtype="float32"
    )
    _raw, capture = runtime.forward_with_capture(token_batch, spec)

    for row, (tok, pad) in enumerate(zip(tokenized, token_batch.meta["offsets"])):
        span = next(s for s in tok.spans if s.kind == "prompt")
        prompt_token_count = span.end - span.start
        assert span.start == 0, "this assertion reads the count off a span that starts at zero"
        (index,) = capture.positions[row]
        assert index - pad == prompt_token_count
        assert index - pad - 1 == prompt_token_count - 1  # the last prompt token, one before


def test_a_multi_position_kind_keeps_the_sequence_and_still_records_where_it_read(signal, batch):
    """The `(B, K)` case. `CaptureMount` gathers one index per row, so `all` cannot be gathered.

    What changes is not the tensor, which is the whole sequence as before, but the position list,
    which used to come back empty and now says which indices are valid on each row.
    """
    spec = CaptureSpec(sites=(SITE,), position=PositionSpec("all"), dtype="float32")
    _raw, capture = signal.runtime.forward_with_capture(batch, spec)
    tensor = capture.tensors[SITE]
    assert tensor.shape[0] == len(ITEMS)
    assert tensor.ndim == 3, "a multi-index kind keeps the sequence"
    assert len(capture.positions) == len(ITEMS)
    assert all(len(p) > 1 for p in capture.positions)
    tokenized = batch.meta["tokenized"]
    offsets = batch.meta["offsets"]
    for row, (tok, pad) in enumerate(zip(tokenized, offsets)):
        assert capture.positions[row] == [i + pad for i in tok.valid_positions()]


def test_full_sequence_still_returns_the_sequence_and_now_says_where(signal, batch):
    spec = CaptureSpec(
        sites=(SITE,),
        position=PositionSpec("explicit", detail=LOCAL_INDEX),
        full_sequence=True,
        dtype="float32",
    )
    _raw, capture = signal.runtime.forward_with_capture(batch, spec)
    assert capture.tensors[SITE].ndim == 3
    assert capture.positions == [[i] for i in padded(batch, LOCAL_INDEX)]


def test_the_final_position_path_is_unchanged(signal, batch):
    """The default is the overwhelming majority of the 22 references to `capture_sites` and it
    must not move. Both the tensor shape and the recorded positions are as they were."""
    spec = CaptureSpec(sites=(SITE,), position=PositionSpec("final"), dtype="float32")
    _raw, capture = signal.runtime.forward_with_capture(batch, spec)
    final_pos = signal.runtime._final_positions(batch.attention_mask)
    assert capture.tensors[SITE].shape == (len(ITEMS), signal.meta.d_model)
    assert capture.positions == [[int(p)] for p in final_pos.tolist()]
    spec_none = CaptureSpec(sites=(SITE,), position=None, dtype="float32")
    _raw2, capture2 = signal.runtime.forward_with_capture(batch, spec_none)
    assert torch.equal(capture.tensors[SITE], capture2.tensors[SITE])
    assert capture.positions == capture2.positions


def test_an_out_of_range_index_is_refused_rather_than_silently_clamped(signal, batch):
    """The baseline clamps: `pos.clamp_(0, T - 1)` inside `CaptureMount._store`.

    A clamp turns an off-by-one into a plausible vector at the last token, which is exactly the
    reading the whole row exists to stop, and it does it without a word. It is also an in-place
    op on the result of `.to(device)`, which is the caller's own tensor when it is already on that
    device, so the clamp silently rewrote the indices the caller passed in.
    """
    too_far = int(batch.input_ids.shape[1]) + 5
    spec = CaptureSpec(
        sites=(SITE,), position=PositionSpec("explicit", detail=too_far), dtype="float32"
    )
    with pytest.raises(IndexError):
        signal.runtime.forward_with_capture(batch, spec)


# ---------------------------------------------------------------------------
# the composing calls above the runtime
# ---------------------------------------------------------------------------


def test_capture_sites_takes_a_position(signal):
    """`measure/battery/_common.py` hard-coded `PositionSpec("final")` with no way past it."""
    from reward_lens.measure.battery._common import capture_sites

    tensors = capture_sites(
        signal, list(ITEMS), (SITE,), position=PositionSpec("explicit", detail=LOCAL_INDEX)
    )
    runtime = signal.runtime
    token_batch = runtime.collate([signal.tokenize(it) for it in ITEMS])
    reference = reference_slice(runtime, token_batch, padded(token_batch, LOCAL_INDEX))
    assert torch.allclose(tensors[SITE], reference, atol=1e-6, rtol=0)


def test_capture_sites_defaults_to_the_final_position(signal):
    from reward_lens.measure.battery._common import capture_sites

    default = capture_sites(signal, list(ITEMS), (SITE,))
    explicit = capture_sites(signal, list(ITEMS), (SITE,), position=PositionSpec("final"))
    assert torch.equal(default[SITE], explicit[SITE])


def test_capture_probe_inputs_takes_a_position(signal):
    """`concepts/probes.py` had no occurrence of `PositionSpec` at all."""
    import numpy as np

    from reward_lens.concepts.probes import capture_probe_inputs

    class Side:
        def __init__(self, text):
            self.text = text

    class Pair:
        def __init__(self, prompt, chosen, rejected, seed):
            self.prompt_text = prompt
            self.chosen = Side(chosen)
            self.rejected = Side(rejected)
            self.seed_id = seed

    view = [Pair("Question one?", "alpha beta gamma", "d e", "s0")]

    def target(item, side):
        return 1 if side == "chosen" else 0

    at_final = capture_probe_inputs(signal, view, target, (SITE,))
    at_three = capture_probe_inputs(
        signal, view, target, (SITE,), position=PositionSpec("explicit", detail=LOCAL_INDEX)
    )
    assert at_final.features[SITE].shape == at_three.features[SITE].shape
    assert not np.allclose(at_final.features[SITE], at_three.features[SITE]), (
        "the position argument was accepted and then ignored"
    )


# ---------------------------------------------------------------------------
# the primitive that had no callers
# ---------------------------------------------------------------------------


def test_position_spec_resolve_is_called_on_the_capture_path(signal, batch, monkeypatch):
    """`PositionSpec.resolve` is defined at `signals/base.py:87` and had zero callers in `src`.

    A capture path that reimplements the resolution beside it is two definitions of one rule, and
    the one that is never called is the one that drifts. This asserts the shipped method is the
    one the runtime uses, by counting calls to it.
    """
    calls: list[str] = []
    original = PositionSpec.resolve

    def counting(self, tokens):
        calls.append(self.kind)
        return original(self, tokens)

    monkeypatch.setattr(PositionSpec, "resolve", counting)
    spec = CaptureSpec(
        sites=(SITE,), position=PositionSpec("explicit", detail=LOCAL_INDEX), dtype="float32"
    )
    signal.runtime.forward_with_capture(batch, spec)
    assert calls == ["explicit"] * len(ITEMS), "one resolve per row, through the shipped method"


# ---------------------------------------------------------------------------
# the policy runtime, which carries the identical collapse
# ---------------------------------------------------------------------------


def test_the_policy_runtime_resolves_positions_the_same_way():
    """`policy/hf.py` and `runtime/hf.py` were structurally identical at the defect and have to
    stay identical at the repair, or the same coordinate means two things on the two sides."""
    from reward_lens.policy.hf import from_pretrained

    policy = from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", contrast=(" yes", " no"))
    runtime = policy.runtime
    tokenized = [policy.tokenize(it) for it in (("count upward from 15", " 16 17"),)]
    token_batch = runtime.collate(tokenized)
    site = Site(0, "resid_post")
    spec = CaptureSpec(sites=(site,), position=PositionSpec("explicit", detail=2), dtype="float32")
    _raw, capture = runtime.forward_with_capture(token_batch, spec)
    assert capture.tensors[site].ndim == 2
    assert capture.positions == [[2]]
    with torch.no_grad():
        out = runtime.model(
            input_ids=token_batch.input_ids,
            attention_mask=token_batch.attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
    reference = out.hidden_states[1][torch.arange(1), torch.tensor([2])].to(torch.float32)
    assert torch.allclose(capture.tensors[site], reference, atol=1e-6, rtol=0)
