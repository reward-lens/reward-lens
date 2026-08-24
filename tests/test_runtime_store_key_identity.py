"""The activation store's key carries the whole capture identity (BLK-003).

The defect this file holds the library to, from `runtime/store.py` at `59a5f5a`::

    position = getattr(spec.position, "kind", "final") if spec.position else "final"
    key = self.key(model_fp, dataset, tuple(spec.sites), position, spec.dtype, intervention_fp)

`spec.position.detail` is never read. For the ``explicit`` and ``judgment`` kinds the detail **is**
the position: it is the resolved token index. So a read at the last prompt token and a read at
``T_pre = 16`` generated tokens, on one input, one bank, one dtype and one intervention, are one
cache entry, and whichever runs second is served the first one's tensor. Part 7.1 of the design
takes exactly those two reads, which is why this is a `STOP` row rather than a latent hazard: the
completion-position vector served under the prompt-position key makes the probe read the exploit's
own tokens.

Two further components of the same identity were also absent and are added here:

* ``spec.full_sequence`` decides whether the capture keeps every token position or only the
  resolved ones. Two specs differing only in that flag produced one key and therefore one shard,
  and the shapes are not even the same. This is not in the ledger; see
  ``chain/repair/proofs/BLK-003/ERRATUM-FULL-SEQUENCE.md``.
* the ``span_ends`` detail names which span kind the ends are taken from, so two span kinds on one
  input collided in the same way ``explicit`` did.

**Why these tests assert on the miss counter.** The row says so, and the reason is that a value
comparison can pass by luck. Two reads at two token positions can return numerically close or even
equal tensors on a small fixture, and a test that only compares values then fails for a reason that
has nothing to do with the cache, or passes while the collision is still there. ``store.misses`` is
the direct observation of whether a second capture was computed. Tensor inequality and the shard
count are here as corroboration, not as the proof; both can also be evaluated at `59a5f5a`, where
the counter does not exist.

The signal is a fake. ``ActivationStore.get_or_compute`` reads exactly two things off a signal,
``meta.fingerprint`` and ``capture(view, spec)``, and errata E-16 records that the method has zero
call sites anywhere in the tree, so this fixture is the caller as well as the subject. The fake
resolves the position through the real ``PositionSpec.resolve`` and returns a tensor that is a
function of the resolved indices, so a served-from-cache tensor is recognisable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# The caller the library does not have (errata E-16)
# ---------------------------------------------------------------------------


class _FixedMeta:
    """The only part of a signal's meta that ``get_or_compute`` reads."""

    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint


class PositionSensitiveSignal:
    """A signal whose capture depends only on the resolved token positions.

    Every capture is recorded in ``self.calls`` so the test can say how many computations happened
    independently of what the store's own counters claim.
    """

    def __init__(self, fingerprint: str = "mfp:fixed-for-this-test") -> None:
        self.meta = _FixedMeta(fingerprint)
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def _tokens(self, view):
        from reward_lens.signals.base import TokenizedInput

        text = " ".join(str(v) for v in view)
        ids = list(range(8))
        return TokenizedInput(input_ids=ids, attention_mask=[1] * len(ids), text=text)

    def capture(self, view, spec):
        from reward_lens.runtime.backend import Capture

        tokens = self._tokens(view)
        indices = spec.position.resolve(tokens) if spec.position else [len(tokens) - 1]
        self.calls.append((getattr(spec.position, "kind", "final"), tuple(indices)))
        rows = len(indices) if spec.full_sequence is False else len(tokens)
        # The payload is a function of the resolved indices, so a tensor served from another
        # position's entry is visible in the values as well as in the counter.
        base = float(sum(indices) + 1)
        tensors = {site: torch.full((rows, 4), base, dtype=torch.float32) for site in spec.sites}
        return [Capture(tensors=tensors, positions=[list(indices)], dtype=spec.dtype)]


def _spec(position=None, *, full_sequence: bool = False):
    from reward_lens.core.types import Site
    from reward_lens.runtime.backend import CaptureSpec

    return CaptureSpec(
        sites=(Site(1, "resid_post"),),
        position=position,
        full_sequence=full_sequence,
        dtype="float32",
    )


def _store():
    from reward_lens.runtime.store import ActivationStore

    return ActivationStore(Path(tempfile.mkdtemp(prefix="rl-store-key-")))


VIEW = [("q0", "the cat sat on the mat")]


# ---------------------------------------------------------------------------
# BLK-003's closure proof
# ---------------------------------------------------------------------------


def test_two_explicit_positions_on_one_input_are_two_computations():
    """BLK-003's closure proof, asserted on the miss counter.

    One input, one model, one site set, one dtype, one intervention; two explicit token indices.
    Two reads must be two computations. At `59a5f5a` the key folded only the position **kind**, so
    both reads produced the string ``"explicit"`` and the second was a hit.
    """
    from reward_lens.signals.base import PositionSpec

    signal = PositionSensitiveSignal()
    store = _store()

    at_prompt_end = _spec(PositionSpec(kind="explicit", detail=[3]))
    at_generated_16 = _spec(PositionSpec(kind="explicit", detail=[6]))

    first = store.get_or_compute(signal, VIEW, at_prompt_end, dataset_id="ds:fixed")
    second = store.get_or_compute(signal, VIEW, at_generated_16, dataset_id="ds:fixed")

    assert store.misses == 2, (
        f"the store recorded {store.misses} miss(es) for two different explicit token positions on "
        f"one input. A second miss is the only evidence the second position was actually read; one "
        f"miss means the second read was served from the first position's entry. "
        f"captures actually computed: {signal.calls}"
    )
    assert store.hits == 0
    assert signal.calls == [("explicit", (3,)), ("explicit", (6,))]

    # Corroboration, evaluable at the baseline where the counter does not exist.
    site = at_prompt_end.sites[0]
    assert not torch.equal(first.get(site), second.get(site))


def test_two_judgment_positions_on_one_input_are_two_computations():
    """The same collision on the other detail-carrying kind.

    ``judgment`` positions are detected per rollout by the signal and passed back through
    ``detail``, so two rollouts of one input whose verdict token lands in different places are the
    ordinary case rather than the exotic one.
    """
    from reward_lens.signals.base import PositionSpec

    signal = PositionSensitiveSignal()
    store = _store()

    store.get_or_compute(signal, VIEW, _spec(PositionSpec("judgment", [2])), dataset_id="ds:fixed")
    store.get_or_compute(signal, VIEW, _spec(PositionSpec("judgment", [5])), dataset_id="ds:fixed")

    assert (store.misses, store.hits) == (2, 0), (
        f"two judgment positions collided: misses={store.misses}, hits={store.hits}"
    )


def test_two_span_kinds_on_one_input_are_two_computations():
    """``span_ends`` carries the span kind in ``detail`` and collided the same way."""
    from reward_lens.core.types import Span
    from reward_lens.signals.base import PositionSpec, TokenizedInput

    class SpanSignal(PositionSensitiveSignal):
        def _tokens(self, view):
            ids = list(range(8))
            return TokenizedInput(
                input_ids=ids,
                attention_mask=[1] * len(ids),
                text=" ".join(str(v) for v in view),
                spans=(Span(kind="receipt", start=0, end=2), Span(kind="critique", start=4, end=6)),
            )

    signal = SpanSignal()
    store = _store()

    store.get_or_compute(
        signal, VIEW, _spec(PositionSpec("span_ends", "receipt")), dataset_id="ds:fixed"
    )
    store.get_or_compute(
        signal, VIEW, _spec(PositionSpec("span_ends", "critique")), dataset_id="ds:fixed"
    )

    assert (store.misses, store.hits) == (2, 0), (
        f"two span kinds collided: misses={store.misses}, hits={store.hits}"
    )


def test_full_sequence_is_part_of_the_capture_identity():
    """``full_sequence`` changes what is captured and was absent from the key.

    Not a ledger row. Found while enumerating the identity tuple BLK-003's method step demands, and
    recorded as an erratum proposal in the row's proof directory. Included here because the row's
    own cheap-fake warning is that one field is added and the collision survives on every pair of
    dimensions nobody enumerated.
    """
    from reward_lens.signals.base import PositionSpec

    signal = PositionSensitiveSignal()
    store = _store()
    position = PositionSpec(kind="explicit", detail=[3])

    resolved_only = store.get_or_compute(
        signal, VIEW, _spec(position, full_sequence=False), dataset_id="ds:fixed"
    )
    whole_sequence = store.get_or_compute(
        signal, VIEW, _spec(position, full_sequence=True), dataset_id="ds:fixed"
    )

    assert (store.misses, store.hits) == (2, 0), (
        f"a full-sequence capture was served from a position-resolved entry: "
        f"misses={store.misses}, hits={store.hits}"
    )
    site = _spec(position).sites[0]
    assert resolved_only.get(site).shape != whole_sequence.get(site).shape


# ---------------------------------------------------------------------------
# The counter has to be able to say "hit", or a miss means nothing
# ---------------------------------------------------------------------------


def test_the_same_position_read_twice_is_a_hit():
    """A key that separated everything would satisfy every test above and destroy the cache."""
    from reward_lens.signals.base import PositionSpec

    signal = PositionSensitiveSignal()
    store = _store()
    spec = _spec(PositionSpec(kind="explicit", detail=[3]))

    store.get_or_compute(signal, VIEW, spec, dataset_id="ds:fixed")
    store.get_or_compute(signal, VIEW, spec, dataset_id="ds:fixed")

    assert (store.misses, store.hits) == (1, 1)
    assert len(signal.calls) == 1, "the second read recomputed; the cache is not caching"


def test_an_equal_position_spec_built_separately_is_still_a_hit():
    """The key is a function of the position's value, not of the spec object's identity."""
    from reward_lens.signals.base import PositionSpec

    signal = PositionSensitiveSignal()
    store = _store()

    store.get_or_compute(
        signal, VIEW, _spec(PositionSpec(kind="explicit", detail=[3])), dataset_id="ds:fixed"
    )
    store.get_or_compute(
        signal, VIEW, _spec(PositionSpec(kind="explicit", detail=[3])), dataset_id="ds:fixed"
    )

    assert (store.misses, store.hits) == (1, 1)


def test_explicit_index_lists_are_order_sensitive_but_not_type_sensitive():
    """``[3]`` and ``(3,)`` are the same read; ``[3, 6]`` and ``[6, 3]`` are not.

    ``PositionSpec.resolve`` returns the indices in the order given, and the capture's rows come
    back in that order, so the order is part of the identity. The container type is not.
    """
    from reward_lens.signals.base import PositionSpec

    signal = PositionSensitiveSignal()
    store = _store()

    store.get_or_compute(signal, VIEW, _spec(PositionSpec("explicit", [3])), dataset_id="ds:f")
    store.get_or_compute(signal, VIEW, _spec(PositionSpec("explicit", (3,))), dataset_id="ds:f")
    assert (store.misses, store.hits) == (1, 1)

    store.get_or_compute(signal, VIEW, _spec(PositionSpec("explicit", [3, 6])), dataset_id="ds:f")
    store.get_or_compute(signal, VIEW, _spec(PositionSpec("explicit", [6, 3])), dataset_id="ds:f")
    assert (store.misses, store.hits) == (3, 1)


# ---------------------------------------------------------------------------
# The refusal: a field added to a key with no refusal is a field with no writer
# ---------------------------------------------------------------------------


def test_key_refuses_a_detail_carrying_kind_passed_as_a_bare_string():
    """``key()`` is public. A caller passing ``"explicit"`` would restore the collision.

    The repair is not "``get_or_compute`` now passes more"; it is that the key cannot be computed
    at all for a kind whose identity lives in its detail unless the detail is supplied.
    """
    from reward_lens.core.types import ModelFP, Site

    store = _store()
    for kind in ("explicit", "judgment", "span_ends"):
        with pytest.raises(ValueError, match="detail"):
            store.key(
                ModelFP("mfp:x"),
                "ds:fixed",
                (Site(1, "resid_post"),),
                kind,
                "float32",
            )


def test_key_still_accepts_the_kinds_that_carry_no_detail():
    """``final`` and ``all`` resolve from the input alone, which is already in the key."""
    from reward_lens.core.types import ModelFP, Site

    store = _store()
    args = (ModelFP("mfp:x"), "ds:fixed", (Site(1, "resid_post"),))
    final = store.key(*args, "final", "float32")
    every = store.key(*args, "all", "float32")
    assert final != every
    assert isinstance(final, str) and len(final) == 32


def test_key_is_stable_across_calls_and_moves_with_every_component():
    """One canonical key per identity, and no component silently ignored."""
    from reward_lens.core.types import ModelFP, Site
    from reward_lens.signals.base import PositionSpec

    store = _store()
    site = Site(1, "resid_post")
    base = dict(
        model_fp=ModelFP("mfp:a"),
        dataset="ds:a",
        sites=(site,),
        position=PositionSpec("explicit", [3]),
        dtype="float32",
        intervention_fp="none",
        full_sequence=False,
    )
    reference = store.key(**base)
    assert reference == store.key(**base), "the key is not a function of its arguments"

    moved = {
        "model_fp": ModelFP("mfp:b"),
        "dataset": "ds:b",
        "sites": (Site(2, "resid_post"),),
        "position": PositionSpec("explicit", [6]),
        "dtype": "float16",
        "intervention_fp": "ifp:patched",
        "full_sequence": True,
    }
    for field, value in moved.items():
        variant = dict(base)
        variant[field] = value
        assert store.key(**variant) != reference, f"the key does not move with {field}"
