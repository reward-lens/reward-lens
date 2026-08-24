"""The model fingerprint separates two checkpoints that share a base directory (BLK-002, BLK-034).

The defect these tests hold the library to: `runtime.fingerprint` resolved a weights directory from
``config._name_or_path`` and hashed the safetensors it found there, and took its adapter component
from a string the caller passed. Under a parameter efficient fine-tune the resolved directory is the
**base** model's, so it is identical for every adapter ever trained on that base, and the string the
three grader-side call sites passed was the adapter object's class name, which is one constant since
family dispatch was removed. Two components of a four-component identity were therefore constant
across a whole training run, and the other two (config, tokenizer) do not move between checkpoints
of one run either. Forty LoRA checkpoints of five seeds shared one ``ModelFP``.

`runtime.store.ActivationStore` is keyed on that fingerprint, so the consequence is not a mislabelled
record. It is that the second checkpoint's capture is never taken: the store finds a shard under the
first checkpoint's key and returns it. Every cross-checkpoint number computed from those activations
is then a comparison of one capture with itself.

Why these tests assert on the store's miss counter rather than on the returned tensors: two distinct
subjects can return numerically close vectors, so a value comparison passes or fails for reasons that
have nothing to do with the cache, and a value comparison that happens to pass on a fixture is exactly
the test that let this defect through. The counter is the direct observation of whether a second
computation happened. The tensor comparisons are here too, as corroboration, not as the proof.

`LoraLinear` below is a structural mock of ``peft.tuners.lora.Linear``: it carries the same
``adapter_layer_names`` contract, the same ``base_layer`` / ``lora_A`` / ``lora_B`` / ``lora_embedding_*``
naming, and the same ``disable_adapters`` and ``merged_adapters`` attributes that the library reads by
duck typing. What it reproduces is the only thing that matters here: a live model whose behaviour is
not determined by the files in the directory its config names.

The mock was written when ``peft`` was declared at ``pyproject.toml:87`` and absent from the
environment, so the whole repair had only ever been exercised against our model of the dependency.
It is installed now, and the cases at the foot of this file run the same closure proof against a
real ``peft.tuners.lora.Linear`` under ``importorskip`` (CB-1061). The mock stays: it is the
coverage a base install without ``peft`` gets, and it is not wrong.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import torch.nn as nn  # noqa: E402
from transformers import LlamaConfig, LlamaForSequenceClassification  # noqa: E402

from reward_lens.runtime.fingerprint import fingerprint  # noqa: E402
from reward_lens.signals.adapters import resolve_adapter  # noqa: E402
from reward_lens.signals.loaders import _build_tokenizer, wrap_hf_model  # noqa: E402

#: The static-check command. ``-I`` skips binary files and ``--exclude-dir`` skips compiled caches,
#: because this test file assembles the forbidden string at runtime and its own ``.pyc`` would
#: otherwise match it. ``build/`` is not searched: it is gitignored and holds a stale copy of every
#: source file, so a bare recursive grep from the repository root reports every hit twice.
_GREP = [
    "grep",
    "-rnI",
    "--fixed-strings",
    "--exclude-dir=__pycache__",
    "--exclude-dir=.git",
    "--exclude-dir=build",
]


# ---------------------------------------------------------------------------
# The fixture: two LoRA checkpoints over one base directory
# ---------------------------------------------------------------------------


class LoraLinear(nn.Module):
    """A structural stand-in for ``peft.tuners.lora.Linear``.

    Mirrors the attributes the library reads by duck typing: ``adapter_layer_names`` (PEFT's own
    declaration of which attributes hold per-adapter weights), the ``base_layer`` indirection that
    renames ``q_proj.weight`` to ``q_proj.base_layer.weight`` in the state dict, per-adapter
    ``nn.ModuleDict`` weights keyed by adapter name, and the ``disable_adapters`` / ``merged_adapters``
    / ``active_adapters`` flags.
    """

    adapter_layer_names = ("lora_A", "lora_B")
    other_param_names = ("r", "lora_alpha", "scaling")

    def __init__(self, base: nn.Linear, rank: int, seed: int, name: str = "default"):
        super().__init__()
        self.base_layer = base
        generator = torch.Generator().manual_seed(seed)
        down = nn.Linear(base.in_features, rank, bias=False)
        up = nn.Linear(rank, base.out_features, bias=False)
        with torch.no_grad():
            down.weight.copy_(torch.randn(down.weight.shape, generator=generator))
            up.weight.copy_(torch.randn(up.weight.shape, generator=generator))
        self.lora_A = nn.ModuleDict({name: down})
        self.lora_B = nn.ModuleDict({name: up})
        self._active_adapter = [name]
        self._disable_adapters = False
        self.merged_adapters: list[str] = []
        self.r = {name: rank}

    @property
    def active_adapters(self) -> list[str]:
        return list(self._active_adapter)

    @property
    def disable_adapters(self) -> bool:
        return self._disable_adapters

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = self.base_layer(x)
        if not self._disable_adapters:
            for name in self._active_adapter:
                out = out + self.lora_B[name](self.lora_A[name](x))
        return out


def _config(tokenizer, d_model: int = 32) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=int(getattr(tokenizer, "vocab_size", 1000) or 1000),
        hidden_size=d_model,
        intermediate_size=2 * d_model,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
        num_labels=1,
        attn_implementation="eager",
    )


def _save_base(config: LlamaConfig, root: Path) -> Path:
    """Write one base checkpoint to disk. Every adapter below sits on top of exactly these files."""
    torch.manual_seed(0)
    base = LlamaForSequenceClassification(config).eval()
    base_dir = root / "base"
    base.save_pretrained(base_dir, safe_serialization=True)
    assert list(base_dir.glob("*.safetensors")), "the fixture must put real weights on disk"
    return base_dir


def _checkpoint(config: LlamaConfig, base_dir: Path, seed: int):
    """One LoRA checkpoint: the base weights of ``base_dir``, plus an adapter that exists only here."""
    torch.manual_seed(0)
    model = LlamaForSequenceClassification(config).eval()
    # What a PEFT wrapper leaves behind: the config still names the base directory, because the
    # config *is* the base model's.
    model.config._name_or_path = str(base_dir)
    for index, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj = LoraLinear(
            layer.self_attn.q_proj, rank=4, seed=seed * 1000 + index
        )
    return model.eval()


@pytest.fixture(scope="module")
def tokenizer():
    return _build_tokenizer("gpt2")


@pytest.fixture(scope="module")
def two_checkpoints(tokenizer):
    """Two checkpoints over one base directory, with weights that are known to differ."""
    root = Path(tempfile.mkdtemp(prefix="rl-fp-"))
    config = _config(tokenizer)
    base_dir = _save_base(config, root)
    first = _checkpoint(config, base_dir, seed=11)
    second = _checkpoint(config, base_dir, seed=143)
    return first, second, base_dir


# ---------------------------------------------------------------------------
# The fixture is what it claims to be
# ---------------------------------------------------------------------------


def test_the_two_checkpoints_have_different_weights_and_the_same_base_directory(two_checkpoints):
    """Establish the premise before asserting anything about it.

    If these two models were identical, or if they did not share a resolved directory, everything
    below would pass for the wrong reason. This is the assertion that stops that.
    """
    first, second, base_dir = two_checkpoints

    assert first.config._name_or_path == str(base_dir)
    assert second.config._name_or_path == str(base_dir)

    first_state = first.state_dict()
    second_state = second.state_dict()
    assert set(first_state) == set(second_state), "the two checkpoints must differ in values only"

    adapter_keys = [k for k in first_state if "lora_" in k]
    assert adapter_keys, "the fixture must actually carry adapter tensors"
    assert any(not torch.equal(first_state[k], second_state[k]) for k in adapter_keys), (
        "the two adapters must hold different weights"
    )

    base_keys = [k for k in first_state if "lora_" not in k]
    assert all(torch.equal(first_state[k], second_state[k]) for k in base_keys), (
        "the base weights must be identical, which is what makes the directory hash collide"
    )

    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        first_logits = first(input_ids=ids).logits
        second_logits = second(input_ids=ids).logits
    assert not torch.allclose(first_logits, second_logits), "different weights, different outputs"


def test_the_adapter_class_name_is_one_constant_and_cannot_separate_anything(two_checkpoints):
    """The root cause under the three grader-side call sites (errata E-15).

    ``resolve_adapter`` has no family dispatch left: it walks the tree and returns a `GraderAdapter`
    on every path. So the class name the call sites used to pass was the same string for every
    checkpoint of every seed, and the caller's own ``adapter_id`` never reached the fingerprint.
    Repairing the call sites without knowing this would have moved a constant from one place to
    another.
    """
    first, second, _ = two_checkpoints
    first_adapter = resolve_adapter(first, "checkpoints/seed-1/step-11")
    second_adapter = resolve_adapter(second, "checkpoints/seed-3/step-143")

    assert first_adapter.__class__ is second_adapter.__class__, (
        "resolve_adapter returns one type on every path, so a class name carries no checkpoint "
        "identity and must not be the adapter component of a fingerprint"
    )
    # The model_name each was constructed with does differ, which is the point: the information was
    # there, on the adapter, and the call sites reached past it for the class instead.
    assert first_adapter.model_name != second_adapter.model_name


# ---------------------------------------------------------------------------
# BLK-002: the fingerprint separates them
# ---------------------------------------------------------------------------


def test_two_checkpoints_over_one_base_do_not_share_a_fingerprint(two_checkpoints, tokenizer):
    """The defect, stated directly. Fails at 59a5f5a with two identical ``mfp:`` ids.

    Both calls pass **the same** adapter id, because that is what the library did: all three
    grader-side sites passed the adapter's class name, which is one string for every checkpoint of
    every seed. Handing this two different ids would test the argument rather than the defect and
    would pass at the baseline, which is the shape of test that let this through in the first place.
    """
    first, second, _ = two_checkpoints
    constant_id = "GraderAdapter"  # what all three call sites passed, for every checkpoint
    first_fp = fingerprint(first, tokenizer, constant_id)
    second_fp = fingerprint(second, tokenizer, constant_id)
    assert first_fp != second_fp, (
        f"two checkpoints with different weights fingerprinted identically as {first_fp}: the "
        f"weight component hashed the shared base directory and the adapter tensors were never read"
    )


def test_the_fingerprint_separates_them_without_help_from_the_caller(two_checkpoints, tokenizer):
    """The separation must not depend on four call sites each passing the right string.

    Three of them passed a class name and a fourth passed a literal. A fingerprint whose only
    checkpoint-varying component is an argument will meet a caller that gets it wrong; the adapter
    tensors are on the model, so they are read here.
    """
    first, second, _ = two_checkpoints
    assert fingerprint(first, tokenizer) != fingerprint(second, tokenizer)


def test_disabling_the_adapter_changes_the_fingerprint(two_checkpoints, tokenizer):
    """The reference pass is a different model from the policy and must fingerprint as one.

    Under LoRA the KL reference is the base model with the adapter disabled, which is the same object
    with a boolean flipped: identical tensors, identical manifest, different forward. Nothing in a
    weight hash records it. This is the identity dimension the gap ledger never enumerated, and
    capturing activations under it would have collided with the policy's own capture.
    """
    first, _, _ = two_checkpoints
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    enabled_fp = fingerprint(first, tokenizer, "checkpoints/seed-1/step-11")
    with torch.no_grad():
        enabled_logits = first(input_ids=ids).logits

    for layer in first.model.layers:
        layer.self_attn.q_proj._disable_adapters = True
    try:
        disabled_fp = fingerprint(first, tokenizer, "checkpoints/seed-1/step-11")
        with torch.no_grad():
            disabled_logits = first(input_ids=ids).logits
    finally:
        for layer in first.model.layers:
            layer.self_attn.q_proj._disable_adapters = False

    assert not torch.allclose(enabled_logits, disabled_logits), "the flag must change the forward"
    assert enabled_fp != disabled_fp, (
        "the adapter-disabled reference read fingerprinted as the policy it is the reference for"
    )


def test_the_fingerprint_is_stable_when_nothing_changed(two_checkpoints, tokenizer):
    """A separating fingerprint that also separates a model from itself is not an identity.

    Cheap to state and worth stating: everything above would also pass if the digest were random.
    """
    first, _, _ = two_checkpoints
    assert fingerprint(first, tokenizer, "id") == fingerprint(first, tokenizer, "id")


# ---------------------------------------------------------------------------
# BLK-002's own closure proof: the store's miss counter
# ---------------------------------------------------------------------------


def _capture_spec():
    from reward_lens.core.types import Site
    from reward_lens.runtime.backend import CaptureSpec

    return CaptureSpec(sites=(Site(1, "resid_post"),), position=None, dtype="float32")


def _signals(two_checkpoints, tokenizer):
    first, second, _ = two_checkpoints
    return (
        wrap_hf_model(
            first,
            tokenizer,
            device="cpu",
            adapter_id="checkpoints/seed-1/step-11",
            architecture="LlamaForSequenceClassification",
            conformance_quickcheck=False,
        ),
        wrap_hf_model(
            second,
            tokenizer,
            device="cpu",
            adapter_id="checkpoints/seed-3/step-143",
            architecture="LlamaForSequenceClassification",
            conformance_quickcheck=False,
        ),
    )


def test_a_second_checkpoint_is_computed_and_not_served_from_the_first(two_checkpoints, tokenizer):
    """BLK-002's closure proof, asserted on the miss counter.

    One input, two checkpoints with known-different weights, one store. Two reads must be two
    computations. The counter is the assertion; the shard count and the capture call count are
    independent witnesses of the same fact, and they are here because they can be evaluated at the
    baseline where the counter does not yet exist.
    """
    from reward_lens.runtime.store import ActivationStore

    first_signal, second_signal = _signals(two_checkpoints, tokenizer)
    assert first_signal.meta.fingerprint != second_signal.meta.fingerprint

    root = Path(tempfile.mkdtemp(prefix="rl-store-"))
    store = ActivationStore(root)
    view = [("q0", "the cat sat on the mat"), ("q1", "a dog ran through the park")]
    spec = _capture_spec()

    calls = {"n": 0}

    def counted(signal):
        original = signal.capture

        def wrapper(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        signal.capture = wrapper  # type: ignore[method-assign]
        return signal

    counted(first_signal)
    counted(second_signal)

    store.get_or_compute(first_signal, view, spec, dataset_id="ds:fixed")
    store.get_or_compute(second_signal, view, spec, dataset_id="ds:fixed")

    shards = sorted(root.rglob("*.safetensors"))

    assert store.misses == 2, (
        f"the store recorded {store.misses} miss(es) for two distinct checkpoints. A second miss is "
        f"the only evidence that the second checkpoint was actually run; one miss means the second "
        f"read was served from the first checkpoint's entry."
    )
    assert store.hits == 0
    assert calls["n"] == 2, f"the signal was asked to capture {calls['n']} time(s), not twice"
    assert len(shards) == 2, f"one shard per distinct subject, found {[p.name for p in shards]}"


def test_the_same_checkpoint_read_twice_is_a_hit(two_checkpoints, tokenizer):
    """The counter has to be able to say hit, or it says nothing when it says miss.

    A store that missed on every read would satisfy the test above and would have broken the cache.
    """
    from reward_lens.runtime.store import ActivationStore

    first_signal, _ = _signals(two_checkpoints, tokenizer)
    root = Path(tempfile.mkdtemp(prefix="rl-store-"))
    store = ActivationStore(root)
    view = [("q0", "the cat sat on the mat")]
    spec = _capture_spec()

    store.get_or_compute(first_signal, view, spec, dataset_id="ds:fixed")
    store.get_or_compute(first_signal, view, spec, dataset_id="ds:fixed")

    assert (store.misses, store.hits) == (1, 1)
    assert store.stats.total == 2
    assert store.stats.hit_rate == 0.5
    store.reset_stats()
    assert store.stats.total == 0


# ---------------------------------------------------------------------------
# BLK-034: the assignment sites, and the deep check that used to be incapable
# ---------------------------------------------------------------------------


def test_no_call_site_passes_an_adapter_class_name_as_an_identity(two_checkpoints, tokenizer):
    """BLK-034's closure proof, run as code rather than typed into a document.

    The gap ledger named two assignment sites. There were three, plus a fourth on the policy side
    passing a hardcoded literal, and all four fed the same fingerprint argument. A repair that fixed
    two of three would have left the dynamics path separating checkpoints while the signals path did
    not, which is worse than the uniform defect because the two would then disagree silently.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    roots = [
        str(root / name)
        for name in ("src", "tests", "studies", "experiments", "examples", "scripts", "tools")
        if (root / name).exists()
    ]
    # Assembled rather than written out, so this file does not itself contain the string it forbids.
    forbidden = "type(" + "adapter)." + "__name__"
    found = subprocess.run(
        [*_GREP, forbidden, *roots],
        capture_output=True,
        text=True,
        check=False,
    )
    assert found.returncode == 1, f"the class-name expression survives at:\n{found.stdout}"

    policy_literal = 'fingerprint(model, tokenizer, "Architecture' + 'View")'
    literal = subprocess.run(
        [*_GREP, policy_literal, *roots],
        capture_output=True,
        text=True,
        check=False,
    )
    assert literal.returncode == 1, f"the policy-side literal survives at:\n{literal.stdout}"


def test_the_caller_s_adapter_id_reaches_the_fingerprint(two_checkpoints, tokenizer):
    """Two loads of one model under two ids must differ, or the id is still being dropped.

    This is the direct test of the clobbering: ``wrap_hf_model`` accepted ``adapter_id``, used it to
    resolve the adapter, and then overwrote the variable with the adapter's class name before
    fingerprinting, so the argument had no effect on identity.
    """
    first, _, _ = two_checkpoints
    left = wrap_hf_model(
        first,
        tokenizer,
        device="cpu",
        adapter_id="checkpoints/seed-1/step-11",
        architecture="LlamaForSequenceClassification",
        conformance_quickcheck=False,
    )
    right = wrap_hf_model(
        first,
        tokenizer,
        device="cpu",
        adapter_id="checkpoints/seed-1/step-231",
        architecture="LlamaForSequenceClassification",
        conformance_quickcheck=False,
    )
    assert left.meta.fingerprint != right.meta.fingerprint
    assert left.meta.lineage["adapter_id"] == "checkpoints/seed-1/step-11"
    assert right.meta.lineage["adapter_id"] == "checkpoints/seed-1/step-231"
    assert "GraderAdapter" not in left.meta.adapter, (
        "the metadata field should describe what the navigation found, not name a class"
    )


def test_the_deep_fingerprint_check_can_catch_an_adapter_swap(two_checkpoints, tokenizer):
    """`CheckpointSequence.verify_fingerprints` has to be able to fail, or it is decoration.

    It recomputed the fingerprint with the same constant the recording used, against a digest that
    hashed the base directory, so swapping the adapter under a checkpoint moved both sides of the
    comparison together and the deep check returned ok. Here the loader is swapped to return the
    other checkpoint and the check must localize it.
    """
    from reward_lens.dynamics.checkpoints import CheckpointSequence

    first_signal, second_signal = _signals(two_checkpoints, tokenizer)

    honest = CheckpointSequence.build(
        [
            (11, first_signal.meta.fingerprint, lambda: first_signal),
            (143, second_signal.meta.fingerprint, lambda: second_signal),
        ]
    )
    assert honest.verify_chain().ok
    assert honest.verify_fingerprints().ok, "an untampered chain must verify"

    swapped = CheckpointSequence.build(
        [
            (11, first_signal.meta.fingerprint, lambda: first_signal),
            # The record still says step 143's fingerprint; the loader now returns step 11's weights.
            (143, second_signal.meta.fingerprint, lambda: first_signal),
        ]
    )
    assert swapped.verify_chain().ok, (
        "the swap is invisible to the shallow chain check, as designed"
    )
    deep = swapped.verify_fingerprints()
    assert not deep.ok, "the deep check served its purpose only if it can fail on a swap"
    assert deep.first_bad_step == 143


# ---------------------------------------------------------------------------
# CB-1061 — the same proof against real `peft`, not against the mock above
#
# Everything before this point runs against `LoraLinear`, a structural stand-in. A stand-in proves
# the code is consistent with our model of the dependency; it proves nothing about the dependency.
# The specific risk it leaves open is narrow and it is the flattering one: if a real
# `peft.tuners.lora.Linear` declared `adapter_layer_names` differently, `_adapter_param_names` would
# return the empty set, the fingerprint would fall back to the base-directory digest plus the
# parameter manifest, and two LoRA checkpoints of the same rank over one base have **identical**
# manifests. The collision this repair exists to remove would come straight back, silently, with
# the repair's own suite still green.
#
# So the case below is run against the installed `peft` and is skipped where it is absent, which is
# why the mock above stays: it is the coverage a base install gets.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_real_peft_checkpoints(tokenizer):
    """Two genuine `peft` LoRA adapters at one rank over one saved base directory.

    `lora_B` initialises to zeros, so two freshly wrapped models are byte-identical and would
    separate on nothing. Both adapters are given their own random weights, which is what a
    checkpoint of a trained adapter is.
    """
    peft = pytest.importorskip("peft")

    root = Path(tempfile.mkdtemp(prefix="rl-fp-real-peft-"))
    config = _config(tokenizer)
    base_dir = _save_base(config, root)

    def _adapter(seed: int):
        torch.manual_seed(0)
        model = LlamaForSequenceClassification(config).eval()
        # As a PEFT wrapper leaves it: the config still names the base directory, because the
        # config *is* the base model's. That is the whole reason the disk digest cannot separate.
        model.config._name_or_path = str(base_dir)
        wrapped = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=4, lora_alpha=8, target_modules=["q_proj"], lora_dropout=0.0, bias="none"
            ),
        )
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                if "lora_" in name:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator))
        return wrapped.eval()

    return _adapter(11), _adapter(143), base_dir


def test_real_peft_declares_the_contract_the_library_reads_off_the_module_tree(
    two_real_peft_checkpoints,
):
    """The assumption the mock encoded, checked against the installed package.

    `_adapter_param_names` reads `adapter_layer_names` off the module tree and never imports
    `peft`. If a real tuner layer did not declare it, the function would return the empty set and
    the fingerprint would lose its adapter component without saying so.
    """
    from reward_lens.runtime.fingerprint import _adapter_param_names

    first, second, _ = two_real_peft_checkpoints
    names = _adapter_param_names(first)
    assert names, "a real peft model contributed no adapter parameter names"
    assert names == _adapter_param_names(second), "the two adapters must be structurally identical"
    assert all("lora_" in name for name in names)


def test_real_peft_two_same_rank_adapters_over_one_base_do_not_share_a_fingerprint(
    two_real_peft_checkpoints, tokenizer
):
    """CB-1061's closure proof. Same rank, same base directory, same config, same tokenizer.

    The four components that were the whole identity before BLK-002 are all constant here, so if
    the adapter tensors did not reach the digest these two would collide.

    Every premise below is read off the models themselves rather than off the library's private
    hashes, so this case is evaluable against a library that has neither `_hash_manifest` nor
    `_adapter_param_names` — which is what the pinned revision it must fail on is.
    """
    first, second, base_dir = two_real_peft_checkpoints
    first_state, second_state = first.state_dict(), second.state_dict()

    # The premise, established before it is used: everything except the adapter weights agrees.
    assert first.config._name_or_path == second.config._name_or_path == str(base_dir)
    assert first.config.to_dict() == second.config.to_dict()
    assert type(first).__name__ == type(second).__name__
    manifest = lambda state: [  # noqa: E731 - a manifest is a list comprehension, not a helper
        (name, str(state[name].dtype), tuple(state[name].shape)) for name in sorted(state)
    ]
    assert manifest(first_state) == manifest(second_state), (
        "two adapters of one rank have identical parameter manifests, which is why the manifest "
        "cannot be what separates them"
    )

    # And the adapter tensors really do differ, so there is something to separate on.
    differing = [
        name
        for name in sorted(first_state)
        if "lora_" in name and not torch.equal(first_state[name], second_state[name])
    ]
    assert differing, "the fixture must give the two adapters different weights"

    assert fingerprint(first, tokenizer) != fingerprint(second, tokenizer)
    # Without help from the caller: no adapter id is passed on either side.
    assert fingerprint(first, tokenizer, "") != fingerprint(second, tokenizer, "")
