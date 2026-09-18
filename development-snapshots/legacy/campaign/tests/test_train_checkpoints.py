"""Checkpoint saving on `train_organism` (torch-gated).

A developmental sweep needs a `CheckpointSequence`, and a sequence needs saved weights to load,
so a recipe that sets ``save_every`` and ``checkpoint_dir`` must leave a ``step_<n>`` directory
trail behind training: step 0 for the initial weights, every interval step, and the final step.
These tests train the tiny CPU trunk for a handful of steps, check the trail on disk and on the
result, and rebuild the trail into a verified `CheckpointSequence` whose checkpoints load and
score, which is the exact round trip the dynamics sweeps rely on.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="train_organism needs torch")

from reward_lens.organisms._tiny import make_micro_trunk
from reward_lens.organisms.foundry import spurious_correlation_organism
from reward_lens.organisms.train import TrainRecipe, train_organism


def _recipe(tmp_path, *, epochs: int, batch_size: int) -> TrainRecipe:
    return TrainRecipe(
        method="full_ft",
        epochs=epochs,
        lr=5e-3,
        batch_size=batch_size,
        seed=0,
        max_length=64,
        device="cpu",
        label="micro-ckpt",
        save_every=2,
        checkpoint_dir=tmp_path / "ckpts",
    )


def test_checkpoint_trail_rebuilds_into_a_sequence(tmp_path):
    # 16 pairs at batch 8 is 2 optimizer steps per epoch; 2 epochs is 4 steps, so save_every=2
    # lands on 0, 2, and 4, with the final step coinciding with the interval (no duplicate).
    view, key = spurious_correlation_organism(rho=0.85, n=16, seed=0)
    trained = train_organism(make_micro_trunk(seed=0), view, _recipe(tmp_path, epochs=2, batch_size=8), key)

    assert trained.checkpoint_steps == [0, 2, 4]
    step_dirs = {p.name for p in (tmp_path / "ckpts").iterdir()}
    assert step_dirs == {"step_0", "step_2", "step_4"}

    # Rebuild the trail from disk alone: reload each step, wrap it as a signal, and chain the
    # sequence. This is the loader pattern every developmental sweep uses.
    from transformers import AutoTokenizer, LlamaForSequenceClassification

    from reward_lens.dynamics import CheckpointSequence
    from reward_lens.signals.loaders import wrap_hf_model

    triples = []
    meta = {}
    for step in trained.checkpoint_steps:
        path = tmp_path / "ckpts" / f"step_{step}"
        signal = wrap_hf_model(
            LlamaForSequenceClassification.from_pretrained(path),
            AutoTokenizer.from_pretrained(path),
            device="cpu",
            conformance_quickcheck=False,
        )

        def loader(_signal=signal):
            return _signal

        triples.append((step, signal.meta.fingerprint, loader))
        meta[step] = {"path": str(path)}

    sequence = CheckpointSequence.build(triples, meta=meta)
    assert sequence.steps == [0, 2, 4]
    assert sequence.verify_chain().ok

    # Training moved the weights, so the initial and final checkpoints are different models.
    assert sequence[0].model_fp != sequence[-1].model_fp

    # A loaded checkpoint scores through the signal path, which is all a sweep asks of it.
    scores = sequence[0].load().score([("q0", "a short reply")]).value.values
    assert scores.shape == (1,)


def test_final_step_is_saved_off_interval(tmp_path):
    # 8 pairs at batch 8 is 1 step per epoch; 3 epochs is 3 steps, so the interval saves land
    # on 0 and 2 and the final step 3 must be saved on its own.
    view, key = spurious_correlation_organism(rho=0.85, n=8, seed=0)
    trained = train_organism(make_micro_trunk(seed=0), view, _recipe(tmp_path, epochs=3, batch_size=8), key)

    assert trained.checkpoint_steps == [0, 2, 3]
    assert (tmp_path / "ckpts" / "step_3").is_dir()


def test_checkpointing_is_off_by_default():
    # The presets do not opt in, so ordinary training must write nothing and report no steps.
    recipe = TrainRecipe.micro()
    assert recipe.save_every is None
    assert recipe.checkpoint_dir is None
