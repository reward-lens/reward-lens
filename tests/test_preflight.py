"""
Pytest wrapper around the experiment preflight runner.

Each experiment becomes one parametrised test so failures show up as
named test cases (``test_experiment_runs[e12_sae_feature_decomposition]``)
rather than as one giant assertion. CI gets a per-experiment regression
gate this way.

The first test in the file is :func:`test_llama_shim_applied` — that's
the §3.1 root cause. If the shim regresses, every other test in this
module is suspect, so it runs first.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from experiments import preflight, registry


@pytest.fixture(scope="module")
def tiny_rm():
    from experiments.utils.tiny_model import make_tiny_reward_model

    return make_tiny_reward_model()


@pytest.fixture(scope="module")
def out_root():
    p = Path(tempfile.mkdtemp(prefix="preflight_pytest_"))
    yield p
    shutil.rmtree(p, ignore_errors=True)


def test_llama_shim_applied():
    """The §3.1 root cause: transformers ≥ 4.45 dropped LLAMA_INPUTS_DOCSTRING.

    ArmoRM / QRM / InternLM2 custom Hub modeling code imports it. The shim
    re-injects it before each model load. If this test fails, the three
    "broken-loading" models from deep_analysisv1 will silently fall back
    to AutoModel and lose their reward heads.
    """
    from reward_lens.model import _patch_llama_modeling_shims

    _patch_llama_modeling_shims()
    from transformers.models.llama import modeling_llama

    for name in ("LLAMA_INPUTS_DOCSTRING", "_CONFIG_FOR_DOC", "LLAMA_START_DOCSTRING"):
        assert hasattr(modeling_llama, name), (
            f"shim missing: modeling_llama.{name} should be defined after shim"
        )


def test_adapter_dispatch_routes():
    """The §7.2 fix: the dispatcher must route the campaign's six models
    to the correct family adapter — ArmoRM/QRM/InternLM2 in particular.

    We can't instantiate model objects without weights, so this is a
    name-based static check that mirrors :func:`get_adapter`'s logic.
    """
    cases = {
        "Skywork/Skywork-Reward-Llama-3.1-8B": "LlamaAdapter",
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2": "LlamaAdapter",
        "Skywork/Skywork-Reward-Gemma-2-27B-v0.2": "Gemma2Adapter",
        "RLHFlow/ArmoRM-Llama3-8B-v0.1": "ArmoRMAdapter",
        "internlm/internlm2-20b-reward": "InternLM2Adapter",
        "nicolinho/QRM-Llama3.1-8B": "LlamaAdapter",
    }
    for mid, want in cases.items():
        # We don't know the actual class_name without loading; pretend it
        # matches the family. The dispatcher prefers model_id-based hints
        # (armorm, qrm, internlm) so this works for the fragile cases.
        cl = mid.split("/")[-1].lower()
        mt = (
            "llama"
            if "llama" in cl
            else ("gemma" if "gemma" in cl else ("internlm2" if "internlm" in cl else ""))
        )
        got = preflight._resolve_adapter_class_name(cl, mt, mid)
        assert got == want, f"{mid}: dispatcher would send to {got}, want {want}"


@pytest.mark.parametrize("name", sorted(registry.list_experiments()))
def test_experiment_runs(name: str, tiny_rm, out_root):
    """Each experiment runs end-to-end against a tiny CPU RewardModel.

    Failures here mean a runner has an API mismatch (e10/e11/e12 style),
    a missing import, or breaks on n=2 pairs (a robustness bug).
    """
    result = preflight._run_experiment(name, out_root, tiny_rm)
    assert result.ok, (
        f"{name} preflight failed in {result.seconds:.1f}s\n"
        f"  detail: {result.detail}\n"
        f"  error:  {result.error}"
    )


# v3 regression tests — one per failure mode we fixed --------------------------


def test_v3_dtype_coercion_qrm_style():
    """The §2.3 deep_analysis_v2 root cause: QRM's regression_layer is fp32
    (because nn.Linear defaults to global default dtype) but the rest of
    the model is bf16, so the forward hits "expected scalar type Float but
    found BFloat16" in the cuBLAS GEMM. ``_coerce_reward_head_dtype`` must
    cast the head to match.
    """
    r = preflight._check_dtype_coercion()
    assert r.ok, f"dtype coercion broken: {r.detail or r.error}"


def test_v3_gemma_soft_cap_disabled():
    """The §2.5 deep_analysis_v2 root cause: SKG27 lens NaN, e09 NaN, and
    e08 dose-response damping all trace to Gemma-2's
    ``final_logit_softcapping`` collapsing the late-layer differential.
    ``RewardModel.from_pretrained`` must null both soft-cap fields.
    """
    r = preflight._check_soft_cap_disabling()
    assert r.ok, f"soft-cap not disabled: {r.detail or r.error}"


def test_v3_lens_zero_diff_robust():
    """When the final-layer differential is numerically zero,
    ``_crystal_frac`` must fall back to the largest-magnitude finite
    differential rather than returning NaN. This was the secondary
    SKG27 NaN propagation path."""
    r = preflight._check_zero_diff_lens()
    assert r.ok, f"lens regressed on zero final-diff: {r.detail or r.error}"


def test_v3_per_model_isolation(out_root):
    """A single broken model must NOT abort the whole experiment loop.
    The v3 manifest_run + swallow_exceptions wiring guarantees the next
    iteration of the loop still runs, with the broken cell marked
    ``status: failed`` in its manifest."""
    r = preflight._check_per_model_isolation(out_root)
    assert r.ok, f"per-model isolation broken: {r.detail or r.error}"


def test_v3_dataset_split_recovery_wired():
    """The dataset loader's split-recovery fallback must be in place:
    when the requested split doesn't exist (e.g. RewardBench-2 has only
    ``test``), the loader should pick a real split rather than returning
    None silently (which used to leave downstream calls scoring on
    whatever empty fallback the consumer used)."""
    r = preflight._check_dataset_split_recovery()
    assert r.ok, f"dataset split recovery missing: {r.detail or r.error}"


def test_v3_internlm2_register():
    """The §2.2 deep_analysis_v2 root cause: InternLM2's config class
    isn't registered with AutoModelForSequenceClassification, so the
    standard load falls back to AutoModel and drops the v_head. The v3
    fix registers the model_type before each load."""
    from reward_lens.model import _register_internlm2_for_seq_classification

    _register_internlm2_for_seq_classification()
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
        )

        # We don't assert on the exact value (transformers may rewrite the
        # internal map keying); the relevant invariant is that "internlm2"
        # is now in the map so AutoModel doesn't reject the config class.
        assert "internlm2" in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES, (
            "InternLM2 not registered for SequenceClassification — "
            "the v_head will be dropped on load"
        )
    except ImportError:
        pytest.skip("transformers internal layout changed; can't assert directly")


def test_v3_manifest_run_swallows_exceptions(tmp_path):
    """The v3 ``manifest_run(swallow_exceptions=True)`` flag should
    suppress exceptions raised inside the with-block and write a
    ``failed`` manifest, so the caller's loop continues to the next
    iteration."""
    import json

    from experiments.utils.io import manifest_run

    cell = tmp_path / "cell"
    cell.mkdir()
    # Should NOT propagate
    with manifest_run(cell, "test_exp", {}, model="m", seed=0, swallow_exceptions=True):
        raise ValueError("simulated forward failure")
    manifest = json.loads((cell / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "ValueError" in (manifest["notes"] or "")


def test_v3_oom_recovery_path_present():
    """``forward_with_cache_batch`` must catch ``CUDA OutOfMemoryError``
    and retry with a halved chunk. We can't trigger an OOM without a
    GPU, so this test just verifies the source contains the recovery
    path — a sanity check that the v3 fix is still wired up."""
    import inspect

    from reward_lens.model import RewardModel

    src = inspect.getsource(RewardModel.forward_with_cache_batch)
    assert "OutOfMemoryError" in src, "forward_with_cache_batch lost its OOM-recovery branch"
