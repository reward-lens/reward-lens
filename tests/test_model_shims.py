"""
Regression tests for the RewardModel loader shims and reward-head dtype
coercion.

These assert on ``reward_lens.model`` behaviour directly: the transformers
shims that keep custom Hub modeling code importable, InternLM2
sequence-classification registration, the fp32-head / bf16-backbone dtype
coercion, and the presence of the OOM-recovery branch. None of them depend on
an experiment runner.
"""

from __future__ import annotations


def test_llama_shim_applied():
    """transformers >= 4.45 dropped ``LLAMA_INPUTS_DOCSTRING``. ArmoRM / QRM /
    InternLM2 custom Hub modeling code imports it, so the shim re-injects it
    before each model load. Without it those models silently fall back to
    ``AutoModel`` and lose their reward heads."""
    from reward_lens.model import _patch_llama_modeling_shims

    _patch_llama_modeling_shims()
    from transformers.models.llama import modeling_llama

    for name in ("LLAMA_INPUTS_DOCSTRING", "_CONFIG_FOR_DOC", "LLAMA_START_DOCSTRING"):
        assert hasattr(modeling_llama, name), (
            f"shim missing: modeling_llama.{name} should be defined after shim"
        )


def test_internlm2_register():
    """InternLM2's config class isn't registered with
    ``AutoModelForSequenceClassification``, so a standard load falls back to
    ``AutoModel`` and drops the v_head. The loader registers the model_type
    before each load."""
    import pytest

    from reward_lens.model import _register_internlm2_for_seq_classification

    _register_internlm2_for_seq_classification()
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
        )
    except ImportError:
        pytest.skip("transformers internal layout changed; can't assert directly")

    assert "internlm2" in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES, (
        "InternLM2 not registered for SequenceClassification — the v_head will be dropped on load"
    )


def test_reward_head_dtype_coercion():
    """A custom reward head built via ``nn.Linear(d, K)`` defaults to fp32 even
    when the backbone is loaded in bf16 (the QRM ``regression_layer`` case),
    which trips "expected scalar type Float but found BFloat16" in the GEMM.
    ``_coerce_reward_head_dtype`` must cast the head to the backbone dtype while
    preserving its shape."""
    import torch
    import torch.nn as nn

    from reward_lens.model import _coerce_reward_head_dtype

    class _Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            # Mimic QRM's regression_layer: an fp32 head on a bf16 backbone.
            self.regression_layer = nn.Linear(8, 19)

    m = _Wrap()
    before_dtype = m.regression_layer.weight.dtype
    before_shape = m.regression_layer.weight.shape
    _coerce_reward_head_dtype(m, torch.bfloat16)

    assert before_dtype == torch.float32
    assert m.regression_layer.weight.dtype == torch.bfloat16
    assert m.regression_layer.weight.shape == before_shape


def test_oom_recovery_path_present():
    """``forward_with_cache_batch`` must catch ``CUDA OutOfMemoryError`` and
    retry with a halved chunk. An OOM can't be triggered without a GPU, so this
    asserts the recovery branch is still wired into the source."""
    import inspect

    from reward_lens.model import RewardModel

    src = inspect.getsource(RewardModel.forward_with_cache_batch)
    assert "OutOfMemoryError" in src, "forward_with_cache_batch lost its OOM-recovery branch"
