"""Tests for reward lens and attribution on mock models."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from reward_lens.model import RewardModel
from reward_lens.model_adapters import LlamaAdapter


class MockLayer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.self_attn = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.self_attn(x)
        mlp_out = self.mlp(x)
        out = x + attn_out + mlp_out
        return (out,)


class MockConfig:
    num_attention_heads = 4
    model_type = "llama"
    hidden_size = 64


class MockBackbone(nn.Module):
    def __init__(self, n_layers=4, d_model=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, d_model)
        self.layers = nn.ModuleList([MockLayer(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)


class MockRewardModelHF(nn.Module):
    """Mock model mimicking HuggingFace AutoModelForSequenceClassification."""

    def __init__(self, n_layers=4, d_model=64):
        super().__init__()
        self.model = MockBackbone(n_layers, d_model)
        self.score = nn.Linear(d_model, 1)
        self.config = MockConfig()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)[0]
        x = self.model.norm(x)
        # Use last token
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = x.shape[0]
            last_hidden = x[torch.arange(batch_size), seq_lengths]
        else:
            last_hidden = x[:, -1]
        logits = self.score(last_hidden).unsqueeze(1)
        return type("Output", (), {"logits": logits})()


def make_mock_reward_model(n_layers=4, d_model=64):
    """Create a RewardModel wrapping a mock model."""
    mock = MockRewardModelHF(n_layers=n_layers, d_model=d_model)
    mock.eval()

    # Minimal tokenizer mock
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.chat_template = None
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"

    def mock_tokenize(text, return_tensors="pt", truncation=True, max_length=2048, padding=False):
        # Simple tokenization: map each character to a token. Handles both a
        # single string (single-pair path) and a list of strings (the batched
        # path used by forward_with_cache_batch / attribute_heads). Batches are
        # left-padded so the final token sits at position T-1 for every row —
        # the contract the batched forward path relies on.
        rows = [text] if isinstance(text, str) else list(text)
        encoded = [[ord(c) % 1000 for c in s[:max_length]] for s in rows]
        width = max(len(r) for r in encoded)
        input_ids, attention_mask = [], []
        for row in encoded:
            pad = width - len(row)
            input_ids.append([0] * pad + row)
            attention_mask.append([0] * pad + [1] * len(row))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

    tokenizer.side_effect = mock_tokenize
    tokenizer.__call__ = mock_tokenize

    adapter = LlamaAdapter()
    device = torch.device("cpu")

    return RewardModel(model=mock, tokenizer=tokenizer, adapter=adapter, device=device)


class TestRewardLens:
    """Tests for the RewardLens analysis."""

    def test_lens_trace_returns_result(self):
        from reward_lens.lens import RewardLens

        rm = make_mock_reward_model(n_layers=4)
        lens = RewardLens(rm)
        result = lens.trace("hello", "good response", "bad response")

        assert result.layers is not None
        assert len(result.layers) == 5  # -1, 0, 1, 2, 3
        assert len(result.reward_lens_preferred) == 5
        assert len(result.reward_lens_dispreferred) == 5
        assert len(result.differential) == 5
        assert len(result.marginal_contributions) == 4  # diff of 5 values
        assert isinstance(result.crystallization_layer, int)

    def test_lens_trace_single(self):
        from reward_lens.lens import RewardLens

        rm = make_mock_reward_model(n_layers=4)
        lens = RewardLens(rm)
        layers, values = lens.trace_single("hello", "response")

        assert len(layers) == 5
        assert len(values) == 5
        assert not np.any(np.isnan(values))


class TestComponentAttribution:
    """Tests for component attribution."""

    def test_attribution_returns_result(self):
        from reward_lens.attribution import ComponentAttribution

        rm = make_mock_reward_model(n_layers=4)
        attrib = ComponentAttribution(rm)
        result = attrib.attribute("hello", "good", "bad")

        assert len(result.component_names) > 0
        assert len(result.contributions_preferred) == len(result.component_names)
        assert len(result.differential_contributions) == len(result.component_names)

    def test_attribution_has_all_types(self):
        from reward_lens.attribution import ComponentAttribution

        rm = make_mock_reward_model(n_layers=4)
        attrib = ComponentAttribution(rm)
        result = attrib.attribute("hello", "good", "bad")

        types = set(result.component_types)
        assert "embed" in types
        assert "attn" in types
        assert "mlp" in types

    def test_top_k(self):
        from reward_lens.attribution import ComponentAttribution

        rm = make_mock_reward_model(n_layers=4)
        attrib = ComponentAttribution(rm)
        result = attrib.attribute("hello", "good", "bad")

        top = result.top_k(k=3, by="differential")
        assert len(top) == 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)

    def test_by_type_filter(self):
        from reward_lens.attribution import ComponentAttribution

        rm = make_mock_reward_model(n_layers=4)
        attrib = ComponentAttribution(rm)
        result = attrib.attribute("hello", "good", "bad")

        attn_only = result.by_type("attn")
        assert all(t == "attn" for t in attn_only.component_types)
        assert len(attn_only.component_names) == 4  # 4 layers


class MockAttentionWithHeads(nn.Module):
    """Attention stub whose output projection (``o_proj``) receives the
    concatenated per-head outputs.

    The plain-``Linear`` ``self_attn`` used by :class:`MockLayer` has no
    ``o_proj`` submodule, so ``capture_heads=True`` finds nothing to hook. This
    variant exposes an ``o_proj`` fed a ``(B, T, n_heads * d_head)`` tensor,
    which is exactly what the per-head attribution path hooks.
    """

    def __init__(self, d_model=64):
        super().__init__()
        # in_proj stands in for QKV + attention; its output is treated as the
        # concatenated per-head values handed to o_proj.
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return self.o_proj(self.in_proj(x))


class MockLayerWithHeads(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.self_attn = MockAttentionWithHeads(d_model)
        self.mlp = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = x + self.self_attn(x) + self.mlp(x)
        return (out,)


def make_head_capable_reward_model(n_layers=4, d_model=64):
    """Like :func:`make_mock_reward_model`, but every layer's attention exposes
    an ``o_proj`` so ``capture_heads=True`` (and ``attribute_heads``) work."""
    rm = make_mock_reward_model(n_layers=n_layers, d_model=d_model)
    rm.model.model.layers = nn.ModuleList([MockLayerWithHeads(d_model) for _ in range(n_layers)])
    rm.model.eval()
    return rm


class TestHeadAttribution:
    """Tests for per-head attention attribution (``attribute_heads``).

    This feature depends on the ``_batch_head_attribution`` helper; these tests
    also guard against the NameError that used to make the method uncallable.
    """

    def test_attribute_heads_runs(self):
        from reward_lens.attribution import ComponentAttribution

        rm = make_head_capable_reward_model(n_layers=4)
        attrib = ComponentAttribution(rm)
        result = attrib.attribute_heads("hello", "good response", "bad response")

        # 4 layers x 4 heads = 16 head components, plus 4 per-layer MLPs.
        head_names = [
            n for n, t in zip(result.component_names, result.component_types) if t == "attn_head"
        ]
        assert len(head_names) == 16
        assert result.component_types.count("mlp") == 4
        assert all(name.startswith("head_L") for name in head_names)

    def test_head_contributions_sum_to_layer_attention(self):
        """Because o_proj is linear and the heads partition its input, the
        per-head contributions for a layer must sum to that layer's total
        attention contribution reported by the standard attribution path."""
        from reward_lens.attribution import ComponentAttribution

        rm = make_head_capable_reward_model(n_layers=4)
        attrib = ComponentAttribution(rm)

        heads = attrib.attribute_heads("hello", "good", "bad")
        components = attrib.attribute("hello", "good", "bad")

        for layer in range(4):
            head_sum = sum(
                c
                for c, t, comp_layer in zip(
                    heads.contributions_preferred,
                    heads.component_types,
                    heads.layer_indices,
                )
                if t == "attn_head" and comp_layer == layer
            )
            attn_total = next(
                c
                for c, t, comp_layer in zip(
                    components.contributions_preferred,
                    components.component_types,
                    components.layer_indices,
                )
                if t == "attn" and comp_layer == layer
            )
            assert head_sum == pytest.approx(attn_total, abs=1e-3)


class TestProjectOntoReward:
    """Tests for the reward projection function."""

    def test_project_returns_scalar(self):
        rm = make_mock_reward_model()
        h = torch.randn(1, rm.d_model)
        proj = rm.project_onto_reward(h)
        assert proj.shape == (1,)

    def test_project_batch(self):
        rm = make_mock_reward_model()
        h = torch.randn(5, rm.d_model)
        proj = rm.project_onto_reward(h)
        assert proj.shape == (5,)

    def test_project_linearity(self):
        """The projection should be linear: proj(a + b) = proj(a) + proj(b)."""
        rm = make_mock_reward_model()
        a = torch.randn(1, rm.d_model)
        b = torch.randn(1, rm.d_model)
        proj_sum = rm.project_onto_reward(a + b)
        sum_proj = rm.project_onto_reward(a) + rm.project_onto_reward(b)
        # Subtract the extra bias term
        bias = rm.reward_bias
        assert torch.allclose(proj_sum, sum_proj - bias, atol=1e-5)
