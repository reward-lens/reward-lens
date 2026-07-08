"""Tests for reward_lens.path_patching — head-level 2-hop path patching."""

import pytest
import torch
import torch.nn as nn

from reward_lens.model import RewardModel
from reward_lens.model_adapters import LlamaAdapter

# ── Mock model with o_proj support ──────────────────────────────────────


class MockAttn(nn.Module):
    """Attention module that has an o_proj, matching Llama arch."""

    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads

    def forward(self, x, **kwargs):
        # Simplified: project through o_proj directly
        return (self.o_proj(x),)


class MockLayer(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.self_attn = MockAttn(d_model, n_heads)
        self.mlp = nn.Linear(d_model, d_model, bias=False)
        self.input_layernorm = nn.LayerNorm(d_model)
        self.post_attention_layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.self_attn(x)[0]
        mlp_out = self.mlp(x)
        return (x + attn_out + mlp_out,)


class MockConfig:
    num_attention_heads = 4
    model_type = "llama"
    hidden_size = 64


class MockBackbone(nn.Module):
    def __init__(self, n_layers=4, d_model=64, n_heads=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, d_model)
        self.layers = nn.ModuleList([MockLayer(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)


class MockRewardModelHF(nn.Module):
    def __init__(self, n_layers=4, d_model=64, n_heads=4):
        super().__init__()
        self.model = MockBackbone(n_layers, d_model, n_heads)
        self.score = nn.Linear(d_model, 1)
        self.config = MockConfig()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)[0]
        x = self.model.norm(x)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = x.shape[0]
            last_hidden = x[torch.arange(batch_size), seq_lengths]
        else:
            last_hidden = x[:, -1]
        logits = self.score(last_hidden).unsqueeze(1)
        return type("Output", (), {"logits": logits})()


def make_mock_with_oproj(n_layers=4, d_model=64, n_heads=4):
    """Create a RewardModel with o_proj support for path patching tests."""
    mock = MockRewardModelHF(n_layers=n_layers, d_model=d_model, n_heads=n_heads)
    mock.eval()

    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.chat_template = None
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"

    def mock_tokenize(text, return_tensors="pt", truncation=True, max_length=2048, padding=False):
        tokens = [ord(c) % 1000 for c in text[:50]]
        input_ids = torch.tensor([tokens])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenizer.side_effect = mock_tokenize
    tokenizer.__call__ = mock_tokenize

    adapter = LlamaAdapter()
    device = torch.device("cpu")
    return RewardModel(model=mock, tokenizer=tokenizer, adapter=adapter, device=device)


# ── Tests ───────────────────────────────────────────────────────────────


class TestPathPatcher:
    """Tests for PathPatcher."""

    def test_basic_patch(self):
        from reward_lens.path_patching import PathPatcher

        rm = make_mock_with_oproj(n_layers=4)
        patcher = PathPatcher(rm)
        result = patcher.patch(
            "hello",
            "good response",
            "bad response",
            sender=("head", 0, 0),
            receiver=("mlp", 2, None),
            mode="noising",
        )
        assert hasattr(result, "path_effect")
        assert hasattr(result, "original_differential")
        assert hasattr(result, "patched_differential")
        assert isinstance(result.path_effect, float)

    def test_denoising_mode(self):
        from reward_lens.path_patching import PathPatcher

        rm = make_mock_with_oproj(n_layers=4)
        patcher = PathPatcher(rm)
        result = patcher.patch(
            "hello",
            "good",
            "bad",
            sender=("head", 0, 0),
            receiver=("mlp", 2, None),
            mode="denoising",
        )
        assert isinstance(result.path_effect, float)

    def test_receiver_must_be_downstream(self):
        from reward_lens.path_patching import PathPatcher

        rm = make_mock_with_oproj(n_layers=4)
        patcher = PathPatcher(rm)
        with pytest.raises(ValueError, match="receiver layer.*must be > sender"):
            patcher.patch(
                "hello",
                "good",
                "bad",
                sender=("head", 2, 0),
                receiver=("mlp", 1, None),
                mode="noising",
            )

    def test_sender_must_be_head(self):
        from reward_lens.path_patching import PathPatcher

        rm = make_mock_with_oproj(n_layers=4)
        patcher = PathPatcher(rm)
        with pytest.raises(NotImplementedError):
            patcher.patch(
                "hello",
                "good",
                "bad",
                sender=("mlp", 0, None),
                receiver=("mlp", 2, None),
            )

    def test_sender_head_required(self):
        from reward_lens.path_patching import PathPatcher

        rm = make_mock_with_oproj(n_layers=4)
        patcher = PathPatcher(rm)
        with pytest.raises(ValueError, match="sender head index is required"):
            patcher.patch(
                "hello",
                "good",
                "bad",
                sender=("head", 0, None),
                receiver=("mlp", 2, None),
            )

    def test_result_dataclass_fields(self):
        from reward_lens.path_patching import PathPatchResult

        r = PathPatchResult(
            sender=("head", 0, 1),
            receiver=("mlp", 3, None),
            mode="noising",
            original_differential=1.5,
            patched_differential=1.2,
            path_effect=0.3,
        )
        assert r.sender == ("head", 0, 1)
        assert r.receiver == ("mlp", 3, None)
        assert r.mode == "noising"
        assert abs(r.path_effect - 0.3) < 1e-10

    def test_invalid_mode_raises(self):
        from reward_lens.path_patching import PathPatcher

        rm = make_mock_with_oproj(n_layers=4)
        patcher = PathPatcher(rm)
        with pytest.raises(ValueError, match="unknown mode"):
            patcher.patch(
                "hello",
                "good",
                "bad",
                sender=("head", 0, 0),
                receiver=("mlp", 2, None),
                mode="invalid_mode",
            )
