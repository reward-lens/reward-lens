"""
Tiny synthetic Llama RewardModel for CPU-only preflight tests.

The deep_analysisv1 campaign discovered that several experiments shipped
with broken APIs (e10 called a non-existent ``analyzer.analyze``; e11
called ``fit_corpus`` instead of ``fit_distribution``). The bugs only
surfaced after hours of GPU time were burned. This module gives the
preflight runner a fast, deterministic, CPU-runnable RewardModel that
lets every experiment exercise its full code path on a tiny config
before the H200 campaign launches.

The tiny model is a real ``LlamaForSequenceClassification`` with
hidden_size=32 and 2 layers, so the LlamaAdapter, the activation hooks,
the lens projection, the patcher, and the SAE all see the same module
tree they will see on a real 8B Skywork model. The only thing that
differs is the magnitude of the numbers — the *shape* of every output is
identical to a production run.
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch


def make_tiny_reward_model(
    *,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    seed: int = 0,
    vocab_size: Optional[int] = None,
    tokenizer_name: str = "gpt2",
    seq_max: int = 256,
    device: Optional[str] = None,
):
    """Construct a tiny Llama-architecture RewardModel for CPU testing.

    The tokenizer defaults to GPT-2 (cached in ~/.cache/huggingface, no
    chat template). The model is a stock LlamaForSequenceClassification
    with ``num_labels=1`` so the standard LlamaAdapter handles it without
    any fallback paths firing.

    Args:
        d_model: hidden size; pick a power of 2 that's a multiple of n_heads.
        n_layers: how many decoder blocks (>=2 so attribution has multiple
            ``mlp_L*`` / ``attn_L*`` rows).
        n_heads: must divide d_model.
        seed: torch manual seed for reproducible reward heads.
        vocab_size: defaults to the tokenizer's vocab size.
        tokenizer_name: any HF tokenizer; gpt2 is small + offline-friendly.
        seq_max: max position embeddings; raise if your test prompts go
            past 256 tokens after templating.
        device: "cpu" / "cuda" / None=auto.

    Returns:
        A ``reward_lens.model.RewardModel`` ready for end-to-end use.
    """
    from transformers import LlamaConfig, LlamaForSequenceClassification, AutoTokenizer

    from reward_lens.model import RewardModel
    from reward_lens.model_adapters import LlamaAdapter

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if vocab_size is None:
        vocab_size = tokenizer.vocab_size

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=d_model,
        intermediate_size=2 * d_model,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=seq_max,
        rms_norm_eps=1e-6,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=1,
        attn_implementation="eager",
    )
    model = LlamaForSequenceClassification(config).to(torch_device).eval()

    return RewardModel(
        model=model,
        tokenizer=tokenizer,
        adapter=LlamaAdapter(),
        device=torch_device,
    )
