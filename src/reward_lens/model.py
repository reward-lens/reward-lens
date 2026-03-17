"""
Reward Model Wrapper with Activation Hooks.

Design decisions:
    1. We build directly on HuggingFace transformers, not TransformerLens.
       Reward models are loaded via AutoModelForSequenceClassification (or custom
       classes for multi-objective models like ArmoRM). TransformerLens assumes an
       unembedding matrix for next-token prediction, which reward models lack.

    2. The hook system is minimal: we register forward hooks on every transformer
       layer to capture residual stream states, attention outputs, and MLP outputs.
       This gives us everything we need for the reward lens, component attribution,
       and activation patching — without the overhead of a full interpretability
       framework.

    3. We auto-detect model architecture by inspecting the module tree. Llama-based
       models have `model.layers`, GPT-2 based have `transformer.h`, etc. The
       adapter system handles the mapping.

    4. The critical abstraction is the "reward direction" — the weight vector of the
       reward head. Every tool in this library ultimately projects onto or decomposes
       along this direction. We extract it once at load time and expose it as a
       first-class attribute.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reward_lens.model_adapters import get_adapter, ModelAdapter


@dataclass
class ActivationCache:
    """Stores activations captured during a forward pass.

    Attributes:
        residual_streams: Dict mapping layer index -> residual stream tensor at final token.
            Shape of each tensor: (batch, d_model).
        attn_outputs: Dict mapping layer index -> attention sublayer output at final token.
            Shape: (batch, d_model).
        mlp_outputs: Dict mapping layer index -> MLP sublayer output at final token.
            Shape: (batch, d_model).
        final_token_positions: The token positions used for reward computation.
            Shape: (batch,).
        raw_residual_streams: Full sequence residual streams (optional, for patching).
            Dict mapping layer index -> (batch, seq_len, d_model).
        raw_attn_outputs: Full sequence attention outputs (optional).
        raw_mlp_outputs: Full sequence MLP outputs (optional).
    """

    residual_streams: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    final_token_positions: Optional[torch.Tensor] = None
    raw_residual_streams: dict[int, torch.Tensor] = field(default_factory=dict)
    raw_attn_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    raw_mlp_outputs: dict[int, torch.Tensor] = field(default_factory=dict)


class RewardModel:
    """Wrapper around a HuggingFace reward model for mechanistic interpretability.

    This class handles:
        1. Loading and wrapping the model with activation hooks
        2. Extracting the reward direction (reward head weights)
        3. Tokenizing preference pairs
        4. Running forward passes with activation caching
        5. Computing reward scores

    Args:
        model: The HuggingFace model.
        tokenizer: The tokenizer.
        adapter: Model-specific adapter for architecture navigation.
        device: Device to run on.

    Example:
        >>> rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
        >>> score = rm.score("What is 2+2?", "2+2 is 4.")
        >>> print(f"Reward: {score:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        adapter: ModelAdapter,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.device = device

        # Extract the reward direction — the single most important vector
        # for all interpretability analysis.
        self._reward_weight, self._reward_bias = adapter.get_reward_head_params(model)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
        **kwargs: Any,
    ) -> "RewardModel":
        """Load a reward model from HuggingFace.

        This is the main entry point. It auto-detects the model architecture and
        selects the appropriate adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            device: Device string ("cuda", "cuda:0", "cpu"). Auto-detected if None.
            torch_dtype: Dtype for model weights. bfloat16 recommended for 8B models.
            trust_remote_code: Required for models like ArmoRM with custom code.
            attn_implementation: Attention implementation ("flash_attention_2", "eager", etc.).
            **kwargs: Additional arguments passed to from_pretrained.

        Returns:
            RewardModel instance ready for analysis.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        load_kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        load_kwargs.update(kwargs)

        # Try loading as sequence classification model first (most common).
        # This covers both standard HF models and custom-class models whose
        # config.json carries an AutoModelForSequenceClassification auto_map entry
        # (e.g. ArmoRM's LlamaForRewardModelWithGating).
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                device_map=str(device) if device.type == "cuda" else device.type,
                **load_kwargs,
            )
        except Exception as e:
            # Only fall back to AutoModel when the SequenceClassification load
            # genuinely fails (e.g. model has no classification head at all).
            # Log the original error so it is not silently swallowed.
            import warnings
            warnings.warn(
                f"AutoModelForSequenceClassification.from_pretrained failed for "
                f"'{model_name_or_path}' with: {type(e).__name__}: {e}\n"
                f"Falling back to AutoModel — reward head may not load correctly.",
                stacklevel=2,
            )
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                model_name_or_path,
                device_map=str(device) if device.type == "cuda" else device.type,
                **load_kwargs,
            )

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        adapter = get_adapter(model, model_name_or_path)
        actual_device = next(model.parameters()).device

        return cls(model=model, tokenizer=tokenizer, adapter=adapter, device=actual_device)

    @property
    def reward_direction(self) -> torch.Tensor:
        """The reward head weight vector — the direction in activation space that
        defines the reward. Shape: (d_model,)."""
        return self._reward_weight

    @property
    def reward_bias(self) -> float:
        """The reward head bias term."""
        return self._reward_bias

    @property
    def d_model(self) -> int:
        """Hidden dimension of the model."""
        return self._reward_weight.shape[0]

    @property
    def n_layers(self) -> int:
        """Number of transformer layers."""
        return self.adapter.n_layers(self.model)

    @property
    def n_heads(self) -> int:
        """Number of attention heads per layer."""
        return self.adapter.n_heads(self.model)

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    def tokenize_conversation(
        self,
        prompt: str,
        response: str,
        max_length: int = 2048,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a (prompt, response) pair using the model's chat template.

        Args:
            prompt: The user prompt.
            response: The assistant response.
            max_length: Maximum sequence length.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors on device.
        """
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # Use chat template if available
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = f"User: {prompt}\nAssistant: {response}"

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def tokenize_raw(self, text: str, max_length: int = 2048) -> dict[str, torch.Tensor]:
        """Tokenize raw text (already formatted).

        Args:
            text: Pre-formatted text string.
            max_length: Maximum sequence length.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors on device.
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def score(
        self,
        prompt: str,
        response: str,
        max_length: int = 2048,
    ) -> float:
        """Compute the scalar reward for a (prompt, response) pair.

        Args:
            prompt: The user prompt.
            response: The assistant response.
            max_length: Maximum sequence length.

        Returns:
            Scalar reward value.
        """
        inputs = self.tokenize_conversation(prompt, response, max_length=max_length)
        with torch.no_grad():
            output = self.model(**inputs)
        return self.adapter.extract_reward(output, inputs).item()

    def score_pair(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        max_length: int = 2048,
    ) -> tuple[float, float]:
        """Score both completions in a preference pair.

        Args:
            prompt: The user prompt.
            preferred: The preferred completion.
            dispreferred: The dispreferred completion.

        Returns:
            Tuple of (preferred_score, dispreferred_score).
        """
        score_w = self.score(prompt, preferred, max_length=max_length)
        score_l = self.score(prompt, dispreferred, max_length=max_length)
        return score_w, score_l

    def forward_with_cache(
        self,
        prompt: str,
        response: str,
        cache_full_sequences: bool = False,
        max_length: int = 2048,
    ) -> tuple[float, ActivationCache]:
        """Run a forward pass and cache all intermediate activations.

        This is the workhorse function for all interpretability analyses.
        It registers hooks on every transformer layer to capture:
        - Residual stream states after each layer
        - Attention sublayer outputs
        - MLP sublayer outputs

        Args:
            prompt: The user prompt.
            response: The assistant response.
            cache_full_sequences: If True, also cache full-sequence activations
                (needed for activation patching). Uses more memory.
            max_length: Maximum sequence length.

        Returns:
            Tuple of (reward_score, activation_cache).
        """
        inputs = self.tokenize_conversation(prompt, response, max_length=max_length)
        return self._forward_with_cache_from_inputs(inputs, cache_full_sequences=cache_full_sequences)

    def forward_with_cache_from_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        cache_full_sequences: bool = False,
    ) -> tuple[float, ActivationCache]:
        """Run forward pass with caching from pre-tokenized inputs.

        Args:
            inputs: Dict with 'input_ids' and 'attention_mask'.
            cache_full_sequences: Whether to cache full sequence activations.

        Returns:
            Tuple of (reward_score, activation_cache).
        """
        return self._forward_with_cache_from_inputs(inputs, cache_full_sequences=cache_full_sequences)

    def _forward_with_cache_from_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        cache_full_sequences: bool = False,
    ) -> tuple[float, ActivationCache]:
        """Internal implementation of forward-with-cache."""
        cache = ActivationCache()

        # Determine the final token position (where the reward is computed).
        # For most models this is the last non-padding token.
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        # Final token position: last token with attention_mask == 1
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        cache.final_token_positions = seq_lengths

        hooks = []

        def make_layer_hook(layer_idx: int):
            """Create a hook that captures the residual stream after a layer."""
            def hook_fn(module, input, output):
                # Different model architectures return different output formats.
                # We use the adapter to extract the hidden state.
                hidden_state = self.adapter.extract_layer_output(output)
                # hidden_state shape: (batch, seq_len, d_model)
                batch_size = hidden_state.shape[0]
                seq_len = hidden_state.shape[1]
                final_pos = seq_lengths.to(hidden_state.device)
                # Clamp positions to valid range to avoid CUDA index errors
                final_pos = final_pos.clamp(0, seq_len - 1)
                # Extract the final-token hidden state
                final_hidden = hidden_state[
                    torch.arange(batch_size, device=hidden_state.device), final_pos
                ]
                cache.residual_streams[layer_idx] = final_hidden.detach()
                if cache_full_sequences:
                    cache.raw_residual_streams[layer_idx] = hidden_state.detach()
            return hook_fn

        def make_attn_hook(layer_idx: int):
            """Create a hook that captures attention sublayer output."""
            def hook_fn(module, input, output):
                hidden_state = self.adapter.extract_attn_output(output)
                batch_size = hidden_state.shape[0]
                seq_len = hidden_state.shape[1]
                final_pos = seq_lengths.to(hidden_state.device)
                # Clamp positions to valid range to avoid CUDA index errors
                final_pos = final_pos.clamp(0, seq_len - 1)
                final_hidden = hidden_state[
                    torch.arange(batch_size, device=hidden_state.device), final_pos
                ]
                cache.attn_outputs[layer_idx] = final_hidden.detach()
                if cache_full_sequences:
                    cache.raw_attn_outputs[layer_idx] = hidden_state.detach()
            return hook_fn

        def make_mlp_hook(layer_idx: int):
            """Create a hook that captures MLP sublayer output."""
            def hook_fn(module, input, output):
                hidden_state = self.adapter.extract_mlp_output(output)
                batch_size = hidden_state.shape[0]
                seq_len = hidden_state.shape[1]
                final_pos = seq_lengths.to(hidden_state.device)
                # Clamp positions to valid range to avoid CUDA index errors
                final_pos = final_pos.clamp(0, seq_len - 1)
                final_hidden = hidden_state[
                    torch.arange(batch_size, device=hidden_state.device), final_pos
                ]
                cache.mlp_outputs[layer_idx] = final_hidden.detach()
                if cache_full_sequences:
                    cache.raw_mlp_outputs[layer_idx] = hidden_state.detach()
            return hook_fn

        # Register hooks on every layer
        layers = self.adapter.get_layers(self.model)
        for idx, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_layer_hook(idx)))
            attn_module = self.adapter.get_attn_module(layer)
            if attn_module is not None:
                hooks.append(attn_module.register_forward_hook(make_attn_hook(idx)))
            mlp_module = self.adapter.get_mlp_module(layer)
            if mlp_module is not None:
                hooks.append(mlp_module.register_forward_hook(make_mlp_hook(idx)))

        # Also capture the embedding output (layer 0 residual stream input)
        embed_module = self.adapter.get_embedding(self.model)
        def embed_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            batch_size = hidden_state.shape[0]
            seq_len = hidden_state.shape[1]
            final_pos = seq_lengths.to(hidden_state.device)
            # Clamp positions to valid range to avoid CUDA index errors
            final_pos = final_pos.clamp(0, seq_len - 1)
            final_hidden = hidden_state[
                torch.arange(batch_size, device=hidden_state.device), final_pos
            ]
            cache.residual_streams[-1] = final_hidden.detach()  # -1 = pre-first-layer
            if cache_full_sequences:
                cache.raw_residual_streams[-1] = hidden_state.detach()

        hooks.append(embed_module.register_forward_hook(embed_hook))

        try:
            with torch.no_grad():
                output = self.model(**inputs)
            reward = self.adapter.extract_reward(output, inputs)
        finally:
            for h in hooks:
                h.remove()

        return reward.item(), cache

    def project_onto_reward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project a hidden state onto the reward direction.

        This is the core "reward lens" operation: given a hidden state h,
        compute w_r^T @ h + b_r.

        Args:
            hidden_state: Tensor of shape (..., d_model).

        Returns:
            Tensor of shape (...) with the projected reward values.
        """
        return (hidden_state @ self._reward_weight.to(hidden_state.device)) + self._reward_bias

    @contextlib.contextmanager
    def hooks(self, hook_fns: dict[str, callable]):
        """Context manager for temporarily registering hooks.

        Args:
            hook_fns: Dict mapping module path strings to hook functions.

        Yields:
            None. Hooks are active within the context.
        """
        handles = []
        for path, fn in hook_fns.items():
            module = self._get_module_by_path(path)
            handles.append(module.register_forward_hook(fn))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _get_module_by_path(self, path: str) -> nn.Module:
        """Get a submodule by dot-separated path."""
        module = self.model
        for attr in path.split("."):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        return module

    def __repr__(self) -> str:
        return (
            f"RewardModel(\n"
            f"  adapter={self.adapter.__class__.__name__},\n"
            f"  n_layers={self.n_layers},\n"
            f"  d_model={self.d_model},\n"
            f"  n_heads={self.n_heads},\n"
            f"  device={self.device},\n"
            f")"
        )
