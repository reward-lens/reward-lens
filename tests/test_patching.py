"""Tests for the activation patching module."""

import numpy as np

from tests.test_analysis import make_mock_reward_model


class TestActivationPatcher:
    """Tests for activation patching."""

    def test_patching_runs(self):
        from reward_lens.patching import ActivationPatcher

        rm = make_mock_reward_model(n_layers=4)
        patcher = ActivationPatcher(rm)
        result = patcher.patch_all_components(
            "hello",
            "good response",
            "bad response",
            mode="noising",
            show_progress=False,
        )
        assert len(result.component_names) > 0
        assert len(result.patch_effects) == len(result.component_names)

    def test_patching_modes(self):
        from reward_lens.patching import ActivationPatcher

        rm = make_mock_reward_model(n_layers=4)
        patcher = ActivationPatcher(rm)

        for mode in ["noising", "denoising", "zero"]:
            result = patcher.patch_all_components(
                "hello",
                "good",
                "bad",
                mode=mode,
                show_progress=False,
            )
            assert len(result.component_names) > 0

    def test_patching_has_correct_component_count(self):
        from reward_lens.patching import ActivationPatcher

        rm = make_mock_reward_model(n_layers=4)
        patcher = ActivationPatcher(rm)
        result = patcher.patch_all_components(
            "hello",
            "good",
            "bad",
            mode="noising",
            show_progress=False,
        )

        # Should have: 4 attn + 4 mlp = 8 components
        assert len(result.component_names) == 8, (
            f"Expected 8 components, got {len(result.component_names)}: {result.component_names}"
        )

    def test_top_k(self):
        from reward_lens.patching import ActivationPatcher

        rm = make_mock_reward_model(n_layers=4)
        patcher = ActivationPatcher(rm)
        result = patcher.patch_all_components(
            "hello",
            "good",
            "bad",
            mode="noising",
            show_progress=False,
        )

        top = result.top_k(k=3)
        assert len(top) == 3
        # Should be sorted by absolute effect
        abs_effects = [abs(e) for _, e in top]
        assert abs_effects == sorted(abs_effects, reverse=True)

    def test_normalized_effects(self):
        from reward_lens.patching import ActivationPatcher

        rm = make_mock_reward_model(n_layers=4)
        patcher = ActivationPatcher(rm)
        result = patcher.patch_all_components(
            "hello",
            "good",
            "bad",
            mode="noising",
            show_progress=False,
        )

        normed = result.normalized_effects()
        # normalized_effects returns an ndarray
        assert isinstance(normed, np.ndarray)
        assert len(normed) == len(result.patch_effects)

    def test_patch_single_component(self):
        from reward_lens.patching import ActivationPatcher

        rm = make_mock_reward_model(n_layers=4)
        patcher = ActivationPatcher(rm)

        # Patch a specific layer's attention
        effect = patcher.patch_single_component(
            "hello", "good", "bad", layer_idx=0, component_type="attn", mode="noising"
        )
        assert isinstance(effect, float)
