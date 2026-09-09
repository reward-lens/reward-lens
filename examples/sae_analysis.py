"""
Example: SAE Training and Feature Analysis on a Reward Model

This script demonstrates:
1. Collecting activations from a reward model
2. Training a TopK SAE on those activations
3. Analyzing which features are aligned with the reward direction
4. Decomposing specific inputs through SAE features

Usage:
    python examples/sae_analysis.py --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2

Requires GPU with 24GB+ VRAM.
"""

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(description="SAE Analysis on Reward Model")
    parser.add_argument(
        "--model",
        type=str,
        default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer to analyze (-1 means use the last layer; auto-detected)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="SAE dictionary size (default: 8x model dimension)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Number of active features (TopK parameter)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=3,
        help="Number of SAE training epochs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sae",
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------------
    print(f"\nLoading model: {args.model}")
    from reward_lens import RewardModel
    rm = RewardModel.from_pretrained(args.model)
    print(rm)

    # Determine target layer
    target_layer = args.layer
    if target_layer == -1:
        # Use the last transformer layer
        target_layer = rm.n_layers - 1
    print(f"\nTarget layer for SAE: {target_layer}")

    # -----------------------------------------------------------------------
    # 2. Collect activations
    # -----------------------------------------------------------------------
    print(f"\nCollecting activations...")
    from reward_lens.sae import ActivationCollector
    from reward_lens.diagnostic_data import get_all_prompts_and_responses

    collector = ActivationCollector(rm)
    data = get_all_prompts_and_responses()

    # For a real analysis, you'd use a much larger dataset.
    # The diagnostic data gives ~40 examples, enough for a demo.
    prompts = [d["prompt"] for d in data]
    responses = [d["response"] for d in data]

    activations = collector.collect(prompts, responses, layer=target_layer)
    print(f"  Collected {activations.shape[0]} activations of dim {activations.shape[1]}")

    # -----------------------------------------------------------------------
    # 3. Train SAE
    # -----------------------------------------------------------------------
    print(f"\nTraining TopK SAE...")
    from reward_lens.sae import SAETrainer

    trainer = SAETrainer(
        d_model=rm.d_model,
        n_features=args.n_features,
        k=args.k,
        lr=3e-4,
        batch_size=min(32, activations.shape[0]),
        device=str(rm.device),
    )
    sae = trainer.train(activations, n_epochs=args.n_epochs)

    # Save the trained SAE
    sae_path = os.path.join(args.output_dir, "trained_sae")
    sae.save(sae_path)
    print(f"  SAE saved to {sae_path}")

    # -----------------------------------------------------------------------
    # 4. Feature Analysis
    # -----------------------------------------------------------------------
    print(f"\nAnalyzing features...")
    from reward_lens.sae import FeatureAnalyzer

    analyzer = FeatureAnalyzer(sae, rm)

    # Top reward-aligned features
    top_features = analyzer.top_reward_features(k=10)
    print("\n  Top 10 reward-aligned features:")
    for feat_idx, alignment in top_features:
        print(f"    Feature {feat_idx}: alignment = {alignment:+.4f}")

    # Bottom (anti-reward) features
    bottom_features = analyzer.bottom_reward_features(k=10)
    print("\n  Top 10 anti-reward features:")
    for feat_idx, alignment in bottom_features:
        print(f"    Feature {feat_idx}: alignment = {alignment:+.4f}")

    # Plot alignment histogram
    analyzer.plot_alignment_histogram(
        save_path=os.path.join(args.output_dir, "alignment_histogram.png")
    )

    # -----------------------------------------------------------------------
    # 5. Decompose specific inputs
    # -----------------------------------------------------------------------
    print(f"\nDecomposing reward for specific inputs...")

    from reward_lens.diagnostic_data import get_diagnostic_pairs
    pair = get_diagnostic_pairs(["helpfulness"])[0]

    # Preferred
    contribs_w, total_w = analyzer.decompose_reward_for_input(
        pair.prompt, pair.preferred, layer=target_layer
    )
    print(f"\n  Preferred response: reconstructed reward ≈ {total_w:.4f}")
    print(f"  Top 5 feature contributions:")
    for feat_idx, contrib in contribs_w[:5]:
        alignment = sae.feature_reward_alignments(rm.reward_direction)[feat_idx].item()
        print(f"    Feature {feat_idx}: contribution = {contrib:+.4f} (alignment = {alignment:+.4f})")

    # Dispreferred
    contribs_l, total_l = analyzer.decompose_reward_for_input(
        pair.prompt, pair.dispreferred, layer=target_layer
    )
    print(f"\n  Dispreferred response: reconstructed reward ≈ {total_l:.4f}")
    print(f"  Top 5 feature contributions:")
    for feat_idx, contrib in contribs_l[:5]:
        alignment = sae.feature_reward_alignments(rm.reward_direction)[feat_idx].item()
        print(f"    Feature {feat_idx}: contribution = {contrib:+.4f} (alignment = {alignment:+.4f})")

    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
