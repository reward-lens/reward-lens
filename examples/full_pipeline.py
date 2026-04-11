"""
Example: Full Reward Model Analysis Pipeline

This script demonstrates the complete reward-lens workflow:
1. Load a reward model
2. Run reward lens analysis
3. Run component attribution
4. Run activation patching
5. Run reward hacking detection
6. Generate a comprehensive dashboard

Usage:
    python examples/full_pipeline.py --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2

This requires a GPU with at least 16GB VRAM for 8B models.
"""

import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Reward Model Analysis Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--skip-patching",
        action="store_true",
        help="Skip activation patching (slow)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load the reward model
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading reward model: {args.model}")
    print(f"{'='*60}\n")

    from reward_lens import RewardModel
    rm = RewardModel.from_pretrained(args.model)
    print(rm)
    print(f"\nReward direction shape: {rm.reward_direction.shape}")
    print(f"Reward direction norm: {rm.reward_direction.norm():.4f}")

    # -----------------------------------------------------------------------
    # 2. Score some diagnostic pairs
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Scoring diagnostic pairs")
    print(f"{'='*60}\n")

    from reward_lens.diagnostic_data import get_diagnostic_pairs

    pairs = get_diagnostic_pairs()
    for pair in pairs[:5]:
        score_w, score_l = rm.score_pair(pair.prompt, pair.preferred, pair.dispreferred)
        correct = "✓" if score_w > score_l else "✗"
        print(f"  [{correct}] {pair.dimension}: Δ = {score_w - score_l:+.4f}")

    # -----------------------------------------------------------------------
    # 3. Reward Lens Analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Running Reward Lens analysis")
    print(f"{'='*60}\n")

    from reward_lens.lens import RewardLens

    lens = RewardLens(rm)

    # Analyze a helpfulness pair
    pair = get_diagnostic_pairs(["helpfulness"])[0]
    result = lens.trace(pair.prompt, pair.preferred, pair.dispreferred)
    print(f"  Preference crystallizes at layer {result.crystallization_layer}")
    print(f"  Final reward (preferred): {result.reward_preferred:.4f}")
    print(f"  Final reward (dispreferred): {result.reward_dispreferred:.4f}")
    result.plot(save_path=os.path.join(args.output_dir, "reward_lens_helpfulness.png"))

    # Analyze a safety pair
    pair = get_diagnostic_pairs(["safety"])[0]
    result_safety = lens.trace(pair.prompt, pair.preferred, pair.dispreferred)
    print(f"\n  Safety pair crystallizes at layer {result_safety.crystallization_layer}")
    result_safety.plot(save_path=os.path.join(args.output_dir, "reward_lens_safety.png"))

    # -----------------------------------------------------------------------
    # 4. Component Attribution
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Running Component Attribution")
    print(f"{'='*60}\n")

    from reward_lens.attribution import ComponentAttribution

    attrib = ComponentAttribution(rm)

    pair = get_diagnostic_pairs(["helpfulness"])[0]
    components = attrib.attribute(pair.prompt, pair.preferred, pair.dispreferred)

    print("  Top 10 components by differential contribution:")
    for name, value in components.top_k(k=10, by="differential"):
        print(f"    {name}: {value:+.4f}")

    components.plot_top_k(
        k=15, save_path=os.path.join(args.output_dir, "attribution_top15.png")
    )
    components.plot_heatmap(
        save_path=os.path.join(args.output_dir, "attribution_heatmap.png")
    )

    # -----------------------------------------------------------------------
    # 5. Activation Patching (optional — slow)
    # -----------------------------------------------------------------------
    if not args.skip_patching:
        print(f"\n{'='*60}")
        print("Running Activation Patching")
        print(f"{'='*60}\n")

        from reward_lens.patching import ActivationPatcher

        patcher = ActivationPatcher(rm)
        pair = get_diagnostic_pairs(["helpfulness"])[0]
        effects = patcher.patch_all_components(
            pair.prompt, pair.preferred, pair.dispreferred, mode="noising"
        )

        print("  Top 10 causally important components:")
        for name, effect in effects.top_k(k=10):
            print(f"    {name}: {effect:+.4f}")

        effects.plot(save_path=os.path.join(args.output_dir, "patching_heatmap.png"))
    else:
        print("\n  Skipping activation patching (use --no-skip-patching to enable)")
        effects = None

    # -----------------------------------------------------------------------
    # 6. Reward Hacking Detection
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Running Reward Hacking Detection")
    print(f"{'='*60}\n")

    from reward_lens.hacking import HackingDetector

    detector = HackingDetector(rm)
    report = detector.scan()
    report.print_summary()

    # -----------------------------------------------------------------------
    # 7. Dashboard
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Generating Dashboard")
    print(f"{'='*60}\n")

    from reward_lens.viz import reward_lens_dashboard

    pair = get_diagnostic_pairs(["helpfulness"])[0]
    lens_result = lens.trace(pair.prompt, pair.preferred, pair.dispreferred)
    attrib_result = attrib.attribute(pair.prompt, pair.preferred, pair.dispreferred)

    reward_lens_dashboard(
        lens_result=lens_result,
        attrib_result=attrib_result,
        patching_result=effects,
        save_path=os.path.join(args.output_dir, "dashboard.png"),
        title=f"Reward Model Analysis: {args.model.split('/')[-1]}",
    )

    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
