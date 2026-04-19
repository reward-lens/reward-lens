"""
Example: Reward Hacking Deep Dive

Goes beyond the automated scan to mechanistically analyze
WHY a reward model is vulnerable to a specific exploit.

Workflow:
1. Run hacking scan to find vulnerabilities
2. For each vulnerability, run reward lens to see which layers respond
3. Run component attribution to find the responsible components
4. (Optional) Run activation patching on top components
5. Generate a detailed vulnerability report

Usage:
    python examples/hacking_deepdive.py --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
"""

import argparse
import os

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Reward Hacking Deep Dive")
    parser.add_argument(
        "--model",
        type=str,
        default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/hacking",
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load model and run hacking scan
    # -----------------------------------------------------------------------
    print(f"\nLoading: {args.model}")
    from reward_lens import RewardModel
    rm = RewardModel.from_pretrained(args.model)

    from reward_lens.hacking import HackingDetector
    detector = HackingDetector(rm)
    report = detector.scan()
    report.print_summary()

    # -----------------------------------------------------------------------
    # 2. Deep dive into each vulnerable dimension
    # -----------------------------------------------------------------------
    vulnerable_dims = report.get_vulnerable_dimensions(threshold=0.3)
    if not vulnerable_dims:
        print("\nNo significant vulnerabilities found! This is a well-calibrated model.")
        return

    from reward_lens.lens import RewardLens
    from reward_lens.attribution import ComponentAttribution
    from reward_lens.hacking import ALL_TESTS

    lens = RewardLens(rm)
    attrib = ComponentAttribution(rm)

    for dim in vulnerable_dims:
        print(f"\n{'='*60}")
        print(f"Deep Dive: {dim.upper()} BIAS")
        print(f"{'='*60}")

        bias_result = report.results[dim]
        test_pairs = ALL_TESTS[dim]

        for pair_idx, pair_data in enumerate(test_pairs):
            prompt = pair_data["prompt"]
            neutral = pair_data["neutral"]
            biased = pair_data["biased"]

            # Reward lens: when does the bias emerge?
            lens_result = lens.trace(prompt, neutral, biased)
            print(f"\n  Pair {pair_idx + 1}:")
            print(f"    Reward (neutral):  {lens_result.reward_preferred:.4f}")
            print(f"    Reward (biased):   {lens_result.reward_dispreferred:.4f}")
            print(f"    Bias emerges at:   Layer {lens_result.crystallization_layer}")

            # Which layers contribute most to the bias?
            top_layers = np.argsort(np.abs(lens_result.marginal_contributions))[-3:][::-1]
            print(f"    Top bias layers:   {[int(lens_result.layers[i+1]) for i in top_layers]}")

            # Component attribution
            # Note: for hacking analysis, the "dispreferred" is actually the biased
            # version (we want to understand what gives it HIGHER reward)
            attrib_result = attrib.attribute(prompt, biased, neutral)
            top_components = attrib_result.top_k(k=5, by="differential")
            print(f"    Components driving bias:")
            for name, val in top_components:
                print(f"      {name}: {val:+.4f}")

            lens_result.plot(
                save_path=os.path.join(args.output_dir, f"{dim}_pair{pair_idx}_lens.png"),
                title=f"{dim.title()} Bias — Reward Lens (Pair {pair_idx + 1})",
            )

        # -----------------------------------------------------------------------
        # 3. Aggregate analysis across pairs
        # -----------------------------------------------------------------------
        print(f"\n  Aggregate {dim} analysis:")
        print(f"    Mean bias: {bias_result.mean_delta:+.4f}")
        print(f"    Effect size: {bias_result.effect_size:.3f}")

        # Check if bias is localized (few components) or distributed (many)
        all_top_components = set()
        for pair_data in test_pairs:
            ar = attrib.attribute(pair_data["prompt"], pair_data["biased"], pair_data["neutral"])
            top = ar.top_k(k=10, by="differential")
            for name, _ in top:
                all_top_components.add(name)

        n_unique = len(all_top_components)
        n_possible = rm.n_layers * 2 + 1  # attn + mlp per layer + embed

        print(f"    Bias spread: {n_unique}/{n_possible} unique components across pairs")
        if n_unique <= 5:
            print(f"    → LOCALIZED bias (few components, potentially fixable via ablation)")
        elif n_unique <= 15:
            print(f"    → MODERATELY distributed bias")
        else:
            print(f"    → WIDELY distributed bias (hard to fix with targeted ablation)")

    print(f"\nOutputs saved to: {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
