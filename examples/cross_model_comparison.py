"""
Example: Cross-Model Comparison

Compare two or more reward models to understand:
- Do they develop preferences at the same layers?
- Do they rely on the same components?
- Where do they disagree?

Usage:
    python examples/cross_model_comparison.py \
        --models Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
                 RLHFlow/ArmoRM-Llama3-8B-v0.1

Requires enough GPU VRAM to load each model sequentially.
"""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Cross-Model Comparison")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
            "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        ],
        help="List of HuggingFace model names",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/comparison",
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from reward_lens import RewardModel
    from reward_lens.lens import RewardLens
    from reward_lens.attribution import ComponentAttribution
    from reward_lens.diagnostic_data import get_diagnostic_pairs

    pairs = get_diagnostic_pairs(["helpfulness", "safety", "verbosity"])

    results_by_model = {}

    for model_name in args.models:
        short_name = model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Analyzing: {short_name}")
        print(f"{'='*60}")

        rm = RewardModel.from_pretrained(model_name)
        lens = RewardLens(rm)
        attrib = ComponentAttribution(rm)

        model_results = {
            "lens": [],
            "attrib": [],
            "crystallization": [],
            "scores": [],
        }

        for pair in pairs:
            lr = lens.trace(pair.prompt, pair.preferred, pair.dispreferred)
            ar = attrib.attribute(pair.prompt, pair.preferred, pair.dispreferred)

            model_results["lens"].append(lr)
            model_results["attrib"].append(ar)
            model_results["crystallization"].append(lr.crystallization_layer)
            model_results["scores"].append(
                (lr.reward_preferred, lr.reward_dispreferred)
            )

            print(f"  [{pair.dimension}] crystal=L{lr.crystallization_layer}  "
                  f"Δ={lr.reward_preferred - lr.reward_dispreferred:+.4f}")

        results_by_model[short_name] = model_results

        # Free GPU memory
        del rm, lens, attrib
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Cross-model analysis
    # -----------------------------------------------------------------------
    model_names = list(results_by_model.keys())
    n_models = len(model_names)

    print(f"\n{'='*60}")
    print("Cross-Model Comparison")
    print(f"{'='*60}")

    # 1. Crystallization comparison
    print("\nCrystallization layers (as fraction of model depth):")
    for name in model_names:
        crystals = results_by_model[name]["crystallization"]
        n_layers = len(results_by_model[name]["lens"][0].layers) - 1  # subtract layer -1
        fractions = [c / n_layers for c in crystals]
        avg = np.mean(fractions)
        print(f"  {name}: avg = {avg:.2f}  per-pair: {[f'{f:.2f}' for f in fractions]}")

    # 2. Agreement on preference direction
    print("\nPreference agreement:")
    for i in range(len(pairs)):
        pair = pairs[i]
        deltas = {}
        for name in model_names:
            w, l = results_by_model[name]["scores"][i]
            deltas[name] = w - l

        signs = [np.sign(deltas[n]) for n in model_names]
        agree = all(s == signs[0] for s in signs)
        symbol = "✓" if agree else "✗"
        delta_str = ", ".join(f"{n}: {deltas[n]:+.4f}" for n in model_names)
        print(f"  [{symbol}] {pair.dimension}: {delta_str}")

    # 3. Formation curve correlation
    print("\nFormation curve correlation (pairwise):")
    if n_models >= 2:
        for pair_idx in range(len(pairs)):
            pair = pairs[pair_idx]
            curves = {}
            for name in model_names:
                lr = results_by_model[name]["lens"][pair_idx]
                # Normalize differential to [0, 1] range
                diff = lr.differential
                if diff[-1] != diff[0]:
                    curves[name] = (diff - diff[0]) / (diff[-1] - diff[0] + 1e-8)
                else:
                    curves[name] = diff - diff[0]

            # Resample to common length for correlation
            common_len = 100
            for name in model_names:
                c = curves[name]
                x_old = np.linspace(0, 1, len(c))
                x_new = np.linspace(0, 1, common_len)
                curves[name] = np.interp(x_new, x_old, c)

            for i_m in range(n_models):
                for j_m in range(i_m + 1, n_models):
                    corr = np.corrcoef(
                        curves[model_names[i_m]], curves[model_names[j_m]]
                    )[0, 1]
                    print(
                        f"  {pair.dimension}: "
                        f"{model_names[i_m]} vs {model_names[j_m]}: r = {corr:.3f}"
                    )

    print(f"\nDone!")


if __name__ == "__main__":
    main()
