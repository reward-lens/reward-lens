"""
Example: Multi-Dimensional Preference Analysis

Analyze how a reward model handles different preference dimensions
(helpfulness, safety, verbosity, sycophancy, etc.) and measure
whether the same circuits are used for different dimensions.

This is the experiment that answers: "Are preference dimensions
computed by separate circuits (modular) or overlapping circuits
(entangled)?" — a question with direct implications for alignment.

Usage:
    python examples/multi_dimension_analysis.py --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
"""

import argparse
import os

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Multi-Dimension Preference Analysis")
    parser.add_argument(
        "--model",
        type=str,
        default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/multi_dim",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top components per dimension for overlap analysis",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------------
    print(f"\nLoading: {args.model}")
    from reward_lens import RewardModel
    rm = RewardModel.from_pretrained(args.model)
    print(rm)

    # -----------------------------------------------------------------------
    # 2. Run reward lens per dimension
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Reward Lens per Preference Dimension")
    print(f"{'='*60}\n")

    from reward_lens.lens import RewardLens
    from reward_lens.diagnostic_data import get_diagnostic_pairs, ALL_DIMENSIONS

    lens = RewardLens(rm)
    crystal_by_dim = {}

    for dim_name in ALL_DIMENSIONS:
        pairs = get_diagnostic_pairs([dim_name])
        if not pairs:
            continue

        # Average crystallization layer across pairs in this dimension
        crystal_layers = []
        for pair in pairs:
            result = lens.trace(pair.prompt, pair.preferred, pair.dispreferred)
            crystal_layers.append(result.crystallization_layer)

        avg_crystal = np.mean(crystal_layers)
        crystal_by_dim[dim_name] = avg_crystal
        print(f"  {dim_name:15s}: avg crystallization layer = {avg_crystal:.1f}")

    # -----------------------------------------------------------------------
    # 3. Component attribution per dimension
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Component Attribution per Dimension")
    print(f"{'='*60}\n")

    from reward_lens.attribution import ComponentAttribution

    attrib = ComponentAttribution(rm)
    top_components_by_dim = {}

    for dim_name in ALL_DIMENSIONS:
        pairs = get_diagnostic_pairs([dim_name])
        if not pairs:
            continue

        # Use first pair for each dimension
        pair = pairs[0]
        result = attrib.attribute(pair.prompt, pair.preferred, pair.dispreferred)
        top = result.top_k(k=args.top_k, by="differential")
        top_set = set(name for name, _ in top)
        top_components_by_dim[dim_name] = top_set

        print(f"\n  {dim_name} — top 5:")
        for name, val in top[:5]:
            print(f"    {name}: {val:+.4f}")

    # -----------------------------------------------------------------------
    # 4. Circuit overlap analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Circuit Overlap Between Dimensions")
    print(f"{'='*60}\n")

    dim_names = sorted(top_components_by_dim.keys())
    n_dims = len(dim_names)
    overlap_matrix = np.zeros((n_dims, n_dims))

    for i, dim_a in enumerate(dim_names):
        for j, dim_b in enumerate(dim_names):
            set_a = top_components_by_dim[dim_a]
            set_b = top_components_by_dim[dim_b]
            if len(set_a | set_b) > 0:
                overlap = len(set_a & set_b) / len(set_a | set_b)
            else:
                overlap = 0.0
            overlap_matrix[i, j] = overlap

    # Print overlap table
    header = f"{'':15s}" + "".join(f"{d[:8]:>10s}" for d in dim_names)
    print(header)
    for i, dim_a in enumerate(dim_names):
        row = f"{dim_a:15s}"
        for j in range(n_dims):
            row += f"{overlap_matrix[i, j]:10.2f}"
        print(row)

    # Plot
    from reward_lens.viz import circuit_overlap_plot

    circuit_overlap_plot(
        overlap_matrix,
        dim_names,
        save_path=os.path.join(args.output_dir, "circuit_overlap.png"),
    )

    # -----------------------------------------------------------------------
    # 5. Interpretation
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Interpretation")
    print(f"{'='*60}")

    # Compute average overlap (excluding diagonal)
    mask = ~np.eye(n_dims, dtype=bool)
    avg_overlap = overlap_matrix[mask].mean()

    print(f"\n  Average circuit overlap (off-diagonal): {avg_overlap:.3f}")
    if avg_overlap > 0.7:
        print("  → Preference dimensions share most circuits (ENTANGLED)")
        print("  → Implication: hard to independently control different preference axes")
    elif avg_overlap > 0.3:
        print("  → Preference dimensions partially overlap (SHARED + SPECIALIZED)")
        print("  → Implication: some modular control possible, shared comprehension base")
    else:
        print("  → Preference dimensions use different circuits (MODULAR)")
        print("  → Implication: good potential for independent preference control")

    # Which dimensions are most entangled?
    max_off_diag = 0.0
    max_pair = ("", "")
    for i in range(n_dims):
        for j in range(i+1, n_dims):
            if overlap_matrix[i, j] > max_off_diag:
                max_off_diag = overlap_matrix[i, j]
                max_pair = (dim_names[i], dim_names[j])

    print(f"\n  Most entangled pair: {max_pair[0]} ↔ {max_pair[1]} (overlap: {max_off_diag:.3f})")

    print(f"\nOutputs saved to: {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
