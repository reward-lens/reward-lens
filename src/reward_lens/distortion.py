"""
Distortion Index — Predictive Reward Hacking Analysis.

Based on "Reward Hacking as Equilibrium under Finite Evaluation" (2603.28063).

The key insight: reward hacking is a structural equilibrium, not a bug to fix.
Under finite evaluation, any optimized agent will under-invest in unmeasured
quality dimensions. The distortion index predicts which dimensions are under-
covered and how severely they will be hacked.

This is fundamentally different from the HackingDetector module:
- HackingDetector: DETECTS existing biases through post-hoc testing
- DistortionAnalyzer: PREDICTS hacking severity BEFORE deployment

The mathematical framework:
1. Quality is multi-dimensional (N dimensions: helpfulness, safety, honesty, ...)
2. Evaluation is finite (K probes, where K << N in practice)
3. Coverage matrix C shows which probes cover which dimensions
4. Distortion index = how under-covered each dimension is relative to others

Agentic amplification: as agents gain tools, quality dimensions scale
combinatorially while evaluation scales linearly → distortion increases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from reward_lens.model import RewardModel
from reward_lens.diagnostic_data import PreferencePair


@dataclass
class DistortionReport:
    """Result of distortion index analysis.

    Attributes:
        quality_dimensions: Names of the quality dimensions analyzed.
        per_dimension_distortion: Distortion index per dimension (higher = more hackable).
        under_covered_dimensions: Dimensions with distortion above threshold.
        predicted_hacking_severity: Overall predicted severity (0-1).
        coverage_matrix: Matrix showing which probes cover which dimensions.
        effective_coverage: Effective coverage per dimension (0-1).
        recommendations: Suggested additional probes to reduce distortion.
    """

    quality_dimensions: list[str]
    per_dimension_distortion: dict[str, float]
    under_covered_dimensions: list[str]
    predicted_hacking_severity: float
    coverage_matrix: np.ndarray
    effective_coverage: dict[str, float]
    recommendations: list[str] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print a formatted distortion analysis summary."""
        print(f"\n{'='*60}")
        print("Distortion Index Analysis — Predicted Hacking Vulnerabilities")
        print(f"{'='*60}")

        print(f"\nOverall Predicted Hacking Severity: {self.predicted_hacking_severity:.2%}")

        print("\nPer-Dimension Analysis:")
        sorted_dims = sorted(
            self.per_dimension_distortion.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for dim, distortion in sorted_dims:
            coverage = self.effective_coverage.get(dim, 0.0)
            risk = "🔴 HIGH RISK" if distortion > 0.7 else (
                "🟡 MODERATE" if distortion > 0.4 else "🟢 LOW"
            )
            print(f"  {dim}:")
            print(f"    Distortion Index: {distortion:.3f} ({risk})")
            print(f"    Effective Coverage: {coverage:.1%}")

        if self.under_covered_dimensions:
            print(f"\n⚠️  Under-Covered Dimensions (likely to be hacked):")
            for dim in self.under_covered_dimensions:
                print(f"    - {dim}")

        if self.recommendations:
            print(f"\n📋 Recommendations:")
            for rec in self.recommendations:
                print(f"    • {rec}")

        print(f"\n{'='*60}")

    def plot(
        self,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (12, 5),
    ) -> None:
        """Plot distortion analysis visualization.

        Args:
            save_path: Optional path to save figure.
            figsize: Figure size.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left: Distortion index bar chart
        dims = list(self.per_dimension_distortion.keys())
        distortions = [self.per_dimension_distortion[d] for d in dims]
        colors = ['#F44336' if d > 0.7 else '#FF9800' if d > 0.4 else '#4CAF50'
                  for d in distortions]

        y_pos = np.arange(len(dims))
        ax1.barh(y_pos, distortions, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(dims)
        ax1.set_xlabel('Distortion Index')
        ax1.set_title('Predicted Hacking Vulnerability by Dimension')
        ax1.axvline(x=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
        ax1.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='High threshold')
        ax1.legend(fontsize=8)
        ax1.set_xlim(0, 1)

        # Right: Coverage heatmap
        if self.coverage_matrix is not None and self.coverage_matrix.size > 0:
            import seaborn as sns
            sns.heatmap(
                self.coverage_matrix.T,
                ax=ax2,
                cmap='YlGnBu',
                xticklabels=[f'P{i}' for i in range(self.coverage_matrix.shape[0])],
                yticklabels=dims,
                cbar_kws={'label': 'Coverage Strength'},
            )
            ax2.set_xlabel('Evaluation Probes')
            ax2.set_title('Coverage Matrix')
        else:
            ax2.text(0.5, 0.5, 'Coverage matrix\nnot available',
                     ha='center', va='center', fontsize=12)
            ax2.set_axis_off()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()


class DistortionAnalyzer:
    """Analyze reward model for predicted hacking vulnerabilities.

    This implements the distortion index framework from "Reward Hacking as
    Equilibrium under Finite Evaluation." It predicts which quality dimensions
    are under-covered by current evaluation and thus likely to be hacked.

    Args:
        model: A RewardModel instance.

    Example:
        >>> analyzer = DistortionAnalyzer(model)
        >>> report = analyzer.compute_distortion_index(
        ...     quality_dimensions=["helpfulness", "safety", "honesty"],
        ...     evaluation_probes={
        ...         "helpfulness": [pair1, pair2],
        ...         "safety": [pair3],
        ...         "honesty": [],  # No probes!
        ...     }
        ... )
        >>> report.print_summary()
        # Shows "honesty" as high distortion (under-covered)
    """

    def __init__(self, model: RewardModel):
        self.model = model

    def compute_distortion_index(
        self,
        quality_dimensions: list[str],
        evaluation_probes: dict[str, list[PreferencePair]],
        cross_dimension_probes: Optional[list[tuple[PreferencePair, list[str]]]] = None,
        coverage_threshold: float = 0.3,
        distortion_threshold: float = 0.5,
        max_length: int = 2048,
    ) -> DistortionReport:
        """Compute distortion index predicting hacking severity per dimension.

        The distortion index for dimension d is:
            D(d) = 1 - C_eff(d) / max(C_eff)

        where C_eff(d) is the effective coverage of dimension d, computed from
        how many probes target that dimension and how well the reward model
        distinguishes them.

        Args:
            quality_dimensions: List of quality dimension names to analyze.
            evaluation_probes: Dict mapping dimension name to list of
                PreferencePair objects that specifically test that dimension.
            cross_dimension_probes: Optional list of (pair, [dimensions]) tuples
                for probes that test multiple dimensions.
            coverage_threshold: Minimum coverage to consider a dimension "covered."
            distortion_threshold: Threshold above which to flag as under-covered.
            max_length: Maximum sequence length for scoring.

        Returns:
            DistortionReport with distortion analysis results.
        """
        n_dims = len(quality_dimensions)

        # Build coverage matrix: rows = probes, cols = dimensions
        all_probes = []
        probe_dim_mappings = []

        for dim in quality_dimensions:
            probes = evaluation_probes.get(dim, [])
            for probe in probes:
                all_probes.append(probe)
                probe_dim_mappings.append([dim])

        if cross_dimension_probes:
            for probe, dims in cross_dimension_probes:
                all_probes.append(probe)
                probe_dim_mappings.append(dims)

        n_probes = len(all_probes)

        if n_probes == 0:
            # No probes at all — maximum distortion for everything
            return DistortionReport(
                quality_dimensions=quality_dimensions,
                per_dimension_distortion={d: 1.0 for d in quality_dimensions},
                under_covered_dimensions=quality_dimensions.copy(),
                predicted_hacking_severity=1.0,
                coverage_matrix=np.zeros((0, n_dims)),
                effective_coverage={d: 0.0 for d in quality_dimensions},
                recommendations=[
                    f"Add evaluation probes for ALL dimensions: {quality_dimensions}"
                ],
            )

        # Initialize coverage matrix
        coverage_matrix = np.zeros((n_probes, n_dims))

        # Compute coverage: score each probe and measure discrimination
        probe_scores = []
        for probe in all_probes:
            score_w, score_l = self.model.score_pair(
                probe.prompt, probe.preferred, probe.dispreferred,
                max_length=max_length
            )
            delta = score_w - score_l
            probe_scores.append(delta)

        probe_scores = np.array(probe_scores)

        # Normalize scores to [0, 1] representing discrimination strength
        if probe_scores.max() > probe_scores.min():
            norm_scores = (probe_scores - probe_scores.min()) / (
                probe_scores.max() - probe_scores.min()
            )
        else:
            norm_scores = np.ones_like(probe_scores) * 0.5

        # Fill coverage matrix
        dim_to_idx = {d: i for i, d in enumerate(quality_dimensions)}
        for probe_idx, dims in enumerate(probe_dim_mappings):
            for dim in dims:
                if dim in dim_to_idx:
                    # Coverage = normalized discrimination strength
                    # Probes that the model distinguishes well provide more coverage
                    coverage_matrix[probe_idx, dim_to_idx[dim]] = max(
                        0, norm_scores[probe_idx]
                    )

        # Compute effective coverage per dimension
        # Using sum with diminishing returns: C_eff = 1 - prod(1 - c_i)
        effective_coverage = {}
        for dim_idx, dim in enumerate(quality_dimensions):
            dim_coverage = coverage_matrix[:, dim_idx]
            if dim_coverage.sum() == 0:
                eff = 0.0
            else:
                # Diminishing returns formula
                eff = 1.0 - np.prod(1.0 - np.clip(dim_coverage, 0, 1))
            effective_coverage[dim] = eff

        # Compute distortion index
        max_coverage = max(effective_coverage.values()) if effective_coverage else 0
        if max_coverage > 0:
            per_dimension_distortion = {
                dim: 1.0 - (cov / max_coverage)
                for dim, cov in effective_coverage.items()
            }
        else:
            per_dimension_distortion = {d: 1.0 for d in quality_dimensions}

        # Identify under-covered dimensions
        under_covered = [
            dim for dim, dist in per_dimension_distortion.items()
            if dist > distortion_threshold
        ]

        # Compute overall severity
        predicted_severity = np.mean(list(per_dimension_distortion.values()))

        # Generate recommendations
        recommendations = []
        for dim in under_covered:
            n_probes_dim = sum(
                1 for mapping in probe_dim_mappings if dim in mapping
            )
            if n_probes_dim == 0:
                recommendations.append(
                    f"Add evaluation probes for '{dim}' (currently 0 probes)"
                )
            else:
                recommendations.append(
                    f"Add more diverse probes for '{dim}' "
                    f"(current {n_probes_dim} probes have low discrimination)"
                )

        return DistortionReport(
            quality_dimensions=quality_dimensions,
            per_dimension_distortion=per_dimension_distortion,
            under_covered_dimensions=under_covered,
            predicted_hacking_severity=predicted_severity,
            coverage_matrix=coverage_matrix,
            effective_coverage=effective_coverage,
            recommendations=recommendations,
        )

    def analyze_agentic_amplification(
        self,
        base_dimensions: list[str],
        tool_count: int,
        base_probes: dict[str, list[PreferencePair]],
        max_length: int = 2048,
    ) -> DistortionReport:
        """Analyze how distortion amplifies in agentic settings.

        As agents gain access to more tools, quality dimensions scale
        combinatorially while evaluation typically scales linearly. This
        method estimates the amplified distortion.

        Args:
            base_dimensions: Base quality dimensions without tools.
            tool_count: Number of tools the agent has access to.
            base_probes: Evaluation probes for base dimensions.
            max_length: Maximum sequence length.

        Returns:
            DistortionReport with amplification-adjusted distortion.
        """
        # Combinatorial expansion: each tool creates interaction dimensions
        # e.g., "safety" becomes "safety_tool1", "safety_tool1_tool2", etc.
        # For tractability, we approximate with a scaling factor

        base_report = self.compute_distortion_index(
            quality_dimensions=base_dimensions,
            evaluation_probes=base_probes,
            max_length=max_length,
        )

        # Amplification factor: quality dims scale as O(2^tools), evals as O(tools)
        # Simplified: amplification = 2^tools / (tools + 1)
        amplification = (2 ** min(tool_count, 10)) / (tool_count + 1)

        # Amplify distortion (capped at 1.0)
        amplified_distortion = {
            dim: min(1.0, dist * np.log2(amplification + 1))
            for dim, dist in base_report.per_dimension_distortion.items()
        }

        # Recalculate under-covered
        under_covered = [
            dim for dim, dist in amplified_distortion.items()
            if dist > 0.5
        ]

        amplified_severity = np.mean(list(amplified_distortion.values()))

        recommendations = base_report.recommendations.copy()
        if tool_count > 3:
            recommendations.insert(
                0,
                f"⚠️ Agentic amplification with {tool_count} tools "
                f"increases distortion ~{amplification:.1f}x. "
                f"Consider tool-specific evaluation probes."
            )

        return DistortionReport(
            quality_dimensions=base_dimensions,
            per_dimension_distortion=amplified_distortion,
            under_covered_dimensions=under_covered,
            predicted_hacking_severity=amplified_severity,
            coverage_matrix=base_report.coverage_matrix,
            effective_coverage=base_report.effective_coverage,
            recommendations=recommendations,
        )

    def compare_evaluation_strategies(
        self,
        quality_dimensions: list[str],
        strategies: dict[str, dict[str, list[PreferencePair]]],
        max_length: int = 2048,
    ) -> dict[str, DistortionReport]:
        """Compare distortion across different evaluation strategies.

        Useful for deciding which evaluation probes to invest in.

        Args:
            quality_dimensions: Quality dimensions to analyze.
            strategies: Dict mapping strategy name to evaluation_probes dict.
            max_length: Maximum sequence length.

        Returns:
            Dict mapping strategy name to DistortionReport.
        """
        results = {}
        for strategy_name, probes in strategies.items():
            results[strategy_name] = self.compute_distortion_index(
                quality_dimensions=quality_dimensions,
                evaluation_probes=probes,
                max_length=max_length,
            )
        return results
