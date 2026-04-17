"""
Reward Term Conflict Analyzer.

Based on "Aligned, Orthogonal or In-conflict: When Can We Safely Optimize CoT?" (2603.30036).

The key insight: when reward signals can be decomposed into multiple terms
(e.g., helpfulness + safety + honesty), the geometric relationship between
these terms determines whether optimization is safe:

1. ALIGNED: Terms point in similar directions.
   → Optimizing one improves others. Safe to optimize jointly.

2. ORTHOGONAL: Terms point in independent directions.
   → Optimizing one doesn't affect others. Monitorability preserved.

3. IN-CONFLICT: Terms point in opposing directions.
   → Optimizing one hurts others. Models may learn to hide reasoning.
   → This is when CoT monitorability breaks down!

For reward models, this helps identify:
- Which reward components may be fighting each other
- Whether the reward structure encourages reasoning obfuscation
- How to decompose multi-objective rewards safely
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch

from reward_lens.model import RewardModel


class RelationshipType(Enum):
    """Classification of relationship between reward terms."""
    ALIGNED = "aligned"
    ORTHOGONAL = "orthogonal"
    IN_CONFLICT = "in_conflict"


@dataclass
class TermPairAnalysis:
    """Analysis of relationship between two reward terms.
    
    Attributes:
        term1: Name of first term.
        term2: Name of second term.
        cosine_similarity: Cosine similarity between directions (-1 to 1).
        relationship: Classification of relationship.
        conflict_severity: How severe the conflict is (0 if not conflicting).
        recommendations: Suggested actions for this pair.
    """
    
    term1: str
    term2: str
    cosine_similarity: float
    relationship: RelationshipType
    conflict_severity: float
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ConflictReport:
    """Complete reward term conflict analysis.
    
    Attributes:
        term_names: Names of analyzed reward terms.
        term_directions: The direction vectors for each term.
        pairwise_analysis: Analysis for each pair of terms.
        relationship_matrix: Matrix of relationship classifications.
        similarity_matrix: Matrix of cosine similarities.
        in_conflict_pairs: List of term pairs that are in conflict.
        overall_conflict_score: 0-1 score of overall conflict severity.
        monitorability_risk: Risk that models will hide reasoning.
        recommendations: Overall recommendations.
    """
    
    term_names: list[str]
    term_directions: dict[str, torch.Tensor]
    pairwise_analysis: list[TermPairAnalysis]
    relationship_matrix: np.ndarray  # n_terms × n_terms, encoded as 0=aligned, 1=orthogonal, 2=conflict
    similarity_matrix: np.ndarray
    in_conflict_pairs: list[tuple[str, str]]
    overall_conflict_score: float
    monitorability_risk: float
    recommendations: list[str] = field(default_factory=list)
    
    def print_summary(self) -> None:
        """Print a formatted conflict analysis summary."""
        print(f"\n{'='*60}")
        print("Reward Term Conflict Analysis")
        print(f"{'='*60}")
        
        print(f"\nOverall Conflict Score: {self.overall_conflict_score:.1%}")
        print(f"Monitorability Risk: {self.monitorability_risk:.1%}")
        
        risk_level = "🔴 HIGH" if self.monitorability_risk > 0.6 else (
            "🟡 MODERATE" if self.monitorability_risk > 0.3 else "🟢 LOW"
        )
        print(f"Risk Level: {risk_level}")
        
        if self.monitorability_risk > 0.5:
            print("\n⚠️  WARNING: High monitorability risk!")
            print("    Models optimizing this reward may learn to hide reasoning.")
        
        print("\nPairwise Relationships:")
        for analysis in self.pairwise_analysis:
            icon = "🔴" if analysis.relationship == RelationshipType.IN_CONFLICT else (
                "⚪" if analysis.relationship == RelationshipType.ORTHOGONAL else "🟢"
            )
            print(f"  {icon} {analysis.term1} ↔ {analysis.term2}:")
            print(f"      Cosine similarity: {analysis.cosine_similarity:+.3f}")
            print(f"      Relationship: {analysis.relationship.value}")
            if analysis.conflict_severity > 0:
                print(f"      Conflict severity: {analysis.conflict_severity:.2f}")
        
        if self.in_conflict_pairs:
            print(f"\n⚠️  In-Conflict Pairs ({len(self.in_conflict_pairs)}):")
            for t1, t2 in self.in_conflict_pairs:
                print(f"    • {t1} vs {t2}")
        
        if self.recommendations:
            print("\n📋 Recommendations:")
            for rec in self.recommendations:
                print(f"  • {rec}")
        
        print(f"\n{'='*60}")
    
    def plot(
        self,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (12, 5),
    ) -> None:
        """Plot similarity matrix and relationship diagram.
        
        Args:
            save_path: Optional path to save figure.
            figsize: Figure size.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Cosine similarity heatmap
        sns.heatmap(
            self.similarity_matrix,
            ax=ax1,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=self.term_names,
            yticklabels=self.term_names,
            square=True,
        )
        ax1.set_title("Cosine Similarity Between Terms")
        
        # Right: Relationship matrix
        # 0 = aligned (green), 1 = orthogonal (white), 2 = conflict (red)
        cmap = plt.cm.colors.ListedColormap(['#4CAF50', '#FFFFFF', '#F44336'])
        im = ax2.imshow(self.relationship_matrix, cmap=cmap, vmin=0, vmax=2)
        
        ax2.set_xticks(range(len(self.term_names)))
        ax2.set_yticks(range(len(self.term_names)))
        ax2.set_xticklabels(self.term_names, rotation=45, ha='right')
        ax2.set_yticklabels(self.term_names)
        ax2.set_title("Relationship Classification")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4CAF50', label='Aligned'),
            Patch(facecolor='#FFFFFF', edgecolor='gray', label='Orthogonal'),
            Patch(facecolor='#F44336', label='In-Conflict'),
        ]
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()


class RewardConflictAnalyzer:
    """Analyze conflicts between reward terms.
    
    This helps identify when reward structures may cause models to
    hide reasoning (in-conflict terms) vs when optimization is safe
    (aligned or orthogonal terms).
    
    Args:
        model: A RewardModel instance.
        
    Example:
        >>> analyzer = RewardConflictAnalyzer(model)
        >>> 
        >>> # For multi-objective models, extract term directions
        >>> terms = {
        ...     "helpfulness": model.get_objective_direction("helpfulness"),
        ...     "safety": model.get_objective_direction("safety"),
        ... }
        >>> 
        >>> # Or learn directions from contrastive pairs
        >>> terms = analyzer.learn_term_directions({
        ...     "helpfulness": [(helpful_pair1, helpful_pair2)],
        ...     "safety": [(safe_pair1, safe_pair2)],
        ... })
        >>> 
        >>> report = analyzer.analyze_conflicts(terms)
        >>> report.print_summary()
    """
    
    # Thresholds for classification
    ALIGNED_THRESHOLD = 0.5      # cos > 0.5 → aligned
    ORTHOGONAL_THRESHOLD = 0.2   # |cos| < 0.2 → orthogonal
    # Between thresholds → slight alignment/conflict
    # cos < -0.3 → in-conflict
    CONFLICT_THRESHOLD = -0.3
    
    def __init__(self, model: RewardModel):
        self.model = model
    
    def analyze_conflicts(
        self,
        reward_terms: dict[str, torch.Tensor],
        aligned_threshold: float = 0.5,
        orthogonal_threshold: float = 0.2,
        conflict_threshold: float = -0.3,
    ) -> ConflictReport:
        """Analyze conflicts between reward term directions.
        
        Args:
            reward_terms: Dict mapping term names to direction vectors.
                Each vector should have shape (d_model,).
            aligned_threshold: Cosine above this → aligned.
            orthogonal_threshold: |Cosine| below this → orthogonal.
            conflict_threshold: Cosine below this → in-conflict.
            
        Returns:
            ConflictReport with full conflict analysis.
        """
        term_names = list(reward_terms.keys())
        n_terms = len(term_names)
        
        # Normalize all directions
        normalized = {}
        for name, direction in reward_terms.items():
            norm = direction.float().norm()
            if norm > 1e-8:
                normalized[name] = direction.float() / norm
            else:
                normalized[name] = direction.float()
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((n_terms, n_terms))
        relationship_matrix = np.zeros((n_terms, n_terms), dtype=int)
        
        pairwise_analysis = []
        in_conflict_pairs = []
        
        for i, name1 in enumerate(term_names):
            for j, name2 in enumerate(term_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    relationship_matrix[i, j] = 0  # aligned with self
                else:
                    cos_sim = (normalized[name1] @ normalized[name2]).item()
                    similarity_matrix[i, j] = cos_sim
                    
                    # Classify relationship
                    if cos_sim > aligned_threshold:
                        rel = RelationshipType.ALIGNED
                        relationship_matrix[i, j] = 0
                        conflict_sev = 0.0
                    elif cos_sim < conflict_threshold:
                        rel = RelationshipType.IN_CONFLICT
                        relationship_matrix[i, j] = 2
                        conflict_sev = abs(cos_sim)
                        if i < j:
                            in_conflict_pairs.append((name1, name2))
                    elif abs(cos_sim) < orthogonal_threshold:
                        rel = RelationshipType.ORTHOGONAL
                        relationship_matrix[i, j] = 1
                        conflict_sev = 0.0
                    else:
                        # Between thresholds - slight alignment or slight conflict
                        if cos_sim > 0:
                            rel = RelationshipType.ALIGNED
                            relationship_matrix[i, j] = 0
                        else:
                            rel = RelationshipType.IN_CONFLICT
                            relationship_matrix[i, j] = 2
                            if i < j:
                                in_conflict_pairs.append((name1, name2))
                        conflict_sev = max(0, -cos_sim)
                    
                    if i < j:  # Only add each pair once
                        recs = self._pair_recommendations(name1, name2, cos_sim, rel)
                        pairwise_analysis.append(TermPairAnalysis(
                            term1=name1,
                            term2=name2,
                            cosine_similarity=cos_sim,
                            relationship=rel,
                            conflict_severity=conflict_sev,
                            recommendations=recs,
                        ))
        
        # Compute overall scores
        n_pairs = n_terms * (n_terms - 1) // 2
        if n_pairs > 0:
            n_conflicts = len(in_conflict_pairs)
            overall_conflict = n_conflicts / n_pairs
            
            # Monitorability risk based on conflict severity
            max_conflict = max(
                (pa.conflict_severity for pa in pairwise_analysis),
                default=0
            )
            monitorability_risk = min(1.0, overall_conflict + 0.5 * max_conflict)
        else:
            overall_conflict = 0.0
            monitorability_risk = 0.0
        
        # Generate overall recommendations
        recommendations = self._overall_recommendations(
            in_conflict_pairs, monitorability_risk, term_names
        )
        
        return ConflictReport(
            term_names=term_names,
            term_directions={k: v.cpu() for k, v in normalized.items()},
            pairwise_analysis=pairwise_analysis,
            relationship_matrix=relationship_matrix,
            similarity_matrix=similarity_matrix,
            in_conflict_pairs=in_conflict_pairs,
            overall_conflict_score=overall_conflict,
            monitorability_risk=monitorability_risk,
            recommendations=recommendations,
        )
    
    def learn_term_directions(
        self,
        term_pairs: dict[str, list[tuple[str, str, str, str]]],
        max_length: int = 2048,
    ) -> dict[str, torch.Tensor]:
        """Learn reward term directions from contrastive pairs.
        
        For each term, provide pairs where the preferred response is better
        on that specific dimension. The direction is learned as the average
        difference in final-layer activations.
        
        Args:
            term_pairs: Dict mapping term name to list of (prompt, preferred, dispreferred, _).
                Each tuple is (prompt, preferred_response, dispreferred_response, _).
            max_length: Maximum sequence length.
            
        Returns:
            Dict mapping term names to learned direction vectors.
        """
        directions = {}
        
        for term_name, pairs in term_pairs.items():
            deltas = []
            
            for pair in pairs:
                prompt, preferred, dispreferred = pair[0], pair[1], pair[2]
                
                # Get activations for both
                _, cache_pref = self.model.forward_with_cache(
                    prompt, preferred, max_length=max_length
                )
                _, cache_disp = self.model.forward_with_cache(
                    prompt, dispreferred, max_length=max_length
                )
                
                # Get final layer activations
                final_layer = self.model.n_layers - 1
                h_pref = cache_pref.residual_streams.get(final_layer)
                h_disp = cache_disp.residual_streams.get(final_layer)
                
                if h_pref is not None and h_disp is not None:
                    delta = (h_pref - h_disp).squeeze().cpu().float()
                    deltas.append(delta)
            
            if deltas:
                # Average direction
                avg_direction = torch.stack(deltas).mean(dim=0)
                directions[term_name] = avg_direction
            else:
                # Zero direction if no pairs
                directions[term_name] = torch.zeros(self.model.d_model)
        
        return directions
    
    def analyze_multi_objective_model(
        self,
        objective_names: Optional[list[str]] = None,
    ) -> ConflictReport:
        """Analyze conflicts in a multi-objective reward model.
        
        For models like ArmoRM that have multiple output dimensions,
        this extracts the direction for each objective and analyzes conflicts.
        
        Args:
            objective_names: Names of objectives. If None, auto-detected.
            
        Returns:
            ConflictReport for the model's objectives.
        """
        # Try to get multi-objective directions from the adapter
        try:
            reward_weight = self.model.reward_direction
            
            if reward_weight.dim() > 1:
                # Multi-dimensional output
                n_objectives = reward_weight.shape[0]
                if objective_names is None:
                    objective_names = [f"objective_{i}" for i in range(n_objectives)]
                
                terms = {
                    name: reward_weight[i]
                    for i, name in enumerate(objective_names[:n_objectives])
                }
            else:
                # Single-objective: just use the main direction
                terms = {"reward": reward_weight}
        except Exception:
            terms = {"reward": self.model.reward_direction}
        
        return self.analyze_conflicts(terms)
    
    def _pair_recommendations(
        self,
        term1: str,
        term2: str,
        cos_sim: float,
        relationship: RelationshipType,
    ) -> list[str]:
        """Generate recommendations for a term pair."""
        recs = []
        
        if relationship == RelationshipType.IN_CONFLICT:
            recs.append(
                f"Terms '{term1}' and '{term2}' are in conflict (cos={cos_sim:.2f}). "
                "Consider reweighting or removing one term."
            )
            if cos_sim < -0.5:
                recs.append(
                    "Strong conflict may cause models to hide reasoning. "
                    "Monitor for obfuscation during training."
                )
        elif relationship == RelationshipType.ORTHOGONAL:
            recs.append(
                f"Terms '{term1}' and '{term2}' are orthogonal. "
                "Safe to optimize independently."
            )
        else:  # ALIGNED
            if cos_sim > 0.8:
                recs.append(
                    f"Terms '{term1}' and '{term2}' are highly aligned (cos={cos_sim:.2f}). "
                    "Consider combining into single term to simplify reward."
                )
        
        return recs
    
    def _overall_recommendations(
        self,
        in_conflict_pairs: list[tuple[str, str]],
        monitorability_risk: float,
        term_names: list[str],
    ) -> list[str]:
        """Generate overall recommendations."""
        recs = []
        
        if monitorability_risk > 0.6:
            recs.append(
                "⚠️ HIGH MONITORABILITY RISK: Reward structure has significant conflicts. "
                "Models may learn to obfuscate reasoning. Consider restructuring reward terms."
            )
        
        if len(in_conflict_pairs) > 0:
            recs.append(
                f"Found {len(in_conflict_pairs)} conflicting term pair(s). "
                "Review these pairs and consider: (1) removing one term, "
                "(2) adjusting weights, or (3) using auxiliary losses to encourage transparency."
            )
        
        if monitorability_risk < 0.2 and len(in_conflict_pairs) == 0:
            recs.append(
                "✅ Reward structure appears safe for CoT optimization. "
                "Terms are either aligned or orthogonal."
            )
        
        return recs


def quick_conflict_check(
    model: RewardModel,
    term_pairs: dict[str, list[tuple[str, str, str]]],
    max_length: int = 2048,
) -> ConflictReport:
    """Convenience function for quick conflict analysis.
    
    Args:
        model: RewardModel instance.
        term_pairs: Dict mapping term name to list of (prompt, preferred, dispreferred).
        max_length: Maximum sequence length.
        
    Returns:
        ConflictReport.
    """
    analyzer = RewardConflictAnalyzer(model)
    directions = analyzer.learn_term_directions(term_pairs, max_length=max_length)
    return analyzer.analyze_conflicts(directions)
