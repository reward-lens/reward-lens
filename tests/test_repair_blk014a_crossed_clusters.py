"""BLK-014-a — `labelled_contrast` silently kept one cluster per prompt group.

The map from group to cluster was built as `{g: c for g, c in zip(grp, clu)}`, a dict comprehension
over pairs, so a group appearing in several clusters kept whichever came last. Two designs show what
that cost, and only one of them failed visibly:

    four runs, every group scored in all four   cluster_level_n = 1, se = nan
    partially crossed, half the groups re-run   cluster_level_n = 2, [0.1009, 1.3379]
                                                bootstrap on the same input [0.6707, 0.7681]

A twelvefold disagreement in interval width, with nothing in the output noticing. The `nan` case at
least fails where somebody can see it; the partially crossed case returns a plausible number.

**The repair is a refusal, not a threshold.** If a contributing group maps to more than one cluster,
the design is not the one this estimator computes, and the answer is a refusal naming the group.

Raw before-and-after at `chain/repair/proofs/BLK-014-a/`.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.measure.ledger.price import CrossedClusterDesign, labelled_contrast

N_GROUPS = 40
K = 6
TRUE_GAP = 0.72
CONTRAST = [("EXPLOIT", "LEGITIMATE")]


def _sample(group_of_row: np.ndarray, cluster_of_row: np.ndarray, seed: int):
    """Rollouts with a real group effect, a real cluster effect and a real label effect.

    Every group is forced to carry both labels so the within-group contrast is estimable on all of
    them, which keeps the crossed and nested cases comparable on everything except the crossing.
    """
    rng = np.random.default_rng(seed)
    n = group_of_row.size
    labels = np.where(rng.random(n) < 0.4, "EXPLOIT", "LEGITIMATE")
    for g in np.unique(group_of_row):
        idx = np.where(group_of_row == g)[0]
        labels[idx[0]] = "EXPLOIT"
        labels[idx[1]] = "LEGITIMATE"
    group_effect = rng.standard_normal(int(group_of_row.max()) + 1) * 2.0
    cluster_effect = rng.standard_normal(int(cluster_of_row.max()) + 1) * 0.5
    values = (
        group_effect[group_of_row]
        + cluster_effect[cluster_of_row]
        + TRUE_GAP * (labels == "EXPLOIT")
        + rng.standard_normal(n) * 0.3
    )
    return values, labels


def _nested():
    """40 groups in 10 clusters of 4, each group in exactly one. The design the estimator assumes."""
    groups = np.repeat(np.arange(N_GROUPS), K)
    clusters = groups // 4
    values, labels = _sample(groups, clusters, seed=1)
    return values, labels, groups, clusters


def _fully_crossed():
    """Four runs, every prompt group scored in all four."""
    groups = np.tile(np.repeat(np.arange(N_GROUPS), K), 4)
    runs = np.repeat(np.arange(4), N_GROUPS * K)
    values, labels = _sample(groups, runs, seed=2)
    return values, labels, groups, runs


def _partially_crossed():
    """Run 0 scores every group; run 1 re-scores the first half. The dangerous case."""
    half = N_GROUPS // 2
    groups = np.concatenate([np.repeat(np.arange(N_GROUPS), K), np.repeat(np.arange(half), K)])
    runs = np.concatenate([np.zeros(N_GROUPS * K, dtype=int), np.ones(half * K, dtype=int)])
    values, labels = _sample(groups, runs, seed=3)
    return values, labels, groups, runs


def _call(sample, **kwargs):
    values, labels, groups, clusters = sample
    return labelled_contrast(
        values,
        labels,
        groups,
        cluster_ids=clusters,
        contrasts=CONTRAST,
        n_resamples=kwargs.pop("n_resamples", 0),
        ci_level=kwargs.pop("ci_level", 0.95),
        **kwargs,
    ).contrast("EXPLOIT", "LEGITIMATE")


# ---------------------------------------------------------------------------
# The refusal, on both crossed designs, analytic and bootstrap alike
# ---------------------------------------------------------------------------


def test_a_fully_crossed_design_is_refused() -> None:
    """Before the repair this returned `cluster_level_n = 1` and `se = nan`."""
    with pytest.raises(CrossedClusterDesign) as caught:
        _call(_fully_crossed())
    assert caught.value.n_crossed == N_GROUPS
    assert caught.value.n_groups == N_GROUPS
    assert len(caught.value.clusters) == 4


def test_a_partially_crossed_design_is_refused() -> None:
    """Before the repair this returned a confident `[0.1009, 1.3379]`-shaped interval."""
    with pytest.raises(CrossedClusterDesign) as caught:
        _call(_partially_crossed())
    # Only the re-scored half is crossed, and the refusal counts them rather than rounding up.
    assert caught.value.n_crossed == N_GROUPS // 2
    assert caught.value.n_groups == N_GROUPS
    assert caught.value.clusters == (0, 1)


def test_the_refusal_names_a_group_that_really_is_crossed() -> None:
    """Not the first group in the array; a group the input genuinely puts in several clusters."""
    values, labels, groups, clusters = _partially_crossed()
    with pytest.raises(CrossedClusterDesign) as caught:
        _call((values, labels, groups, clusters))
    named = caught.value.group
    assert set(np.unique(clusters[groups == named]).tolist()) == set(caught.value.clusters)
    assert len(caught.value.clusters) > 1


def test_the_bootstrap_path_refuses_too() -> None:
    """The resampling unit is the same group-to-cluster map, so it fails the same way."""
    with pytest.raises(CrossedClusterDesign):
        _call(_partially_crossed(), n_resamples=500, seed=7)


def test_the_refusal_is_a_value_error() -> None:
    """Callers already catching this module's structural refusals keep working."""
    with pytest.raises(ValueError):
        _call(_fully_crossed())


# ---------------------------------------------------------------------------
# The escape route the refusal names has to be a real one
# ---------------------------------------------------------------------------


def test_clustering_at_the_group_level_is_accepted_on_a_crossed_sample() -> None:
    """The message says to pass `group_ids` as `cluster_ids`. That must actually work."""
    values, labels, groups, _ = _fully_crossed()
    got = _call((values, labels, groups, groups))
    assert got.cluster_level_n == N_GROUPS
    assert np.isfinite(got.standard_error)
    assert got.ci_low < TRUE_GAP < got.ci_high


def test_a_degenerate_crossed_group_does_not_refuse() -> None:
    """A group carrying one label contributes no score, so its crossing cannot move the interval.

    Scoped deliberately: refusing on it would make the refusal depend on which contrast was asked
    for rather than on whether the answer is computable. On the record as a test, not as a comment.
    """
    values, labels, groups, clusters = _nested()
    extra_values = np.array([1.0, 1.2, 0.9, 1.1])
    extra_labels = np.array(["EXPLOIT"] * 4)  # one label only: degenerate
    extra_groups = np.array([N_GROUPS, N_GROUPS, N_GROUPS, N_GROUPS])
    extra_clusters = np.array([0, 1, 2, 3])  # and crossed across four clusters

    got = labelled_contrast(
        np.concatenate([values, extra_values]),
        np.concatenate([labels, extra_labels]),
        np.concatenate([groups, extra_groups]),
        cluster_ids=np.concatenate([clusters, extra_clusters]),
        contrasts=CONTRAST,
        n_resamples=0,
        ci_level=0.95,
    ).contrast("EXPLOIT", "LEGITIMATE")
    assert got.n_groups_degenerate == 1
    assert got.n_groups_informative == N_GROUPS


# ---------------------------------------------------------------------------
# The nested design must be untouched, and must agree with a bootstrap on the same input
# ---------------------------------------------------------------------------


def test_the_nested_interval_matches_a_bootstrap_on_the_same_input() -> None:
    """The second half of the child blocker's closure proof.

    At ten clusters a Student-t interval and a cluster percentile bootstrap are not identical and
    should not be asserted equal; what they must not be is an order of magnitude apart, which is
    what the collapsed map produced. Both cover the truth and their widths agree to within 40%.
    """
    sample = _nested()
    analytic = _call(sample)
    booted = _call(sample, n_resamples=4000, seed=7)

    assert analytic.cluster_level_n == 10
    assert booted.cluster_level_n == 10
    assert analytic.ci_low < TRUE_GAP < analytic.ci_high
    assert booted.ci_low < TRUE_GAP < booted.ci_high

    wide = analytic.ci_high - analytic.ci_low
    narrow = booted.ci_high - booted.ci_low
    ratio = max(wide, narrow) / min(wide, narrow)
    assert ratio < 1.4, f"interval widths {wide:.4f} and {narrow:.4f} disagree by {ratio:.2f}x"


def test_the_clustered_standard_error_is_the_one_the_formula_says() -> None:
    """Recomputed here from the group scores, so the nested path is not checked against itself."""
    values, labels, groups, clusters = _nested()
    got = _call((values, labels, groups, clusters))

    diffs, weights, cluster_of_group = [], [], []
    for g in np.unique(groups):
        m = groups == g
        high = m & (labels == "EXPLOIT")
        low = m & (labels == "LEGITIMATE")
        n_high, n_low = int(high.sum()), int(low.sum())
        if n_high == 0 or n_low == 0:
            continue
        diffs.append(values[high].mean() - values[low].mean())
        weights.append(n_high * n_low / (n_high + n_low))
        cluster_of_group.append(np.unique(clusters[m]).item())
    diffs = np.asarray(diffs)
    weights = np.asarray(weights)
    cluster_of_group = np.asarray(cluster_of_group)

    point = float(np.sum(weights * diffs) / np.sum(weights))
    labels_c = np.unique(cluster_of_group)
    scores = np.asarray(
        [
            float(np.sum(weights[cluster_of_group == c] * (diffs[cluster_of_group == c] - point)))
            for c in labels_c
        ]
    )
    n_c = labels_c.size
    expected = float(np.sqrt(n_c / (n_c - 1) * np.sum(scores**2)) / np.sum(weights))

    assert got.value == pytest.approx(point, rel=1e-12)
    assert got.standard_error == pytest.approx(expected, rel=1e-12)


def test_a_nested_sample_with_one_cluster_per_group_is_the_finest_clustering() -> None:
    """Cluster at the group level and every group is its own cluster; nothing is crossed."""
    values, labels, groups, _ = _nested()
    got = _call((values, labels, groups, groups))
    assert got.cluster_level_n == N_GROUPS
    assert got.n_groups_informative == N_GROUPS
