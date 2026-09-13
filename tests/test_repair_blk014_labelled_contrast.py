"""BLK-014 — no single call decomposed an advantage by a behavioural label.

`selection_differential` returns `Cov_group(A, f)`. For a 0/1 label that is the within-group
covariance, which carries a factor of `p(1 - p)`: it is the fixed-effects contrast **multiplied by
the label's within-group variance**, with no per-population means, no counts and no interval. So
Part 6.1's `Delta_gap` was reachable only by dividing one number by another in the experiment
directory, which the gap ledger's own rule says does not close a gap.

Part 6.1's estimand, verbatim from `chain/parts/part_06.md:23`:

    Delta_gap = E[A^vuln - A^corr | EXPLOIT] - E[A^vuln - A^corr | LEGITIMATE]

    estimated as a regression of `A^vuln - A^corr` on the label with prompt-group fixed effects,
    clustered at the group level.

and Part 6.3 at `:202` makes it three-way: "the exploit-versus-failure contrast reported alongside
the exploit-versus-legitimate one, so a reader can see which of the two any claim leans on". The
three labels are `EXPLOIT`, `LEGITIMATE` and `FAIL` (`chain/parts/part_03.md:107-113`).

**The bootstrap settings are not chosen here.** BLK-065 records that eleven registered rows resolve
on a bootstrap interval and no document states the replicate count, the interval type or the
resampling level, and that the resampling level is genuinely contradictory across Parts 5.1, 6.1 and
8.4. So `cluster_ids`, `n_resamples` and `ci_level` are all required arguments with no defaults.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.measure.ledger.price import (
    LabelledContrast,
    labelled_contrast,
)

EXPLOIT, LEGIT, FAIL = "EXPLOIT", "LEGITIMATE", "FAIL"


def _balanced_fixture(
    *, n_groups: int = 40, per_label: int = 3, means: dict[str, float], sd: float, seed: int
):
    """Every group holds the same count of every label, so the FE contrast is the raw difference.

    Balanced on purpose: it is the case where the hand-computed difference of per-population means
    and the fixed-effects estimator provably coincide, which is what makes the closure proof's
    "equals the hand-computed difference" checkable rather than approximate.
    """
    rng = np.random.default_rng(seed)
    values, labels, groups = [], [], []
    for g in range(n_groups):
        offset = rng.normal(0.0, 2.0)  # a real group effect, which the FE estimator must remove
        for label, mu in means.items():
            for _ in range(per_label):
                values.append(mu + offset + rng.normal(0.0, sd))
                labels.append(label)
                groups.append(g)
    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(labels, dtype=object),
        np.asarray(groups, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# The closure proof, first half: the contrast equals the hand-computed difference
# ---------------------------------------------------------------------------


def test_the_contrast_equals_the_hand_computed_difference_of_population_means() -> None:
    """BLK-014's closure proof, first half, on a fixture with known per-population means."""
    values, labels, groups = _balanced_fixture(
        means={EXPLOIT: 1.0, LEGIT: 0.25, FAIL: -0.5}, sd=0.4, seed=14
    )
    out = labelled_contrast(
        values,
        labels,
        groups,
        cluster_ids=groups,
        contrasts=((EXPLOIT, LEGIT), (EXPLOIT, FAIL)),
        n_resamples=200,
        ci_level=0.95,
        seed=0,
    )
    by_label = {p.label: p for p in out.populations}
    hand_el = by_label[EXPLOIT].mean - by_label[LEGIT].mean
    hand_ef = by_label[EXPLOIT].mean - by_label[FAIL].mean
    contrasts = {(c.high, c.low): c for c in out.contrasts}
    assert contrasts[(EXPLOIT, LEGIT)].value == pytest.approx(hand_el, abs=1e-12)
    assert contrasts[(EXPLOIT, FAIL)].value == pytest.approx(hand_ef, abs=1e-12)


def test_the_per_population_means_are_the_ones_that_were_planted() -> None:
    """G9 asks for per-population means. They have to be the means, not something proportional.

    Recovered up to one common shift, which is all a pooled mean can be in the presence of a group
    effect: the fixture plants a per-group offset with a standard deviation of 2 over forty groups,
    so the realised mean offset is of order 0.3 and lands on every population equally. The shift
    cancels out of every contrast, which is the point of the fixed effects, and it does not cancel
    out of a mean, which is why the means are reported separately rather than as a contrast each.
    """
    values, labels, groups = _balanced_fixture(
        means={EXPLOIT: 1.0, LEGIT: 0.25, FAIL: -0.5}, sd=0.2, seed=15
    )
    out = labelled_contrast(
        values,
        labels,
        groups,
        cluster_ids=groups,
        contrasts=((EXPLOIT, LEGIT),),
        n_resamples=50,
        ci_level=0.95,
        seed=0,
    )
    by_label = {p.label: p for p in out.populations}
    assert set(by_label) == {EXPLOIT, LEGIT, FAIL}
    planted = {EXPLOIT: 1.0, LEGIT: 0.25, FAIL: -0.5}
    shifts = [by_label[k].mean - planted[k] for k in planted]
    assert max(shifts) - min(shifts) < 0.05, f"the shift is not common to every label: {shifts}"
    assert abs(np.mean(shifts)) < 1.0, "the shift is far larger than the planted group effect"


def test_the_counts_are_reported_per_population() -> None:
    """G9 asks for counts too, and a mean with no n behind it is not a measurement."""
    values, labels, groups = _balanced_fixture(
        means={EXPLOIT: 1.0, LEGIT: 0.25, FAIL: -0.5}, sd=0.3, seed=16, n_groups=40, per_label=3
    )
    out = labelled_contrast(
        values,
        labels,
        groups,
        cluster_ids=groups,
        contrasts=((EXPLOIT, LEGIT),),
        n_resamples=50,
        ci_level=0.95,
        seed=0,
    )
    for population in out.populations:
        assert population.n == 120
        assert population.n_groups == 40
        assert np.isfinite(population.standard_error)


def test_the_fixed_effects_contrast_removes_the_group_effect() -> None:
    """A group effect ten times the contrast must not move the answer.

    The fixture plants a per-group offset with a standard deviation of 20 against a contrast of
    0.75. An estimator that pooled across groups would still be unbiased here but would carry an
    enormous standard error; one that failed to remove the offset would be visibly wrong.
    """
    rng = np.random.default_rng(17)
    values, labels, groups = [], [], []
    for g in range(60):
        offset = rng.normal(0.0, 20.0)
        for label, mu in ((EXPLOIT, 1.0), (LEGIT, 0.25)):
            for _ in range(4):
                values.append(mu + offset + rng.normal(0.0, 0.3))
                labels.append(label)
                groups.append(g)
    out = labelled_contrast(
        np.asarray(values),
        np.asarray(labels, dtype=object),
        np.asarray(groups),
        cluster_ids=np.asarray(groups),
        contrasts=((EXPLOIT, LEGIT),),
        n_resamples=200,
        ci_level=0.95,
        seed=0,
    )
    c = out.contrasts[0]
    assert c.value == pytest.approx(0.75, abs=0.05)
    assert c.standard_error < 0.1, "the group effect was not removed"


def test_the_within_group_and_pooled_contrasts_differ_when_the_groups_are_unbalanced() -> None:
    """The reason the fixed-effects estimator is the one Part 6.1 registers.

    Groups whose label mix correlates with their level make the pooled difference of means wrong and
    leave the within-group one right. Both are on the result so a reader can see the gap.
    """
    rng = np.random.default_rng(18)
    values, labels, groups = [], [], []
    for g in range(60):
        # Groups with a high offset are mostly exploits; groups with a low one are mostly legit.
        high = g % 2 == 0
        offset = 5.0 if high else -5.0
        counts = {EXPLOIT: 6, LEGIT: 2} if high else {EXPLOIT: 2, LEGIT: 6}
        for label, mu in ((EXPLOIT, 1.0), (LEGIT, 0.25)):
            for _ in range(counts[label]):
                values.append(mu + offset + rng.normal(0.0, 0.3))
                labels.append(label)
                groups.append(g)
    out = labelled_contrast(
        np.asarray(values),
        np.asarray(labels, dtype=object),
        np.asarray(groups),
        cluster_ids=np.asarray(groups),
        contrasts=((EXPLOIT, LEGIT),),
        n_resamples=200,
        ci_level=0.95,
        seed=0,
    )
    c = out.contrasts[0]
    assert c.value == pytest.approx(0.75, abs=0.06), "the within-group contrast should be right"
    assert c.pooled_value > 3.0, "the pooled difference should be badly confounded here"
    assert abs(c.pooled_value - c.value) > 3.0


# ---------------------------------------------------------------------------
# The closure proof, second half: the interval covers at its nominal rate
# ---------------------------------------------------------------------------


def test_the_interval_covers_at_its_nominal_rate_over_one_thousand_draws() -> None:
    """BLK-014's closure proof, second half.

    One thousand independent fixtures drawn from a known truth, each contrast computed with its
    clustered interval, and the fraction of intervals containing the truth compared against the
    nominal 95%. The band is a binomial one at n = 1000: a coverage of exactly 0.95 has a standard
    error of 0.0069, so a three-sigma band is roughly [0.929, 0.971].

    The analytic clustered interval is used here rather than the bootstrap one because a thousand
    bootstraps of a thousand fixtures is minutes of test time for the same statement; the test below
    checks that the bootstrap interval agrees with this one.
    """
    truth = 0.75
    covered = 0
    draws = 1000
    for draw in range(draws):
        values, labels, groups = _balanced_fixture(
            means={EXPLOIT: 1.0, LEGIT: 0.25}, sd=1.0, seed=10_000 + draw, n_groups=30, per_label=3
        )
        out = labelled_contrast(
            values,
            labels,
            groups,
            cluster_ids=groups,
            contrasts=((EXPLOIT, LEGIT),),
            n_resamples=0,
            ci_level=0.95,
            seed=draw,
        )
        c = out.contrasts[0]
        if c.ci_low <= truth <= c.ci_high:
            covered += 1
    coverage = covered / draws
    assert 0.925 <= coverage <= 0.975, f"coverage {coverage:.3f} against a nominal 0.95"


def test_the_bootstrap_interval_agrees_with_the_analytic_one() -> None:
    """The two routes to the same interval, so the coverage result above transfers to the bootstrap."""
    values, labels, groups = _balanced_fixture(
        means={EXPLOIT: 1.0, LEGIT: 0.25}, sd=1.0, seed=21, n_groups=60, per_label=4
    )
    common = dict(cluster_ids=groups, contrasts=((EXPLOIT, LEGIT),), ci_level=0.95, seed=3)
    analytic = labelled_contrast(values, labels, groups, n_resamples=0, **common).contrasts[0]
    booted = labelled_contrast(values, labels, groups, n_resamples=4000, **common).contrasts[0]
    assert booted.value == pytest.approx(analytic.value, abs=1e-12)
    half_a = (analytic.ci_high - analytic.ci_low) / 2
    half_b = (booted.ci_high - booted.ci_low) / 2
    assert half_b == pytest.approx(half_a, rel=0.25), f"{half_b:.4f} against {half_a:.4f}"
    assert booted.ci_method == "cluster-percentile"
    assert analytic.ci_method == "clustered-t"
    assert booted.n_resamples == 4000


# ---------------------------------------------------------------------------
# BLK-065: nothing about the bootstrap is chosen here
# ---------------------------------------------------------------------------


def test_the_bootstrap_settings_are_required_arguments() -> None:
    values, labels, groups = _balanced_fixture(means={EXPLOIT: 1.0, LEGIT: 0.25}, sd=0.4, seed=22)
    with pytest.raises(TypeError):
        labelled_contrast(  # type: ignore[call-arg]
            values, labels, groups, cluster_ids=groups, contrasts=((EXPLOIT, LEGIT),)
        )
    with pytest.raises(TypeError):
        labelled_contrast(  # type: ignore[call-arg]
            values, labels, groups, contrasts=((EXPLOIT, LEGIT),), n_resamples=10, ci_level=0.95
        )


def test_the_resampling_level_is_separate_from_the_grouping_level() -> None:
    """BLK-065 records Parts 5.1, 6.1 and 8.4 naming three different resampling units.

    The fixed-effects grouping and the resampling cluster are therefore two arguments, and passing a
    coarser cluster (a run holding several prompt groups) widens the interval, which is the whole
    substance of the contradiction BLK-065 records.
    """
    values, labels, groups = _balanced_fixture(
        means={EXPLOIT: 1.0, LEGIT: 0.25}, sd=1.0, seed=23, n_groups=60, per_label=3
    )
    runs = groups // 12  # five runs of twelve prompt groups each
    common = dict(contrasts=((EXPLOIT, LEGIT),), n_resamples=0, ci_level=0.95, seed=0)
    by_group = labelled_contrast(values, labels, groups, cluster_ids=groups, **common).contrasts[0]
    by_run = labelled_contrast(values, labels, groups, cluster_ids=runs, **common).contrasts[0]
    assert by_group.value == pytest.approx(by_run.value, abs=1e-12), (
        "the point estimate is the same"
    )
    assert by_group.cluster_level_n == 60
    assert by_run.cluster_level_n == 5
    assert by_group.n_groups_informative == by_run.n_groups_informative == 60


def test_the_interval_covers_at_five_clusters_too_which_it_does_not_under_a_normal() -> None:
    """The half of BLK-065 that is about the resampling level, and the reason the interval uses a t.

    A cluster-robust standard error is an average of `C` squared scores, so at small `C` it is
    downward-biased and a normal critical value gives an interval that is too narrow. Five is not a
    corner case here: it is the number of runs this design has, and Parts 5.1 and 8.4 both name the
    run as the unit of replication.

    Measured on this estimator over 1,000 draws, with the same fixture read at both levels:

        normal critical value :  60 clusters 0.955   5 clusters 0.897
        t on C - 1 df         :  60 clusters 0.955   5 clusters 0.950

    So under a normal the five-cluster interval under-covers by five points, and the direction is
    the flattering one. The estimator uses the t and this test holds that line.
    """
    truth = 0.75
    covered_groups = covered_runs = 0
    draws = 1000
    for draw in range(draws):
        values, labels, groups = _balanced_fixture(
            means={EXPLOIT: 1.0, LEGIT: 0.25}, sd=1.0, seed=50_000 + draw, n_groups=60, per_label=3
        )
        runs = groups // 12  # five runs of twelve prompt groups each
        common = dict(contrasts=((EXPLOIT, LEGIT),), n_resamples=0, ci_level=0.95, seed=0)
        g = labelled_contrast(values, labels, groups, cluster_ids=groups, **common).contrasts[0]
        r = labelled_contrast(values, labels, groups, cluster_ids=runs, **common).contrasts[0]
        covered_groups += g.ci_low <= truth <= g.ci_high
        covered_runs += r.ci_low <= truth <= r.ci_high
    at_groups, at_runs = covered_groups / draws, covered_runs / draws
    assert 0.925 <= at_groups <= 0.975, f"coverage at 60 clusters {at_groups:.3f}"
    assert 0.925 <= at_runs <= 0.975, (
        f"coverage at 5 clusters {at_runs:.3f}; a normal critical value gives 0.897 here"
    )


def test_the_critical_value_is_a_t_and_widens_as_the_clusters_run_out() -> None:
    """The mechanism behind the test above, isolated so a regression to a normal is visible."""
    from scipy.stats import norm
    from scipy.stats import t as student_t

    values, labels, groups = _balanced_fixture(
        means={EXPLOIT: 1.0, LEGIT: 0.25}, sd=1.0, seed=31, n_groups=60, per_label=3
    )
    common = dict(contrasts=((EXPLOIT, LEGIT),), n_resamples=0, ci_level=0.95, seed=0)
    by_group = labelled_contrast(values, labels, groups, cluster_ids=groups, **common).contrasts[0]
    by_run = labelled_contrast(
        values, labels, groups, cluster_ids=groups // 12, **common
    ).contrasts[0]

    for c, clusters in ((by_group, 60), (by_run, 5)):
        assert c.ci_method == "clustered-t"
        half = (c.ci_high - c.ci_low) / 2
        assert half / c.standard_error == pytest.approx(
            float(student_t.ppf(0.975, df=clusters - 1)), abs=1e-9
        )
    # At five clusters the t is far wider than the normal, which is the whole correction.
    assert float(student_t.ppf(0.975, df=4)) / float(norm.ppf(0.975)) > 1.4


def test_the_interval_settings_travel_on_the_result() -> None:
    """A row resolved on a bootstrap whose count and level are not recorded cannot be reproduced."""
    values, labels, groups = _balanced_fixture(means={EXPLOIT: 1.0, LEGIT: 0.25}, sd=0.4, seed=24)
    c = labelled_contrast(
        values,
        labels,
        groups,
        cluster_ids=groups,
        contrasts=((EXPLOIT, LEGIT),),
        n_resamples=500,
        ci_level=0.90,
        seed=7,
    ).contrasts[0]
    assert c.ci_level == 0.90
    assert c.n_resamples == 500
    assert c.ci_method == "cluster-percentile"


# ---------------------------------------------------------------------------
# Refusals and degenerate inputs
# ---------------------------------------------------------------------------


def test_a_contrast_naming_an_absent_label_raises() -> None:
    values, labels, groups = _balanced_fixture(means={EXPLOIT: 1.0, LEGIT: 0.25}, sd=0.4, seed=25)
    with pytest.raises(ValueError, match="ABSENT"):
        labelled_contrast(
            values,
            labels,
            groups,
            cluster_ids=groups,
            contrasts=((EXPLOIT, "ABSENT"),),
            n_resamples=0,
            ci_level=0.95,
            seed=0,
        )


def test_a_contrast_with_no_group_carrying_both_labels_raises() -> None:
    """Part 6.1's estimand is over "prompt groups carrying variation in the exploit label".

    With no such group the within-group contrast is not estimable, and returning a pooled difference
    instead would answer a different question under the same name.
    """
    rng = np.random.default_rng(26)
    values, labels, groups = [], [], []
    for g in range(20):
        label = EXPLOIT if g % 2 == 0 else LEGIT
        for _ in range(4):
            values.append(rng.normal(1.0 if label == EXPLOIT else 0.25, 0.3))
            labels.append(label)
            groups.append(g)
    with pytest.raises(ValueError, match="no group"):
        labelled_contrast(
            np.asarray(values),
            np.asarray(labels, dtype=object),
            np.asarray(groups),
            cluster_ids=np.asarray(groups),
            contrasts=((EXPLOIT, LEGIT),),
            n_resamples=0,
            ci_level=0.95,
            seed=0,
        )


def test_the_degenerate_group_count_is_reported_rather_than_dropped_silently() -> None:
    """Groups carrying only one of the two labels contribute nothing and are counted.

    `selection_differential` already reports `n_degenerate` for the same reason and this keeps the
    convention: a contrast estimated on eight of forty groups is a different measurement from one
    estimated on all forty, and the two look identical without the count.
    """
    rng = np.random.default_rng(27)
    values, labels, groups = [], [], []
    for g in range(40):
        present = (EXPLOIT, LEGIT) if g < 8 else (EXPLOIT,)
        for label in present:
            for _ in range(3):
                values.append(rng.normal(1.0 if label == EXPLOIT else 0.25, 0.3))
                labels.append(label)
                groups.append(g)
    c = labelled_contrast(
        np.asarray(values),
        np.asarray(labels, dtype=object),
        np.asarray(groups),
        cluster_ids=np.asarray(groups),
        contrasts=((EXPLOIT, LEGIT),),
        n_resamples=0,
        ci_level=0.95,
        seed=0,
    ).contrasts[0]
    assert c.n_groups_informative == 8
    assert c.n_groups_degenerate == 32


def test_non_finite_values_are_dropped_and_counted() -> None:
    values, labels, groups = _balanced_fixture(means={EXPLOIT: 1.0, LEGIT: 0.25}, sd=0.4, seed=28)
    values = values.copy()
    values[:5] = np.nan
    out = labelled_contrast(
        values,
        labels,
        groups,
        cluster_ids=groups,
        contrasts=((EXPLOIT, LEGIT),),
        n_resamples=0,
        ci_level=0.95,
        seed=0,
    )
    assert out.n_dropped == 5
    assert np.isfinite(out.contrasts[0].value)


def test_the_result_is_a_frozen_record() -> None:
    values, labels, groups = _balanced_fixture(means={EXPLOIT: 1.0, LEGIT: 0.25}, sd=0.4, seed=29)
    out = labelled_contrast(
        values,
        labels,
        groups,
        cluster_ids=groups,
        contrasts=((EXPLOIT, LEGIT),),
        n_resamples=0,
        ci_level=0.95,
        seed=0,
    )
    assert isinstance(out, LabelledContrast)
    with pytest.raises(Exception):
        out.n_dropped = 0  # type: ignore[misc]


def test_the_shipped_differential_is_left_alone() -> None:
    """`selection_differential` is untouched; it answers a different question and still answers it."""
    from reward_lens.measure.ledger.price import selection_differential

    rng = np.random.default_rng(30)
    features = rng.standard_normal((120, 3))
    advantages = features[:, 0] * 0.5 + rng.standard_normal(120) * 0.2
    group_ids = np.repeat(np.arange(30), 4)
    d = selection_differential(features, advantages, group_ids, ["a", "b", "c"])
    assert d.n_groups == 30
    assert np.isfinite(d.value).all()
