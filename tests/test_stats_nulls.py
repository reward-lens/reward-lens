"""Tests for reward_lens.stats.nulls — random-direction, shuffle, and RUM nulls."""

import numpy as np

from reward_lens.stats.nulls import (
    random_direction_cosines,
    random_direction_null,
    rum_identifiability_null,
    shuffle_null,
)


class TestRandomDirection:
    def test_cosines_concentrate_near_zero_high_d(self):
        cos = random_direction_cosines(d=1000, n=5000, seed=0)
        assert cos.shape == (5000,)
        assert abs(np.mean(cos)) < 0.05  # symmetric about 0
        assert np.mean(np.abs(cos)) < 0.05  # concentrated near 0 in high d
        assert np.max(np.abs(cos)) < 0.5

    def test_low_d_more_spread_than_high_d(self):
        lo = random_direction_cosines(d=3, n=5000, seed=1)
        hi = random_direction_cosines(d=300, n=5000, seed=1)
        assert np.std(lo) > np.std(hi)

    def test_high_observed_cosine_small_p(self):
        res = random_direction_null(observed_cos=0.9, d=512, n=10000, seed=0)
        assert res["p_value"] < 0.01
        assert res["null_mean"] < 0.1
        assert 0.0 <= res["null_p95"] <= 1.0

    def test_zero_cosine_large_p(self):
        res = random_direction_null(observed_cos=0.0, d=512, n=5000, seed=0)
        assert res["p_value"] > 0.5


class TestShuffleNull:
    def test_group_mean_difference(self):
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])
        labels = np.array([0] * 50 + [1] * 50)

        def mean_gap(v, lab):
            return abs(v[lab == 1].mean() - v[lab == 0].mean())

        res = shuffle_null(values, labels, mean_gap, n=2000, seed=1)
        assert res["p_value"] < 0.01
        assert res["observed"] > res["null_mean"]

    def test_no_grouping_signal(self):
        rng = np.random.default_rng(2)
        values = rng.normal(size=100)
        labels = np.array([0] * 50 + [1] * 50)

        def mean_gap(v, lab):
            return abs(v[lab == 1].mean() - v[lab == 0].mean())

        res = shuffle_null(values, labels, mean_gap, n=2000, seed=3)
        assert res["p_value"] > 0.1


class TestRUMNull:
    def test_distribution_in_unit_range(self):
        null = rum_identifiability_null(d=64, k=4, n=300, seed=0)
        assert null.shape == (300,)
        assert np.all(null >= 0.0) and np.all(null <= 1.0)
        # unrelated k-subspaces in high d overlap only a little (~ k/d)
        assert np.mean(null) < 0.5

    def test_alignment_grows_with_k_over_d(self):
        low = rum_identifiability_null(d=128, k=2, n=200, seed=0)
        high = rum_identifiability_null(d=16, k=8, n=200, seed=0)
        assert np.mean(high) > np.mean(low)
