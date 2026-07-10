"""Tests for reward_lens.stats.effects.bca_bootstrap — the BCa interval."""

import math

import numpy as np

from reward_lens.stats.effects import bca_bootstrap, bootstrap_ci


class TestBCaBootstrap:
    def test_finite_bounds_bracket_point_skewed(self):
        rng = np.random.default_rng(0)
        # exponential is strongly right-skewed, where the percentile method is
        # biased and BCa earns its keep
        values = rng.exponential(scale=2.0, size=200)
        res = bca_bootstrap(values, seed=1, n_resamples=4000)
        assert res.method == "bca"
        assert math.isfinite(res.ci_low) and math.isfinite(res.ci_high)
        assert res.ci_low < res.point < res.ci_high

    def test_matches_percentile_on_symmetric(self):
        rng = np.random.default_rng(2)
        values = rng.normal(loc=5.0, scale=1.0, size=300)
        bca = bca_bootstrap(values, seed=3, n_resamples=4000)
        pct = bootstrap_ci(values, seed=3, n_resamples=4000)
        # on a symmetric sample the bias correction and acceleration are ~0, so
        # the two intervals should nearly coincide
        assert abs(bca.ci_low - pct.ci_low) < 0.1
        assert abs(bca.ci_high - pct.ci_high) < 0.1

    def test_small_n_returns_nan(self):
        res = bca_bootstrap([42.0], seed=0)
        assert math.isnan(res.ci_low)
        assert math.isnan(res.ci_high)
        assert res.method == "bca"

    def test_constant_input_is_degenerate_not_crash(self):
        # a constant sample has a zero jackknife spread; BCa must stay finite
        res = bca_bootstrap([3.0, 3.0, 3.0, 3.0], seed=0, n_resamples=1000)
        assert res.point == 3.0
        assert res.ci_low == 3.0 and res.ci_high == 3.0
