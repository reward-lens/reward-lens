"""Tests for reward_lens.stats.roc — ROC/PR, calibration, operating points."""

import math

import numpy as np

from reward_lens.stats.roc import calibration_curve, operating_point, roc_pr


class TestROC:
    def test_perfect_classifier_auc_one(self):
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])
        labels = np.array([0, 0, 0, 1, 1, 1])
        res = roc_pr(scores, labels)
        assert abs(res.auc - 1.0) < 1e-12
        assert abs(res.average_precision - 1.0) < 1e-9

    def test_perfectly_wrong_auc_zero(self):
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0, 1, 1, 1])
        res = roc_pr(scores, labels)
        assert abs(res.auc - 0.0) < 1e-12

    def test_random_auc_near_half(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=4000)
        labels = rng.integers(0, 2, size=4000)
        res = roc_pr(scores, labels)
        assert abs(res.auc - 0.5) < 0.05

    def test_one_class_returns_nan_auc(self):
        res = roc_pr(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]))
        assert math.isnan(res.auc)
        assert math.isnan(res.average_precision)

    def test_all_ties_give_half(self):
        # every pair is a tie, so each contributes exactly 0.5 to the AUC
        scores = np.zeros(10)
        labels = np.array([0, 1] * 5)
        res = roc_pr(scores, labels)
        assert abs(res.auc - 0.5) < 1e-12

    def test_nan_scores_dropped(self):
        scores = np.array([0.1, np.nan, 0.3, 0.9, 0.8, 0.7])
        labels = np.array([0, 0, 0, 1, 1, 1])
        res = roc_pr(scores, labels)
        # dropping the NaN row leaves a still-perfectly-separable problem
        assert abs(res.auc - 1.0) < 1e-12


class TestOperatingPoint:
    def test_respects_fpr_cap(self):
        rng = np.random.default_rng(1)
        scores = np.concatenate([rng.normal(0, 1, 500), rng.normal(3, 1, 500)])
        labels = np.array([0] * 500 + [1] * 500)
        res = roc_pr(scores, labels)
        op = operating_point(res, target_fpr=0.05)
        assert op["fpr"] <= 0.05 + 1e-9
        assert 0.0 <= op["tpr"] <= 1.0

    def test_perfect_classifier_full_tpr_at_cap(self):
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])
        labels = np.array([0, 0, 0, 1, 1, 1])
        res = roc_pr(scores, labels)
        op = operating_point(res, target_fpr=0.05)
        assert op["fpr"] <= 0.05 + 1e-9
        assert abs(op["tpr"] - 1.0) < 1e-12

    def test_degenerate_roc_returns_nan(self):
        res = roc_pr(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]))
        op = operating_point(res, target_fpr=0.05)
        assert math.isnan(op["tpr"])


class TestCalibration:
    def test_ece_in_unit_interval(self):
        rng = np.random.default_rng(2)
        probs = rng.uniform(size=500)
        labels = (rng.uniform(size=500) < probs).astype(int)
        cal = calibration_curve(probs, labels, n_bins=10)
        assert 0.0 <= cal["ece"] <= 1.0
        assert cal["bin_centers"].shape == (10,)
        assert cal["bin_counts"].sum() == 500

    def test_well_calibrated_low_ece(self):
        rng = np.random.default_rng(3)
        probs = rng.uniform(size=5000)
        labels = (rng.uniform(size=5000) < probs).astype(int)
        cal = calibration_curve(probs, labels, n_bins=10)
        assert cal["ece"] < 0.1

    def test_miscalibrated_high_ece(self):
        # always predict 0.99, always wrong -> ECE near 0.99
        probs = np.full(200, 0.99)
        labels = np.zeros(200, dtype=int)
        cal = calibration_curve(probs, labels, n_bins=10)
        assert cal["ece"] > 0.9
