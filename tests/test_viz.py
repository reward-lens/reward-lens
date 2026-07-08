"""Tests for visualization functions (smoke tests — check they don't crash)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")


class TestSetupStyle:
    def test_setup_does_not_crash(self):
        from reward_lens.viz import setup_style

        setup_style()


class TestCircuitOverlapPlot:
    def test_basic_plot(self, tmp_path):
        from reward_lens.viz import circuit_overlap_plot

        overlap = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
        labels = ["helpfulness", "safety", "verbosity"]

        path = str(tmp_path / "overlap.png")
        circuit_overlap_plot(overlap, labels, save_path=path)

        import os

        assert os.path.exists(path)


class TestDashboard:
    def test_dashboard_lens_only(self, tmp_path):
        """Dashboard should work even with patching=None."""
        from reward_lens.lens import RewardLensResult
        from reward_lens.viz import reward_lens_dashboard

        lens_result = RewardLensResult(
            layers=np.array([-1, 0, 1, 2, 3]),
            reward_lens_preferred=np.array([0.0, 0.2, 0.5, 0.8, 1.0]),
            reward_lens_dispreferred=np.array([0.0, 0.1, 0.3, 0.4, 0.5]),
            reward_preferred=1.0,
            reward_dispreferred=0.5,
        )

        path = str(tmp_path / "dashboard.png")
        reward_lens_dashboard(
            lens_result=lens_result,
            save_path=path,
        )

        import os

        assert os.path.exists(path)
