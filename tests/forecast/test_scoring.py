from __future__ import annotations

from decimal import Decimal

import pytest


def test_t1_brier_and_fraction_wis_have_hand_computed_values() -> None:
    from reward_lens.forecast.scoring import score_fraction, score_t1

    assert score_t1(0.7, True) == Decimal("0.09")
    # Five equally weighted quantiles of a point mass at 0.5: WIS is 0.1 at y=0.6.
    assert score_fraction((0.1, 0.25, 0.5, 0.75, 0.9), (0.5,) * 5, 0.6) == Decimal("0.1")


def test_fraction_scoring_refuses_brier_and_unregistered_quantiles() -> None:
    from reward_lens.forecast.scoring import score_fraction

    with pytest.raises(ValueError, match="CRPS or WIS"):
        score_fraction((0.5,), (0.4,), 0.6)
