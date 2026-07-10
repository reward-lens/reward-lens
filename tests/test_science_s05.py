"""S5 capacity theory of bias: the coherence/Welch law, proven on planted geometry, run as a study.

This exercises the cheap-and-deep headline of the capacity science. Every object is exact by
construction: the criterion coherences are planted, the effective dimension is a participation ratio
of a known Gram, and the reward is built from named channels plus a known interference term. The
tests pin the four mechanisms where the answer is known (the simplex tight frame meets the Welch
floor with equality, random over-packed frames sit above it, the inline steer makes contamination
equal to coherence to machine precision, the dark reward rises with K/d_eff, and a best-of-n policy
mines the interference channel), then run the study end to end and confirm it emits REGISTERED
Evidence, confirms its registered predictions, and folds T12 on the scoreboard.

The real-model arms (the ArmoRM nineteen-objective matrix, the population Welch-curve fit, the
SteeringIntervention residual-stream contamination, and the interference-hacking best-of-n on real
models) are population/GPU-gated: they are built and proven on organisms here, marked, and skipped,
so no real-model number is invented.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.measure.indices.coherence import (
    coherence_matrix,
    effective_dimension,
    max_offdiagonal_coherence,
    welch_bound,
)
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s05_capacity.analysis import (
    _ANCHOR_CORRELATION,
    _CONTAM_STEER,
    _anchor_directions,
    _contamination_arm,
    _dark_reward_sweep,
    _inline_contamination,
    _interference_hacking,
    _production_contamination_status,
    _random_frame,
    _simplex_etf,
    _welch_floor_rows,
    build_spec,
)

# ---------------------------------------------------------------------------
# The Welch floor (Appendix A9), proven on planted frames
# ---------------------------------------------------------------------------


def test_planted_rubric_coherence_is_exact():
    """The foundry's planted rubric has an exact pairwise cosine, and K <= d makes the floor vacuous."""
    dirs, corr = _anchor_directions()
    mu = coherence_matrix(dirs)
    k = dirs.shape[0]
    off = mu[~np.eye(k, dtype=bool)]
    # Every off-diagonal equals the planted correlation to machine precision.
    assert np.allclose(off, corr, atol=1e-9)
    assert abs(float(np.max(np.abs(off - corr)))) < 1e-9
    # K <= d, so the Welch bound is zero: there is room for the criteria to be non-orthogonal without
    # violating any floor. Correlated criteria at full rank do not owe a floor.
    assert welch_bound(k, dirs.shape[1]) == 0.0
    assert corr == _ANCHOR_CORRELATION


def test_simplex_etf_meets_welch_floor_with_equality():
    """A simplex tight frame is the Welch equality case: max coherence equals the bound exactly."""
    for k in (3, 4, 6, 10, 16):
        v = _simplex_etf(k)
        d = k - 1
        assert v.shape == (k, d)
        mu = coherence_matrix(v)
        max_coh = max_offdiagonal_coherence(mu)
        bound = welch_bound(k, d)
        # Equality to machine precision: the floor formula is verified against the planted geometry.
        assert abs(max_coh - bound) < 1e-9
        assert abs(max_coh - 1.0 / (k - 1)) < 1e-9
        # The frame is over-packed: K exceeds its effective dimension (d_eff = d = k - 1).
        assert k > effective_dimension(v) + 1e-9


def test_random_overpacked_frames_sit_at_or_above_the_floor():
    """Any K > d frame must clear the Welch floor; random frames sit strictly above it."""
    for k, d in ((6, 3), (10, 4), (20, 6), (30, 8), (50, 10)):
        v = _random_frame(k, d, seed=0)
        max_coh = max_offdiagonal_coherence(coherence_matrix(v))
        bound = welch_bound(k, d)
        assert bound > 0.0  # K > d, so the bound is a real floor
        assert max_coh >= bound - 1e-9
        assert k > effective_dimension(v)


def test_welch_floor_rows_min_slack_nonnegative_and_etf_exact():
    """The study's floor rows never fall below the bound, and the ETF rows meet it to machine precision."""
    rows = _welch_floor_rows()
    assert all(row.overpacked for row in rows)
    assert min(row.slack for row in rows) >= -1e-9
    etf_gap = max(abs(row.slack) for row in rows if row.label.startswith("simplex"))
    assert etf_gap < 1e-9


# ---------------------------------------------------------------------------
# Contamination equals coherence (Appendix A9), by construction
# ---------------------------------------------------------------------------


def test_inline_steer_makes_contamination_equal_coherence():
    """Steering criterion j and reading criterion k gives C_jk = c * mu_jk to machine precision."""
    v = _random_frame(8, 20, seed=1)
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((300, 20))
    mu = coherence_matrix(v)
    contam = _inline_contamination(v, latents, _CONTAM_STEER)
    off = ~np.eye(8, dtype=bool)
    # The first-order linear identity holds exactly: contamination is the coherence, scaled by the
    # steer. This is the mechanism A9 names, proven on a planted rubric where mu is exact.
    assert np.max(np.abs(contam - _CONTAM_STEER * mu)[off]) < 1e-9
    corr = float(np.corrcoef(contam[off], (_CONTAM_STEER * mu)[off])[0, 1])
    assert corr > 0.999


def test_contamination_arm_correlation_and_slope():
    """The noisy-readout contamination still tracks coherence at high correlation and slope c."""
    arm = _contamination_arm()
    assert arm["contamination_coherence_corr"] > 0.9
    assert abs(arm["contamination_slope"] - _CONTAM_STEER) < 0.05
    assert arm["exact_first_order_deviation"] < 1e-9


# ---------------------------------------------------------------------------
# Dark reward grows with K/d_eff (Appendix A10), on planted organisms
# ---------------------------------------------------------------------------


def test_dark_reward_grows_with_k_over_deff():
    """The unmediated variance fraction rises with K/d_eff and is zero for orthogonal criteria."""
    dark = _dark_reward_sweep()
    assert dark["dark_reward_kdeff_spearman"] > 0.8
    # The low anchor is near-orthonormal (K <= d), so there is no interference and no dark reward; the
    # over-packed high end leaks a large fraction of its variance into the unmediated channel.
    assert dark["dark_low"] < 0.05
    assert dark["dark_high"] > dark["dark_low"] + 0.5
    ratios = [r["k_over_deff"] for r in dark["rungs"]]
    assert max(ratios) > min(ratios)  # the sweep really varies K/d_eff


# ---------------------------------------------------------------------------
# Policies mine the interference terms (the hack invisible to per-criterion audits)
# ---------------------------------------------------------------------------


def test_interference_hacking_mines_the_dark_channel():
    """Best-of-n on the true reward puts most of its gain in the dark channel, rising with K/d_eff."""
    hacking = _interference_hacking()
    assert hacking["interference_dark_share"] > 0.5
    assert hacking["dark_share_kdeff_spearman"] > 0.8
    # The dark share grows from the least to the most over-packed rung.
    assert hacking["dark_share_high"] > hacking["dark_share_low"]


# ---------------------------------------------------------------------------
# Production wiring: the SteeringIntervention contract
# ---------------------------------------------------------------------------


def test_production_contamination_contract_compiles():
    """The SteeringIntervention(direction, site, strength) contract imports and compiles when present."""
    status = _production_contamination_status()
    if status["steer_module"] == "present":
        # A fingerprinted, site-addressed steer is constructible; the real-model run stays gated.
        assert status["fingerprint"] == status["compiled_fingerprint"]
        assert status["contract"] == "SteeringIntervention(direction, site, strength)"
    else:
        # If the module is absent the inline proof stands and the production arm is pending.
        assert "pending" in status["note"]


# ---------------------------------------------------------------------------
# The study runs, registers, and folds the scoreboard
# ---------------------------------------------------------------------------


def test_s05_runs_and_registers(tmp_path):
    """The study runs end to end, confirms its registered predictions, and emits REGISTERED Evidence."""
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # All four arms confirm and nothing dies: the Welch floor holds, the ETFs meet it exactly,
    # contamination tracks coherence, the dark reward grows, and the policy mines interference.
    assert result.outcomes["H1-welch-floor-holds"] == "confirmed", result.metrics
    assert result.outcomes["H2-etf-meets-floor-exactly"] == "confirmed", result.metrics
    assert result.outcomes["H3-contamination-scales-with-coherence"] == "confirmed", result.metrics
    assert result.outcomes["H4-dark-reward-grows-with-k-over-deff"] == "confirmed", result.metrics
    assert result.outcomes["H5-policies-mine-interference"] == "confirmed", result.metrics
    assert not result.killed, result.killed_by

    # The registered metrics match the proofs.
    assert result.metrics["welch_floor_min_slack"] >= -1e-9
    assert result.metrics["etf_equality_gap_max"] < 1e-9
    assert result.metrics["contamination_coherence_corr"] > 0.9
    assert result.metrics["dark_reward_kdeff_spearman"] > 0.8
    assert result.metrics["interference_dark_share"] > 0.5

    # The headline capacity-law Evidence is REGISTERED and traces to the four proof arms.
    law = store.find(observable="S05.CapacityLaw")
    assert law and law[0].trust is TrustLevel.REGISTERED
    parents = store.parents(law[0])
    assert {p.observable for p in parents} == {
        "S05.WelchFloor",
        "S05.Contamination",
        "S05.DarkReward",
        "S05.InterferenceHacking",
    }

    # The gated real-model arms are recorded as REGISTERED Evidence carrying no invented number.
    gated = store.find(observable="S05.GatedRealArms")
    assert gated and gated[0].value["armorm_nineteen_objective"]["gated"] is True
    assert gated[0].value["population_welch_curve"]["gated"] is True


def test_s05_updates_scoreboard(tmp_path):
    """The confirmed outcomes fold T12 (the coherence/Welch candidate law) to confirmed."""
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    assert board.rows["T12"].status == "confirmed"
    assert board.rows["T12"].adjudicating_evidence

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report
