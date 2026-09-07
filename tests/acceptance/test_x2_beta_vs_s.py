"""X2 acceptance: `beta` against `S` as a predictor of realised drift, and the `h2` half refused.

The clause, in full:

    *The campaign's stored forecasts are re-run with `beta` in place of `S` under the within-group
    operator, scored against the drift the card recorded, and `PREDICTIONS.md` P1 is resolved
    whichever way it comes out. The `h2` half is refused rather than dropped, and the refusal names
    what the archive cannot support and what would answer it.*

Most of this file reads the released artifacts under `experiments/x2_release/`, because the release
is the deliverable. Four tests run live code: the reproduction gate, the estimator import, the
prediction arithmetic, and the figure guard.

Producing the artifacts:

    python -m experiments.x2_beta_vs_s --allow-dirty

It reads the campaign store read-only. No GPU, no network, no hosted model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.x2_beta_vs_s import (
    CAMPAIGN_STORE,
    OPERATORS,
    STUDY,
    DriftBank,
    bon_drift,
    load_bank,
    refuse_h2,
    spearman,
    spearman_grid_step,
)
from experiments.x3_transfer import Bound, require_bound
from reward_lens.artifacts.claims import check_files
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.store import EvidenceStore

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "experiments" / "x2_release"

pytestmark = pytest.mark.skipif(
    not (RELEASE / "manifest.json").exists(),
    reason=(
        f"no release under {RELEASE}. Produce it with `python -m experiments.x2_beta_vs_s`; this "
        f"file checks the published artifacts, and skipping it leaves them unchecked rather than "
        f"checked."
    ),
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def store() -> EvidenceStore:
    return EvidenceStore(RELEASE / "evidence")


@pytest.fixture(scope="module")
def rows(store: EvidenceStore) -> dict:
    return {ev.observable.split("x2.", 1)[-1]: ev for ev in store}


# ---------------------------------------------------------------------------
# P1 is resolved, whichever way it came out
# ---------------------------------------------------------------------------


def test_p1_is_resolved_and_the_release_says_which_row_it_resolves(manifest: dict) -> None:
    assert manifest["resolves"] == "PREDICTIONS.md P1, frozen at c746e9f"
    assert "H-P1" in manifest["outcomes"]
    assert manifest["outcomes"]["H-P1"] in {"confirmed", "refuted", "void"}


def test_p1_is_adjudicated_against_the_registered_rule(manifest: dict) -> None:
    """The rule is: the point estimate is at least 0.10 *and* the interval excludes zero.

    Checked against the released numbers rather than trusted, so a verdict that disagreed with its
    own metrics would fail here.
    """
    metrics = manifest["metrics"]
    delta = metrics["delta_min_over_models__within_group__raw"]
    verdict = manifest["outcomes"]["H-P1"]
    assert verdict == ("confirmed" if delta >= 0.10 else "refuted")


def test_the_registered_threshold_was_above_the_arithmetic_ceiling(manifest: dict) -> None:
    """The first reason P1 fails is that no estimator could have confirmed it on this bank."""
    metrics = manifest["metrics"]
    ceiling = metrics["max_achievable_delta__within_group__raw"]
    assert ceiling < 0.10
    assert manifest["kill_outcomes"]["K-threshold-unreachable"] == "fired"
    # The ceiling is exactly the headroom the realised S ranking leaves.
    assert ceiling == pytest.approx(1.0 - metrics["rho_S_min__within_group__raw"])


def test_the_spearman_grid_is_why_the_ceiling_is_where_it_is() -> None:
    """One adjacent transposition on seven items moves Spearman by 12/(k(k^2-1))."""
    assert spearman_grid_step(7) == pytest.approx(0.0357142857, abs=1e-9)
    # And the value one step below perfect is the card's own confirmatory statistic.
    assert 1.0 - spearman_grid_step(7) == pytest.approx(0.9642857142857, abs=1e-9)


def test_both_operators_and_both_unit_conventions_are_reported(manifest: dict) -> None:
    """E17 settles within-group as primary and asks for pooled as the contrast, so both ship."""
    metrics = manifest["metrics"]
    for operator in OPERATORS:
        for units in ("raw", "standardised"):
            suffix = f"__{operator}__{units}"
            for key in ("rho_S_min", "rho_beta_min", "delta_min_over_models"):
                assert np.isfinite(metrics[f"{key}{suffix}"]), f"{key}{suffix} is missing"


def test_the_within_group_operator_is_the_primary_one(manifest: dict) -> None:
    """The registered metric names the operator, so the primary reading cannot drift."""
    spec = json.loads((RELEASE / "freeze.json").read_text(encoding="utf-8"))["spec"]
    p1 = next(h for h in spec["hypotheses"] if h["id"] == "H-P1")
    assert p1["prediction"]["metric"].endswith("__within_group__raw")
    assert p1["prediction"]["threshold"] == 0.10
    assert p1["prediction"]["comparator"] == ">="


def test_the_second_must_beat_is_checked_separately(manifest: dict) -> None:
    """P1 names two things to beat and losing to one is not the same as losing to both."""
    assert manifest["outcomes"]["H-beats-random"] in {"confirmed", "refuted", "void"}
    assert np.isfinite(manifest["metrics"]["delta_beta_minus_random__within_group__raw"])


# ---------------------------------------------------------------------------
# The h2 half is refused, not dropped
# ---------------------------------------------------------------------------


def test_the_h2_half_is_refused_with_a_reason_and_a_remedy(store: EvidenceStore) -> None:
    refusals = [ev for ev in store if ev.observable.startswith("x2.refusal.")]
    assert refusals, "the h2 half must be refused, and a refusal that is not recorded is a drop"
    h2 = next(ev for ev in refusals if "h2" in ev.observable)
    assert h2.value["reason"] == "ACCESS_INSUFFICIENT"
    assert "G" in h2.value["detail"] and "C = G + N" in h2.value["detail"]
    assert "BACKWARD" in h2.value["remedy"]
    assert len(h2.value["remedy"]) > 80


def test_the_refusal_is_a_reading_and_never_an_exception() -> None:
    """`Reading = Evidence | Refusal`. It is not raised, not None, not a zero."""
    bank = DriftBank(
        feature_names=("a", "b"),
        grouped_features_a=np.zeros((2, 2, 2)),
        grouped_rewards_a={},
        grouped_features_b=np.zeros((2, 2, 2)),
        grouped_rewards_b={},
        stored_chi={},
        stored_drift={},
        bon_n=16,
        n_prompts_total=4,
        evidence_ids={},
        reproduction={},
    )
    got = refuse_h2(bank, {})
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert got.statistics["erratum"] == 18.0


def test_the_findings_carry_the_h2_refusal_in_prose() -> None:
    text = (RELEASE / "FINDINGS-x2.md").read_text(encoding="utf-8")
    assert "## The `h2` half, refused" in text
    assert "ACCESS_INSUFFICIENT" in text
    assert "Remedy:" in text


# ---------------------------------------------------------------------------
# The bank, and the two reproductions that gate everything above
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=f"no campaign store at {CAMPAIGN_STORE}",
)
def test_the_stored_chi_reproduces_bit_exactly_on_the_cards_own_half() -> None:
    """Approximate agreement is not enough: `C` has to be measured on the same rows as `S`.

    `load_bank` raises when it does not, so reaching this assertion is most of the test.
    """
    bank = load_bank()
    assert set(bank.reproduction["chi_exact"].values()) == {True}
    assert len(bank.feature_names) == 7
    assert bank.bon_n == 16
    for model, rel in bank.reproduction["drift_max_rel"].items():
        assert rel < 1e-9, f"{model} drift reproduces only to {rel}"


def test_the_release_records_that_both_reproductions_passed(rows: dict) -> None:
    reproduction = rows["bank"].value["reproduction"]
    assert set(reproduction["chi_exact"].values()) == {True}
    assert all(v < 1e-9 for v in reproduction["drift_max_rel"].values())
    assert rows["bank"].value["n_prompts_total"] == 2000
    assert rows["bank"].value["bon_n"] == 16


def test_best_of_n_drift_is_the_order_statistic_weighting() -> None:
    """At `n = 1` the drift is zero by construction, and it grows with `n`."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 4, 3))
    rewards = features[:, :, 0] + 0.1 * rng.normal(size=(200, 4))
    at_one = bon_drift(features, rewards, 1)
    assert np.allclose(at_one, 0.0, atol=1e-12)
    at_four = bon_drift(features, rewards, 4)
    at_sixteen = bon_drift(features, rewards, 16)
    # The rewarded feature is pushed up, and harder at higher n.
    assert at_sixteen[0] > at_four[0] > 0.0


def test_the_forecast_is_fitted_on_one_half_and_scored_on_the_other(rows: dict) -> None:
    """Nothing here scores a prediction against the data that produced it."""
    value = rows["bank"].value
    assert value["n_prompts_half"] * 2 == value["n_prompts_total"]


# ---------------------------------------------------------------------------
# The solve is W5.3's, not a second copy
# ---------------------------------------------------------------------------


def test_beta_comes_from_the_shipped_estimator() -> None:
    """`measure/indices/chi.py` owns `beta = C^-1 S`; this experiment imports it.

    A second implementation in an experiment is the duplication the build brief calls out by name,
    so this asserts the import rather than the arithmetic.
    """
    import experiments.x2_beta_vs_s as x2
    from reward_lens.measure.indices import chi

    assert x2.selection_gradient is chi.selection_gradient
    assert x2.feature_covariance is chi.feature_covariance
    assert x2.differential is chi.differential
    assert x2.susceptibility is chi.susceptibility


def test_the_two_differentials_agree_up_to_a_positive_scalar() -> None:
    """The card called the population form and W5.3's is unbiased, so rankings are identical.

    This is the check the module docstring promises rather than assumes: `n/(n-1)` is a positive
    scalar, and a positive scalar cannot reorder anything.
    """
    from reward_lens.measure.indices.chi import differential, susceptibility

    rng = np.random.default_rng(3)
    features = rng.normal(size=(400, 5))
    reward = features @ rng.normal(size=5) + rng.normal(size=400)
    population = susceptibility(features, reward)
    unbiased = differential(features, reward, operator="pooled")
    ratio = unbiased / population
    assert np.allclose(ratio, ratio[0])
    assert ratio[0] > 0
    assert spearman(population, unbiased) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Freeze before measure
# ---------------------------------------------------------------------------


def test_the_freeze_predates_every_measurement(store: EvidenceStore, manifest: dict) -> None:
    frozen_at = json.loads((RELEASE / "freeze.json").read_text(encoding="utf-8"))["frozen_at"]
    assert frozen_at == manifest["frozen_at"]
    for ev in store:
        assert ev.created_at >= frozen_at, f"{ev.observable} predates the freeze it claims"
        assert ev.provenance.study == manifest["study_id"]


def test_the_spec_hash_changes_when_a_threshold_changes() -> None:
    """The property that makes a preregistration one: an edit is visible as a new id."""
    from dataclasses import replace

    from reward_lens.studies.freeze import freeze

    original = freeze(STUDY, repo_dir=str(REPO))
    hypotheses = list(STUDY.hypotheses)
    moved = replace(hypotheses[0], prediction=replace(hypotheses[0].prediction, threshold=0.05))
    edited = freeze(replace(STUDY, hypotheses=(moved, *hypotheses[1:])), repo_dir=str(REPO))
    assert original.study_id != edited.study_id
    assert original.spec_hash != edited.spec_hash


def test_a_provisional_freeze_says_so_on_the_page(manifest: dict) -> None:
    text = (RELEASE / "FINDINGS-x2.md").read_text(encoding="utf-8")
    if manifest["clean_tree"]:
        assert "This freeze is provisional" not in text
    else:
        assert "This freeze is provisional" in text
        assert manifest["git_sha"].endswith("+dirty")


# ---------------------------------------------------------------------------
# Power, figures, and the binding of every number in the prose
# ---------------------------------------------------------------------------


def test_the_power_calculation_is_at_the_realized_n(rows: dict) -> None:
    power = rows["power"].value
    assert power["n_prompts_half"] == 1000
    assert power["n_responses_half"] == 4000
    assert power["registered_threshold"] == 0.10
    assert power["q_effect_headroom"] < 1.0
    assert power["resolved"] is False
    # The arithmetic ceiling and the resampling agree that the registered effect was unreachable.
    # Not exactly zero: the ceiling is computed at the *observed* S ranking, and under resampling
    # S itself moves, so a draw in which S lands lower leaves more headroom than the point estimate
    # does. One draw in a couple of thousand doing that is the resampling agreeing, not disagreeing.
    assert power["empirical_power_at_threshold"] < 0.01


def test_a_figure_refuses_to_plot_a_bare_float() -> None:
    good = Bound(value=0.5, evidence="ev:abc", field="x")
    with pytest.raises(TypeError, match="carries no evidence id"):
        require_bound([good, 0.5], "test")
    assert require_bound([good], "test") == [good]


def test_the_published_figures_exist() -> None:
    assert (RELEASE / "figures" / "predictive.png").exists()
    assert (RELEASE / "figures" / "delta.png").exists()


def test_the_findings_section_has_no_unbound_number(store: EvidenceStore) -> None:
    report = check_files([RELEASE / "FINDINGS-x2.md"], store)
    assert report.ok, report.render()
    assert report.results, "the findings section contains no claims at all"


def test_the_honest_number_and_the_flattering_one_are_both_on_the_page(manifest: dict) -> None:
    """Publishing only the convention that looks best is the failure this checks against."""
    text = (RELEASE / "FINDINGS-x2.md").read_text(encoding="utf-8")
    metrics = manifest["metrics"]
    honest = metrics["delta_min_over_models__within_group__raw"]
    flattering = metrics["delta_min_over_models__within_group__standardised"]
    assert flattering >= honest
    for value in (honest, flattering):
        assert f"value={value:.4f}" in text, f"{value} is not claimed anywhere in the findings"
