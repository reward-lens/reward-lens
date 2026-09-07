"""X9 acceptance: `VERIF-PRM`'s below-chance AUC re-scored against a position-stratified null.

The clause, in full:

    *`VERIF-PRM`'s stored artifacts are re-scored against a position-stratified null; the re-scoring
    reproduces the published pooled AUC exactly; the stratification's own control check is produced
    rather than assumed; the direction is reported in the direction it came out, with an interval on
    the move rather than on the two endpoints separately; the verdict rule is shown to return the
    other answers on cases whose true explanation is known; the answer is shown not to depend on the
    analysis choices or on the verdict-rule correction SPEC-ERRATA E42 item 9 made after the number
    was first produced; and the published findings section contains no number the experiment's own
    evidence store cannot verify.*

Most of this file reads the released artifacts under `experiments/x9_position/` rather than
rebuilding them, because the release is the deliverable. The parts cheap enough to rebuild live are
rebuilt live.

The load-bearing test in this file is
`test_the_answer_is_bit_identical_to_what_was_measured_before_the_correction`. It imports W3.6's own
pinned constants rather than restating them, so the released numbers and the pins cannot drift apart
without a failure here.

Producing the artifacts:

    python -m experiments.x9_position

It reads one evidence store read-only, needs no network, no GPU and no hosted model, and takes about
thirty seconds.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from experiments.x9_position import (
    CAMPAIGN_STORE,
    HIGHER_IS_POSITIVE,
    OBSERVABLE,
    PINNED_BEFORE_THE_CORRECTION,
    REPO,
    ROSTER,
    SLICE,
    Ledger,
    _verdict_under_the_old_ordering,
    controls,
    load_stored,
)
from reward_lens.artifacts.claims import check_files, find_unbound_numbers
from reward_lens.core.store import EvidenceStore
from reward_lens.measure.labels import load_step_scores, rescore_against_position
from reward_lens.measure.labels.position import CHANCE

RELEASE = REPO / "experiments" / "x9_position"

pytestmark = pytest.mark.skipif(
    not (RELEASE / "manifest.json").exists(),
    reason=(
        f"no release under {RELEASE}. Produce it with `python -m experiments.x9_position`; this "
        f"file checks the published artifacts, and skipping it leaves them unchecked rather than "
        f"checked."
    ),
)

needs_campaign = pytest.mark.skipif(
    not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=f"the campaign evidence store is not at {CAMPAIGN_STORE}",
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def store() -> EvidenceStore:
    return EvidenceStore(RELEASE / "evidence", readonly=True)


@pytest.fixture(scope="module")
def rows(store: EvidenceStore) -> dict[str, Any]:
    return {store.get(i).observable: store.get(i).value for i in list(store._index)}


# ---------------------------------------------------------------------------
# Clause 1: the re-scoring reproduces what the card published
# ---------------------------------------------------------------------------


def test_the_rescoring_reproduces_the_published_pooled_auc(rows: dict[str, Any]) -> None:
    """Exactly, not approximately. Without this the rest is about a different object."""
    r = rows["x9.rescoring"]
    assert r["uniform_auc"] == r["stored_dense_localization_auc"]
    assert r["position_only_auc"] == r["stored_position_baseline_auc"]
    assert r["uniform_abs_error"] == 0.0
    assert r["position_only_abs_error"] == 0.0
    assert r["n_items"] == r["stored_n_items_with_error"] == 2221
    assert r["n_candidates"] == r["stored_n_pooled_steps"] == 19783


def test_the_registered_hypothesis_is_read_off_the_stored_row(rows: dict[str, Any]) -> None:
    """The threshold argued about is the frozen one, not one remembered from a document."""
    r = rows["x9.rescoring"]
    assert r["registered_comparator"] == ">"
    assert r["registered_threshold"] == 0.7
    assert r["registered_outcome"] == "refuted"
    assert r["card_killed"] is True


# ---------------------------------------------------------------------------
# Clause 2: the stratification controls what it claims to
# ---------------------------------------------------------------------------


def test_the_stratifications_own_control_check_is_produced(rows: dict[str, Any]) -> None:
    """Within an exact-position stratum every position comparison is a tie, so this must be 0.5.

    Carried as an output rather than only asserted, because a stratification that silently failed
    to control its covariate is the failure this instrument exists to catch and a self-check nobody
    can read is not a check.
    """
    assert rows["x9.rescoring"]["exact_position_check"] == pytest.approx(CHANCE, abs=1e-12)
    assert rows["x9.sensitivity"]["exact_position_check"] == pytest.approx(CHANCE, abs=1e-12)


# ---------------------------------------------------------------------------
# Clause 3: the direction, reported the way it came out
# ---------------------------------------------------------------------------


def test_conditioning_moved_the_reading_away_from_chance(rows: dict[str, Any]) -> None:
    """The finding, and it is the opposite of the direction the specification illustrated."""
    r = rows["x9.rescoring"]
    assert r["stratified_auc"] < r["uniform_auc"] < CHANCE
    assert r["moved_away_from_chance"] is True
    assert abs(r["stratified_auc"] - CHANCE) > abs(r["uniform_auc"] - CHANCE)
    assert r["verdict"] == "below chance, and not from position"


def test_the_interval_is_on_the_move_and_excludes_zero(rows: dict[str, Any]) -> None:
    """Both halves on the same resampled items, so the two statistics' correlation is carried.

    The verdict turns on whether this interval excludes zero rather than on the sign of the point
    estimate, which is what stops a resampling artifact from being named as a confound.
    """
    r = rows["x9.rescoring"]
    assert r["confound_size"] < 0.0
    assert r["confound_ci_high"] < 0.0, "the move is away from chance, with room to spare"
    assert r["confound_ci_excludes_zero"] is True
    assert r["ci_low"] <= r["stratified_auc"] <= r["ci_high"]
    assert not (r["ci_low"] <= CHANCE <= r["ci_high"]), "the reading is not noise"


def test_the_inversion_is_reported_and_not_claimed(rows: dict[str, Any]) -> None:
    """`1 - AUC` is arithmetic. That the model should be re-read that way is not, and is not claimed.

    The row carries the inverted values and the fact that the pooled inversion clears the registered
    threshold. The findings section states plainly that settling it needs the model re-run.
    """
    r = rows["x9.rescoring"]
    assert r["inverted_stratified_auc"] == pytest.approx(1.0 - r["stratified_auc"], abs=1e-15)
    assert r["inverted_uniform_auc"] == pytest.approx(1.0 - r["uniform_auc"], abs=1e-15)
    assert r["inverted_clears_registered_threshold"] is True
    text = (RELEASE / "FINDINGS-x9.md").read_text(encoding="utf-8")
    assert "not claimed as established" in text
    assert "needs the model run again" in text or "model re-run" in text


# ---------------------------------------------------------------------------
# Clause 4: the answer survives the correction to its own machinery
# ---------------------------------------------------------------------------


def test_the_answer_is_bit_identical_to_what_was_measured_before_the_correction(
    rows: dict[str, Any],
) -> None:
    """X9's values against W3.6's own pins, imported rather than restated.

    SPEC-ERRATA E42 item 9 reordered the two tests in L5's verdict rule after this reading was first
    taken. The reordering changes only readings whose conditioned statistic is above chance, and this
    one is 0.2190, so the numbers should not move at all. Bit-identical is the right standard: a rank
    statistic recomputed over fixed stored numbers has no floating-point freedom between runs.

    Importing the constants from `test_w3_6_labels` is the citation. If either side is edited without
    the other, this fails.
    """
    from tests.acceptance.test_w3_6_labels import (
        MEASURED_CI,
        MEASURED_CONFOUND_SIZE,
        MEASURED_EXACT_POSITION_AUC,
        MEASURED_STRATIFIED_AUC,
        STORED_DENSE_LOCALIZATION_AUC,
        STORED_N_ITEMS_WITH_ERROR,
        STORED_N_POOLED_STEPS,
        STORED_POSITION_BASELINE_AUC,
    )

    r = rows["x9.rescoring"]
    assert r["uniform_auc"] == STORED_DENSE_LOCALIZATION_AUC
    assert r["position_only_auc"] == STORED_POSITION_BASELINE_AUC
    assert r["stratified_auc"] == MEASURED_STRATIFIED_AUC
    assert r["exact_position_auc"] == MEASURED_EXACT_POSITION_AUC
    assert (r["ci_low"], r["ci_high"]) == MEASURED_CI
    assert r["confound_size"] == MEASURED_CONFOUND_SIZE
    assert r["n_items"] == STORED_N_ITEMS_WITH_ERROR
    assert r["n_candidates"] == STORED_N_POOLED_STEPS

    # And the experiment's own copy of the pins agrees with W3.6's, so the two cannot drift.
    assert PINNED_BEFORE_THE_CORRECTION["stratified_auc"] == MEASURED_STRATIFIED_AUC
    assert PINNED_BEFORE_THE_CORRECTION["uniform_auc"] == STORED_DENSE_LOCALIZATION_AUC
    assert PINNED_BEFORE_THE_CORRECTION["ci_low"] == MEASURED_CI[0]
    assert PINNED_BEFORE_THE_CORRECTION["ci_high"] == MEASURED_CI[1]


def test_the_correction_is_run_against_the_reading_rather_than_assumed_harmless(
    rows: dict[str, Any],
) -> None:
    r = rows["x9.survives_the_correction"]
    assert r["all_bit_identical"] is True
    assert r["n_bit_identical"] == r["n_fields_compared"] == 9
    assert r["moved_fields"] == []
    assert r["verdict_unchanged"] is True
    assert r["verdict_now"] == r["verdict_under_the_old_ordering"]
    assert r["reading_is_below_chance"] is True
    assert "E42 item 9" in r["erratum"]
    assert "test_w3_6_labels" in r["asserted_in"]


def test_the_superseded_ordering_does_differ_where_the_erratum_says_it_does() -> None:
    """The other half of the claim, which the reading itself cannot show.

    Saying "the correction does not move this reading" is only informative if the correction moves
    something. Above chance, with a confound interval excluding zero, the two orderings disagree:
    the old one returns "localises" and the corrected one names the confound. That is the case
    SPEC-ERRATA E42 item 9 found unreachable.
    """
    old = _verdict_under_the_old_ordering(0.8291, 0.6298, 0.5949, 0.6685, (-0.24, -0.16))
    assert old == "localises"
    from reward_lens.measure.labels.position import _verdict

    new, _ = _verdict(0.8291, 0.6298, 0.8400, 0.5949, 0.6685, (-0.24, -0.16))
    assert new == "localises, and position inflated the pooled reading"
    assert new != old

    # And below chance, which is where this experiment's reading sits, they agree.
    below_old = _verdict_under_the_old_ordering(0.2821, 0.2190, 0.2069, 0.2314, (-0.067, -0.059))
    below_new, _ = _verdict(0.2821, 0.2190, 0.3446, 0.2069, 0.2314, (-0.067, -0.059))
    assert below_old == below_new == "below chance, and not from position"


# ---------------------------------------------------------------------------
# Clause 5: the verdict rule returns the other answers when they are the true ones
# ---------------------------------------------------------------------------


def test_the_rule_names_each_explanation_when_it_is_the_true_one(rows: dict[str, Any]) -> None:
    """Four localisers built so one explanation is true by construction.

    A rule that returned "below chance, and not from position" on everything would return it on
    VERIF-PRM too and nothing would follow.
    """
    r = rows["x9.controls"]
    assert r["noise_reads_as_noise"] is True
    assert r["inverted_reads_as_inverted"] is True
    assert r["confound_is_named_when_it_is_there"] is True
    assert r["confound_is_named_above_chance"] is True
    assert r["n_distinct_verdicts"] >= 3

    # The pure position confound is pulled back to chance; the real artifact is pushed away from it.
    confound = r["by_kind"]["position_only"]
    assert abs(confound["stratified_auc"] - CHANCE) < abs(confound["uniform_auc"] - CHANCE)
    real = rows["x9.rescoring"]
    assert abs(real["stratified_auc"] - CHANCE) > abs(real["uniform_auc"] - CHANCE)


def test_the_controls_rebuild_live() -> None:
    """The synthetic half runs without the campaign store, so it can never silently stop running."""
    ledger = Ledger(Path(tempfile.mkdtemp(prefix="x9-acceptance-")), "test")
    result = controls(ledger, n_boot=100)
    assert result["noise_reads_as_noise"]
    assert result["inverted_reads_as_inverted"]
    assert result["confound_is_named_above_chance"]


# ---------------------------------------------------------------------------
# Clause 6: the answer does not depend on the analysis choices
# ---------------------------------------------------------------------------


def test_the_direction_holds_under_every_analysis_choice(rows: dict[str, Any]) -> None:
    r = rows["x9.sensitivity"]
    assert r["n_settings"] >= 5
    assert r["every_setting_below_chance"] is True
    assert r["every_setting_moved_away_from_chance"] is True
    assert r["every_setting_agrees_on_the_verdict"] is True
    assert r["spread_of_stratified_auc"] < 0.05
    assert r["exact_position_below_chance"] is True


def test_the_two_pooled_statistics_are_reported_separately(rows: dict[str, Any]) -> None:
    """A localisation study that pools raw step scores publishes a different number under one name.

    Both are carried. Substituting one for the other silently is the naming problem this reports.
    """
    r = rows["x9.sensitivity"]
    assert r["uniform_auc_standardised"] != r["uniform_auc_raw"]
    assert r["standardisation_gap"] > 0.0
    assert r["uniform_auc_standardised"] == rows["x9.rescoring"]["stored_dense_localization_auc"]


# ---------------------------------------------------------------------------
# Clause 7: published means it reaches Evidence with its comparators
# ---------------------------------------------------------------------------


def test_the_number_reaches_a_reading_with_the_baselines_a_card_prints(
    rows: dict[str, Any],
) -> None:
    r = rows["x9.reading"]
    assert r["quantity"] == "labels.position_prior"
    assert r["n_baselines"] == 2
    assert r["baseline_uniform_prior"] == rows["x9.rescoring"]["stored_dense_localization_auc"]
    assert r["baseline_position_only"] == rows["x9.rescoring"]["stored_position_baseline_auc"]
    assert r["uncertainty_method"]
    assert "0.2821" in r["render"], "the published number appears on the rendered reading"


def test_every_number_in_the_findings_section_is_bound_to_a_row(store: EvidenceStore) -> None:
    """`FINDINGS.md` runs the claims checker with no baseline, so this has to hold with none here."""
    path = RELEASE / "FINDINGS-x9.md"
    report = check_files([path], store)
    assert report.ok, report.render()
    assert len(report.results) >= 30, "a findings section with no claims in it proves nothing"
    assert find_unbound_numbers(path.read_text(encoding="utf-8")) == []


def test_the_figures_name_the_evidence_rows_they_drew_from() -> None:
    figures = sorted((RELEASE / "figures").glob("*.png"))
    assert figures, "no figures were produced"
    for figure in figures:
        caption = figure.with_suffix(".txt")
        assert caption.exists(), f"{figure.name} has no caption naming its evidence"
        assert "ev:" in caption.read_text(encoding="utf-8"), figure.name


def test_the_release_names_the_artifact_it_read(manifest: dict[str, Any]) -> None:
    assert manifest["observable"] == OBSERVABLE
    assert manifest["slice"] == SLICE
    assert manifest["roster_key"] == ROSTER
    assert manifest["higher_is_positive"] is HIGHER_IS_POSITIVE
    assert manifest["n_boot"] == 1000, "the headline reading is at the count W3.6 pinned"
    assert manifest["n_items"] == 2221
    assert manifest["git_sha"]


# ---------------------------------------------------------------------------
# The campaign store itself, read live
# ---------------------------------------------------------------------------


@needs_campaign
def test_the_slice_read_is_the_whole_run_and_not_one_of_its_shards() -> None:
    """Seven `::partNNNN` rows sit beside `processbench-full`. Scoring a shard is a different claim."""
    series = load_step_scores(
        str(CAMPAIGN_STORE),
        observable=OBSERVABLE,
        slice_name=SLICE,
        roster_key=ROSTER,
        higher_is_positive=HIGHER_IS_POSITIVE,
    )
    prior = rescore_against_position(series, n_boot=50, seed=0)
    assert prior.n_items == 2221
    assert prior.n_candidates == 19783
    # The point estimates do not depend on the bootstrap count; only the interval does.
    assert prior.uniform_auc == PINNED_BEFORE_THE_CORRECTION["uniform_auc"]
    assert prior.stratified_auc == PINNED_BEFORE_THE_CORRECTION["stratified_auc"]


@needs_campaign
def test_the_store_this_reads_is_the_one_that_holds_the_adjudication() -> None:
    """SPEC-ERRATA E3: the other campaign store carries no adjudication row."""
    stored = load_stored(CAMPAIGN_STORE)
    assert stored["outcomes"]["H-verif-loc"] == "refuted"
    assert stored["thresholds"]["H-verif-loc"] == 0.7
    assert stored["killed"] is True
    assert stored["killed_by"] == ["K-verif"]
    assert stored["metrics"]["dense_localization_auc"] == 0.2821441611813727
