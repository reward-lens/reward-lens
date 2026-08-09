"""The five invalidation rows of section 6.1, one test each.

Every expected set below was written by hand from the table and the corpus in `conftest.py`. None
of them is derived from `reward_lens.store`; a reviewer can check each against the two columns:

| Change | Invalidated | Still standing |
|---|---|---|
| judge revision | judge-dependent scores and comparisons | grader-source checks, independent outcome checks |
| target checkpoint | selection and training evidence that read its samples or state | evaluator-side evidence |
| outcome label version | estimates and decisions derived from those labels | measurements not derived from labels |
| acceptance policy | the decision | every measurement |
| grader source | source checks, reach, exploits, replay | forecast entries that named a different subject |
"""

from __future__ import annotations

import pytest

from .conftest import ALL_IDS

# --- row 1: judge revision ----------------------------------------------------------------------

JUDGE_REVISION_STALE = {
    "soundness.judge_agreement",  # a judge-dependent score
    "soundness.judge_comparison",  # a judge-dependent comparison
}
JUDGE_REVISION_STANDING = {
    "validity.grader_source",  # a grader-source check: the table names it as standing
    "validity.replay_determinism",
    "calibration.outcome_agreement",  # an independent outcome check: the table names it too
    "reward_statistics.label_derived_rate",
    "signal.selection_stress",
    "signal.training_pressure",
    "reach.exposure_inventory",
    "exploits.attack_witness",
    "framing.task_families",
    "forecast.this_subject",
    "forecast.other_subject",
}

# --- row 2: target checkpoint -------------------------------------------------------------------

TARGET_CHECKPOINT_STALE = {
    "soundness.judge_agreement",  # read the checkpoint's samples
    "signal.selection_stress",  # selection evidence: read its state and its samples
    "signal.training_pressure",  # training evidence: read its state
    "exploits.attack_witness",  # read the checkpoint's samples
}
TARGET_CHECKPOINT_STANDING = {
    "validity.grader_source",  # evaluator-side: never read the checkpoint
    "validity.replay_determinism",
    "soundness.judge_comparison",  # evaluator-side: scorer config and task mixture only
    "calibration.outcome_agreement",
    "reward_statistics.label_derived_rate",
    "reach.exposure_inventory",
    "framing.task_families",
    "forecast.this_subject",
    "forecast.other_subject",
}

# --- row 3: outcome label version ---------------------------------------------------------------

OUTCOME_LABEL_STALE = {
    "calibration.outcome_agreement",
    "reward_statistics.label_derived_rate",  # an estimate derived from those labels
}
OUTCOME_LABEL_STANDING = {
    "validity.grader_source",
    "validity.replay_determinism",
    "soundness.judge_agreement",
    "soundness.judge_comparison",
    "signal.selection_stress",
    "signal.training_pressure",
    "reach.exposure_inventory",
    "exploits.attack_witness",
    "framing.task_families",
    "forecast.this_subject",
    "forecast.other_subject",
}

# --- row 4: acceptance policy -------------------------------------------------------------------

ACCEPTANCE_POLICY_STALE: set[str] = set()
ACCEPTANCE_POLICY_STANDING = set(ALL_IDS)  # every measurement stands; the decision alone falls

# --- row 5: grader source -----------------------------------------------------------------------

GRADER_SOURCE_STALE = {
    "validity.grader_source",  # a source check
    "validity.replay_determinism",  # replay
    "reach.exposure_inventory",  # reach
    "exploits.attack_witness",  # exploits
    "forecast.this_subject",  # a forecast that named this subject
}
GRADER_SOURCE_STANDING = {
    "soundness.judge_agreement",
    "soundness.judge_comparison",
    "signal.selection_stress",
    "signal.training_pressure",
    "calibration.outcome_agreement",
    "reward_statistics.label_derived_rate",
    "framing.task_families",
    "forecast.other_subject",  # a forecast entry that named a different subject
}

ROWS = [
    ("judge_revision", JUDGE_REVISION_STALE, JUDGE_REVISION_STANDING),
    ("target_checkpoint", TARGET_CHECKPOINT_STALE, TARGET_CHECKPOINT_STANDING),
    ("outcome_label_version", OUTCOME_LABEL_STALE, OUTCOME_LABEL_STANDING),
    ("acceptance_policy", ACCEPTANCE_POLICY_STALE, ACCEPTANCE_POLICY_STANDING),
    ("grader_source", GRADER_SOURCE_STALE, GRADER_SOURCE_STANDING),
]


def _check(project, assay, change, stale_expected, standing_expected):
    stale, standing = project.stale(assay, {change})
    assert set(stale) == stale_expected
    assert set(standing) == standing_expected
    assert set(stale) | set(standing) == set(ALL_IDS)
    assert not set(stale) & set(standing)
    assert len(stale) + len(standing) == len(ALL_IDS)


def test_row_judge_revision(project, corpus_assay):
    _check(project, corpus_assay, "judge_revision", JUDGE_REVISION_STALE, JUDGE_REVISION_STANDING)


def test_row_target_checkpoint(project, corpus_assay):
    _check(
        project,
        corpus_assay,
        "target_checkpoint",
        TARGET_CHECKPOINT_STALE,
        TARGET_CHECKPOINT_STANDING,
    )


def test_row_outcome_label_version(project, corpus_assay):
    _check(
        project, corpus_assay, "outcome_label_version", OUTCOME_LABEL_STALE, OUTCOME_LABEL_STANDING
    )


def test_row_acceptance_policy(project, corpus_assay):
    _check(
        project,
        corpus_assay,
        "acceptance_policy",
        ACCEPTANCE_POLICY_STALE,
        ACCEPTANCE_POLICY_STANDING,
    )


def test_row_grader_source(project, corpus_assay):
    _check(project, corpus_assay, "grader_source", GRADER_SOURCE_STALE, GRADER_SOURCE_STANDING)


# --- the rest of the rule -----------------------------------------------------------------------


def test_no_change_leaves_everything_standing(project, corpus_assay):
    stale, standing = project.stale(corpus_assay, set())
    assert stale == ()
    assert set(standing) == set(ALL_IDS)


def test_two_changes_take_the_union(project, corpus_assay):
    stale, _ = project.stale(corpus_assay, {"judge_revision", "grader_source"})
    assert set(stale) == JUDGE_REVISION_STALE | GRADER_SOURCE_STALE


def test_the_returned_order_is_the_record_order(project, corpus_assay):
    stale, standing = project.stale(corpus_assay, {"grader_source"})
    order = [entry.entry_id for entry in corpus_assay.entries()]
    assert list(stale) == [i for i in order if i in GRADER_SOURCE_STALE]
    assert list(standing) == [i for i in order if i in GRADER_SOURCE_STANDING]


def test_a_digest_name_is_accepted_in_place_of_a_change_kind(project, corpus_assay):
    by_kind, _ = project.stale(corpus_assay, {"grader_source"})
    by_digest, _ = project.stale(corpus_assay, {"digest:source"})
    bare, _ = project.stale(corpus_assay, {"source"})
    assert set(by_kind) == set(by_digest) == set(bare) == GRADER_SOURCE_STALE


def test_an_unknown_change_token_is_refused(project, corpus_assay):
    with pytest.raises(ValueError) as excinfo:
        project.stale(corpus_assay, {"vibes"})
    assert "vibes" in str(excinfo.value)


def test_the_five_rows_are_closed_under_derivation():
    """Invalidation is transitive; the table's five rows are already closed under `DERIVES_FROM`."""
    from reward_lens.store.invalidation import PERTURBS, close

    for change, digests in PERTURBS.items():
        assert close(digests) == digests, change


def test_samples_derive_from_the_distribution_and_the_policy():
    from reward_lens.store.invalidation import close

    assert close(frozenset({"task_distribution"})) == frozenset({"task_distribution", "samples"})
    assert close(frozenset({"policy"})) == frozenset({"policy", "samples"})


def test_the_decision_falls_where_the_table_says_it_does(project, corpus_assay):
    assert project.decision_stale(corpus_assay, {"acceptance_policy"}) is True
    assert project.decision_stale(corpus_assay, {"outcome_label_version"}) is True
    assert project.decision_stale(corpus_assay, set()) is False


def test_the_decision_is_stale_when_a_reason_it_rests_on_goes_stale(project, record_dict):
    from reward_lens.contracts import Assay

    record_dict["decision"]["reasons"] = ["required_stale:validity.grader_source"]
    assay = Assay.model_validate(record_dict)
    assert project.decision_stale(assay, {"grader_source"}) is True
    assert project.decision_stale(assay, {"judge_revision"}) is False


# --- D-11 through the public API ------------------------------------------------------------------
#
# The tests above put a change to `partition()` by name. These put the same five rows through the
# project: a real input moves on disk or in the project file, and the store is asked what still
# stands. Two rows have no input to move, for reasons the design fixed rather than overlooked, and
# each of those tests says which half of its row a file can reach and which half only a declared
# change can. See `## Premise checks` in the handoff.

#: Changing the response bank moves `samples` alone: the half of row 2 a file can reach.
SAMPLES_STALE = {
    "soundness.judge_agreement",
    "signal.selection_stress",
    "exploits.attack_witness",
}
#: Changing the task set moves `task_distribution`, and the samples drawn from it derive from it.
TASK_SET_STALE = SAMPLES_STALE | {
    "soundness.judge_comparison",
    "reward_statistics.label_derived_rate",
    "framing.task_families",
}


@pytest.fixture()
def measured(project, record_dict):
    """A measurement of the version this project declares, as the store would have recorded it."""
    from reward_lens.contracts import Assay

    from .test_project import align

    return Assay.model_validate(align(record_dict, project))


def _fires(project, measured, moved: set[str], stale_expected: set[str]) -> None:
    """The project reports exactly `moved`, and exactly `stale_expected` stops standing."""
    changed = project.changed_since(measured)
    assert changed == frozenset(moved)
    stale, standing = project.stale(measured, changed)  # invalidation.partition
    assert set(stale) == stale_expected
    assert set(standing) == set(ALL_IDS) - stale_expected
    assert set(stale) | set(standing) == set(ALL_IDS)
    reuse = project.reuse(measured)
    assert set(reuse.stale_entry_ids) == stale_expected
    assert set(reuse.reused_entry_ids) == set(ALL_IDS) - stale_expected


def test_d11_a_judge_revision_through_the_project(project, measured, project_root):
    """Row 1: the judge is named and weighted by the scorer configuration, so revising it is row 1."""
    from reward_lens.store import Project

    from .conftest import JUDGE_WEIGHT, REVISED_JUDGE_WEIGHT, revise_config

    revise_config(project_root, JUDGE_WEIGHT, REVISED_JUDGE_WEIGHT)
    _fires(Project.open(project_root), measured, {"scorer_config"}, JUDGE_REVISION_STALE)


def test_d11_a_new_response_bank_through_the_project(project, measured, project_root):
    """Row 2, the half a file can reach: a different checkpoint's samples."""
    from .conftest import revise_responses

    revise_responses(project_root)
    _fires(project, measured, {"samples"}, SAMPLES_STALE)


def test_d11_a_target_checkpoint_declared_takes_the_training_evidence_too(project, measured):
    """Row 2, whole: `policy` is absent from every project by design, so the change is declared."""
    assert project.version().digests.to_dict()["policy"] is None
    assert project.changed_since(measured) == frozenset()  # no file can move it
    stale, standing = project.stale(measured, {"target_checkpoint"})
    assert set(stale) == TARGET_CHECKPOINT_STALE
    assert set(standing) == TARGET_CHECKPOINT_STANDING
    assert "signal.training_pressure" in set(stale) - SAMPLES_STALE


def test_d11_a_revised_outcome_protocol_through_the_project(project, measured, project_root):
    """Row 3: the protected suite is the labels, and revising it is an outcome label version."""
    from .conftest import revise_outcome

    revise_outcome(project_root)
    _fires(project, measured, {"outcome_protocol"}, OUTCOME_LABEL_STALE)


def test_d11_a_revised_acceptance_policy_leaves_every_measurement_standing(
    project, measured, project_root
):
    """Row 4: no digest carries the acceptance policy, and no measurement rests on it."""
    from reward_lens.store import Project

    from .conftest import ACCEPTANCE, REVISED_ACCEPTANCE, revise_config

    revise_config(project_root, ACCEPTANCE, REVISED_ACCEPTANCE)
    reopened = Project.open(project_root)
    assert reopened.config.success != project.config.success  # the project did change
    _fires(reopened, measured, set(), ACCEPTANCE_POLICY_STALE)
    assert reopened.decision_stale(measured, {"acceptance_policy"}) is True


def test_d11_a_changed_grader_source_through_the_project(project, measured, project_root):
    """Row 5: the source moves, and the forecast that named another subject still stands."""
    from .conftest import revise_grader

    revise_grader(project_root)
    _fires(project, measured, {"source"}, GRADER_SOURCE_STALE)
    assert "forecast.other_subject" in GRADER_SOURCE_STANDING


def test_d11_a_changed_task_set_carries_through_to_the_samples(project, measured, project_root):
    """Not a row of its own: the one derivation the table states, put through the project."""
    from .conftest import revise_tasks

    revise_tasks(project_root)
    _fires(project, measured, {"task_distribution"}, TASK_SET_STALE)
