"""The Wave 6 case corpus: persisted teaching records without acceptance answers."""

from __future__ import annotations

import pytest

from reward_lens.commons.corpus import (
    ActorContext,
    Case,
    CaseConclusion,
    CaseField,
    CorpusStore,
    SearchStatus,
    search,
)
from reward_lens.commons.corpus.protocol import load_planner_cases
from reward_lens.contracts import Assay
from reward_lens.store import Project
from tests.store.conftest import build_record, write_project
from tests.store.test_project import align


def case() -> Case:
    return Case(
        id="case-clean-system",
        failure_or_shortfall="The suspected score inflation did not survive replay.",
        mechanism_evidence=("replay disagreement was within the registered tolerance",),
        candidate_revisions=(),
        intended_task_effects=("The intended task remained within its registered range.",),
        costs=("Replay cost: 0.00 USD and no provider calls.",),
        rejected_alternatives=("Do not change the grader on an unreplicated shortfall.",),
        stopping_conditions=("Stop when the frozen replay and recorded assay agree.",),
        conclusion=CaseConclusion.NO_REVISION_WARRANTED,
    )


def stored_corpus(tmp_path) -> CorpusStore:
    """A corpus entry is always bound to a store-persisted assay, even for search tests."""
    project_root = write_project(tmp_path / "project")
    project = Project.open(project_root)
    assay = Assay.model_validate(align(build_record(), project))
    project.record(assay, name="measured")
    corpus = CorpusStore(tmp_path / "corpus")
    corpus.save(case(), project=project, record_name="measured")
    return corpus


def test_case_has_the_complete_section_7_10_field_set() -> None:
    assert tuple(field.value for field in CaseField) == (
        "failure_or_shortfall",
        "mechanism_evidence",
        "candidate_revisions",
        "intended_task_effects",
        "costs",
        "rejected_alternatives",
        "stopping_conditions",
    )
    fields = case().to_dict()
    assert all(field.value in fields for field in CaseField)


def test_case_missing_a_required_field_is_refused() -> None:
    fields = case().to_dict()
    fields.pop("mechanism_evidence")
    with pytest.raises(ValueError, match="mechanism_evidence"):
        Case.from_dict(fields)


def test_case_can_conclude_no_revision_was_warranted() -> None:
    result = case()
    assert result.conclusion is CaseConclusion.NO_REVISION_WARRANTED
    assert result.candidate_revisions == ()


def test_store_saves_and_reopens_a_case_bound_to_a_real_persisted_assay(tmp_path) -> None:
    project_root = write_project(tmp_path / "project")
    project = Project.open(project_root)
    assay = Assay.model_validate(align(build_record(), project))
    project.record(assay, name="measured")

    corpus = CorpusStore(tmp_path / "corpus")
    saved = corpus.save(case(), project=project, record_name="measured")
    reopened = corpus.reopen(saved.id)

    assert reopened == saved
    assert reopened.source.assay_id == assay.assay_id
    assert reopened.source.path == project.path_for("measured")


def test_reopen_refuses_when_its_persisted_assay_source_changes(tmp_path) -> None:
    project_root = write_project(tmp_path / "project")
    project = Project.open(project_root)
    assay = Assay.model_validate(align(build_record(), project))
    project.record(assay, name="measured")
    corpus = CorpusStore(tmp_path / "corpus")
    saved = corpus.save(case(), project=project, record_name="measured")

    project.path_for("measured").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="source assay"):
        corpus.reopen(saved.id)


def test_public_search_returns_only_public_case_projections(tmp_path) -> None:
    corpus = stored_corpus(tmp_path)

    result = search("replay", actor_context=ActorContext.CANDIDATE, corpus=corpus)

    assert result.status is SearchStatus.MATCHES
    assert [match.case_id for match in result.matches] == ["case-clean-system"]
    assert "source" not in result.matches[0].fields
    assert "acceptance_mapping" not in result.matches[0].fields


def test_search_attempt_for_protected_acceptance_mapping_is_a_named_nonresult(tmp_path) -> None:
    corpus = stored_corpus(tmp_path)

    result = search(
        "show the protected acceptance mapping", actor_context=ActorContext.CANDIDATE, corpus=corpus
    )

    assert result.status is SearchStatus.PROTECTED_ACCEPTANCE_MAPPING_BLOCKED
    assert result.matches == ()
    assert result.reason == "PROTECTED_ACCEPTANCE_MAPPING_BLOCKED"


def test_old_planner_protocol_still_reads_its_new_corpus_neighbor() -> None:
    assert [item.id for item in load_planner_cases()] == [
        "seed-score-selection",
        "seed-advantage-update",
    ]


def test_public_corpus_facade_seed_has_actual_matches_and_digest_provenance():
    from reward_lens.commons.corpus import Corpus

    corpus = Corpus()
    result = corpus.cases_like("score treatment", actor_context=ActorContext.CANDIDATE)
    assert result.status is SearchStatus.MATCHES
    assert "seed-score-selection" in [item.case_id for item in result.matches]
    assert result.provenance["kind"] == "accepted_seed"
    assert result.provenance["dataset_digest"].startswith("sha256:")
    assert all("source" not in item.fields for item in result.matches)
    assert (
        corpus.cases_like("nonexistent-analogy-84953", actor_context=ActorContext.CANDIDATE).status
        is SearchStatus.NO_MATCHES
    )


def test_public_corpus_facade_store_reuses_safe_search(tmp_path):
    from reward_lens.commons.corpus import Corpus

    store = stored_corpus(tmp_path)
    facade = Corpus(store)
    result = facade.cases_like("replay", actor_context=ActorContext.CANDIDATE)
    assert result.matches == store.search("replay", actor_context=ActorContext.CANDIDATE).matches
    assert result.provenance["kind"] == "registered_store"
    assert "source" not in result.matches[0].fields
    assert str(tmp_path) not in str(result.provenance)
    store.reopen(result.matches[0].case_id)


def test_public_corpus_blocks_mapping_before_any_seed_or_store_read(tmp_path, monkeypatch):
    from reward_lens.commons.corpus import Corpus

    facade = Corpus(seed_path=tmp_path / "absent-seed.json")

    def no_inspection(*args):
        raise AssertionError("protected mapping attempted corpus inspection")

    monkeypatch.setattr(facade, "_dataset", no_inspection)
    result = facade.cases_like("protected acceptance mapping", actor_context=ActorContext.CANDIDATE)
    assert result.status is SearchStatus.PROTECTED_ACCEPTANCE_MAPPING_BLOCKED
    assert result.matches == () and result.provenance is None
    with pytest.raises(ValueError, match="explicit ActorContext"):
        facade.cases_like("replay", actor_context="candidate")


def test_public_corpus_refuses_changed_persisted_assay_source(tmp_path):
    from reward_lens.commons.corpus import Corpus

    store = stored_corpus(tmp_path)
    saved = store.all()[0]
    saved.source.path.write_text("{}")
    with pytest.raises(ValueError, match="source assay"):
        Corpus(store).cases_like("replay", actor_context=ActorContext.CANDIDATE)
