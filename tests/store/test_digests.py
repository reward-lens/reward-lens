"""The nine digests: computed over canonical bytes, stable, specific, and `None` where absent."""

from __future__ import annotations

from pathlib import Path

from .conftest import write_project

NINE = (
    "source",
    "environment",
    "scorer_config",
    "task_distribution",
    "samples",
    "outcome_protocol",
    "policy",
    "training_semantics",
    "instrument_method",
)

DIGEST = r"^sha256:[0-9a-f]{64}$"


def test_the_nine_are_the_nine(project):
    from reward_lens.store import DIGEST_NAMES

    assert DIGEST_NAMES == NINE
    assert tuple(project.version().digests.to_dict()) == NINE


def test_each_present_digest_is_a_sha256_over_canonical_bytes(project):
    import re

    digests = project.version().digests.to_dict()
    for name in NINE:
        value = digests[name]
        if value is not None:
            assert re.match(DIGEST, value), name


def test_a_full_project_leaves_only_the_checkpoint_absent(project):
    digests = project.version().digests.to_dict()
    assert digests["policy"] is None  # a project does not bind a checkpoint; an experiment does
    assert digests["training_semantics"] is None  # no trainer is declared
    for name in ("source", "environment", "scorer_config", "task_distribution", "samples",
                 "outcome_protocol", "instrument_method"):
        assert digests[name] is not None, name


def test_instrument_method_is_never_absent(project):
    assert project.version().digests.instrument_method is not None


def test_the_digests_are_stable_across_calls_and_reopens(project, project_root):
    from reward_lens.store import Project

    first = project.version().digests.to_dict()
    second = project.version().digests.to_dict()
    third = Project.open(project_root).version().digests.to_dict()
    assert first == second == third


def _changed(before: dict, after: dict) -> set[str]:
    return {name for name in NINE if before[name] != after[name]}


def test_editing_the_grader_moves_the_source_digest_alone(project, project_root):
    before = project.version().digests.to_dict()
    (project_root / "grader.py").write_text("def score(t, r):\n    return 0.5\n", encoding="utf-8")
    assert _changed(before, project.version().digests.to_dict()) == {"source"}


def test_editing_the_task_file_moves_the_task_distribution_alone(project, project_root):
    before = project.version().digests.to_dict()
    (project_root / "tasks.jsonl").write_text('{"prompt": "three"}\n', encoding="utf-8")
    assert _changed(before, project.version().digests.to_dict()) == {"task_distribution"}


def test_editing_the_responses_moves_the_samples_digest_alone(project, project_root):
    before = project.version().digests.to_dict()
    (project_root / "responses.jsonl").write_text('{"text": "c"}\n', encoding="utf-8")
    assert _changed(before, project.version().digests.to_dict()) == {"samples"}


def test_editing_the_protected_suite_moves_the_outcome_protocol_alone(project, project_root):
    before = project.version().digests.to_dict()
    (project_root / "outcome" / "test_solution.py").write_text("def test_x():\n    pass\n",
                                                               encoding="utf-8")
    assert _changed(before, project.version().digests.to_dict()) == {"outcome_protocol"}


def test_editing_the_lockfile_moves_the_environment_alone(project, project_root):
    before = project.version().digests.to_dict()
    (project_root / "requirements.txt").write_text("pytest==9.0.0\n", encoding="utf-8")
    assert _changed(before, project.version().digests.to_dict()) == {"environment"}


def test_changing_the_scorer_configuration_moves_the_scorer_config(tmp_path: Path):
    from .conftest import PROJECT_YAML
    from reward_lens.store import Project

    a = Project.open(write_project(tmp_path / "a")).version().digests.to_dict()
    b_root = write_project(
        tmp_path / "b", yaml_text=PROJECT_YAML.replace("none_to_nan", "bool_as_score")
    )
    b = Project.open(b_root).version().digests.to_dict()
    assert _changed(a, b) == {"scorer_config"}


def test_a_declared_trainer_gives_the_training_semantics_a_digest(tmp_path: Path):
    from .conftest import PROJECT_YAML
    from reward_lens.store import Project

    yaml_text = PROJECT_YAML.replace("  kind: composite\n", "  kind: composite\n  trainer: trl\n")
    digests = Project.open(write_project(tmp_path / "t", yaml_text=yaml_text)).version().digests
    assert digests.training_semantics is not None


def test_a_project_with_no_outcome_and_no_responses_says_so(tmp_path: Path):
    from reward_lens.store import Project

    yaml_text = (
        "reward:\n  kind: plain\n  entry: grader.py:score\n"
        "tasks:\n  path: tasks.jsonl\n  prompt: prompt\n"
        "outcome: null\n"
    )
    digests = Project.open(write_project(tmp_path / "bare", yaml_text=yaml_text)).version().digests
    assert digests.outcome_protocol is None
    assert digests.samples is None
    assert digests.source is not None


def test_the_version_digest_is_over_the_nine_and_the_identity(project, project_root):
    version = project.version()
    assert version.digest() == project.version().digest()
    (project_root / "grader.py").write_text("def score(t, r):\n    return 0.25\n", encoding="utf-8")
    assert project.version().digest() != version.digest()


def test_the_version_takes_its_id_and_parents_from_the_config(tmp_path: Path):
    from .conftest import PROJECT_YAML
    from reward_lens.store import Project

    yaml_text = PROJECT_YAML + "version:\n  id: v7\n  parents:\n    - v6\n"
    version = Project.open(write_project(tmp_path / "v", yaml_text=yaml_text)).version()
    assert version.id == "v7"
    assert version.parents == ("v6",)


def test_an_undeclared_version_id_is_derived_from_the_digest(project):
    version = project.version()
    assert version.id == "v-" + version.digest().removeprefix("sha256:")[:12]
    assert version.parents == ()


def test_the_version_becomes_a_subject_the_contracts_accept(project):
    from reward_lens.contracts.models import Digests, VersionRef

    version = project.version()
    assert isinstance(version.to_subject_digests(), Digests)
    assert isinstance(version.to_version_ref(), VersionRef)


def test_the_instrument_method_digest_follows_the_method_set():
    from reward_lens.store.digests import instrument_method_digest

    empty = instrument_method_digest(())
    assert empty == instrument_method_digest(())
    assert instrument_method_digest(("rl.validity.replay",)) != empty


def test_nothing_here_uses_python_json_for_hashed_bytes():
    """D-10: `contracts.canonical_bytes` is the only producer of hashed bytes."""
    from pathlib import Path as P

    source = P(__file__).resolve().parents[2] / "src" / "reward_lens" / "store"
    for path in sorted(source.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "json.dumps" not in text, path.name
