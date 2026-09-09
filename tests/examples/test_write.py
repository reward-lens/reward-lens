"""What `write()` puts on disk, and what it refuses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from reward_lens.contracts.errors import RewardLensError
from reward_lens.contracts.project import ProjectConfig
from reward_lens.examples.code_reward import MANIFEST, build, check_tasks, write

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "reward_lens" / "examples" / "code_reward"
PROJECT_SCHEMA = REPO / "schema" / "project" / "1.0" / "rewardlens.schema.json"

EXPECTED_TOP_LEVEL = {"grader.py", "tasks.jsonl", "responses.jsonl", "rewardlens.yaml", "outcome"}

#: The files `build()` copies as they are, named here rather than imported, so that a file moving
#: between copied and computed fails a test instead of quietly changing what the byte check covers.
COPIED_FILES = ("grader.py", "rewardlens.yaml", "outcome/test_solution.py", "outcome/check.py")


def _computed_files(root: Path) -> set[Path]:
    """Every file under `root` that `build()` computes, relative to it."""
    found = {Path("tasks.jsonl"), Path("responses.jsonl")}
    for path in (root / "outcome").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            relative = path.relative_to(root)
            if relative.as_posix() not in COPIED_FILES:
                found.add(relative)
    return found


def test_write_produces_the_tree(demo) -> None:
    assert {p.name for p in demo.root.iterdir() if p.name != "__pycache__"} == EXPECTED_TOP_LEVEL
    assert (demo.root / "outcome" / "test_solution.py").is_file()
    assert (demo.root / "outcome" / "check.py").is_file()
    assert len(list((demo.root / "outcome" / "known_good").glob("*.py"))) == 12
    assert len(list((demo.root / "outcome" / "known_wrong").glob("*.py"))) == 12


def test_write_returns_every_file_it_wrote(demo) -> None:
    returned = sorted(p.resolve() for p in demo.written)
    on_disk = sorted(
        p.resolve() for p in demo.root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
    assert returned == on_disk
    assert all(isinstance(p, Path) and p.is_absolute() for p in demo.written)


def test_forty_tasks_and_one_hundred_and_twenty_responses(demo) -> None:
    assert len(demo.tasks) == 40
    assert len(demo.responses) == 120
    assert {t["task_id"] for t in demo.tasks} == {r["task_id"] for r in demo.responses}
    assert all(t["tests"] for t in demo.tasks)
    assert all(t["prompt"].strip() for t in demo.tasks)


def test_responses_carry_no_label_for_the_finding(demo) -> None:
    """D-63: a demo whose finding is hardcoded is a lie. Nothing on a response names the exploit."""
    for response in demo.responses:
        assert set(response) == {"response_id", "task_id", "text"}


def _as_far_as_the_layer_knows(raw: dict[str, Any], *, knows_name: bool) -> dict[str, Any]:
    """`raw`, minus the one key a layer may not have a field for yet.

    A-015 gives top-level `name` to P-CONTRACT and has the example declare it ahead of the field
    landing. Both layers forbid extra keys, so until one of them carries `name` the rest of the
    file is what can validate against it; the day the field lands, the whole file does, with no
    edit here. What the name itself has to say is asserted below, off the file as written.
    """
    return raw if knows_name else {key: value for key, value in raw.items() if key != "name"}


def test_rewardlens_yaml_validates_under_both_layers(demo) -> None:
    raw = yaml.safe_load((demo.root / "rewardlens.yaml").read_text(encoding="utf-8"))
    typed = _as_far_as_the_layer_knows(raw, knows_name="name" in ProjectConfig.model_fields)
    config = ProjectConfig.model_validate(typed)
    assert config.reward.shape == "plain"
    # A-017: the record's input-handling sentence is a claim about a named trainer, so the example
    # declares one and this is what makes the declaration load-bearing rather than decorative.
    assert config.reward.trainer == "trl"
    assert config.outcome is not None and config.outcome.kind == "protected_test_suite"
    assert config.tasks.path == "tasks.jsonl"

    import jsonschema_rs

    schema = json.loads(PROJECT_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema_rs.validator_for(schema)
    validator.validate(_as_far_as_the_layer_knows(raw, knows_name="name" in schema["properties"]))


def test_the_yaml_declares_the_name_beside_the_version(demo) -> None:
    """A-015: a report's subject line reads `<name> <label>`, so the file declares both.

    The label is the version id with `<name>-` stripped when it starts with it, which is only a
    rule the file can honour if the two agree. That agreement is what the last assertion is.
    """
    raw = yaml.safe_load((demo.root / "rewardlens.yaml").read_text(encoding="utf-8"))
    assert raw["name"] == "code-reward"
    assert raw["version"]["id"] == "code-reward-v1"
    assert raw["version"]["id"].startswith(f"{raw['name']}-")


def test_the_manifest_describes_the_written_tree(demo) -> None:
    """A-015: `init` prints MANIFEST, so the manifest has to be the tree, in the order it prints."""
    assert [name for name, _ in MANIFEST] == [
        "grader.py",
        "tasks.jsonl",
        "responses.jsonl",
        "outcome/",
        "rewardlens.yaml",
    ]
    for name, description in MANIFEST:
        assert (demo.root / name).exists(), name
        assert description == description.strip() and description

    top_level = {p.name for p in demo.root.iterdir() if p.name != "__pycache__"}
    assert top_level == EXPECTED_TOP_LEVEL
    assert {Path(name).name for name, _ in MANIFEST} == top_level


def test_the_manifest_counts_are_the_ones_the_bank_produces(demo) -> None:
    """The two counts are written, not computed at print time, so a drifting bank fails here."""
    described = dict(MANIFEST)
    assert described["tasks.jsonl"] == f"{len(demo.tasks)} code tasks"
    assert described["responses.jsonl"] == f"{len(demo.responses)} sampled responses from a small model"


def test_example_package_is_under_two_megabytes() -> None:
    total = sum(p.stat().st_size for p in PACKAGE.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert total < 2 * 1024 * 1024, f"{total} bytes"


def test_written_tree_is_under_two_megabytes(demo) -> None:
    total = sum(
        p.stat().st_size for p in demo.root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
    assert total < 2 * 1024 * 1024, f"{total} bytes"


def test_refuses_a_non_empty_destination(tmp_path: Path) -> None:
    (tmp_path / "already-here.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(RewardLensError) as caught:
        write(tmp_path)
    assert caught.value.code == "RL0001"
    assert caught.value.exit_code == 4
    assert (tmp_path / "already-here.txt").read_text(encoding="utf-8") == "mine"


def test_force_writes_into_a_non_empty_destination(tmp_path: Path) -> None:
    (tmp_path / "already-here.txt").write_text("mine", encoding="utf-8")
    written = write(tmp_path, force=True)
    assert (tmp_path / "grader.py").is_file()
    lines = (tmp_path / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len([json.loads(x) for x in lines if x.strip()]) == 40
    assert all(p.is_file() for p in written)


def test_a_task_without_tests_is_rejected(demo) -> None:
    broken = [dict(demo.tasks[0]), dict(demo.tasks[1])]
    broken[1]["tests"] = []
    with pytest.raises(RewardLensError) as caught:
        check_tasks(broken)
    assert caught.value.code == "RL0001"
    assert broken[1]["task_id"] in str(caught.value)


def test_a_task_with_a_malformed_case_is_rejected(demo) -> None:
    broken = dict(demo.tasks[0])
    broken["tests"] = [{"args": [1]}]
    with pytest.raises(RewardLensError) as caught:
        check_tasks([broken])
    assert caught.value.code == "RL0001"


def test_the_shipped_bank_is_reproducible(tmp_path: Path) -> None:
    """Nothing in the bank was hand-tuned: recomputing the computed files gives the same bytes.

    Only the files `build()` computes are compared. The four it copies are checked in the next test,
    because a copy matching its source says nothing about whether anything was tuned by hand.
    """
    build(tmp_path)
    generated = _computed_files(PACKAGE)
    assert not {p.as_posix() for p in generated} & set(COPIED_FILES)
    assert len(generated) == 27, sorted(p.as_posix() for p in generated)
    for relative in sorted(generated):
        assert (tmp_path / relative).read_bytes() == (PACKAGE / relative).read_bytes(), relative.as_posix()


def test_the_static_files_are_copied_unchanged(tmp_path: Path) -> None:
    """The other half, kept apart: these four are copied, and every file is in one half or the other."""
    from reward_lens.examples.code_reward._bank import STATIC_FILES

    assert set(STATIC_FILES) == set(COPIED_FILES), "a file changed category, so the split above moved"
    written = build(tmp_path)
    for name in COPIED_FILES:
        assert (tmp_path / name).read_bytes() == (PACKAGE / name).read_bytes(), name
    relative = {Path(p).relative_to(tmp_path.resolve()) for p in written}
    assert relative == _computed_files(PACKAGE) | {Path(name) for name in COPIED_FILES}
