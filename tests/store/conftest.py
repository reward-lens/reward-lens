"""Fixtures for the store tests: a project on disk, and a record whose entries span the table.

The corpus below is written by hand. The five invalidation tests read their expected stale and
standing sets off section 6.1's table and off this corpus, never off `reward_lens.store`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = REPO_ROOT / "schema" / "assay" / "1.0" / "fixtures" / "reference.json"

#: The version digest the reference record's subject names; every entry about this reward-system
#: version carries it as `subject_ref`.
SUBJECT = "sha256:" + "2" * 64
#: A different subject, for the forecast entry that row 5 leaves standing.
OTHER_SUBJECT = "sha256:" + "3" * 64

SECTIONS = (
    "validity",
    "soundness",
    "reach",
    "exploits",
    "framing",
    "reward_statistics",
    "signal",
    "trace",
    "forecast",
    "calibration",
)

#: entry_id, kind, depends_on, subject_ref, scope. Hand-written to cover every cell of the
#: section 6.1 table: a grader-source check, a replay check, judge-dependent scores and
#: comparisons, selection and training evidence that read the checkpoint, an independent outcome
#: check, a label-derived estimate, a reach inventory, an exploit witness, a task-mixture entry
#: that no row touches, and two forecast entries that differ only in the subject they named.
CORPUS: tuple[tuple[str, str, tuple[str, ...], str, str], ...] = (
    ("validity.grader_source", "check", ("digest:source",), SUBJECT, "evaluator_comparison"),
    (
        "validity.replay_determinism",
        "check",
        ("digest:source", "digest:environment"),
        SUBJECT,
        "evaluator_comparison",
    ),
    (
        "soundness.judge_agreement",
        "estimate",
        ("digest:scorer_config", "digest:samples"),
        SUBJECT,
        "evaluator_comparison",
    ),
    (
        "soundness.judge_comparison",
        "estimate",
        ("digest:scorer_config", "digest:task_distribution"),
        SUBJECT,
        "evaluator_comparison",
    ),
    (
        "signal.selection_stress",
        "estimate",
        ("digest:policy", "digest:samples"),
        SUBJECT,
        "selection_stress",
    ),
    (
        "signal.training_pressure",
        "estimate",
        ("digest:policy", "digest:training_semantics"),
        SUBJECT,
        "reconstructed_training_pressure",
    ),
    (
        "calibration.outcome_agreement",
        "check",
        ("digest:outcome_protocol",),
        SUBJECT,
        "evaluator_comparison",
    ),
    (
        "reward_statistics.label_derived_rate",
        "estimate",
        ("digest:outcome_protocol", "digest:task_distribution"),
        SUBJECT,
        "evaluator_comparison",
    ),
    ("reach.exposure_inventory", "check", ("digest:source",), SUBJECT, "evaluator_comparison"),
    (
        "exploits.attack_witness",
        "witness",
        ("digest:source", "digest:samples"),
        SUBJECT,
        "selection_stress",
    ),
    (
        "framing.task_families",
        "check",
        ("digest:task_distribution",),
        SUBJECT,
        "evaluator_comparison",
    ),
    ("forecast.this_subject", "check", ("digest:source",), SUBJECT, "evaluator_comparison"),
    (
        "forecast.other_subject",
        "check",
        ("digest:source",),
        OTHER_SUBJECT,
        "evaluator_comparison",
    ),
)

ALL_IDS: tuple[str, ...] = tuple(row[0] for row in CORPUS)


def reference_dict() -> dict:
    """The frozen reference record, as data."""
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def build_record(corpus=CORPUS) -> dict:
    """The reference record with `corpus` installed as its measurement, as data."""
    base = reference_dict()
    templates = {e["kind"]: e for section in base["measurement"].values() for e in section}
    measurement: dict[str, list] = {section: [] for section in SECTIONS}
    for entry_id, kind, depends_on, subject_ref, scope in corpus:
        entry = copy.deepcopy(templates[kind])
        entry.update(
            entry_id=entry_id,
            section=entry_id.split(".", 1)[0],
            depends_on=list(depends_on),
            subject_ref=subject_ref,
            scope=scope,
        )
        measurement[entry_id.split(".", 1)[0]].append(entry)
    base["measurement"] = measurement
    base["holes"] = []
    return base


@pytest.fixture()
def record_dict() -> dict:
    return build_record()


@pytest.fixture()
def corpus_assay(record_dict):
    from reward_lens.contracts import Assay

    return Assay.model_validate(record_dict)


#: A composite reward, because one of its two components is a model judge: the scorer
#: configuration is where a judge is named and weighted, so row 1 of the table has something real
#: to revise that is not the grader's source.
PROJECT_YAML = """\
reward:
  kind: composite
  entry: grader.py:score
  components:
    - name: judge
      entry: judge.py:judge
      weight: 0.7
    - name: tests
      entry: grader.py:score
      weight: 0.3
  watch:
    - none_to_nan
tasks:
  path: tasks.jsonl
  prompt: prompt
responses:
  path: responses.jsonl
outcome:
  kind: protected_test_suite
  path: outcome
success: the grader agrees with the protected suite
"""

GRADER = "def score(task, response):\n    return 1.0 if response.strip() else 0.0\n"
JUDGE = "def judge(task, response):\n    return 1.0 if len(response) > 1 else 0.0\n"

#: The acceptance policy as the project declares it, and a revision of it. Neither appears in any
#: of the nine digests, which is row 4 of the table: the acceptance policy invalidates the decision
#: and no measurement.
ACCEPTANCE = "success: the grader agrees with the protected suite"
REVISED_ACCEPTANCE = "success: the grader agrees with the protected suite on every task family"

#: The judge's weight in the scorer configuration, and a revision of it.
JUDGE_WEIGHT = "      weight: 0.7"
REVISED_JUDGE_WEIGHT = "      weight: 0.4"


def write_project(root: Path, *, yaml_text: str = PROJECT_YAML) -> Path:
    """A project on disk: config, grader, judge, tasks, responses, protected suite, lockfile."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "rewardlens.yaml").write_text(yaml_text, encoding="utf-8")
    (root / "grader.py").write_text(GRADER, encoding="utf-8")
    (root / "judge.py").write_text(JUDGE, encoding="utf-8")
    (root / "tasks.jsonl").write_text('{"prompt": "one"}\n{"prompt": "two"}\n', encoding="utf-8")
    (root / "responses.jsonl").write_text('{"text": "a"}\n{"text": "b"}\n', encoding="utf-8")
    outcome = root / "outcome"
    outcome.mkdir(exist_ok=True)
    (outcome / "test_solution.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (root / "requirements.txt").write_text("pytest==9.1.1\n", encoding="utf-8")
    return root


REVISED_GRADER = "def score(t, r):\n    return 0.0\n"


def revise_grader(root: Path) -> None:
    """Rewrite the grader's source: the change row 5 names."""
    (root / "grader.py").write_text(REVISED_GRADER, encoding="utf-8")


def revise_tasks(root: Path) -> None:
    """Add a task: the task distribution moves, and the samples drawn from it with it."""
    (root / "tasks.jsonl").write_text(
        '{"prompt": "one"}\n{"prompt": "two"}\n{"prompt": "three"}\n', encoding="utf-8"
    )


def revise_responses(root: Path) -> None:
    """Replace the response bank: the samples a different checkpoint produced."""
    (root / "responses.jsonl").write_text('{"text": "c"}\n{"text": "d"}\n', encoding="utf-8")


def revise_outcome(root: Path) -> None:
    """Revise the protected suite: the labels every outcome-derived estimate rests on."""
    (root / "outcome" / "test_solution.py").write_text(
        "def test_ok():\n    assert True\n\n\ndef test_also():\n    assert 1 == 1\n",
        encoding="utf-8",
    )


def revise_config(root: Path, old: str, new: str) -> None:
    """Edit one declaration in the project file, the way an owner would, before reopening."""
    path = root / "rewardlens.yaml"
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} is not in {path}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    return write_project(tmp_path / "demo")


@pytest.fixture()
def project(project_root: Path):
    from reward_lens.store import Project

    return Project.open(project_root)


@pytest.fixture()
def state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`$XDG_STATE_HOME` pointed at the test's own directory, read at call time."""
    home = tmp_path / "state"
    home.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(home))
    return home
