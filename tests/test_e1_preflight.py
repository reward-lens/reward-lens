"""The E1 preflight's own machinery, and the guard that keeps `SEALED/` sealed.

Two kinds of test here and they earn their place differently.

The first kind is ordinary: the column enumerator, the configuration adjudicator and the artifact
helpers, each on inputs small enough that the right answer is obvious by inspection. They exist
because a preflight whose own tools are untested cannot distinguish a finding from a bug, which is
the failure the whole exercise is written against.

The second is the `SEALED/` guard, and it is a repo scan rather than a unit test. The analysis
package the gate document names does not exist yet, so there is nothing to point an import check at.
A test that skipped until it did would be a test that was green on the day it mattered least and
absent on the day it mattered most, so this one runs unconditionally over every Python file in the
tree and fails the moment anything outside the sealed directory reads from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiments.e1_preflight import artifacts, columns, configs

REPO = Path(__file__).resolve().parents[1]
SEALED = REPO / "e1_preflight" / "SEALED"


# ---------------------------------------------------------------------------
# The SEALED guard
# ---------------------------------------------------------------------------


#: Files allowed to name the sealed directory: the package that writes it, the guard that checks it,
#: and the prose that explains what it is for. Anything else naming it is the leak this test exists
#: to catch, and adding a name here is a decision rather than a formality.
SEALED_WRITERS = {
    "experiments/e1_preflight/report.py",
    "experiments/e1_preflight/__init__.py",
    "tests/test_e1_preflight.py",
    # The quantity-based guard and its tests. They name the directory because they protect it.
    "src/reward_lens/artifacts/sealed.py",
    "tests/test_sealed_guard.py",
}


def _python_files() -> list[Path]:
    out: list[Path] = []
    for root in ("src", "tests", "experiments", "tools", "scripts"):
        base = REPO / root
        if base.exists():
            out.extend(sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts))
    return out


def test_nothing_outside_the_sealed_directory_reads_it():
    """No module may read `e1_preflight/SEALED`. Per-mechanism late-window outcomes are E1's targets.

    The preflight was permitted to touch the late window, because testing reward closure across the
    full series requires it. What it may not do is carry a per-mechanism outcome forward: the
    trajectories, onsets and rank order of the three hack mechanisms, and the trajectory of
    `cot_mentions_hack`, are what E1 is meant to predict rather than to have read.

    **This test is necessary and it is not sufficient, and that is now written down rather than
    assumed.** It watches one channel: naming the directory. It cannot see a file that reconstructs
    the protected quantity without referring to it, and three files in this tree did exactly that
    for a year while this test was green. `tests/test_sealed_guard.py` watches the other channel,
    the arithmetic, over every file at every extension. Both are kept because a reference and a
    reconstruction are different leaks and neither guard sees the other's.
    """
    offenders: list[str] = []
    for path in _python_files():
        rel = str(path.relative_to(REPO))
        if rel in SEALED_WRITERS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "SEALED" in text:
            offenders.append(rel)
    assert offenders == [], (
        "these modules name the sealed directory: "
        + ", ".join(offenders)
        + ". A per-mechanism late-window quantity that leaves e1_preflight/SEALED/ is a held-out "
        "target that has been read. If a decision genuinely needs one, that decision belongs in "
        "the analysis document under a frozen rule, not in code."
    )


@pytest.mark.skipif(not SEALED.exists(), reason="the preflight has not been run in this tree")
def test_the_sealed_directory_holds_counts_and_never_a_trajectory():
    """What may be recorded is that these columns exist and are populated. Not how they moved."""
    payload = json.loads((SEALED / "null_counts.json").read_text(encoding="utf-8"))
    assert payload
    for run in payload.values():
        assert set(run) == {"rh_always_equal", "rh_exit", "rh_conftest", "cot_mentions_hack"}
        for column in run.values():
            assert set(column) == {"present", "n_null"}
            assert isinstance(column["n_null"], int) or column["n_null"] is None


# ---------------------------------------------------------------------------
# Column enumeration, including the JSON a column listing would hide
# ---------------------------------------------------------------------------


@pytest.fixture
def frame():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "flag": [0, 1, 1, 0],
            "score": [0.0, 0.25, 0.75, 1.0],
            "constant": [3, 3, 3, 3],
            "name": ["a", "b", "a", "c"],
            "nested": ['{"exit": true, "conftest": false}'] * 3
            + ['{"exit": false, "conftest": true}'],
            "sometimes_null": [1.0, None, 3.0, None],
        }
    )


def test_a_json_string_column_has_its_nested_keys_enumerated(frame):
    """Trap four: a column holding serialised JSON hides its own fields from a column listing."""
    got = columns.enumerate_columns(frame)["nested"]
    assert set(got.json_keys) == {"exit", "conftest"}
    assert got.json_keys["exit"] == 4
    assert set(got.json_value_sets["exit"]) == {True, False}


def test_a_plain_string_column_reports_no_json_keys(frame):
    assert columns.enumerate_columns(frame)["name"].json_keys == {}


def test_nulls_are_counted_and_kept_out_of_the_range(frame):
    got = columns.enumerate_columns(frame)["sometimes_null"]
    assert got.n_null == 2
    assert got.minimum == 1.0 and got.maximum == 3.0


def test_a_constant_column_is_not_a_composition_candidate(frame):
    got = columns.enumerate_columns(frame)
    assert "constant" not in columns.numeric_candidates(got)
    assert "score" in columns.numeric_candidates(got)
    assert "name" not in columns.numeric_candidates(got)


def test_binary_and_quarter_valued_columns_are_told_apart(frame):
    got = columns.enumerate_columns(frame)
    assert got["flag"].is_binary
    assert not got["score"].is_binary


# ---------------------------------------------------------------------------
# Configuration adjudication, which must never turn on a filename
# ---------------------------------------------------------------------------


ADAPTER = {
    "r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
}


def test_all_linear_is_expanded_rather_than_compared_as_a_string():
    modules, expanded = configs.expand_target_modules("all-linear")
    assert expanded
    assert modules == frozenset(configs.ALL_LINEAR_OLMO)
    assert configs.expand_target_modules(["q_proj"]) == (frozenset({"q_proj"}), False)


def test_a_config_matching_on_every_compared_field_matches():
    verdict = configs.ConfigVerdict(
        "anything.yaml",
        configs.adjudicate(
            {
                "peft_config": {
                    "r": 32,
                    "lora_alpha": 32,
                    "lora_dropout": 0.0,
                    "target_modules": "all-linear",
                }
            },
            adapter=ADAPTER,
        ),
    )
    assert verdict.matches
    assert verdict.n_disagree == 0


def test_a_config_that_differs_on_dropout_is_reported_as_differing_on_dropout():
    """The concrete form of adjudicating by field: one wrong value, named, not a similarity score."""
    verdict = configs.ConfigVerdict(
        "grid_t1.0_g16_eh0.5.yaml",
        configs.adjudicate(
            {
                "peft_config": {
                    "r": 32,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                }
            },
            adapter=ADAPTER,
        ),
    )
    assert not verdict.matches
    assert {c.field for c in verdict.failures()} == {"peft.lora_dropout", "peft.target_modules"}


def test_a_weight_vector_that_cannot_reach_the_observed_reward_is_refuted_by_arithmetic():
    """The strongest disagreement available, and it does not depend on any fit."""
    comparisons = configs.adjudicate(
        {"reward_weights": [1.0, 4.0, 0.0], "peft_config": {}},
        adapter=ADAPTER,
        reward_ceiling=5.953125,
    )
    ceiling = next(c for c in comparisons if c.field == "reward_weights.max_attainable")
    assert ceiling.configured == 5.0
    assert ceiling.agrees is False


def test_a_weight_vector_that_can_reach_it_is_not_refuted():
    comparisons = configs.adjudicate(
        {"reward_weights": [1.0, 4.0, 1.0], "peft_config": {}},
        adapter=ADAPTER,
        reward_ceiling=5.953125,
    )
    assert next(c for c in comparisons if c.field == "reward_weights.max_attainable").agrees is True


def test_env_reward_weights_are_read_when_reward_weights_is_absent():
    comparisons = configs.adjudicate(
        {"env_reward_weights": {"apps": [1.0, 4.0], "mbpp": [1.0, 4.0]}, "peft_config": {}},
        adapter=ADAPTER,
        reward_ceiling=6.0,
    )
    assert (
        next(c for c in comparisons if c.field == "reward_weights.max_attainable").configured == 5.0
    )


def test_distinct_weightings_collapses_identical_vectors():
    """One line that settles whether the publisher varied the weights across its grid at all."""
    got = configs.distinct_reward_weightings(
        {
            "a.yaml": {"reward_weights": [1.0, 4.0]},
            "b.yaml": {"reward_weights": [1, 4]},
            "c.yaml": {"reward_weights": [2.0, 4.0]},
        }
    )
    assert got == {"1, 4": ["a.yaml", "b.yaml"], "2, 4": ["c.yaml"]}


# ---------------------------------------------------------------------------
# Artifact bookkeeping
# ---------------------------------------------------------------------------


def test_checkpoint_directories_are_read_from_the_file_listing():
    listing = [
        "README.md",
        "checkpoint-10/trainer_state.json",
        "checkpoint-400/x",
        "checkpoint-20/y",
        "notes/checkpoint-9",
    ]
    assert artifacts.checkpoint_directories(listing) == [10, 20, 400]


def test_a_run_that_stops_early_reports_a_shorter_checkpoint_list():
    """The companion run tops out at 390, and anything hard-coding 400 silently drops it."""
    listing = [f"checkpoint-{s}/trainer_state.json" for s in range(10, 400, 10)]
    steps = artifacts.checkpoint_directories(listing)
    assert steps[-1] == 390 and len(steps) == 39
    assert artifacts.checkpoint_stride(steps) == (10, True)


def test_an_uneven_stride_is_reported_as_uneven():
    assert artifacts.checkpoint_stride([10, 20, 45]) == (10, False)


def test_the_manifest_records_content_and_not_paths(tmp_path):
    art = artifacts.Artifact(
        key="k",
        repo_id="org/repo",
        repo_type="dataset",
        filename="data/x.parquet",
        revision="abc",
        path="/somewhere/local",
        n_bytes=10,
        sha256="deadbeef",
    )
    manifest = artifacts.manifest_from(
        {
            "run": {
                "model_repo": "org/m",
                "dataset_repo": "org/repo",
                "model_revision": "a",
                "dataset_revision": "b",
                "checkpoints": [10],
                "artifacts": {"dataset/x": art},
            }
        }
    )
    assert "path" not in manifest["run"]["files"]["dataset/x"]
    assert manifest["run"]["files"]["dataset/x"]["sha256"] == "deadbeef"


def test_a_changed_hash_is_reported_rather_than_swallowed(tmp_path):
    def build(sha: str):
        art = artifacts.Artifact("k", "org/repo", "dataset", "x.parquet", "abc", "/p", 10, sha)
        return {
            "run": {
                "model_repo": "org/m",
                "dataset_repo": "org/repo",
                "model_revision": "a",
                "dataset_revision": "b",
                "checkpoints": [],
                "artifacts": {"dataset/x": art},
            }
        }

    path = tmp_path / "hashes.json"
    artifacts.write_manifest(build("aaaa1111"), path)
    assert artifacts.verify_against_manifest(build("aaaa1111"), path) == []
    problems = artifacts.verify_against_manifest(build("bbbb2222"), path)
    assert len(problems) == 1 and "sha256" in problems[0]


def test_sha256_matches_hashlib_on_a_small_file(tmp_path):
    import hashlib

    path = tmp_path / "f.bin"
    path.write_bytes(b"the artifact is what it contains")
    assert artifacts.sha256_of(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_observed_groups_per_step_is_recorded_as_unadjudicable_rather_than_ignored():
    """The parameter was declared and never read, which is a comparison nobody could see was absent.

    It is recorded rather than dropped because the absence is the finding. A configuration's own
    fields do not predict the number of prompt groups per step: the count is the per-device batch
    size times the accumulation steps times the device count, divided by `num_generations`, and the
    device count lives in a launcher configuration with a different schema. `adjudicate` compares
    one configuration mapping at a time, so no mapping it is ever handed carries both. Reporting
    that as an incomparable row is a different statement from not reporting it, and only one of the
    two survives a reader asking why the batch geometry was never checked.
    """
    comparisons = configs.adjudicate(
        {"num_generations": 16},
        adapter={},
        observed_group_size=16,
        observed_groups_per_step=4,
    )
    by_field = {c.field: c for c in comparisons}
    assert "groups_per_step" in by_field
    row = by_field["groups_per_step"]
    assert row.observed == 4
    assert row.configured is None
    assert row.agrees is None
    assert "no configuration field predicts" in row.note

    # And it is absent rather than null when the caller has nothing to compare against, so a run
    # that could not measure it does not report an incomparable row it never had.
    without = configs.adjudicate({"num_generations": 16}, adapter={}, observed_group_size=16)
    assert "groups_per_step" not in {c.field for c in without}
