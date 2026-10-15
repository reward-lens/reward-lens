"""P-CONNECT's acceptance: four callable shapes recognised by fixture, and four refusals.

Every shape here is established by running `inspect.signature` over a fixture callable that ships
in `reward_lens.examples.connect_fixture`. No test imports `trl`, `verifiers`, `transformers` or
`verl`, and one test asserts that detection itself imports none of them either: D-65 forbids
recognising a shape by importing the framework that defines it.

The `verifiers` discriminator is checked against a pinned upstream source fixture. The two
regions the commission names (`rubric.py:106-109` for the group test, `:200-211` for the
name-subset binding) are lifted out of that file with `ast` and executed, so the comparison is
against the framework's code rather than against a retyping of it.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import get_origin

import pytest

from reward_lens import errors
from reward_lens.contracts import EntryProvenance, RewardLensError
from reward_lens.examples import connect_fixture
from reward_lens.examples.connect_fixture import rewards, silent, unsupported, verifiers_like
from reward_lens.examples.connect_fixture import shapes as fixture_shapes
from reward_lens.graders.base import Grader, Manifest, ValidityFinding
from reward_lens.graders.connect import (
    DECLARED_SHAPES,
    ConnectorCapabilityState,
    ConnectorSelectionRanking,
    Detection,
    PretrainedModelBinding,
    Shape,
    ShapeUnsupported,
    bind,
    bound_names,
    detect,
    detect_project,
    is_group_rubric,
    overrides_from_project_file,
    render_detect_transcript,
)
from reward_lens.outcome import TaskValidity, UnqualifiedReason, unqualified

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "fleet" / "golden" / "wave-2" / "connect" / "init-detect.txt"
FROZEN_TARGET = REPO / "fleet" / "golden" / "target" / "init-detect.txt"
VENDORED_RUBRIC = REPO / "tests" / "fixtures" / "verifiers" / "rubric.py"
VENDORED_RUBRIC_SHA256 = "979c163bf7c9da86822738a66c4b04341652c82a652a4b502a8ae10c9c455604"


# --- the four shapes, one fixture callable each ---------------------------------------------------


def test_single_fn_is_recognised_from_its_fixture_signature() -> None:
    found = detect(fixture_shapes.single_fn)
    assert found.shape is Shape.SINGLE_FN
    assert found.declared == "plain"
    assert found.entry == "reward_lens.examples.connect_fixture.shapes:single_fn"
    assert any("inspect.signature" in line for line in found.evidence)
    assert any("(prompt, completion, **kw)" in line for line in found.evidence)


def test_batch_fn_is_recognised_from_its_fixture_signature() -> None:
    found = detect(fixture_shapes.batch_fn)
    assert found.shape is Shape.BATCH_FN
    assert found.declared == "trl"
    assert any("prompts" in line and "completions" in line for line in found.evidence)


def test_async_trl_is_recognised_by_iscoroutinefunction_on_its_own_fixture() -> None:
    found = detect(fixture_shapes.async_trl_fn)
    assert found.shape is Shape.ASYNC_TRL
    assert found.declared == "trl"
    assert found.awaits is True
    assert any("iscoroutinefunction" in line for line in found.evidence)
    # the sync batch fixture is a different callable, so the async branch is not the batch branch
    assert detect(fixture_shapes.batch_fn).shape is Shape.BATCH_FN


def test_pretrained_model_is_recognised_from_a_stub_not_a_download() -> None:
    found = detect(fixture_shapes.scoring_head_stub())
    assert found.shape is Shape.PRETRAINED_MODEL
    assert any("forward(input_ids" in line for line in found.evidence)
    assert any("logits" in line for line in found.evidence)


def test_pretrained_model_binding_scores_an_actual_local_head_with_declared_transform() -> None:
    """The connector routes a detected head through the reviewed adapter, not a generic row."""
    from reward_lens.graders import reward_model

    model, tokenizer = reward_model.tiny_local_model({"good": 2.0})
    # Detection only reads the loaded head's transformer-style surface.  No model is fetched.
    model.config = object()
    binding = PretrainedModelBinding(
        tokenizer=tokenizer,
        template="{prompt} {response}",
        preprocessing=reward_model.Preprocessing(max_length=8, batch_size=1),
        transform=lambda value: value * 3.0 + 1.0,
        direction=reward_model.ScoreDirection.LOWER_IS_BETTER,
    )

    grader = bind(detect(model), pretrained=binding)
    envelope = grader.score({"prompt": "question"}, "good")

    assert isinstance(grader, reward_model.RewardModel)
    assert envelope.verdict == "scored"
    assert envelope.observations["raw_score"] == 2.0
    assert envelope.observations["transformed_score"] == 7.0
    assert envelope.score == 7.0
    assert envelope.observations["direction"] == "lower_is_better"
    assert envelope.observations["preprocessing"] == binding.preprocessing.to_dict()


def test_pretrained_model_without_its_explicit_binding_refuses_before_generic_call() -> None:
    found = detect(fixture_shapes.scoring_head_stub())

    with pytest.raises(RewardLensError) as caught:
        bind(found)

    assert caught.value.code == "RL0003"
    assert caught.value.exit_code == 4
    assert caught.value.context["field"] == "reward.pretrained"
    assert "reward.pretrained" in caught.value.message


def test_non_pretrained_bind_keeps_the_optional_reward_model_runtime_unimported() -> None:
    program = (
        "import sys\n"
        "from reward_lens.graders.connect import bind, detect\n"
        "from reward_lens.examples.connect_fixture.shapes import single_fn\n"
        "bind(detect(single_fn))\n"
        "assert 'reward_lens.graders.reward_model' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=str(REPO)
    )
    assert done.returncode == 0, done.stderr


def test_each_shape_has_a_fixture_of_its_own() -> None:
    targets = {
        Shape.SINGLE_FN: fixture_shapes.single_fn,
        Shape.BATCH_FN: fixture_shapes.batch_fn,
        Shape.ASYNC_TRL: fixture_shapes.async_trl_fn,
        Shape.PRETRAINED_MODEL: fixture_shapes.scoring_head_stub(),
    }
    assert {shape: detect(target).shape for shape, target in targets.items()} == {
        shape: shape for shape in targets
    }
    entries = {detect(target).entry for target in targets.values()}
    assert len(entries) == 4


def test_detection_imports_no_framework() -> None:
    """Forbidden: a framework imported to recognise its shape. Checked in a clean interpreter."""
    program = (
        "import sys\n"
        "from reward_lens.graders.connect import detect\n"
        "from reward_lens.examples.connect_fixture import shapes, unsupported\n"
        "for target in (shapes.single_fn, shapes.batch_fn, shapes.async_trl_fn,\n"
        "               shapes.scoring_head_stub(), unsupported.compute_score,\n"
        "               unsupported.StepEnvironment()):\n"
        "    detect(target)\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in\n"
        "                {'trl', 'verifiers', 'transformers', 'verl', 'torch'})\n"
        "print(','.join(leaked))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=str(REPO), check=True
    )
    assert done.stdout.strip() == ""


def test_nothing_asks_which_framework_before_looking() -> None:
    """D-65's forbidden surface: no question to the user stands between `detect` and the signature."""
    for fn in (detect, bind, render_detect_transcript, detect_project):
        source = inspect.getsource(fn)
        assert "input(" not in source
        assert "?" not in "".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )


# --- the golden connect transcript ----------------------------------------------------------------


def _fixture_project(tmp_path: Path) -> Path:
    root = tmp_path / "my-env"
    connect_fixture.write(root)
    return root


def _golden_command_and_body(path: Path) -> tuple[list[str], str]:
    """A golden's `$ ` line as argv, and the output block under it.

    Read the way `fleet/golden/run_demo.py:parse_golden` reads it: the first line is the
    invocation and everything after it is what the command has to print.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0].startswith("$ reward-lens "), f"{path} does not open with an invocation line"
    return lines[0][len("$ reward-lens ") :].split(), "\n".join(lines[1:])


def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run this checkout's CLI with the interpreter executing the test suite."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("REWARD_LENS_")}
    environment.pop("PYTHONPATH", None)
    environment.update(
        NO_COLOR="1",
        REWARD_LENS_NO_PROGRESS="1",
        REWARD_LENS_COLOR="never",
        REWARD_LENS_FORMAT="text",
        COLUMNS="100",
        TERM="dumb",
    )
    return subprocess.run(
        [sys.executable, "-m", "reward_lens.cli.main", *argv],
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_init_detect_produces_the_golden_connect_transcript(tmp_path: Path) -> None:
    """The verb writes the golden, not the renderer: the command in the file is the one that runs."""
    connect_fixture.write(tmp_path / "my-env")
    argv, expected = _golden_command_and_body(GOLDEN)
    assert argv == ["init", "./my-env", "--detect"]
    done = _run(argv, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout == expected
    assert (tmp_path / "my-env" / "rewardlens.yaml").is_file()


def test_the_transcript_counts_come_from_the_fixture(tmp_path: Path) -> None:
    root = _fixture_project(tmp_path)
    project = detect_project(root)
    assert [d.entry for d in project.callables] == [
        "rewards:correctness_reward",
        "rewards:format_reward",
        "rewards:length_penalty",
    ]
    assert [d.declared for d in project.callables] == ["trl", "trl", "plain"]
    assert project.task_rows == 312
    assert project.task_fields == ["prompt", "answer", "tests"]
    assert project.outcome is None
    assert (root / "data" / "train.jsonl").read_text().count("\n") == 312


def test_the_transcript_says_the_same_thing_as_the_frozen_target(tmp_path: Path) -> None:
    """The controller promotes my transcript to the target. Substance must already agree."""
    root = _fixture_project(tmp_path)
    text = "$ reward-lens init ./my-env --detect\n\n" + render_detect_transcript(
        detect_project(root), displayed="./my-env"
    )
    squash = lambda s: "\n".join(" ".join(line.split()) for line in s.splitlines())  # noqa: E731
    assert squash(text) == squash(FROZEN_TARGET.read_text())


def test_init_detect_writes_the_project_file_the_transcript_shows(tmp_path: Path) -> None:
    root = _fixture_project(tmp_path)
    project = detect_project(root)
    written = project.write_project_file(displayed="./my-env")
    assert written == root / "rewardlens.yaml"
    body = written.read_text()
    assert "watch: [none_to_nan, exception_to_zero, bool_as_score]" in body
    assert "outcome: null" in body
    shown = render_detect_transcript(project, displayed="./my-env")
    for line in body.splitlines():
        assert ("    " + line) in shown


# --- the overrides, and the refusal a bad one makes ----------------------------------------------


def test_reward_shape_overrides_detection() -> None:
    found = detect(fixture_shapes.single_fn)
    assert found.shape is Shape.SINGLE_FN
    grader = bind(found, overrides={"shape": "trl"})
    assert grader.describe().shape == "trl"
    assert grader.detection.declared == "trl"
    assert grader.detection.shape is Shape.BATCH_FN
    assert any("reward.shape" in line for line in grader.detection.evidence)


def test_reward_entry_overrides_detection() -> None:
    found = detect(fixture_shapes.single_fn)
    grader = bind(
        found,
        overrides={"entry": "reward_lens.examples.connect_fixture.shapes:batch_fn"},
    )
    assert grader.detection.entry == "reward_lens.examples.connect_fixture.shapes:batch_fn"
    assert grader.detection.shape is Shape.BATCH_FN
    assert any("reward.entry" in line for line in grader.detection.evidence)


def test_rewardlens_yaml_shape_and_entry_survive_read_write_bind_round_trip(tmp_path: Path) -> None:
    """The two frozen reward keys remain configuration, rather than detect-only metadata."""
    project_file = tmp_path / "rewardlens.yaml"
    project_file.write_text(
        "reward:\n  shape: trl\n  entry: reward_lens.examples.connect_fixture.shapes:batch_fn\n",
        encoding="utf-8",
    )

    overrides = overrides_from_project_file(project_file)
    grader = bind(detect(fixture_shapes.single_fn), overrides=overrides)

    assert overrides == {
        "shape": "trl",
        "entry": "reward_lens.examples.connect_fixture.shapes:batch_fn",
    }
    assert grader.detection.declared == "trl"
    assert grader.detection.entry.endswith(":batch_fn")
    assert grader.describe().shape == "trl"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"shape": "megatron"}, "reward.shape"),
        ({"shape": ""}, "reward.shape"),
        ({"entry": "no-colon-here"}, "reward.entry"),
        ({"entry": "reward_lens.examples.connect_fixture.shapes:absent"}, "reward.entry"),
    ],
)
def test_a_bad_override_refuses_at_exit_4_naming_the_field(overrides: dict, field: str) -> None:
    found = detect(fixture_shapes.single_fn)
    with pytest.raises(RewardLensError) as caught:
        bind(found, overrides=overrides)
    assert caught.value.code == "RL0003"
    assert caught.value.exit_code == 4
    assert field in caught.value.message
    assert caught.value.context["field"] == field


def test_the_override_vocabulary_is_d65s_four_declared_values() -> None:
    assert DECLARED_SHAPES == ("plain", "trl", "verifiers", "inspect")
    found = detect(fixture_shapes.single_fn)
    for declared in DECLARED_SHAPES:
        assert bind(found, overrides={"shape": declared}).detection.declared == declared


# --- the three silent failure modes of D-65, each with an executed witness ------------------------


def test_none_to_nan_is_a_finding_with_an_executed_witness() -> None:
    found = detect(silent.returns_none)
    finding = next(f for f in found.warnings if f.code == "RL0213")
    assert isinstance(finding, ValidityFinding)
    assert finding.rule == "none_to_nan"
    assert finding.trainer == "trl"
    assert "nan" in finding.trainer_behaviour.lower()
    assert finding.witness["returned"] == [1.0, None]
    assert finding.witness["index"] == 1
    assert finding.witness["called"] is True


def test_exception_to_zero_is_a_finding_with_an_executed_witness() -> None:
    found = detect(silent.raises_inside)
    finding = next(f for f in found.warnings if f.code == "RL0214")
    assert finding.rule == "exception_to_zero"
    assert finding.trainer == "verifiers"
    assert "0.0" in finding.trainer_behaviour
    assert finding.witness["exception"] == "ZeroDivisionError"
    assert finding.witness["would_have_scored"] == 0.0
    assert finding.witness["called"] is True


def test_bool_as_score_is_a_finding_with_an_executed_witness() -> None:
    found = detect(silent.returns_bool)
    finding = next(f for f in found.warnings if f.code == "RL0215")
    assert finding.rule == "bool_as_score"
    assert finding.witness["returned"] is True
    assert finding.witness["float_of_it"] == 1.0
    assert finding.witness["called"] is True


def test_the_three_modes_are_the_watch_list_of_the_transcript(tmp_path: Path) -> None:
    root = _fixture_project(tmp_path)
    shown = render_detect_transcript(detect_project(root), displayed="./my-env")
    for rule in ("none_to_nan", "exception_to_zero", "bool_as_score"):
        assert rule in shown
    codes = {"RL0213", "RL0214", "RL0215"}
    seen = set()
    for target in (silent.returns_none, silent.raises_inside, silent.returns_bool):
        seen |= {f.code for f in detect(target).warnings}
    assert codes <= seen


def test_a_clean_fixture_raises_no_silent_mode_finding() -> None:
    assert detect(fixture_shapes.single_fn).warnings == []


# --- the verifiers discriminator, against the framework's own source -----------------------------


def _framework_is_group_func():
    """`Rubric._is_group_func` (rubric.py:106-123) lifted out and executed, not retyped."""
    tree = ast.parse(VENDORED_RUBRIC.read_text(), str(VENDORED_RUBRIC))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_is_group_func"
    )
    node.decorator_list = []
    node.returns = None
    node.args.args = [a for a in node.args.args if a.arg != "self"]
    for arg in node.args.args:
        arg.annotation = None
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"inspect": inspect, "get_origin": get_origin}
    exec(compile(module, str(VENDORED_RUBRIC), "exec"), namespace)
    return namespace["_is_group_func"]


def _framework_bound_names(fn, available: dict) -> set[str]:
    """The binding of `_call_individual_reward_func` (rubric.py:200-211), executed as written."""
    tree = ast.parse(VENDORED_RUBRIC.read_text(), str(VENDORED_RUBRIC))
    enclosing = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_call_individual_reward_func"
    )
    branch = next(n for n in ast.walk(enclosing) if isinstance(n, ast.If))
    assign = next(
        n
        for n in ast.walk(enclosing)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "allowed"
    )
    namespace = {"sig": inspect.signature(fn), "merged": dict(available)}

    def run(node):
        expression = ast.Expression(body=node)
        ast.fix_missing_locations(expression)
        return eval(compile(expression, str(VENDORED_RUBRIC), "eval"), namespace)  # noqa: S307

    return set(available) if run(branch.test) else set(run(assign.value))


def test_the_pinned_verifiers_source_fixture_matches_its_upstream_digest() -> None:
    assert VENDORED_RUBRIC.is_file(), (
        f"the pinned verifiers source fixture is missing at {VENDORED_RUBRIC}"
    )
    assert hashlib.sha256(VENDORED_RUBRIC.read_bytes()).hexdigest() == VENDORED_RUBRIC_SHA256


@pytest.mark.parametrize(
    "fn",
    [
        verifiers_like.individual_reward,
        verifiers_like.group_reward,
        fixture_shapes.single_fn,
        fixture_shapes.batch_fn,
        rewards.correctness_reward,
        rewards.length_penalty,
    ],
)
def test_is_group_rubric_matches_the_framework_on_the_same_callable(fn) -> None:
    assert is_group_rubric(fn) == _framework_is_group_func()(fn)


@pytest.mark.parametrize(
    "fn",
    [
        verifiers_like.individual_reward,
        verifiers_like.group_reward,
        fixture_shapes.single_fn,
        rewards.correctness_reward,
    ],
)
def test_bound_names_matches_the_framework_on_the_same_callable(fn) -> None:
    available = {
        "prompt": "p",
        "completion": "c",
        "answer": "a",
        "state": {},
        "prompts": ["p"],
        "completions": ["c"],
        "answers": ["a"],
        "task": "t",
        "info": {},
    }
    assert bound_names(fn, available) == _framework_bound_names(fn, available)


def test_the_discriminator_splits_the_verifiers_fixture_both_ways() -> None:
    assert is_group_rubric(verifiers_like.group_reward) is True
    assert is_group_rubric(verifiers_like.individual_reward) is False
    assert detect(verifiers_like.group_reward).declared == "verifiers"
    assert detect(verifiers_like.group_reward).shape is Shape.BATCH_FN
    assert detect(verifiers_like.individual_reward).shape is Shape.SINGLE_FN


def test_bound_names_takes_everything_when_the_callable_has_var_keyword() -> None:
    available = {"prompt": "p", "completion": "c", "nobody_asked": 1}
    assert bound_names(fixture_shapes.var_keyword_reward, available) == set(available)


# --- the refusals: verl and a step() environment --------------------------------------------------


def test_verl_compute_score_is_refused_at_exit_5_naming_the_shape_and_the_page() -> None:
    """`detect` returns under the frozen call; RL0710 is raised where the detection is used."""
    found = detect(unsupported.compute_score)
    with pytest.raises(ShapeUnsupported) as caught:
        bind(found, overrides=None)
    error = caught.value
    assert error.code == "RL0710"
    assert error.exit_code == 5
    assert "compute_score" in error.message
    assert "verl" in error.message
    assert (
        error.message
        == errors.make(
            "RL0710", shape="compute_score", entry=found.entry, detail=found.reason
        ).message
    )
    assert (
        error.remediation
        == errors.make(
            "RL0710", shape="compute_score", entry=found.entry, detail=found.reason
        ).remediation
    )


def test_a_step_environment_is_refused_at_exit_5_naming_the_shape_and_the_page() -> None:
    found = detect(unsupported.StepEnvironment())
    with pytest.raises(ShapeUnsupported) as caught:
        bind(found, overrides=None)
    error = caught.value
    assert error.code == "RL0710"
    assert error.exit_code == 5
    assert "step()" in error.message
    assert (
        error.message
        == errors.make("RL0710", shape="step()", entry=found.entry, detail=found.reason).message
    )


def test_an_unsupported_detection_carries_the_reason_and_the_page() -> None:
    """Interfaces section 1 freezes `detect(target) -> Detection`, so UNSUPPORTED is reachable."""
    found = detect(unsupported.compute_score)
    assert found.shape is Shape.UNSUPPORTED
    assert found.reason
    assert found.docs_page == "docs/errors/RL0710.md"
    assert found.warnings == []
    assert "refuse" not in inspect.signature(detect).parameters


def test_binding_an_unsupported_detection_refuses_rather_than_guessing() -> None:
    found = detect(unsupported.StepEnvironment())
    with pytest.raises(ShapeUnsupported) as caught:
        bind(found, overrides=None)
    assert caught.value.exit_code == 5


def test_init_detect_reports_rl0710_when_the_project_holds_a_shape_it_will_not_adapt(
    tmp_path: Path,
) -> None:
    """The second place the refusal has to reach a reader: the verb, on a real directory."""
    project = tmp_path / "verl-env"
    (project / "data").mkdir(parents=True)
    (project / "rewards.py").write_text(
        "def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:\n"
        "    return 1.0 if ground_truth in solution_str else 0.0\n"
    )
    (project / "data" / "train.jsonl").write_text('{"prompt": "p", "answer": "a"}\n')
    done = _run(["init", "./verl-env", "--detect"], cwd=tmp_path)
    assert done.returncode == 5, done.stdout
    assert "RL0710" in done.stderr
    assert "reward-lens explain RL0710" in done.stderr
    assert not (project / "rewardlens.yaml").exists()


# --- bind returns a wave-1 Grader -----------------------------------------------------------------


def test_bind_returns_a_grader_whose_manifest_records_the_shape() -> None:
    grader: Grader = bind(detect(fixture_shapes.single_fn), overrides=None)
    manifest = grader.describe()
    assert isinstance(manifest, Manifest)
    assert manifest.shape == "plain"
    assert manifest.family == "connected"
    assert manifest.source_digest.startswith("sha256:")
    assert grader.validate_input({"prompt": "p"}, "c") is None
    refused = grader.validate_input({}, "c")
    assert refused is not None and refused.field == "prompt"


def test_a_bound_grader_scores_through_the_protocol() -> None:
    grader = bind(detect(fixture_shapes.single_fn), overrides=None)
    envelope = grader.score({"prompt": "abc"}, "abcd", sandbox=None, limits=None)
    assert envelope.verdict == "scored"
    assert envelope.score == pytest.approx(0.75)
    assert envelope.manifest_digest == grader.describe().digest()
    assert any("in this process" in limitation for limitation in envelope.limitations)


def test_a_bound_batch_grader_scores_one_row_of_its_batch() -> None:
    grader = bind(detect(fixture_shapes.batch_fn), overrides=None)
    envelope = grader.score({"prompt": "abc"}, "abcd", sandbox=None, limits=None)
    assert envelope.verdict == "scored"
    assert envelope.score is not None


def test_manifest_and_provenance_use_family_and_preserve_calling_convention() -> None:
    grader = bind(detect(fixture_shapes.async_trl_fn), overrides=None)
    envelope = grader.score({"prompt": "abc"}, "abcd", sandbox=None, limits=None)

    assert grader.describe().shape == "trl"
    assert envelope.provenance["shape"] == "trl"
    assert envelope.provenance["convention"] == "trl_async"


@pytest.mark.parametrize(
    ("target", "code", "context"),
    [
        (silent.returns_none, "RL0213", {}),
        (silent.raises_inside, "RL0214", {"exception": "ZeroDivisionError"}),
        (silent.returns_bool, "RL0215", {}),
    ],
)
def test_connector_finding_messages_are_the_catalogue_messages(target, code, context) -> None:
    found = detect(target)
    finding = next(finding for finding in found.warnings if finding.code == code)
    assert finding.message == errors.make(code, entry=found.entry, **context).message


def _unqualified_outcome():
    return unqualified(
        item_id="task-timeout",
        reason=UnqualifiedReason.TIMEOUT,
        protocol=None,
        provenance=EntryProvenance(
            started="2026-09-26T00:00:00Z", duration_s=0.1, sandbox_tier="L3", offline=True
        ),
        scope="evaluator_comparison",
        task_validity=TaskValidity.UNKNOWN,
    )


def test_connector_preserves_an_unqualified_outcome_and_rejects_boolean_like_payloads() -> None:
    grader = bind(detect(fixture_shapes.single_fn), overrides=None)
    envelope = grader.score({"prompt": "abc"}, "abcd", sandbox=None, limits=None)

    recorded = grader.attach_outcome(envelope, _unqualified_outcome())

    assert recorded.observations["outcome"]["state"] == "UNQUALIFIED"
    assert recorded.observations["outcome"]["reason"] == "timeout"
    assert recorded.observations["outcome"]["task_validity"] == "UNKNOWN"
    with pytest.raises(TypeError, match="Outcome"):
        grader.attach_outcome(envelope, False)  # type: ignore[arg-type]


def test_connector_capability_probe_keeps_mutation_absence_scoped_to_its_check() -> None:
    grader = bind(detect(fixture_shapes.single_fn), overrides=None)

    report = grader.probe_capabilities(
        task={"prompt": "abc"}, baseline_response="abcd", candidate_response="a"
    )

    assert report.status("source_mutation").state is ConnectorCapabilityState.ABSENT
    assert report.status("black_box_repeatability").state is ConnectorCapabilityState.AVAILABLE
    assert report.status("selection").state is ConnectorCapabilityState.AVAILABLE
    assert report.status("source_mutation").evidence == ()
    assert len(report.status("black_box_repeatability").evidence) == 2
    assert len(report.status("selection").evidence) == 2


def test_stateful_repeatability_is_available_but_records_a_false_comparison() -> None:
    class AlternatingReward:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, prompt, completion, **kwargs):
            self.calls += 1
            return float(self.calls)

    grader = bind(detect(AlternatingReward(), probe_entry=False), overrides=None)
    report = grader.probe_capabilities(
        task={"prompt": "p"}, baseline_response="same", candidate_response="candidate"
    )

    repeatability = report.status("black_box_repeatability")
    selection = report.status("selection")
    assert repeatability.state is ConnectorCapabilityState.AVAILABLE
    assert repeatability.repeatable is False
    assert [envelope.score for envelope in repeatability.evidence] == [1.0, 2.0]
    assert selection.state is ConnectorCapabilityState.AVAILABLE
    assert selection.ranking is ConnectorSelectionRanking.CANDIDATE_HIGHER
    assert [envelope.score for envelope in selection.evidence] == [1.0, 3.0]


def test_selection_probe_records_an_observed_score_tie_without_an_outcome_claim() -> None:
    grader = bind(detect(fixture_shapes.single_fn), overrides=None)
    report = grader.probe_capabilities(
        task={"prompt": "abc"}, baseline_response="abcd", candidate_response="abcd"
    )

    selection = report.status("selection")
    assert selection.state is ConnectorCapabilityState.AVAILABLE
    assert selection.ranking is ConnectorSelectionRanking.TIE
    assert [envelope.score for envelope in selection.evidence] == [0.75, 0.75]


def test_a_bound_grader_reports_the_silent_mode_it_meets_as_a_finding() -> None:
    grader = bind(detect(silent.returns_bool), overrides=None)
    envelope = grader.score({"prompt": "abc"}, "abcd", sandbox=None, limits=None)
    assert {f.code for f in envelope.findings} == {"RL0215"}
    assert envelope.findings[0].witness["called"] is True


# --- the Detection shape itself -------------------------------------------------------------------


def test_detection_is_the_frozen_shape_of_interfaces_section_1() -> None:
    found = detect(fixture_shapes.single_fn)
    assert isinstance(found, Detection)
    assert isinstance(found.shape, Shape)
    assert isinstance(found.entry, str)
    assert isinstance(found.evidence, list) and all(isinstance(e, str) for e in found.evidence)
    assert isinstance(found.warnings, list)
    assert [s.name for s in Shape] == [
        "SINGLE_FN",
        "BATCH_FN",
        "ASYNC_TRL",
        "PRETRAINED_MODEL",
        "UNSUPPORTED",
    ]


def test_detect_accepts_a_string_entry_and_a_path() -> None:
    by_string = detect("reward_lens.examples.connect_fixture.shapes:batch_fn")
    assert by_string.shape is Shape.BATCH_FN
    by_path = detect(Path(rewards.__file__))
    assert by_path.shape is Shape.BATCH_FN
    assert by_path.entry == "rewards:correctness_reward"


# --- TRL's keyword-bound forms, and the inspect family (attempt 2) ---------------------------------


@pytest.mark.parametrize(
    "fixture",
    [
        fixture_shapes.trl_declares_only_what_it_uses,
        fixture_shapes.trl_in_any_order,
        fixture_shapes.trl_keyword_only,
    ],
    ids=["declares_only_what_it_uses", "in_any_order", "keyword_only"],
)
def test_every_trl_keyword_form_is_read_as_trl_and_probed_without_a_fabricated_finding(
    fixture,
) -> None:
    """TRL binds by keyword, so declaring only what you use is the convention, not a departure."""
    found = detect(fixture)
    assert found.declared == "trl"
    assert found.shape is Shape.BATCH_FN
    assert found.warnings == []


def test_a_trl_reward_that_declares_only_completions_scores_rather_than_erroring() -> None:
    grader = bind(detect(fixture_shapes.trl_declares_only_what_it_uses), overrides=None)
    envelope = grader.score({"prompt": "two plus two"}, "four")
    assert envelope.verdict == "scored"
    assert envelope.score == 1.0
    assert envelope.findings == ()


def test_the_inspect_family_is_reachable_from_a_fixture_that_detects_as_such() -> None:
    """`state`/`target` is inspect's pair, and it is read before the verifiers name list."""
    found = detect(fixture_shapes.inspect_scorer)
    assert found.declared == "inspect"
    assert found.awaits is True
    assert found.warnings == []


def test_a_verifiers_signature_is_still_verifiers_and_not_trl() -> None:
    """The order of the checks does not cost the family that shares `prompts` and `completions`."""
    assert detect(verifiers_like.individual_reward).declared == "verifiers"
    assert detect(verifiers_like.group_reward).declared == "verifiers"
