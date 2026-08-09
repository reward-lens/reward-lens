"""`runs`, `describe`, `init`, identifier validation, and the seam to `reward_lens.api`."""

from __future__ import annotations

import json
import pathlib

import pytest

from reward_lens.cli.registry import ORDER

CONTRACTED = list(ORDER)


# --- runs -------------------------------------------------------------------------------------

def test_runs_list_shows_a_run_this_machine_holds(cli, state_home):
    from reward_lens.store import RunDir

    run_id = RunDir.create(project=None, name="nightly").run_id
    listed = cli("runs", "list", "--format", "json")
    assert listed.exit_code == 0
    ids = [row["run_id"] for row in json.loads(listed.stdout)["runs"]]
    assert run_id in ids


def test_runs_show_prints_a_resume_command_that_runs(cli, tmp_path, state_home):
    """The printed line is a command, not a description of one, so the test runs it.

    It needs a project to run in: a run whose manifest names one prints that path in the resume
    command, and `audit` on a directory that holds no `rewardlens.yaml` would rightly refuse.
    """
    from reward_lens.store import RunDir

    project = tmp_path / "project"
    assert cli("init", "--example", "code-reward", str(project), "--format", "json").exit_code == 0
    run_id = RunDir.create(project=str(project), name="nightly").run_id
    shown = cli("runs", "show", run_id, "--format", "text")
    assert shown.exit_code == 0
    resume = [line.strip() for line in shown.stdout.splitlines() if line.strip().startswith("reward-lens ")]
    assert resume, shown.stdout
    again = cli(*resume[0].split()[1:], "--dry-run", "--format", "json")
    assert again.exit_code in (0, 2), again.stderr


def test_runs_rm_removes_the_run_once_the_caller_has_said_yes(cli, state_home):
    from reward_lens.store import RunDir

    run_id = RunDir.create(project=None, name="scratch").run_id
    assert cli("runs", "rm", run_id, "--yes", "--format", "json").exit_code == 0
    assert cli("runs", "show", run_id, "--format", "json").exit_code == 4


def test_runs_show_on_an_unknown_id_is_a_usage_error(cli, state_home):
    run = cli("runs", "show", "run-ffffffffffff", "--format", "json")
    assert run.exit_code == 4
    assert run.stderr.startswith("error: ")


# --- describe ---------------------------------------------------------------------------------

def test_describe_emits_the_versioned_command_tree(cli):
    run = cli("describe")
    assert run.exit_code == 0
    tree = json.loads(run.stdout)
    assert tree["schema_version"] == "command-tree/1.0"
    assert [command["name"] for command in tree["commands"]] == CONTRACTED
    for command in tree["commands"]:
        assert command["summary"]
        assert isinstance(command["parameters"], list)
        assert command["output"] and command["side_effect"]


def test_describe_round_trips(cli):
    from reward_lens.cli import describe

    printed = json.loads(cli("describe").stdout)
    assert describe.tree() == printed
    assert json.loads(json.dumps(printed)) == printed
    assert describe.from_dict(printed) == printed


def test_describe_names_the_env_twin_of_every_flag_that_has_one(cli):
    tree = json.loads(cli("describe").stdout)
    audit = next(c for c in tree["commands"] if c["name"] == "audit")
    twins = {p["name"]: p.get("env") for p in audit["parameters"]}
    assert twins["--format"] == "REWARD_LENS_FORMAT"
    assert twins["--offline"] == "REWARD_LENS_OFFLINE"
    assert twins["--max-budget-usd"] == "REWARD_LENS_MAX_BUDGET_USD"


# --- identifiers, D-24 ------------------------------------------------------------------------

BAD = {
    "control-character": "run-000\x07111",
    "path-traversal": "../../etc/passwd",
    "double-encoded": "run-%252e%252e%252fetc",
}

# A null byte cannot cross a subprocess argv at all: CPython raises `ValueError: embedded null
# byte` in the exec path, so the CLI is never reached and the subprocess fixture can say nothing
# about the refusal. The case is real, so it runs in process instead.
NULL_BYTE = "run-000\x00111"


@pytest.mark.parametrize("kind", sorted(BAD))
def test_a_bad_identifier_is_refused_and_the_field_is_named(cli, state_home, kind):
    run = cli("runs", "show", BAD[kind], "--format", "json")
    assert run.exit_code == 4, (kind, run.stderr)
    assert "RL0002" in run.stderr, (kind, run.stderr)
    assert "run_id" in run.stderr, (kind, run.stderr)


def test_a_null_byte_in_an_identifier_is_refused_and_the_field_is_named(invoke, state_home):
    run = invoke("runs", "show", NULL_BYTE, "--format", "json")
    assert run.exit_code == 4, run.stderr
    assert "RL0002" in run.stderr, run.stderr
    assert "run_id" in run.stderr, run.stderr


def test_a_subprocess_cannot_carry_a_null_byte_at_all(cli, state_home):
    """Why the case above is in process: the refusal is the operating system's, not the CLI's."""
    with pytest.raises(ValueError, match="null byte"):
        cli("runs", "show", NULL_BYTE, "--format", "json")


def test_a_good_identifier_passes(state_home):
    from reward_lens.cli.ids import validate_identifier

    assert validate_identifier("run-0123456789ab", field="run_id") == "run-0123456789ab"


# --- init -------------------------------------------------------------------------------------

def test_init_example_writes_the_example(cli, tmp_path):
    dest = tmp_path / "demo"
    run = cli("init", "--example", "code-reward", str(dest), "--format", "json")
    assert run.exit_code == 0, run.stderr
    assert (dest / "rewardlens.yaml").exists()
    assert (dest / "grader.py").exists()


def test_init_with_an_unknown_example_names_the_ones_that_exist(cli, tmp_path):
    run = cli("init", "--example", "no-such", str(tmp_path / "d"), "--format", "json")
    assert run.exit_code == 4
    assert "code-reward" in run.stderr


def test_init_detect_connects_the_reward_system_already_here(cli, tmp_path):
    """C-015: the wave-1 refusal is retired. `--detect` now reads the directory and reports."""
    from reward_lens.examples import connect_fixture

    connect_fixture.write(tmp_path / "my-env")
    run = cli("init", "./my-env", "--detect", cwd=tmp_path, env={"REWARD_LENS_FORMAT": "text"})
    assert run.exit_code == 0, run.stderr
    assert run.stdout.splitlines()[0] == "  Found      rewards.py  ·  3 callables recognised"


# --- export's two vocabularies (C-014) --------------------------------------------------------

def test_export_names_its_artefact_with_kind_and_leaves_format_alone(cli):
    """`--kind` is the artefact; `--format` is the envelope, on this verb as on every other."""
    from reward_lens.cli.verbs import export as verb

    help_text = cli("export", "--help")
    assert help_text.exit_code == 0
    assert help_text.stderr == "", help_text.stderr
    assert "--kind" in help_text.stdout
    assert verb.KINDS == ("html", "sarif", "badge", "revision", "bundle")


@pytest.mark.parametrize("kind", ["html", "badge", "revision", "bundle"])
def test_export_format_is_an_alias_for_kind_only_where_the_two_cannot_collide(kind):
    """The commission writes `export --format html` once; that line still means `--kind html`."""
    from reward_lens.cli.verbs.export import read_kind

    aliased = {"format_": kind, "kind": "bundle"}
    read_kind(aliased)
    assert aliased == {"format_": None, "kind": kind}

    for shared in ("text", "json", "jsonl", "sarif", "github"):
        opts = {"format_": shared, "kind": "bundle"}
        read_kind(opts)
        assert opts == {"format_": shared, "kind": "bundle"}, shared


def test_export_format_json_is_the_envelope_and_not_the_bundle(cli):
    run = cli("export", "--format", "json")
    assert run.exit_code in (0, 5)
    assert json.loads(run.stdout)["schema_version"] == "assay-result/1.0"
    assert "used more than once" not in run.stderr


def test_export_still_refuses_a_word_that_is_neither_a_format_nor_a_kind(cli):
    run = cli("export", "--format", "nonsense")
    assert run.exit_code == 4
    assert "nonsense is not an output format" in run.stderr


# --- the seam ---------------------------------------------------------------------------------

SEAM = {
    "audit": ("audit", ["audit", "."]),
    "trace": ("trace", ["trace", "run-0123456789ab"]),
    "compare": ("compare", ["compare", "a.assay.json", "b.assay.json"]),
    "forecast": ("forecast_issue", ["forecast", "issue", "the claim"]),
    "improve": ("improve", ["improve", "."]),
    "open": ("open_record", ["open", "a.assay.json"]),
    "export": ("export", ["export", "--kind", "bundle"]),
    "import": ("open_record", ["import", "a.assay.json"]),
    "doctor": ("doctor", ["doctor"]),
}


@pytest.mark.parametrize("verb", sorted(SEAM))
def test_every_measuring_verb_passes_through_reward_lens_api(invoke, monkeypatch, verb):
    """What this measures is the seam, not which engines happen to have landed.

    A-026 has four of these verbs refusing before they parse, because their engines are not in
    this build. An empty engine table makes no verb engine-gated, so every verb reaches the seam
    and the spy still answers the question this test was written to ask.
    """
    from reward_lens import api
    from reward_lens.cli import availability

    monkeypatch.setattr(availability, "engines", dict)

    seen: list[str] = []
    name, argv = SEAM[verb]
    original = getattr(api, name)

    def spy(*args, **kwargs):
        seen.append(name)
        raise _Stop

    monkeypatch.setattr(api, name, spy)
    try:
        invoke(*argv, "--format", "json")
    except _Stop:
        pass
    assert seen == [name], f"{verb} did not reach reward_lens.api.{name}"
    assert getattr(api, name) is spy and original is not None


class _Stop(BaseException):
    pass
