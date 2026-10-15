"""Rung 1's lower-level math and public persisted Compare journey."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from reward_lens.bank import BankProvenance, Response, ResponseBank, TaskId
from reward_lens.contracts import UsageError, digest, validate_record
from reward_lens.product.compare.registry import Rung, get
from reward_lens.product.compare.rung1 import rescore
from reward_lens.render.projections import project
from reward_lens.render.projections.types import View

_CASE_EXTENSION = "https://reward-lens.github.io/case/v1"


def _bank() -> ResponseBank:
    first = TaskId("first")
    second = TaskId("second")
    return ResponseBank(
        id="known-pairs",
        digest="bank-digest",
        provenance=BankProvenance("fixture.jsonl", None, None, "source-digest"),
        tasks=(first, second),
        responses={
            first: (
                Response("a", first, "a", {"v1": Decimal("1"), "v2": Decimal("2")}),
                Response("b", first, "b", {"v1": Decimal("3"), "v2": Decimal("4")}),
            ),
            second: (
                Response("c", second, "c", {"v1": Decimal("3"), "v2": Decimal("6")}),
                Response("d", second, "d", {"v1": Decimal("5"), "v2": Decimal("8")}),
            ),
        },
    )


def _write_bank(destination: Path) -> Path:
    destination.write_text(
        "\n".join(
            (
                json.dumps({"name": "known-pairs"}),
                json.dumps(
                    {
                        "task_id": "one",
                        "responses": [
                            {"text": "a", "scores": {"v1": "1", "v2": "2"}},
                            {"text": "b", "scores": {"v1": "3", "v2": "4"}},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "two",
                        "responses": [
                            {"text": "c", "scores": {"v1": "3", "v2": "6"}},
                            {"text": "d", "scores": {"v1": "5", "v2": "8"}},
                        ],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _project(destination: Path) -> Path:
    from reward_lens.examples import REGISTRY

    module_name, _, attribute = REGISTRY["code-reward"].partition(":")
    getattr(importlib.import_module(module_name), attribute)(destination)
    return destination


def _replay_config(destination: Path, version: str, *, offset: str) -> Path:
    destination.write_text(
        json.dumps({"id": version, "weights": {"length_reward": "1"}, "offset": offset}),
        encoding="utf-8",
    )
    return destination


def _replay_run(destination: Path) -> Path:
    source = Path(__file__).resolve().parents[3] / "tests/fixtures/grpo_run/short"
    shutil.copytree(source, destination)
    return destination


def _replay_identity(record):
    return record.extensions["https://reward-lens.github.io/extensions/compare-replay/1"]


def _replay_entry(record):
    return next(
        entry
        for entry in record.measurement.reward_statistics
        if entry.entry_id == "reward_statistics.advantage_replay"
    )


def _assert_case_evidence(record) -> None:
    """Assert the Case view points at final saved R3 entries, not an aggregate cost."""
    extension = record.extensions[_CASE_EXTENSION]
    decisive = _replay_entry(record)
    assert extension["decision"]["entry_id"] == decisive.entry_id
    assert extension["decision"]["digest"] == digest(decisive)

    pair = extension["decisive_pair"]
    costs = extension["candidate_costs"]
    assert len(pair) == len(costs) == 2
    assert [row["candidate_id"] for row in pair] == ["v1", "v2"]
    assert [row["entry_id"] for row in pair] == [row["entry_id"] for row in costs]
    assert [row["digest"] for row in pair] == [row["digest"] for row in costs]
    assert len({row["entry_id"] for row in pair}) == 2

    entries = {entry.entry_id: entry for entry in record.measurement.reward_statistics}
    candidate_ids = {row["entry_id"] for row in pair}
    assert candidate_ids <= set(entries)
    for pair_row, cost_row in zip(pair, costs, strict=True):
        entry = entries[pair_row["entry_id"]]
        assert entry.kind == "witness"
        assert entry.subject_ref == decisive.subject_ref == record.subject.version.digest
        assert pair_row["digest"] == digest(entry)
        counters = entry.to_dict()["provenance"]["counters"]
        assert {field: cost_row[field] for field in ("wall_s", "cpu_s", "usd", "api_calls")} == {
            field: counters[field] for field in ("wall_s", "cpu_s", "usd", "api_calls")
        }

    projection = project(record, View.CASE)
    assert projection.fields["decisive_pair"] == pair
    assert projection.fields["candidate_costs"] == costs
    assert projection.source_pointers == (
        "/assay_id",
        "/extensions/https:~1~1reward-lens.github.io~1case~1v1/decisive_pair",
        "/extensions/https:~1~1reward-lens.github.io~1case~1v1/candidate_costs",
    )


def _assert_bundle(project_root: Path, name: str, record) -> None:
    from reward_lens.store import Project

    project = Project.open(project_root)
    path = project.path_for(name)
    assert path.is_file()
    reopened = project.open_record(name)
    assert reopened.assay_id == record.assay_id
    validate_record(reopened.to_dict(), raw=path.read_bytes())
    assert reopened.assay_id == record.assay_id

    manifest_path = path.with_name("manifest.json")
    report_path = path.with_suffix("").with_suffix(".assay.html")
    assert report_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["assay"]["id"] == record.assay_id
    artifact = manifest["artifacts"][0]
    assert artifact["path"] == path.name
    assert artifact["sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert record.assay_id in report_path.read_text(encoding="utf-8")


def _assert_loose_bundle(root: Path, name: str, record) -> None:
    from reward_lens import contracts

    path = root / "assays" / f"{name}.assay.json"
    assert path.is_file()
    reopened = contracts.Assay.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert reopened.assay_id == record.assay_id
    validate_record(reopened.to_dict(), raw=path.read_bytes())
    assert (root / "assays" / f"{name}.assay.html").is_file()


def test_rescore_uses_task_paired_differences_for_the_clustered_standard_error() -> None:
    result = rescore(_bank(), "v1", "v2")

    assert result.scope == "evaluator_comparison"
    assert result.reading.cluster_unit == "task"
    assert result.reading.task_differences == {"first": Decimal("1"), "second": Decimal("3")}
    assert result.reading.mean_difference == Decimal("2.000000")
    assert result.reading.standard_error == Decimal("1.000000")
    assert result.reading.interval is None
    assert result.scorecard.rows[0].value.reason.startswith(
        "not measured: fewer than 15 task clusters"
    )


def test_rescore_refuses_no_bank_with_p_bank_code() -> None:
    with pytest.raises(UsageError) as caught:
        rescore(None, "v1", "v2")
    assert getattr(caught.value, "code", None) == "RL0134"
    assert getattr(caught.value, "exit_code", None) == 4


def test_sdk_dispatch_refuses_a_missing_bank_without_inventing_one_from_versions() -> None:
    from reward_lens import api

    with pytest.raises(UsageError) as caught:
        api.compare(api.CompareRequest(baseline="v1", candidate="v2"))
    assert caught.value.code == "RL0134"


def test_sdk_compare_request_persists_a_schema_valid_assay_and_report(tmp_path: Path) -> None:
    from reward_lens import api, contracts

    root = _project(tmp_path / "sdk-project")
    source = _write_bank(root / "known-pairs.jsonl")
    record = api.compare(
        api.CompareRequest(
            baseline="v1",
            candidate="v2",
            bank=source,
            project=root,
            name="sdk-compare",
        )
    )

    assert isinstance(record, contracts.Assay)
    entry = record.measurement.reward_statistics[0]
    assert entry.result["version_a"] == "v1"
    assert entry.result["version_b"] == "v2"
    assert entry.depends_on == []
    assert entry.subject_ref == record.subject.version.digest
    assert record.subject.version.id == "code-reward-v1"
    bank_identity = record.extensions["https://reward-lens.github.io/extensions/compare-bank/1"]
    assert bank_identity["canonical_digest"].startswith("sha256:")
    assert bank_identity["source_digest"].startswith("sha256:")
    assert record.decision.state == "unresolved"
    assert "required_missing:reward_statistics" not in record.decision.reasons
    _assert_bundle(root, "sdk-compare", record)


def test_changed_foreign_response_bank_bytes_change_sealed_input_not_project_identity(
    tmp_path: Path,
) -> None:
    from reward_lens import api, contracts
    from reward_lens.bank import load_bank

    root = _project(tmp_path / "configured-project")
    source = _write_bank(tmp_path / "known-pairs.jsonl")
    first = api.compare(
        api.CompareRequest(
            baseline="v1", candidate="v2", bank=source, project=root, name="first-bank"
        )
    )
    first_path = root / "assays" / "first-bank.assay.json"
    assert (
        contracts.Assay.model_validate(json.loads(first_path.read_text(encoding="utf-8"))).assay_id
        == first.assay_id
    )
    first_bank = load_bank(source, name=source.stem)
    configured_samples = first.subject.digests.samples
    assert configured_samples != "sha256:" + first_bank.digest
    first_identity = first.extensions["https://reward-lens.github.io/extensions/compare-bank/1"]
    first_entry_identity = first.measurement.reward_statistics[0].result["response_bank"]
    assert first_entry_identity["canonical_digest"] == "sha256:" + first_bank.digest
    assert first_identity["source_digest"] == "sha256:" + first_bank.provenance.digest

    source.write_text(
        source.read_text(encoding="utf-8").replace('"v2": "2"', '"v2": "9"'), encoding="utf-8"
    )
    second = api.compare(
        api.CompareRequest(
            baseline="v1", candidate="v2", bank=source, project=root, name="changed-bank"
        )
    )
    second_path = root / "assays" / "changed-bank.assay.json"
    assert second_path.is_file()
    assert (
        contracts.Assay.model_validate(json.loads(second_path.read_text(encoding="utf-8"))).assay_id
        == second.assay_id
    )
    second_bank = load_bank(source, name=source.stem)
    second_identity = second.extensions["https://reward-lens.github.io/extensions/compare-bank/1"]
    assert second.subject.digests.samples == configured_samples
    assert second.subject.digests.samples == first.subject.digests.samples
    assert second_identity["canonical_digest"] == "sha256:" + second_bank.digest
    assert second_identity["canonical_digest"] != first_identity["canonical_digest"]
    assert second_identity["source_digest"] != first_identity["source_digest"]
    assert second.assay_id != first.assay_id


@pytest.mark.parametrize(
    ("rungs", "required_field"),
    (
        ((2,), "project"),
        ((4,), "controlled_update_config"),
        ((1, 4), "controlled_update_config"),
    ),
)
def test_sdk_compare_routes_every_explicit_rung_through_registry(
    tmp_path: Path, rungs, required_field
) -> None:
    """Implemented rungs refuse their real missing public input, never RL0703."""
    from reward_lens import api

    source = _write_bank(tmp_path / "known-pairs.jsonl")
    request = api.CompareRequest(baseline="v1", candidate="v2", bank=source, rungs=rungs)

    with pytest.raises(UsageError) as caught:
        api.compare(request)
    assert caught.value.code == "RL0131"
    assert caught.value.exit_code == 4
    assert caught.value.context["field"] == required_field


def test_sdk_compare_explicit_rung_one_runs_the_same_registered_implementation(
    tmp_path: Path,
) -> None:
    from reward_lens import api

    root = _project(tmp_path / "explicit-rung-project")
    source = _write_bank(root / "known-pairs.jsonl")
    record = api.compare(
        api.CompareRequest(
            baseline="v1",
            candidate="v2",
            bank=source,
            project=root,
            rungs=(1,),
            name="explicit-rung",
        )
    )

    assert record.measurement.reward_statistics[0].value == 2.0
    _assert_bundle(root, "explicit-rung", record)


def test_sdk_combines_real_rung_one_and_three_entries_in_one_persisted_assay(
    tmp_path: Path,
) -> None:
    from reward_lens import api

    root = _project(tmp_path / "combined-project")
    bank = _write_bank(root / "known-pairs.jsonl")
    run_dir = _replay_run(tmp_path / "sdk-run")
    baseline = _replay_config(root / "v1.json", "v1", offset="0")
    candidate = _replay_config(root / "v2.json", "v2", offset="-3")

    record = api.compare(
        api.CompareRequest(
            baseline="v1",
            candidate="v2",
            bank=bank,
            project=root,
            rungs=(1, 3),
            run_dir=run_dir,
            baseline_config=baseline,
            candidate_config=candidate,
            name="sdk-rungs-one-three",
        )
    )

    entries = {entry.entry_id: entry for entry in record.measurement.reward_statistics}
    assert "reward_statistics.evaluator_rescore" in entries
    assert "reward_statistics.advantage_replay" in entries
    assert (
        entries["reward_statistics.evaluator_rescore"]
        .result["response_bank"]["canonical_digest"]
        .startswith("sha256:")
    )
    assert "response_bank" not in entries["reward_statistics.advantage_replay"].result
    assert entries["reward_statistics.advantage_replay"].result["verdict"] == "unchanged pressure"
    replay_inputs = _replay_identity(record)
    replay_entry_inputs = entries["reward_statistics.advantage_replay"].result["replay_inputs"]
    assert replay_entry_inputs == replay_inputs
    assert replay_inputs["run"]["source_tree_digest"].startswith("sha256:")
    assert replay_inputs["run"]["parsed_evidence_digest"].startswith("sha256:")
    assert replay_inputs["baseline"]["source_digest"].startswith("sha256:")
    assert replay_inputs["baseline"]["canonical_declaration_digest"].startswith("sha256:")
    assert replay_inputs["candidate"]["source_digest"].startswith("sha256:")
    assert replay_inputs["candidate"]["canonical_declaration_digest"].startswith("sha256:")
    _assert_bundle(root, "sdk-rungs-one-three", record)
    _assert_case_evidence(record)
    from reward_lens.store import Project

    _assert_case_evidence(Project.open(root).open_record("sdk-rungs-one-three"))


def test_cli_combines_real_rung_one_and_three_entries_in_one_persisted_assay(
    tmp_path: Path,
) -> None:
    from click.testing import CliRunner

    from reward_lens.cli.main import cli

    root = _project(tmp_path / "cli-combined-project")
    bank = _write_bank(root / "known-pairs.jsonl")
    run_dir = _replay_run(tmp_path / "cli-run")
    baseline = _replay_config(root / "v1.json", "v1", offset="0")
    candidate = _replay_config(root / "v2.json", "v2", offset="-3")
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "v1",
            "v2",
            "--rung",
            "1",
            "--rung",
            "3",
            "--bank",
            str(bank),
            "--run-dir",
            str(run_dir),
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--project",
            str(root),
            "--name",
            "cli-rungs-one-three",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2, result.output
    document = json.loads(result.output)
    assert document["artifacts"]["assay"] == "assays/cli-rungs-one-three.assay.json"
    from reward_lens.store import Project

    record = Project.open(root).open_record("cli-rungs-one-three")
    entry_ids = {entry.entry_id for entry in record.measurement.reward_statistics}
    assert "reward_statistics.evaluator_rescore" in entry_ids
    assert "reward_statistics.advantage_replay" in entry_ids
    replay_inputs = _replay_identity(record)
    assert _replay_entry(record).result["replay_inputs"] == replay_inputs
    assert replay_inputs["run"]["source_tree_digest"].startswith("sha256:")
    assert replay_inputs["baseline"]["declaration"]["id"] == "v1"
    assert replay_inputs["candidate"]["declaration"]["id"] == "v2"
    _assert_bundle(root, "cli-rungs-one-three", record)
    _assert_case_evidence(record)


def test_sdk_replay_provenance_changes_for_closed_run_and_each_replay_declaration(
    tmp_path: Path,
) -> None:
    """A path label cannot hide changed run or configuration bytes after persistence."""
    from reward_lens import api

    root = _project(tmp_path / "replay-provenance-project")
    bank = _write_bank(root / "known-pairs.jsonl")
    run_dir = _replay_run(tmp_path / "mutable-run")
    baseline = _replay_config(root / "v1.json", "v1", offset="0")
    candidate = _replay_config(root / "v2.json", "v2", offset="-3")

    def compare(name: str):
        return api.compare(
            api.CompareRequest(
                baseline="v1",
                candidate="v2",
                bank=bank,
                project=root,
                rungs=(1, 3),
                run_dir=run_dir,
                baseline_config=baseline,
                candidate_config=candidate,
                name=name,
            )
        )

    first = compare("replay-initial")
    first_identity = _replay_identity(first)
    first_method = _replay_entry(first).method.params_digest
    _assert_bundle(root, "replay-initial", first)

    run_json = next(run_dir.rglob("run.json"))
    run_json.write_bytes(run_json.read_bytes() + b"\n")
    changed_run = compare("replay-changed-run")
    changed_run_identity = _replay_identity(changed_run)
    assert (
        changed_run_identity["run"]["source_tree_digest"]
        != first_identity["run"]["source_tree_digest"]
    )
    assert (
        changed_run_identity["run"]["parsed_evidence_digest"]
        == first_identity["run"]["parsed_evidence_digest"]
    )
    assert changed_run_identity["baseline"] == first_identity["baseline"]
    assert changed_run.assay_id != first.assay_id
    assert _replay_entry(changed_run).method.params_digest != first_method

    _replay_config(baseline, "v1", offset="1")
    changed_baseline = compare("replay-changed-baseline")
    changed_baseline_identity = _replay_identity(changed_baseline)
    assert changed_baseline_identity["baseline"] != changed_run_identity["baseline"]
    assert changed_baseline_identity["candidate"] == changed_run_identity["candidate"]
    assert changed_baseline.assay_id != changed_run.assay_id
    assert (
        _replay_entry(changed_baseline).method.params_digest
        != _replay_entry(changed_run).method.params_digest
    )

    _replay_config(candidate, "v2", offset="-2")
    changed_candidate = compare("replay-changed-candidate")
    changed_candidate_identity = _replay_identity(changed_candidate)
    assert changed_candidate_identity["candidate"] != changed_baseline_identity["candidate"]
    assert changed_candidate_identity["baseline"] == changed_baseline_identity["baseline"]
    assert changed_candidate.assay_id != changed_baseline.assay_id
    assert (
        _replay_entry(changed_candidate).method.params_digest
        != _replay_entry(changed_baseline).method.params_digest
    )
    _assert_bundle(root, "replay-changed-candidate", changed_candidate)


def test_projectless_rung_three_keeps_its_sealed_run_input_unchanged_on_repeat(
    tmp_path: Path,
) -> None:
    """A projectless bundle belongs beside the sealed run tree, not inside it."""
    from reward_lens import api

    run_dir = _replay_run(tmp_path / "closed-run")
    baseline = _replay_config(tmp_path / "v1.json", "v1", offset="0")
    candidate = _replay_config(tmp_path / "v2.json", "v2", offset="-3")

    def compare(name: str):
        return api.compare(
            api.CompareRequest(
                baseline="v1",
                candidate="v2",
                rungs=(3,),
                run_dir=run_dir,
                baseline_config=baseline,
                candidate_config=candidate,
                name=name,
            )
        )

    first = compare("projectless-first")
    second = compare("projectless-second")
    first_inputs = _replay_identity(first)
    second_inputs = _replay_identity(second)
    assert second_inputs["run"]["source_tree_digest"] == first_inputs["run"]["source_tree_digest"]
    assert (
        second_inputs["run"]["parsed_evidence_digest"]
        == first_inputs["run"]["parsed_evidence_digest"]
    )
    assert _replay_entry(second).method.params_digest == _replay_entry(first).method.params_digest
    assert not (run_dir / "assays").exists()
    _assert_loose_bundle(tmp_path, "projectless-first", first)
    _assert_loose_bundle(tmp_path, "projectless-second", second)


def test_cli_projectless_rung_three_reports_parent_bundle_paths(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from reward_lens.cli.main import cli

    run_dir = _replay_run(tmp_path / "cli-closed-run")
    baseline = _replay_config(tmp_path / "v1.json", "v1", offset="0")
    candidate = _replay_config(tmp_path / "v2.json", "v2", offset="-3")
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "v1",
            "v2",
            "--rung",
            "3",
            "--run-dir",
            str(run_dir),
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--name",
            "cli-projectless",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2, result.output
    document = json.loads(result.output)
    assert document["artifacts"]["assay"] == "assays/cli-projectless.assay.json"
    assert not (run_dir / "assays").exists()
    assert (tmp_path / document["artifacts"]["assay"]).is_file()


def test_registry_declares_all_six_rungs_and_loads_only_real_registered_engines() -> None:
    assert [(rung.name, rung.value) for rung in Rung] == [
        ("RE_SCORE", 1),
        ("SELECTION_STRESS", 2),
        ("ADVANTAGE_REPLAY", 3),
        ("SAME_BATCH_SWAP", 4),
        ("SHORT_FORK", 5),
        ("TRANSFER", 6),
    ]
    assert {rung: get(rung).__module__ for rung in Rung} == {
        Rung.RE_SCORE: "reward_lens.product.compare.rung1",
        Rung.SELECTION_STRESS: "reward_lens.product.compare.rung2",
        Rung.ADVANTAGE_REPLAY: "reward_lens.product.compare.rung3",
        Rung.SAME_BATCH_SWAP: "reward_lens.product.compare.registry",
        Rung.SHORT_FORK: "reward_lens.product.compare.rung5",
        Rung.TRANSFER: "reward_lens.product.compare.rung6",
    }


def test_real_compare_cli_persists_and_reports_the_same_assay(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from reward_lens.cli.main import cli

    root = _project(tmp_path / "cli-project")
    source = _write_bank(root / "known-pairs.jsonl")
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "v1",
            "v2",
            "--rung",
            "1",
            "--bank",
            str(source),
            "--project",
            str(root),
            "--name",
            "cli-compare",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2, result.output
    document = json.loads(result.output)
    assert document["command"] == "compare"
    assert document["decision"]["state"] == "unresolved"
    assert document["artifacts"]["assay"] == "assays/cli-compare.assay.json"
    assert document["artifacts"]["report"] == "assays/cli-compare.assay.html"
    from reward_lens.store import Project

    record = Project.open(root).open_record("cli-compare")
    _assert_bundle(root, "cli-compare", record)
    from reward_lens.cli import availability

    assert availability.absent("compare") is False


def test_package_reexports_remain_public_without_polluting_root_help_imports() -> None:
    from reward_lens.product.compare import Scorecard

    assert Scorecard.__name__ == "Scorecard"
    probe = (
        "import sys;"
        "sys.argv=['reward-lens','--help'];"
        "import reward_lens.cli.main as main;"
        "\ntry:\n main.main()\nexcept SystemExit:\n pass\n"
        "print('PYDANTIC=' + str('pydantic' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.rsplit("PYDANTIC=", 1)[1].strip() == "False"
