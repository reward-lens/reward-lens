"""A-026 and D-28: a run that returned a stored record says so, and exits on the stored verdict.

Two arms, because the two halves of that claim fail in different ways. The first hands the verb a
`last_reuse` of its own, so the `Reused` line, the seven `execution.reused` fields and the exit
code are tested without depending on what the store happens to hold. The second audits the written
example twice and asks the store for the reuse itself, so the first arm cannot pass over a seam
that nothing real ever reaches.

The exit code is the point of the second assertion in each arm. A-026 supersedes A-016 there: the
code is the verdict's, so a reused audit of the example prints `unresolved` and exits 2, exactly as
the run that measured it did, and two readers of one assay never disagree about whether it passed.
What a reused run never returns is an error class (4, 6): the saying that it did no work is the
`Reused` line and `execution.reused`, not the status.
"""

from __future__ import annotations

import dataclasses

from reward_lens.cli.verbs._shared import REUSED_FIELDS


@dataclasses.dataclass
class Reuse:
    """What `reward_lens.api.last_reuse()` returns: the seven fields, and nothing derived."""

    name: str = "code-reward"
    assay_id: str = "assay-7f3c9d"
    subject_version: str = "code-reward-v1"
    record_path: str = "./demo/.rewardlens/assays/assay-7f3c9d.json"
    rendered_report: bool = False
    wrote_manifest: bool = False
    says: str = "the stored record of code-reward v1; nothing it depends on has changed"


def example(cli, tmp_path):
    project = tmp_path / "demo"
    assert cli("init", "--example", "code-reward", str(project)).exit_code == 0
    return project


# --- the seam, with a reuse the test supplies ---------------------------------------------------


def test_a_reused_run_says_so_above_the_record_and_exits_on_the_stored_verdict(
    cli, invoke, monkeypatch, tmp_path
):
    from reward_lens import api

    project = example(cli, tmp_path)
    monkeypatch.setattr(api, "last_reuse", lambda: Reuse())
    run = invoke("audit", str(project), "--format", "text")

    assert run.exit_code == 2, run.stdout
    assert f"  Reused    {Reuse().says}" in run.stdout
    assert "  Verdict   " in run.stdout


def test_the_envelope_carries_the_seven_reused_fields_verbatim(cli, invoke, monkeypatch, tmp_path):
    from reward_lens import api

    project = example(cli, tmp_path)
    monkeypatch.setattr(api, "last_reuse", lambda: Reuse())
    run = invoke("audit", str(project), "--format", "json")

    reused = run.json()["execution"]["reused"]
    assert list(reused) == list(REUSED_FIELDS)
    assert reused == dataclasses.asdict(Reuse())
    assert run.exit_code == 2


def test_a_run_that_measured_carries_no_reused_block(cli, tmp_path):
    """The line is a claim about this run: a run that measured must not make it."""
    project = example(cli, tmp_path)
    run = cli("audit", str(project), "--format", "json")

    assert run.json()["execution"]["reused"] is None
    assert run.exit_code == 2


# --- the seam, with the store's own reuse -------------------------------------------------------


def test_a_second_audit_of_the_written_example_returns_the_stored_record(cli, tmp_path):
    project = example(cli, tmp_path)
    first = cli("audit", str(project), "--format", "json")
    second = cli("audit", str(project), "--format", "json")

    reused = second.json()["execution"]["reused"]
    assert reused is not None, "the second audit of an unchanged subject measured again"
    assert list(reused) == list(REUSED_FIELDS)
    assert second.exit_code == 2, second.stdout
    assert second.exit_code == first.exit_code, "the two runs disagree about the same assay"
    assert second.json()["decision"] == first.json()["decision"]


def test_the_second_audit_prints_the_stored_verdict_over_the_reused_line(cli, tmp_path):
    project = example(cli, tmp_path)
    first = cli("audit", str(project), "--format", "json")
    second = cli("audit", str(project), env={"REWARD_LENS_FORMAT": "text"})

    verdict = first.json()["decision"]["state"]
    assert verdict == "unresolved"
    assert second.exit_code == 2, second.stdout
    assert second.exit_code == first.exit_code, "the two runs disagree about the same assay"
    assert f"  Verdict   {verdict}" in second.stdout
    assert "  Reused    " in second.stdout
