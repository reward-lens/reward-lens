"""The publisher accepts two separately qualified CI runs and no partial matrix."""

from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "verify_publish_ci", ROOT / "scripts/verify_publish_ci.py"
)
assert SPEC is not None and SPEC.loader is not None
verify_publish_ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_publish_ci)

RELEASE_SHA = "2" * 40
PUBLISHER_SHA = "3" * 40
WORKFLOW_ID = 365594709


def _ci_pair(sha: str, run_id: int, attempt: int) -> tuple[dict, dict]:
    url = f"https://api.github.com/repos/reward-lens/reward-lens/actions/runs/{run_id}"
    run = {
        "id": run_id,
        "name": "reward-lens release validation",
        "repository": {"full_name": "reward-lens/reward-lens"},
        "head_repository": {"full_name": "reward-lens/reward-lens"},
        "head_sha": sha,
        "head_branch": "main",
        "event": "push",
        "path": ".github/workflows/ci.yml",
        "workflow_id": WORKFLOW_ID,
        "status": "completed",
        "conclusion": "success",
        "run_attempt": attempt,
        "url": url,
    }
    jobs = [
        {
            "id": run_id * 100 + index,
            "run_id": run_id,
            "run_attempt": attempt,
            "head_sha": sha,
            "head_branch": "main",
            "run_url": url,
            "workflow_name": "reward-lens release validation",
            "name": name,
            "status": "completed",
            "conclusion": "success",
        }
        for index, name in enumerate(sorted(verify_publish_ci.JOB_NAMES), start=1)
    ]
    return run, {"total_count": len(jobs), "jobs": jobs}


def _valid_inputs() -> dict:
    release_run, release_jobs = _ci_pair(RELEASE_SHA, 38943117681, 1)
    publisher_run, publisher_jobs = _ci_pair(PUBLISHER_SHA, 39000000000, 2)
    return {
        "workflow": {
            "id": WORKFLOW_ID,
            "name": "reward-lens release validation",
            "path": ".github/workflows/ci.yml",
            "state": "active",
        },
        "release_run": release_run,
        "release_jobs": release_jobs,
        "publisher_run": publisher_run,
        "publisher_jobs": publisher_jobs,
        "release_sha": RELEASE_SHA,
        "publisher_sha": PUBLISHER_SHA,
        "release_ci_run_id": 38943117681,
        "publisher_ci_run_id": 39000000000,
    }


def test_separate_frozen_release_and_publisher_controller_ci_pass() -> None:
    verify_publish_ci.validate_publish_ci(**_valid_inputs())


@pytest.mark.parametrize(
    "mutation",
    [
        "release_sha",
        "publisher_sha",
        "short_sha",
        "run_id",
        "repository",
        "head_repository",
        "workflow_id",
        "workflow_path",
        "event",
        "branch",
        "pending_run",
        "failed_run",
        "missing_job",
        "duplicate_job",
        "wrong_total_count",
        "old_attempt",
        "wrong_job_sha",
        "failed_job",
        "pending_job",
    ],
)
def test_bad_release_or_publisher_ci_is_rejected(mutation: str) -> None:
    inputs = copy.deepcopy(_valid_inputs())
    release = inputs["release_run"]
    publisher = inputs["publisher_run"]
    job = inputs["publisher_jobs"]["jobs"][0]
    if mutation == "release_sha":
        release["head_sha"] = PUBLISHER_SHA
    elif mutation == "publisher_sha":
        publisher["head_sha"] = RELEASE_SHA
    elif mutation == "short_sha":
        inputs["publisher_sha"] = PUBLISHER_SHA[:12]
    elif mutation == "run_id":
        publisher["id"] = 42
    elif mutation == "repository":
        publisher["repository"]["full_name"] = "attacker/reward-lens"
    elif mutation == "head_repository":
        publisher["head_repository"]["full_name"] = "attacker/reward-lens"
    elif mutation == "workflow_id":
        publisher["workflow_id"] = 42
    elif mutation == "workflow_path":
        publisher["path"] = ".github/workflows/lookalike.yml"
    elif mutation == "event":
        publisher["event"] = "pull_request"
    elif mutation == "branch":
        publisher["head_branch"] = "release"
    elif mutation == "pending_run":
        publisher["status"] = "in_progress"
    elif mutation == "failed_run":
        publisher["conclusion"] = "failure"
    elif mutation == "missing_job":
        inputs["publisher_jobs"]["jobs"].pop()
    elif mutation == "duplicate_job":
        inputs["publisher_jobs"]["jobs"][1]["name"] = job["name"]
    elif mutation == "wrong_total_count":
        inputs["publisher_jobs"]["total_count"] = 14
    elif mutation == "old_attempt":
        job["run_attempt"] = 1
    elif mutation == "wrong_job_sha":
        job["head_sha"] = RELEASE_SHA
    elif mutation == "failed_job":
        job["conclusion"] = "failure"
    elif mutation == "pending_job":
        job["status"] = "queued"
    with pytest.raises(ValueError):
        verify_publish_ci.validate_publish_ci(**inputs)


def test_workflow_pins_same_official_action_and_preflight_image() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    action = re.search(r"uses: pypa/gh-action-pypi-publish@([0-9a-f]{40})", workflow)
    image = re.search(
        r"PUBLISHER_IMAGE_TAG: ghcr.io/pypa/gh-action-pypi-publish:([0-9a-f]{40})", workflow
    )
    assert action is not None and image is not None
    assert action.group(1) == image.group(1) == "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"
    assert "sha256:a68d05519f6d7e47372aeaddab80b851b69afa89be179ec41775c72c4e3ab2d5" in workflow
    assert "verify-metadata: true" in workflow
    assert "skip-existing: false" in workflow
    assert "attestations: true" in workflow
    assert "publisher_ci_run_id:" in workflow
    assert "python scripts/verify_publish_ci.py" in workflow
    assert (
        workflow.index("ref: ${{ github.sha }}")
        < workflow.index("python scripts/verify_publish_ci.py")
        < workflow.index("ref: ${{ inputs.release_sha }}")
    )
    assert workflow.index("Check exact pinned publisher image") < workflow.index("  preflight:")
