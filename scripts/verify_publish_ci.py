"""Validate two raw GitHub CI runs for the immutable release and its publisher controller.

The release run owns the wheel and sdist. The later controller run qualifies the
main commit that executes publish.yml. Neither identity substitutes for the other.
This module uses only the standard library so it can run before release checkout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY = "reward-lens/reward-lens"
WORKFLOW_PATH = ".github/workflows/ci.yml"
JOB_NAMES = frozenset(
    {
        "source-quality",
        "secrets",
        "frontend",
        "public-inputs",
        "cpu-dev (3.11)",
        "cpu-dev (3.12)",
        "cpu-dev (3.13)",
        "cpu-all-extras (3.11)",
        "cpu-all-extras (3.12)",
        "cpu-all-extras (3.13)",
        "trl-110-conformance",
        "trl-113-conformance",
        "docs",
        "package",
        "release-ready",
    }
)
FULL_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_ci_run(
    run: dict[str, Any],
    jobs_page: dict[str, Any],
    *,
    label: str,
    expected_sha: str,
    expected_run_id: int,
    workflow: dict[str, Any],
) -> None:
    """Require one exact successful main-push attempt and its complete 15-job matrix."""
    _require(FULL_SHA.fullmatch(expected_sha) is not None, f"{label}: invalid expected SHA")
    _require(type(expected_run_id) is int and expected_run_id > 0, f"{label}: invalid run ID")
    _require(run.get("id") == expected_run_id, f"{label}: wrong run ID")
    for field in ("repository", "head_repository"):
        repo = run.get(field)
        _require(
            isinstance(repo, dict) and repo.get("full_name") == REPOSITORY,
            f"{label}: wrong {field}",
        )
    _require(run.get("head_sha") == expected_sha, f"{label}: wrong head SHA")
    _require(run.get("head_branch") == "main", f"{label}: not main")
    _require(run.get("event") == "push", f"{label}: not a push")
    _require(run.get("path") == WORKFLOW_PATH, f"{label}: wrong workflow path")
    _require(run.get("workflow_id") == workflow["id"], f"{label}: wrong workflow ID")
    _require(run.get("name") == workflow["name"], f"{label}: wrong workflow name")
    _require(
        run.get("status") == "completed" and run.get("conclusion") == "success",
        f"{label}: CI not successful",
    )
    attempt = run.get("run_attempt")
    _require(type(attempt) is int and attempt > 0, f"{label}: invalid attempt")
    jobs = jobs_page.get("jobs")
    _require(isinstance(jobs, list), f"{label}: jobs missing")
    _require(
        jobs_page.get("total_count") == len(JOB_NAMES) and len(jobs) == len(JOB_NAMES),
        f"{label}: incomplete or paginated job set",
    )
    names: set[str] = set()
    job_ids: set[int] = set()
    for job in jobs:
        _require(isinstance(job, dict), f"{label}: malformed job")
        name = job.get("name")
        _require(isinstance(name, str) and name in JOB_NAMES, f"{label}: unexpected job")
        _require(name not in names, f"{label}: duplicate job name")
        names.add(name)
        job_id = job.get("id")
        _require(
            type(job_id) is int and job_id > 0 and job_id not in job_ids,
            f"{label}: duplicate or invalid job ID",
        )
        job_ids.add(job_id)
        _require(job.get("run_id") == expected_run_id, f"{label}: job from another run")
        _require(job.get("run_attempt") == attempt, f"{label}: job from another attempt")
        _require(job.get("head_sha") == expected_sha, f"{label}: job on another SHA")
        _require(job.get("head_branch") == "main", f"{label}: job off main")
        _require(job.get("run_url") == run.get("url"), f"{label}: job run URL mismatch")
        _require(job.get("workflow_name") == workflow["name"], f"{label}: job workflow mismatch")
        _require(
            job.get("status") == "completed" and job.get("conclusion") == "success",
            f"{label}: job not successful: {name}",
        )
    _require(names == JOB_NAMES, f"{label}: required jobs missing")


def validate_publish_ci(
    *,
    workflow: dict[str, Any],
    release_run: dict[str, Any],
    release_jobs: dict[str, Any],
    publisher_run: dict[str, Any],
    publisher_jobs: dict[str, Any],
    release_sha: str,
    publisher_sha: str,
    release_ci_run_id: int,
    publisher_ci_run_id: int,
) -> None:
    _require(workflow.get("path") == WORKFLOW_PATH, "wrong CI workflow path")
    _require(type(workflow.get("id")) is int and workflow["id"] > 0, "invalid CI workflow ID")
    _require(workflow.get("name") == "reward-lens release validation", "wrong CI workflow name")
    _require(workflow.get("state") == "active", "CI workflow inactive")
    validate_ci_run(
        release_run,
        release_jobs,
        label="release",
        expected_sha=release_sha,
        expected_run_id=release_ci_run_id,
        workflow=workflow,
    )
    validate_ci_run(
        publisher_run,
        publisher_jobs,
        label="publisher",
        expected_sha=publisher_sha,
        expected_run_id=publisher_ci_run_id,
        workflow=workflow,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workflow", "release-run", "release-jobs", "publisher-run", "publisher-jobs"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("release-sha", "publisher-sha"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("release-ci-run-id", "publisher-ci-run-id"):
        parser.add_argument(f"--{name}", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        validate_publish_ci(
            workflow=json.loads(args.workflow.read_text(encoding="utf-8")),
            release_run=json.loads(args.release_run.read_text(encoding="utf-8")),
            release_jobs=json.loads(args.release_jobs.read_text(encoding="utf-8")),
            publisher_run=json.loads(args.publisher_run.read_text(encoding="utf-8")),
            publisher_jobs=json.loads(args.publisher_jobs.read_text(encoding="utf-8")),
            release_sha=args.release_sha,
            publisher_sha=args.publisher_sha,
            release_ci_run_id=args.release_ci_run_id,
            publisher_ci_run_id=args.publisher_ci_run_id,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Publisher CI validation failed: {error}", file=sys.stderr)
        return 1
    print("PASS: release and publisher-controller main CI runs each have 15 exact successful jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
