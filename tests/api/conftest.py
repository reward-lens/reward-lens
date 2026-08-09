"""Shared fixtures for the P-SDK suite.

A verb that is pointed at a directory holding no `rewardlens.yaml` rightly refuses it (RL0001),
so a request that means a project gets one: `project_dir` writes the shipped example through the
public `examples.REGISTRY`, the same table `reward-lens init --example` resolves.

Which engines a build holds is not this suite's subject. `engines_absent` takes the verbs whose
engine a test wants gone and points each at a module nobody ships, which is the one lookup
`_dispatch.engine` performs. A test that wants the SDK's own fallback asks for it that way rather
than relying on a packet not having landed yet.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

#: Every function section 8.0 names, in the order it names them.
SECTION_8_0 = (
    "audit",
    "trace",
    "compare",
    "forecast_issue",
    "forecast_resolve",
    "forecast_ledger",
    "improve",
    "open_record",
    "export",
    "doctor",
    "dry_run",
)

#: The verbs that return a record (`export` raises, `doctor` and `dry_run` do not return an
#: Assay, `open_record` reads one rather than producing it).
RECORD_VERBS = (
    "audit",
    "trace",
    "compare",
    "forecast_issue",
    "forecast_resolve",
    "forecast_ledger",
    "improve",
)

#: The ten sections of the record, in schema order.
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

#: The example `reward-lens init --example` writes by default.
EXAMPLE = "code-reward"


def write_example(dest: Path, name: str = EXAMPLE) -> Path:
    """Write one shipped example into `dest`, resolved through the public registry.

    The registry is the table the CLI's `init --example` reads, so a test that builds a project
    this way builds the one a user gets, without importing the CLI to do it.
    """
    from reward_lens.examples import REGISTRY

    module_name, _, attribute = REGISTRY[name].partition(":")
    getattr(importlib.import_module(module_name), attribute)(dest)
    return dest


def engines_absent(monkeypatch: pytest.MonkeyPatch, *verbs: str) -> None:
    """Make each named verb's engine absent through the seam that resolves it.

    `_dispatch.engine` reads one entry of `ENGINES` and loads it; pointing that entry at a module
    nobody ships is the absence itself, not a simulation of it. With no verb named, every verb in
    the table goes.
    """
    from reward_lens.api import _dispatch

    for verb in verbs or tuple(_dispatch.ENGINES):
        _, _, attribute = _dispatch.ENGINES[verb].partition(":")
        monkeypatch.setitem(
            _dispatch.ENGINES, verb, f"reward_lens.product.nothing_here:{attribute}"
        )


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """A real reward-lens project: the shipped example, written into a fresh directory."""
    return write_example(tmp_path / "demo")


@pytest.fixture
def requests_by_verb(project_dir: Path) -> dict[str, object]:
    """One built request per record-returning verb."""
    from reward_lens import api

    return {
        "audit": api.AuditRequest(path=project_dir),
        "trace": api.TraceRequest(run="run-0123456789ab", project=project_dir),
        "compare": api.CompareRequest(baseline="v1", candidate="v2", project=project_dir),
        "forecast_issue": api.ForecastIssueRequest(
            claim="held-out success rises", project=project_dir
        ),
        "forecast_resolve": api.ForecastResolveRequest(forecast_id="f-0001", project=project_dir),
        "forecast_ledger": api.ForecastLedgerRequest(project=project_dir),
        "improve": api.ImproveRequest(path=project_dir),
    }
