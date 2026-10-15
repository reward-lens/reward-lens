"""The public MCP proposal route reaches qualified saved Audit evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from reward_lens import contracts
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.integrations.mcp import dispatch

_ROOT = Path(__file__).parents[2]


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "outcome" / "controls").mkdir(parents=True)
    (root / "outcome" / "targets").mkdir()
    for path in (
        root / "outcome" / "controls" / "good.case",
        root / "outcome" / "controls" / "wrong.case",
        root / "outcome" / "targets" / "target.case",
    ):
        path.write_text(path.name, encoding="utf-8")
    reward_source = (
        "def score(task, response):\n    return 1.0 if response == 'paid-but-wrong' else 0.0\n"
    )
    (root / "grader.py").write_text(reward_source, encoding="utf-8")
    task = {"id": "target", "prompt": "return the reference", "reference": "correct"}
    (root / "tasks.jsonl").write_text(json.dumps(task) + "\n", encoding="utf-8")
    evaluator = root / "outcome" / "owner_evaluator.py"
    evaluator.write_text(
        "def score(task, response):\n    return 1.0 if response == task['reference'] else 0.0\n",
        encoding="utf-8",
    )
    evaluator_digest = PythonGrader.from_path(evaluator).describe().source_digest
    reward_digest = contracts.digest({"source": reward_source})
    execution = {
        "version": 1,
        "evaluator": {"path": "outcome/owner_evaluator.py", "entrypoint": "score"},
        "check": {
            "owner": "assay-owner",
            "source_digest": evaluator_digest,
            "interface": "reward_lens.graders.python_grader.PythonGrader.score/v1",
            "partition_id": "controls-v1",
        },
        "candidate_owner": "candidate",
        "audited_grader": {"owner": "candidate", "source_digest": reward_digest},
        "subject_ref": reward_digest,
        "controls": {
            "partition_path": "outcome/controls",
            "partition_id": "controls-v1",
            "known_good": [{"item_id": "good.case", "task": task, "response": "correct"}],
            "known_wrong": [{"item_id": "wrong.case", "task": task, "response": "wrong"}],
        },
        "target": {
            "partition_path": "outcome/targets",
            "partition_id": "target-v1",
            "tasks": [{"item_id": "target.case", "task": task}],
        },
    }
    (root / "outcome" / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
    (root / "rewardlens.yaml").write_text(
        "reward:\n  entry: grader.py\ntasks:\n  path: tasks.jsonl\n  prompt: prompt\n"
        "outcome:\n  kind: protected_test_suite\n  path: outcome/controls\n"
        "  execution: outcome/execution.json\n",
        encoding="utf-8",
    )
    from reward_lens.product.audit import registry
    from reward_lens.product.audit.run import _planned_project_method_ids
    from reward_lens.store import Project

    project = Project.open(root)
    execution["subject_ref"] = project.version(
        methods=_planned_project_method_ids(registry.discover())
    ).digest()
    (root / "outcome" / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
    return root


def test_project_proposal_persists_and_reopens_actual_supervised_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    root = _project(tmp_path)
    result = dispatch(
        None,
        "tools/call",
        {
            "name": "seeker.propose",
            "arguments": {
                "projectPath": str(root),
                "taskId": "target",
                "response": "paid-but-wrong",
                "rationale": "real candidate",
                "spend": "none",
            },
        },
    )
    body = result.structured_content
    assert body["resultType"] == "seeker_proposal"
    saved = Path(body["path"])
    assert saved.is_file()
    assert body["result"]["evidence"]["actor"] == "agent"
    outcome = body["result"]["evidence"]["outcome_evidence"]
    assert outcome["state"] == "FAIL"
    assert outcome["receipt"]["receipt_digest"].startswith("sha256:")


def test_project_route_refuses_ambiguous_legacy_session(tmp_path: Path) -> None:
    result = dispatch(
        None,
        "tools/call",
        {
            "name": "seeker.propose",
            "arguments": {
                "session": "legacy",
                "projectPath": str(tmp_path),
                "taskId": "one",
                "response": "x",
            },
        },
    )
    assert result.structured_content["resultType"] == "refused"


def test_stdio_project_proposal_executes_and_returns_saved_evidence(tmp_path: Path) -> None:
    """A real MCP child sends its candidate through the public saved-Audit route."""
    root = _project(tmp_path)
    script = f"""\
import asyncio
import json
import sys
from reward_lens.integrations import mcp as _reward_lens_mcp
from mcp import Client
from mcp.client.stdio import StdioServerParameters

async def main():
    async with Client(
        StdioServerParameters(
            command=sys.executable,
            args=['-u', '-m', 'reward_lens.integrations.mcp'],
            cwd='.',
            env={{'PYTHONDONTWRITEBYTECODE': '1',
                 'XDG_STATE_HOME': {str(tmp_path / "state-stdio")!r}}},
        ), mode='auto', read_timeout_seconds=20,
    ) as client:
        result = await client.call_tool('seeker.propose', {{
            'projectPath': {str(root)!r}, 'taskId': 'target',
            'response': 'paid-but-wrong', 'rationale': 'stdio candidate', 'spend': 'none',
        }})
        body = result.structured_content
        assert body['resultType'] == 'seeker_proposal'
        assert body['result']['evidence']['actor'] == 'agent'
        print(json.dumps({{'path': body['path'], 'evidence': body['result']['evidence']}}))

asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr
    body = json.loads(completed.stdout)
    saved = Path(body["path"])
    assert saved.is_file()
    assert body["evidence"]["outcome_evidence"]["receipt"]["receipt_digest"].startswith("sha256:")


def test_public_cli_discovers_lists_and_persists_project_proposal(tmp_path: Path) -> None:
    """The installed public command keeps one verified SDK root through its child."""
    root = _project(tmp_path)
    command = Path(sys.executable).parent / "reward-lens"
    assert command.is_file(), "the test interpreter must expose the public CLI entry point"
    script = f"""\
import asyncio
import json
from reward_lens.integrations import mcp as _reward_lens_mcp
from mcp import Client
from mcp.client.stdio import StdioServerParameters

async def main():
    async with Client(
        StdioServerParameters(
            command={str(command)!r}, args=['mcp'], cwd='.',
            env={{'XDG_STATE_HOME': {str(tmp_path / "state-cli")!r},
                 'PYTHONDONTWRITEBYTECODE': '1'}},
        ), mode='auto', read_timeout_seconds=20,
    ) as client:
        discovered = client.session.discover_result
        assert discovered is not None
        assert discovered.supported_versions == ['2026-07-28']
        listed = await client.list_tools()
        assert {{tool.name for tool in listed.tools}} >= {{'audit', 'doctor', 'seeker.propose'}}
        result = await client.call_tool('seeker.propose', {{
            'projectPath': {str(root)!r}, 'taskId': 'target',
            'response': 'paid-but-wrong', 'rationale': 'public CLI candidate', 'spend': 'none',
        }})
        body = result.structured_content
        assert body['resultType'] == 'seeker_proposal'
        assert body['result']['evidence']['actor'] == 'agent'
        assert body['result']['evidence']['outcome_evidence']['receipt']['receipt_digest'].startswith('sha256:')
        print(json.dumps({{'path': body['path'], 'evidence': body['result']['evidence']}}))

asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=35,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr
    body = json.loads(completed.stdout)
    saved = Path(body["path"])
    assert saved.is_file()
    assert body["evidence"]["outcome_evidence"]["receipt"]["receipt_digest"].startswith("sha256:")
