"""The public ``audit`` CLI carries its closed seeker configuration into a real local actor."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _Handler(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler spelling
        size = int(self.headers["Content-Length"])
        self.rfile.read(size)
        type(self).calls += 1
        body = json.dumps(
            {
                "choices": [{"message": {"content": "wrong"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_audit_cli_carries_seeker_config_to_the_local_protocol_actor(
    invoke, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    grader = project / "grader.py"
    grader.write_text("def score(task, response): return 1.0\n", encoding="utf-8")
    tasks = project / "tasks.jsonl"
    tasks.write_text('{"id":"t1","prompt":"p"}\n', encoding="utf-8")
    (project / "rewardlens.yaml").write_text(
        "reward:\n  entry: grader.py\n  kind: plain\ntasks:\n  path: tasks.jsonl\n  prompt: prompt\noutcome: null\n",
        encoding="utf-8",
    )
    _Handler.calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "seeker.json"
    config.write_text(
        json.dumps(
            {
                "mode": "local",
                "model": "cli-fixture",
                "endpoint": f"http://127.0.0.1:{server.server_port}/v1",
                "max_proposals": 1,
                "recorded_proposals": [],
            }
        ),
        encoding="utf-8",
    )
    try:
        result = invoke(
            "audit",
            str(project),
            "--seeker",
            "local",
            "--seeker-config",
            str(config),
            "--no-offline",
            "--exit-zero",
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert result.exit_code == 0, result.stderr
    assert _Handler.calls == 1
    assert json.loads(result.stdout)["execution"]["state"] == "complete"


def test_audit_cli_runs_recorded_agent_without_a_cap_or_paid_confirmation(
    invoke, tmp_path: Path
) -> None:
    project = tmp_path / "agent-project"
    project.mkdir()
    (project / "grader.py").write_text(
        "def score(task, response): return 1.0 if response == 'proposal' else 0.0\n",
        encoding="utf-8",
    )
    (project / "tasks.jsonl").write_text('{"id":"t1","prompt":"p"}\n', encoding="utf-8")
    (project / "rewardlens.yaml").write_text(
        "reward:\n  entry: grader.py\n  kind: plain\ntasks:\n  path: tasks.jsonl\n  prompt: prompt\noutcome: null\n",
        encoding="utf-8",
    )
    config = tmp_path / "agent.json"
    config.write_text(
        json.dumps(
            {
                "mode": "agent",
                "model": "calling-agent",
                "max_proposals": 1,
                "recorded_proposals": ["proposal"],
            }
        ),
        encoding="utf-8",
    )

    result = invoke(
        "audit", str(project), "--seeker", "agent", "--seeker-config", str(config), "--exit-zero"
    )

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["execution"]["state"] == "complete"
    assay = next((project / "assays").glob("*.assay.json"))
    saved = json.loads(assay.read_text(encoding="utf-8"))
    actor = saved["extensions"]["https://reward-lens.dev/extensions/seeker"]["actor_record"]
    assert actor["actor"] == "agent"
    assert actor["spent_usd"] == "0.00"
    assert saved["intent"]["attack_budget"] == {"seeker_calls": 0, "usd": "0.00"}
    assert saved["cost"]["usd"] == "0.00"
    assert saved["cost"]["api_calls"] == 0


def test_real_cli_budget_stop_seals_one_seeker_partial_and_leaves_the_run_resumable(
    cli, tmp_path: Path
) -> None:
    """The same paid Audit invocation retains its first proposal before it returns RL0501."""
    project = tmp_path / "budget-project"
    project.mkdir()
    (project / "grader.py").write_text(
        "def score(task, response): return 1.0 if response == 'paid-but-wrong' else 0.0\n",
        encoding="utf-8",
    )
    (project / "tasks.jsonl").write_text('{"id":"t1","prompt":"p"}\n', encoding="utf-8")
    (project / "rewardlens.yaml").write_text(
        "reward:\n  entry: grader.py\n  kind: plain\ntasks:\n  path: tasks.jsonl\n  prompt: prompt\noutcome: null\n",
        encoding="utf-8",
    )
    config = tmp_path / "budget-api.json"
    config.write_text(
        json.dumps(
            {
                "mode": "api",
                "model": "recorded-cap-fixture",
                "max_proposals": 2,
                "recorded_proposals": ["paid-but-wrong", "second-never-requested"],
                "pricing": {
                    "usd_per_1k_input": "0",
                    "usd_per_1k_output": "10",
                    "max_output_tokens": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state"
    result = cli(
        "audit",
        str(project),
        "--seeker",
        "api",
        "--seeker-config",
        str(config),
        "--max-budget-usd",
        "0.01",
        "--yes",
        env={"XDG_STATE_HOME": str(state)},
    )

    assert result.exit_code == 6
    assert "RL0501" in result.stderr
    manifests = list((state / "reward-lens" / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["state"] == "running"
    assert "record_path" not in manifest
    partial = manifests[0].parent / "partial" / "exploits.seeker.json"
    actor = json.loads(partial.read_text(encoding="utf-8"))["entries"][-1]["result"]["actor_record"]
    assert actor["stopped"] == "budget_exhausted"
    assert actor["calls"] == len(actor["proposals"]) == 1
