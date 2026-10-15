"""The public audit runner's configured seeker journey, using a real local HTTP fixture."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reward_lens import api, contracts
from reward_lens.contracts import ProjectConfig
from reward_lens.execution import default_sandbox
from reward_lens.graders.python_grader import PythonGrader
from reward_lens.product import audit
from reward_lens.store import Project

GRADER = """
def score(task, response):
    return 1.0 if response == 'paid-but-wrong' else 0.0
"""


class _CompletionHandler(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler spelling
        size = int(self.headers["Content-Length"])
        type(self).calls.append(json.loads(self.rfile.read(size).decode("utf-8")))
        payload = json.dumps(
            {
                "choices": [{"message": {"content": "paid-but-wrong"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _server() -> tuple[ThreadingHTTPServer, threading.Thread]:
    _CompletionHandler.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CompletionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _inputs(tmp_path: Path, endpoint: str) -> tuple[Path, Path, Path]:
    grader = tmp_path / "grader.py"
    grader.write_text(GRADER, encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"id":"one","prompt":"find a response"}\n', encoding="utf-8")
    config = tmp_path / "seeker.json"
    config.write_text(
        json.dumps(
            {
                "mode": "local",
                "model": "fixture-local",
                "endpoint": endpoint,
                "max_proposals": 1,
                "recorded_proposals": [],
            }
        ),
        encoding="utf-8",
    )
    return grader, tasks, config


def test_local_config_constructs_an_actor_and_persists_its_real_http_meter(tmp_path: Path) -> None:
    server, thread = _server()
    try:
        grader, tasks, config = _inputs(tmp_path, f"http://127.0.0.1:{server.server_port}/v1")
        record = audit.run(
            api.AuditRequest(
                path=grader,
                tasks=tasks,
                seeker="local",
                seeker_config=config,
                offline=False,
            ),
            sandbox=default_sandbox(),
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert len(_CompletionHandler.calls) == 1
    seeker = record.extensions["https://reward-lens.dev/extensions/seeker"]
    actor = seeker["actor_record"]
    assert seeker["config"]["config_digest"].startswith("sha256:")
    assert actor["actor"] == "local"
    assert actor["calls"] == 1
    assert actor["proposals"][0]["inputs"]["request"]["model"] == "fixture-local"
    entry = next(item for item in record.measurement.exploits if item.kind == "estimate")
    assert entry.result["actor_record"] == actor
    assert record.intent.attack_budget.seeker_calls == 1
    assert record.cost.usd == "0.00"


def test_changed_config_does_not_reuse_the_previous_actor_measurement(tmp_path: Path) -> None:
    server, thread = _server()
    try:
        grader, tasks, config = _inputs(tmp_path, f"http://127.0.0.1:{server.server_port}/v1")
        first = api.AuditRequest(
            path=grader, tasks=tasks, seeker="local", seeker_config=config, offline=False
        )
        audit.run(first, sandbox=default_sandbox())
        data = json.loads(config.read_text(encoding="utf-8"))
        data["model"] = "fixture-local-v2"
        config.write_text(json.dumps(data), encoding="utf-8")
        second = audit.run(
            api.AuditRequest(
                path=grader, tasks=tasks, seeker="local", seeker_config=config, offline=False
            ),
            sandbox=default_sandbox(),
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert len(_CompletionHandler.calls) == 2
    assert (
        second.extensions["https://reward-lens.dev/extensions/seeker"]["config"]["model"]
        == "fixture-local-v2"
    )


def test_api_recorded_fixture_persists_actual_spend_and_call_count(
    tmp_path: Path,
) -> None:
    grader = tmp_path / "grader.py"
    grader.write_text(GRADER, encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"id":"one","prompt":"find a response"}\n', encoding="utf-8")
    config = tmp_path / "api.json"
    config.write_text(
        json.dumps(
            {
                "mode": "api",
                "model": "recorded-api",
                "max_proposals": 1,
                "recorded_proposals": ["paid-but-wrong", "another"],
                "pricing": {
                    "usd_per_1k_input": "10",
                    "usd_per_1k_output": "10",
                    "max_output_tokens": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    record = audit.run(
        api.AuditRequest(
            path=grader,
            tasks=tasks,
            seeker="api",
            seeker_config=config,
            max_budget_usd="1.00",
        ),
        sandbox=default_sandbox(),
    )

    actor = record.extensions["https://reward-lens.dev/extensions/seeker"]["actor_record"]
    assert actor["calls"] == 1
    assert actor["spent_usd"] == "0.02"
    assert actor["stopped"] == "proposals_exhausted"
    assert record.intent.attack_budget.seeker_calls == 1
    assert record.intent.attack_budget.usd == "0.02"
    assert record.cost.usd == "0.02"
    assert record.cost.api_calls == 1


@pytest.mark.parametrize(
    "unsafe",
    (
        "https://token@example.test/v1",
        "https://example.test/v1?api_key=secret",
        "https://example.test/v1#token",
    ),
)
def test_unsafe_endpoint_is_refused_before_a_config_or_assay_can_persist(
    tmp_path: Path, unsafe: str
) -> None:
    grader = tmp_path / "grader.py"
    grader.write_text(GRADER, encoding="utf-8")
    config = tmp_path / "unsafe.json"
    config.write_text(
        json.dumps(
            {
                "mode": "local",
                "model": "unsafe-fixture",
                "endpoint": unsafe,
                "max_proposals": 1,
                "recorded_proposals": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(contracts.UsageError) as refused:
        audit.run(
            api.AuditRequest(path=grader, seeker="local", seeker_config=config, offline=False),
            sandbox=default_sandbox(),
        )

    assert refused.value.context["field"] == "seeker_config.endpoint"
    assert not (tmp_path / "assays").exists()


def test_plain_file_seeker_uses_the_accepted_python_grader(monkeypatch, tmp_path: Path) -> None:
    """A declared `score(task, response)` must retain both arguments through the reward route."""
    grader, tasks, config = _inputs(tmp_path, "http://127.0.0.1:1/v1")
    config.write_text(
        json.dumps(
            {
                "mode": "agent",
                "model": "fixture-agent",
                "max_proposals": 1,
                "recorded_proposals": ["paid-but-wrong"],
            }
        ),
        encoding="utf-8",
    )
    import reward_lens.instruments.seeker.protocol as protocol

    observed: dict[str, object] = {}
    original = protocol.GraderScorer.__init__

    def capture(self: object, target: object, **kwargs: object) -> None:
        observed["target"] = target
        original(self, target, **kwargs)

    monkeypatch.setattr(protocol.GraderScorer, "__init__", capture)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    record = audit.run(
        api.AuditRequest(
            path=grader,
            tasks=tasks,
            seeker="agent",
            seeker_config=config,
        ),
        sandbox=default_sandbox(),
    )
    assert type(observed["target"]) is PythonGrader
    score = record.extensions["https://reward-lens.dev/extensions/seeker"]["actor_record"][
        "scores"
    ][0]
    assert score["exploit"] is False
    assert score["outcome_evidence"]["state"] == "UNQUALIFIED"


def test_project_supervised_target_receipt_is_required_for_a_public_exploit(
    monkeypatch, tmp_path: Path
) -> None:
    """The positive path is a real supervisor-backed target FAIL, not a host bool callback."""
    root = tmp_path / "project"
    controls, targets = root / "outcome" / "controls", root / "outcome" / "targets"
    controls.mkdir(parents=True)
    targets.mkdir()
    for path in (controls / "good.case", controls / "wrong.case", targets / "target.case"):
        path.write_text(path.name, encoding="utf-8")
    reward_source = (
        "def score(task, response):\n    return 1.0 if response == 'paid-but-wrong' else 0.0\n"
    )
    (root / "grader.py").write_text(reward_source, encoding="utf-8")
    (root / "tasks.jsonl").write_text(
        '{"id":"target","prompt":"return the reference","reference":"correct"}\n',
        encoding="utf-8",
    )
    evaluator = root / "outcome" / "owner_evaluator.py"
    evaluator.write_text(
        "def score(task, response):\n    return 1.0 if response == task['reference'] else 0.0\n",
        encoding="utf-8",
    )
    evaluator_digest = PythonGrader.from_path(evaluator).describe().source_digest
    reward_digest = contracts.digest({"source": reward_source})
    task = {"id": "target", "prompt": "return the reference", "reference": "correct"}
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
        "capability_floor": {
            "partition_path": "outcome/targets",
            "partition_id": "target-v1",
            "candidates": [{"item_id": "target.case", "task": task, "response": "correct"}],
        },
    }
    (root / "outcome" / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
    project = Project(
        root,
        ProjectConfig.model_validate(
            {
                "reward": {"entry": "grader.py"},
                "tasks": {"path": "tasks.jsonl", "prompt": "prompt"},
                "outcome": {
                    "kind": "protected_test_suite",
                    "path": "outcome/controls",
                    "execution": "outcome/execution.json",
                    "held_out_capability": {
                        "partition_id": "target-v1",
                        "path": "outcome/targets",
                        "floor": 1.0,
                    },
                },
            }
        ),
    )
    from reward_lens.product.audit import registry
    from reward_lens.product.audit.run import _planned_project_method_ids

    expected_subject_ref = project.version(
        methods=_planned_project_method_ids(registry.discover())
    ).digest()
    execution["subject_ref"] = expected_subject_ref
    (root / "outcome" / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
    config = root / "seeker.json"
    config.write_text(
        json.dumps(
            {
                "mode": "agent",
                "model": "sdk-recorded",
                "max_proposals": 1,
                "recorded_proposals": ["paid-but-wrong"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    record = audit.run(
        api.AuditRequest(path=root, seeker="agent", seeker_config=config),
        project=project,
        sandbox=default_sandbox(),
    )

    actor = record.extensions["https://reward-lens.dev/extensions/seeker"]["actor_record"]
    score = actor["scores"][0]
    assert score["exploit"] is True
    assert score["outcome_evidence"]["state"] == "FAIL"
    assert score["outcome_evidence"]["receipt"]["receipt_digest"].startswith("sha256:")
    assert actor["outcome_protocol"]["state"] == "QUALIFIED"
    assert record.intent.outcome_check.state == "qualified"
    assert record.subject.version.digest == expected_subject_ref
    floor = next(
        entry
        for entry in record.measurement.soundness
        if entry.entry_id == "soundness.capability_floor"
    )
    assert floor.check is not None and floor.check.passed is True
    assert floor.result["held"] is True
