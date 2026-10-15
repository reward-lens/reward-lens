"""Offline conformance controls for the observational remote adapter."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reward_lens.contracts import CapabilityUnavailable, Counters
from reward_lens.execution import Result
from reward_lens.execution.linux import LinuxSandbox
from reward_lens.graders.base import CapabilityVector, Limits
from reward_lens.graders.remote import (
    REMOTE_REQUEST_SCHEMA_VERSION,
    REMOTE_RESPONSE_SCHEMA_VERSION,
    ExpiredRemoteObservation,
    HttpRemoteTransport,
    ProbeCase,
    RemoteAttempt,
    RemoteGrader,
    RemoteRequest,
    RemoteResponse,
)

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


class FixtureSandbox:
    """A declared execution context; no test represents it as service containment."""

    tier = "T0"


class NoRunSandbox:
    """Deliberately incomplete context: a provider route must reject it before I/O."""

    tier = "T0"


class BreachSandbox:
    """A governed run reporting a supervisor stop, used only for the refusal branch."""

    tier = "T0"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, argv, *, limits, cwd: Path, env, stdin=None, write_roots=()):
        self.calls += 1
        return Result(
            exit_code=-9,
            stdout=b"",
            stderr=b"wall limit exceeded",
            counters=Counters(calls=1),
            breach="wall",
            tier=self.tier,
        )


class FixtureEndpoint:
    """In-process attempt transport with a caller-selected observed scope."""

    def __init__(
        self,
        *,
        failures: int = 0,
        revision: str = "fixture-r17",
        costs: tuple[float | None, ...] = (None,),
        transport_scope: str = "in_process_fixture",
    ) -> None:
        self.failures = failures
        self.revision = revision
        self.costs = costs
        self.transport_scope = transport_scope
        self.requests: list[RemoteRequest] = []

    def attempt(self, request: RemoteRequest, *, sandbox, limits: Limits) -> RemoteAttempt:
        self.requests.append(request)
        assert sandbox.tier == "T0"
        if len(self.requests) <= self.failures:
            raise OSError("fixture temporarily unavailable")
        index = min(len(self.requests) - 1, len(self.costs) - 1)
        response = RemoteResponse(
            schema_version=REMOTE_RESPONSE_SCHEMA_VERSION,
            request_id=request.request_id,
            score=0.75,
            provider_revision=self.revision,
            cost_usd=self.costs[index],
        )
        return RemoteAttempt.from_response(
            request, response, transport_scope=self.transport_scope, limits=limits, sandbox=sandbox
        )


def _grader(endpoint: FixtureEndpoint, *, now: datetime = NOW) -> RemoteGrader:
    return RemoteGrader(
        provider="fixture://local/score",
        transport=endpoint,
        configured_revision="configured-r0",
        validity=timedelta(minutes=5),
        max_attempts=2,
        clock=lambda: now,
    )


def _score(grader: RemoteGrader, *, limits: Limits = Limits(wall_s=2)):
    return grader.score({"prompt": "Say hello"}, "hello", sandbox=FixtureSandbox(), limits=limits)


def test_request_and_response_are_versioned_and_deeply_snapshotted() -> None:
    task = {"prompt": "p", "nested": {"kept": True}}
    response = {"text": "x", "nested": {"kept": True}}
    request = RemoteRequest(REMOTE_REQUEST_SCHEMA_VERSION, "r-1", task, response)
    reply = RemoteResponse(REMOTE_RESPONSE_SCHEMA_VERSION, "r-1", 0.5, "fixture-r17", None)
    task["nested"]["kept"] = False
    response["nested"]["kept"] = False
    assert RemoteRequest.from_dict(request.to_dict()) == request
    assert request.to_dict()["task"]["nested"]["kept"] is True
    assert request.to_dict()["response"]["nested"]["kept"] is True
    assert RemoteResponse.from_dict(reply.to_dict()) == reply


def test_provider_revision_is_observed_from_response_not_configuration() -> None:
    envelope = _score(_grader(FixtureEndpoint(revision="observed-r17")))
    assert envelope.observations["provider_revision"] == "observed-r17"
    assert envelope.observations["provider_revision"] != "configured-r0"


def test_every_observation_has_window_and_expired_reuse_is_refused() -> None:
    endpoint = FixtureEndpoint()
    grader = _grader(endpoint)
    envelope = _score(grader)
    assert envelope.observations["validity_window"] == {
        "observed_at": "2026-09-26T12:00:00Z",
        "expires_at": "2026-09-26T12:05:00Z",
    }
    with pytest.raises(ExpiredRemoteObservation):
        grader.replay(
            envelope, sandbox=FixtureSandbox(), limits=Limits(), now=NOW + timedelta(minutes=6)
        )
    assert len(endpoint.requests) == 1


def test_attempt_receipts_preserve_each_retry_cost_limit_and_outcome() -> None:
    endpoint = FixtureEndpoint(failures=1, costs=(None, 0.02))
    envelope = _score(_grader(endpoint))
    attempts = envelope.observations["attempts"]
    assert len(endpoint.requests) == 2
    assert envelope.counters.calls == envelope.counters.api_calls == 2
    assert attempts[0]["outcome"] == "provider_unavailable"
    assert attempts[0]["cost"] == {"usd": None, "availability": "unavailable"}
    assert attempts[0]["limits"]["wall_s"] == 2
    assert attempts[0]["sandbox_tier"] == "T0"
    assert attempts[1]["outcome"] == "response"
    assert attempts[1]["cost"] == {"usd": 0.02, "availability": "reported_by_response"}
    assert envelope.observations["cost"] == {
        "reported_usd": 0.02,
        "availability": "partially_reported",
        "attempts": 2,
    }


def test_unavailable_provider_uses_real_shared_rl0711_refusal() -> None:
    with pytest.raises(CapabilityUnavailable) as caught:
        _score(_grader(FixtureEndpoint(failures=2)))
    assert caught.value.code == "RL0711"
    assert caught.value.exit_code == 5
    assert caught.value.context["provider"] == "fixture://local/score"
    assert caught.value.context["operation"] == "score"
    assert "attempt 2" in caught.value.context["detail"]


def test_raw_request_and_response_bytes_survive_post_call_mutation() -> None:
    endpoint = FixtureEndpoint()
    grader = _grader(endpoint)
    task = {"prompt": "p", "nested": {"before": 1}}
    response = {"answer": "x", "nested": {"before": 1}}
    envelope = grader.score(task, response, sandbox=FixtureSandbox(), limits=Limits(wall_s=2))
    request_ref, response_ref = envelope.artifacts
    before_request = grader.raw_artifact(request_ref["digest"])
    before_response = grader.raw_artifact(response_ref["digest"])
    task["nested"]["before"] = 2
    response["nested"]["before"] = 2
    endpoint.requests[0].task["nested"]["before"] = 3
    assert grader.raw_artifact(request_ref["digest"]) == before_request
    assert grader.raw_artifact(response_ref["digest"]) == before_response
    assert b'"before":1' in before_request


def test_transport_scope_comes_from_attempt_not_a_blanket_fixture_label() -> None:
    envelope = _score(_grader(FixtureEndpoint(transport_scope="http_endpoint")))
    assert envelope.observations["transport_scope"] == "http_endpoint"
    assert envelope.observations["transport_scope"] != "in_process_fixture"


def test_http_transport_serialises_versioned_wire_against_local_service() -> None:
    seen: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            size = int(self.headers["Content-Length"])
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(size))
            response = json.dumps(
                {
                    "schema_version": REMOTE_RESPONSE_SCHEMA_VERSION,
                    "request_id": seen["body"]["request_id"],
                    "score": 0.4,
                    "provider_revision": "local-http-r1",
                    "cost_usd": None,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format: str, *args) -> None:  # noqa: ANN001
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        transport = HttpRemoteTransport(f"http://127.0.0.1:{server.server_port}/score")
        grader = RemoteGrader(provider="local-http", transport=transport, clock=lambda: NOW)
        envelope = grader.score(
            {"prompt": "Say hello"},
            "hello",
            sandbox=LinuxSandbox(tier="T0"),
            limits=Limits(wall_s=2, network=True),
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert seen["path"] == "/score"
    assert seen["body"]["schema_version"] == REMOTE_REQUEST_SCHEMA_VERSION
    assert envelope.score == 0.4
    assert envelope.observations["transport_scope"] == "http_loopback"
    execution = envelope.observations["attempts"][0]["execution"]
    assert execution["exit_code"] == 0
    assert execution["tier"] == "T0"
    assert execution["network_requested"] is True


def test_http_transport_refuses_missing_sandbox_run_before_any_provider_exchange() -> None:
    request = RemoteRequest(REMOTE_REQUEST_SCHEMA_VERSION, "r-1", {"prompt": "p"}, "x")
    transport = HttpRemoteTransport("http://127.0.0.1:9/score")
    with pytest.raises(TypeError, match="Sandbox.run"):
        transport.attempt(request, sandbox=NoRunSandbox(), limits=Limits(network=True))


def test_http_transport_refuses_a_governed_limit_breach() -> None:
    request = RemoteRequest(REMOTE_REQUEST_SCHEMA_VERSION, "r-1", {"prompt": "p"}, "x")
    sandbox = BreachSandbox()
    transport = HttpRemoteTransport("http://127.0.0.1:9/score")
    with pytest.raises(OSError, match="breach"):
        transport.attempt(request, sandbox=sandbox, limits=Limits(network=True, wall_s=0.01))
    assert sandbox.calls == 1


def test_actual_probe_suite_and_repeatability_are_observed() -> None:
    endpoint = FixtureEndpoint()
    probe = _grader(endpoint).probe(
        (ProbeCase("baseline", {"prompt": "p"}, "x"), ProbeCase("repeat", {"prompt": "p"}, "x")),
        sandbox=FixtureSandbox(),
        limits=Limits(wall_s=2),
    )
    assert probe["passed"] is True
    assert probe["repeatability"] == {
        "n": 2,
        "identical_scores": True,
        "identical_provider_revisions": True,
    }


def test_manifest_is_black_box_observational_and_refuses_unproven_claim() -> None:
    manifest = _grader(FixtureEndpoint()).describe()
    assert manifest.family == "BLACK_BOX_OBSERVATIONAL"
    assert manifest.access_level == "black_box_observational"
    with pytest.raises(Exception, match="RL0702"):
        CapabilityVector(cost_metering=True, probed={})


def test_emitted_record_makes_no_source_or_containment_claim() -> None:
    emitted = _score(_grader(FixtureEndpoint())).to_dict()
    rendered = repr(emitted).lower()
    assert "source_visible" not in rendered
    assert "containment" not in rendered
    assert emitted["provenance"]["access_level"] == "black_box"
