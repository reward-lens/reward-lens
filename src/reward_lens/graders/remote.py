"""Versioned remote observations with evidence-bound transport receipts.

The adapter knows only request and response wire records.  It has no source access
to a provider and does not represent a caller's Sandbox as containment of a remote
service.  A transport receives the real Limits/Sandbox call context, records that
context per attempt, and reports the scope it actually observed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit

from reward_lens.contracts import Counters, canonical_bytes, digest
from reward_lens.errors import make
from reward_lens.graders.base import (
    CapabilityVector,
    EvidenceEnvelope,
    InvalidInput,
    Limits,
    Manifest,
    Sandbox,
)

REMOTE_REQUEST_SCHEMA_VERSION = "reward-lens.remote-request/1"
REMOTE_RESPONSE_SCHEMA_VERSION = "reward-lens.remote-response/1"

__all__ = [
    "REMOTE_REQUEST_SCHEMA_VERSION",
    "REMOTE_RESPONSE_SCHEMA_VERSION",
    "ExpiredRemoteObservation",
    "HttpRemoteTransport",
    "ProbeCase",
    "RemoteAttempt",
    "RemoteGrader",
    "RemoteRequest",
    "RemoteResponse",
    "RemoteTransport",
]


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_stamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _raw_digest(raw: bytes) -> str:
    return "sha256:" + sha256(raw).hexdigest()


def _limits_receipt(limits: Limits) -> dict[str, Any]:
    return asdict(limits)


def _sandbox_tier(sandbox: Sandbox) -> str:
    # The protocol's concrete implementations expose tier. A test double may
    # instead truthfully leave it unreported; that is not upgraded to a claim.
    return str(getattr(sandbox, "tier", "unreported"))


_HTTP_CLIENT = """
import json
import sys
from urllib.request import Request, urlopen

endpoint = sys.argv[1]
timeout = float(sys.argv[2])
body = sys.stdin.buffer.read()
request = Request(
    endpoint,
    data=body,
    headers={"Content-Type": "application/json", "Accept": "application/json"},
    method="POST",
)
with urlopen(request, timeout=timeout) as opened:
    sys.stdout.buffer.write(opened.read())
"""


def _execution_receipt(result: Any, *, network_requested: bool) -> dict[str, Any]:
    counters = result.counters.model_dump(exclude_none=True)
    return {
        "exit_code": result.exit_code,
        "breach": result.breach,
        "tier": result.tier,
        "sandbox_os": result.sandbox_os,
        "counters": counters,
        "network_requested": network_requested,
    }


@dataclass(frozen=True)
class RemoteRequest:
    """The explicit, deeply snapshotted request wire record."""

    schema_version: str
    request_id: str
    task: dict[str, Any]
    response: str | dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != REMOTE_REQUEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported remote request schema {self.schema_version!r}")
        if not self.request_id:
            raise ValueError("remote request_id is required")
        object.__setattr__(self, "task", deepcopy(dict(self.task)))
        object.__setattr__(self, "response", deepcopy(self.response))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "task": deepcopy(self.task),
            "response": deepcopy(self.response),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RemoteRequest":
        return cls(
            str(data["schema_version"]),
            str(data["request_id"]),
            dict(data["task"]),
            data["response"],
        )

    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class RemoteResponse:
    """The versioned response whose provider revision is an observation."""

    schema_version: str
    request_id: str
    score: float | None
    provider_revision: str
    cost_usd: float | None

    def __post_init__(self) -> None:
        if self.schema_version != REMOTE_RESPONSE_SCHEMA_VERSION:
            raise ValueError(f"unsupported remote response schema {self.schema_version!r}")
        if not self.request_id or not self.provider_revision:
            raise ValueError("remote response needs request_id and observed provider_revision")
        if self.score is not None and (isinstance(self.score, bool) or not isfinite(self.score)):
            raise ValueError("remote score must be a finite number or None")
        if self.cost_usd is not None and (self.cost_usd < 0 or not isfinite(self.cost_usd)):
            raise ValueError("remote cost must be a non-negative finite number or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "score": self.score,
            "provider_revision": self.provider_revision,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RemoteResponse":
        return cls(
            str(data["schema_version"]),
            str(data["request_id"]),
            data.get("score"),
            str(data["provider_revision"]),
            data.get("cost_usd"),
        )


@dataclass(frozen=True)
class RemoteAttempt:
    """One immutable response receipt with the actual supplied call limits."""

    response: RemoteResponse
    transport_scope: str
    request_bytes: bytes
    response_bytes: bytes
    limits: dict[str, Any]
    sandbox_tier: str
    execution: dict[str, Any] | None = None

    @classmethod
    def from_response(
        cls,
        request: RemoteRequest,
        response: RemoteResponse,
        *,
        transport_scope: str,
        limits: Limits,
        sandbox: Sandbox,
        execution: dict[str, Any] | None = None,
    ) -> "RemoteAttempt":
        if not transport_scope:
            raise ValueError("transport must report its observed scope")
        return cls(
            response=response,
            transport_scope=transport_scope,
            request_bytes=canonical_bytes(request.to_dict()),
            response_bytes=canonical_bytes(response.to_dict()),
            limits=_limits_receipt(limits),
            sandbox_tier=_sandbox_tier(sandbox),
            execution=deepcopy(execution),
        )


class RemoteTransport(Protocol):
    """One concrete transport attempt. It returns a receipt, not just a score."""

    def attempt(
        self, request: RemoteRequest, *, sandbox: Sandbox, limits: Limits
    ) -> RemoteAttempt: ...


class HttpRemoteTransport:
    """A concrete JSON-over-HTTP provider route.

    It serialises the versioned request into stdin for a sandboxed Python HTTP
    client. `limits.wall_s` is the client's HTTP timeout and the sandbox's own
    wall limit. `limits.network` must explicitly allow the route. The receipt is
    the `Sandbox.run` result; it makes no assertion that the remote service itself
    is contained.
    """

    transport_scope = "http_endpoint"

    def __init__(
        self,
        endpoint: str,
        *,
        transport_scope: str = "http_endpoint",
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("remote HTTP endpoint must be an http(s) URL")
        self.endpoint = endpoint
        host = urlsplit(endpoint).hostname
        self.transport_scope = (
            "http_loopback" if host in {"127.0.0.1", "::1", "localhost"} else transport_scope
        )

    def attempt(self, request: RemoteRequest, *, sandbox: Sandbox, limits: Limits) -> RemoteAttempt:
        if not limits.network:
            raise OSError("remote HTTP transport requires Limits(network=True)")
        run = getattr(sandbox, "run", None)
        if not callable(run):
            raise TypeError("remote HTTP transport requires a Sandbox.run execution seam")
        body = canonical_bytes(request.to_dict())
        with tempfile.TemporaryDirectory(prefix="reward-lens-remote-") as scratch:
            result = run(
                [sys.executable, "-B", "-s", "-c", _HTTP_CLIENT, self.endpoint, str(limits.wall_s)],
                limits=limits,
                cwd=Path(scratch),
                env={"PYTHONHASHSEED": "0"},
                stdin=body,
                write_roots=(),
            )
        execution = _execution_receipt(result, network_requested=True)
        if result.breach is not None:
            raise OSError(f"sandbox reported breach {result.breach}")
        if result.exit_code != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise OSError(f"sandboxed remote HTTP client exited {result.exit_code}: {detail}")
        try:
            decoded = json.loads(result.stdout)
        except (TypeError, ValueError) as caught:
            raise OSError(f"remote HTTP response was not versioned JSON: {caught}") from caught
        if not isinstance(decoded, dict):
            raise OSError("remote HTTP response was not an object")
        response = RemoteResponse.from_dict(decoded)
        return RemoteAttempt.from_response(
            request,
            response,
            transport_scope=self.transport_scope,
            limits=limits,
            sandbox=sandbox,
            execution=execution,
        )


@dataclass(frozen=True)
class ProbeCase:
    name: str
    task: dict[str, Any]
    response: str | dict[str, Any]


class ExpiredRemoteObservation(ValueError):
    """A response cannot be reused after its explicit validity interval."""


class RemoteGrader:
    """A black-box `Grader` with per-attempt observational evidence."""

    def __init__(
        self,
        *,
        provider: str,
        transport: RemoteTransport,
        configured_revision: str | None = None,
        validity: timedelta = timedelta(minutes=5),
        max_attempts: int = 1,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not provider:
            raise ValueError("remote provider identity is required")
        if validity <= timedelta(0) or max_attempts < 1:
            raise ValueError("remote validity and max_attempts must be positive")
        self.provider = provider
        self.transport = transport
        self.configured_revision = configured_revision
        self.validity = validity
        self.max_attempts = max_attempts
        self.clock = clock
        self._raw: dict[str, bytes] = {}
        self._manifest = Manifest(
            family="BLACK_BOX_OBSERVATIONAL",
            implementation_revision="remote-adapter/2",
            input_schema_digest=digest({"task": "object", "response": ["string", "object"]}),
            output_schema_digest=digest(
                RemoteResponse(
                    REMOTE_RESPONSE_SCHEMA_VERSION, "schema", 0.0, "observed", None
                ).to_dict()
            ),
            shape="remote_black_box",
            score_domain="unbounded",
            determinism_class="observed_repeatability_required",
            state_model="remote_provider_unobserved",
            access_level="black_box_observational",
            declared_inputs=("task", "response"),
            declared_outputs=("score", "provider_revision", "validity_window"),
            network_policy="transport_scope_observed_per_attempt",
            secret_policy="transport_scope_unobserved",
            replay_mode="validity_window_reuse_only",
            capabilities=CapabilityVector(probed={}),
        )

    def describe(self) -> Manifest:
        return self._manifest

    def validate_input(self, task: dict, response: str | dict) -> None | InvalidInput:
        if not isinstance(task, dict) or not task.get("prompt"):
            return InvalidInput(field="prompt", reason="the remote request needs a task prompt")
        if not isinstance(response, (str, dict)):
            return InvalidInput(field="response", reason="a remote response is text or a mapping")
        return None

    def source(self) -> None:
        return None

    def reset(self) -> None:
        return None

    def raw_artifact(self, raw_digest: str) -> bytes:
        raw = self._raw[raw_digest]
        if _raw_digest(raw) != raw_digest:
            raise ValueError("retained remote raw exchange no longer matches its digest")
        return raw

    def score(
        self, task: dict, response: str | dict, *, sandbox: Sandbox, limits: Limits
    ) -> EvidenceEnvelope:
        started = self.clock()
        invalid = self.validate_input(task, response)
        if invalid is not None:
            return self._envelope(
                None, None, started, self.clock(), "invalid_input", [], (invalid.reason,)
            )
        request = RemoteRequest(
            REMOTE_REQUEST_SCHEMA_VERSION,
            digest({"provider": self.provider, "task": task, "response": response}),
            task,
            response,
        )
        request_bytes = canonical_bytes(request.to_dict())
        request_raw_digest = self._retain(request_bytes)
        attempts: list[dict[str, Any]] = []
        failures: list[str] = []
        for number in range(1, self.max_attempts + 1):
            try:
                receipt = self.transport.attempt(request, sandbox=sandbox, limits=limits)
                if receipt.response.request_id != request.request_id:
                    raise OSError("remote response request_id does not match submitted request")
                response_raw_digest = self._retain(receipt.response_bytes)
                attempts.append(
                    self._attempt_record(
                        number,
                        "response",
                        receipt.transport_scope,
                        receipt.limits,
                        receipt.sandbox_tier,
                        request_raw_digest,
                        response_raw_digest,
                        receipt.response.cost_usd,
                        execution=receipt.execution,
                    )
                )
                return self._envelope(
                    request,
                    receipt,
                    started,
                    self.clock(),
                    "scored" if receipt.response.score is not None else "unscored",
                    attempts,
                    tuple(failures),
                )
            except Exception as caught:
                detail = f"attempt {number}: {type(caught).__name__}: {caught}"
                failures.append(detail)
                attempts.append(
                    self._attempt_record(
                        number,
                        "provider_unavailable",
                        str(getattr(self.transport, "transport_scope", "unobserved")),
                        _limits_receipt(limits),
                        _sandbox_tier(sandbox),
                        request_raw_digest,
                        None,
                        None,
                        detail,
                    )
                )
        raise make("RL0711", provider=self.provider, operation="score", detail="; ".join(failures))

    def replay(
        self,
        envelope: EvidenceEnvelope,
        *,
        sandbox: Sandbox,
        limits: Limits,
        now: datetime | None = None,
    ) -> EvidenceEnvelope:
        del sandbox, limits
        window = envelope.observations.get("validity_window")
        if not isinstance(window, dict) or "expires_at" not in window:
            raise ExpiredRemoteObservation("remote observation has no reusable validity window")
        if (now or self.clock()) >= _parse_stamp(str(window["expires_at"])):
            raise ExpiredRemoteObservation("remote observation validity window has expired")
        return envelope

    def probe(
        self, suite: tuple[ProbeCase, ...] | list[ProbeCase], *, sandbox: Sandbox, limits: Limits
    ) -> dict[str, Any]:
        if not suite:
            raise ValueError("remote probe suite must contain at least one case")
        cases = []
        for case in suite:
            envelope = self.score(case.task, case.response, sandbox=sandbox, limits=limits)
            cases.append(
                {
                    "name": case.name,
                    "score": envelope.score,
                    "provider_revision": envelope.observations.get("provider_revision"),
                    "validity_window": envelope.observations.get("validity_window"),
                }
            )
        return {
            "passed": True,
            "cases": cases,
            "repeatability": {
                "n": len(cases),
                "identical_scores": len({row["score"] for row in cases}) == 1,
                "identical_provider_revisions": len({row["provider_revision"] for row in cases})
                == 1,
            },
        }

    def _retain(self, raw: bytes) -> str:
        immutable = bytes(raw)
        raw_digest = _raw_digest(immutable)
        self._raw[raw_digest] = immutable
        return raw_digest

    @staticmethod
    def _attempt_record(
        number: int,
        outcome: str,
        transport_scope: str,
        limits: dict[str, Any],
        sandbox_tier: str,
        request_raw_digest: str,
        response_raw_digest: str | None,
        cost_usd: float | None,
        detail: str | None = None,
        execution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "number": number,
            "outcome": outcome,
            "transport_scope": transport_scope,
            "limits": dict(limits),
            "sandbox_tier": sandbox_tier,
            "request_digest": request_raw_digest,
            "cost": {
                "usd": cost_usd,
                "availability": "reported_by_response" if cost_usd is not None else "unavailable",
            },
        }
        if response_raw_digest is not None:
            out["response_digest"] = response_raw_digest
        if detail is not None:
            out["detail"] = detail
        if execution is not None:
            out["execution"] = deepcopy(execution)
        return out

    def _envelope(
        self,
        request: RemoteRequest | None,
        receipt: RemoteAttempt | None,
        started: datetime,
        ended: datetime,
        verdict: str,
        attempts: list[dict[str, Any]],
        errors: tuple[str, ...],
    ) -> EvidenceEnvelope:
        reported = [a["cost"]["usd"] for a in attempts if a["cost"]["usd"] is not None]
        availability = (
            "reported"
            if len(reported) == len(attempts) and attempts
            else ("partially_reported" if reported else "unavailable")
        )
        observations: dict[str, Any] = {
            "validity_window": {
                "observed_at": _stamp(ended),
                "expires_at": _stamp(ended + self.validity),
            },
            "attempts": attempts,
            "cost": {
                "reported_usd": round(sum(reported), 6) if reported else None,
                "availability": availability,
                "attempts": len(attempts),
            },
        }
        artifacts: tuple[dict[str, str], ...] = ()
        if request is not None and receipt is not None:
            request_digest = self._retain(receipt.request_bytes)
            response_digest = self._retain(receipt.response_bytes)
            artifacts = (
                {"kind": "remote_request", "digest": request_digest},
                {"kind": "remote_response", "digest": response_digest},
            )
            observations["provider_revision"] = receipt.response.provider_revision
            observations["transport_scope"] = receipt.transport_scope
        return EvidenceEnvelope(
            run_id=digest(
                {
                    "provider": self.provider,
                    "request": request.to_dict() if request else None,
                    "start": _stamp(started),
                }
            ),
            subject_digest=digest(
                {"provider": self.provider, "request": request.to_dict() if request else None}
            ),
            manifest_digest=self._manifest.digest(),
            request_digest=request.digest() if request else digest({"invalid": True}),
            start=_stamp(started),
            end=_stamp(ended),
            seed=None,
            verdict=verdict,
            score=receipt.response.score if receipt else None,
            observations=observations,
            artifacts=artifacts,
            counters=Counters(calls=len(attempts), api_calls=len(attempts)),
            provenance={"provider": self.provider, "access_level": "black_box"},
            errors=errors,
            limitations=("remote service behaviour is only response-observed",),
        )
