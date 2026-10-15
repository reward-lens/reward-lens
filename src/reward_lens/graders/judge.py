"""LLM-judge adapter with identity, raw-exchange references, and a calibration battery.

The adapter is deliberately a calibrated measurement family, not an outcome oracle.  Its local
fixture transport exercises the protocol without representing a commercial-provider result.  A
live transport is usable only after its dedicated probe passed and the call has both a network
allowance and an explicit key-and-budget witness.
"""

from __future__ import annotations

import datetime as _datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from reward_lens import errors
from reward_lens.contracts import Counters, canonical_bytes, digest
from reward_lens.execution import Result
from reward_lens.graders.base import (
    CAPABILITY_KEYS,
    AdapterCapabilityUnproven,
    CapabilityVector,
    EvidenceEnvelope,
    InvalidInput,
    Limits,
    Manifest,
    Sandbox,
)
from reward_lens.graders.connect import bind, detect

__all__ = [
    "JudgeAdapter",
    "JudgeBattery",
    "JudgeBatteryEntry",
    "JudgeBatteryReport",
    "JudgeIdentity",
    "ProviderReply",
    "SandboxJSONTransport",
]


def _now() -> str:
    return _datetime.datetime.now(tz=_datetime.timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class JudgeIdentity:
    """The versioned identity needed to interpret one judge result."""

    provider: str
    model: str
    prompt_digest: str
    settings_digest: str

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_digest": self.prompt_digest,
            "settings_digest": self.settings_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JudgeIdentity":
        return cls(
            provider=str(data["provider"]),
            model=str(data["model"]),
            prompt_digest=str(data["prompt_digest"]),
            settings_digest=str(data["settings_digest"]),
        )


@dataclass(frozen=True)
class ProviderReply:
    """One decoded provider reply and the execution/cost receipt that produced it."""

    raw: object
    counters: Counters = field(default_factory=Counters)
    transport_scope: str = "local_protocol_fixture"
    attempts: tuple[Mapping[str, Any], ...] = ()


class SandboxJSONTransport:
    """Versioned JSON provider wire transport executed through the supplied sandbox.

    Its command is deliberately injected. Production supplies a credential-aware client command;
    tests supply a fake sandbox that returns a wire reply without making a network request.
    """

    def __init__(self, argv: tuple[str, ...], *, cwd: Path) -> None:
        self.argv = tuple(argv)
        self.cwd = Path(cwd)

    def __call__(
        self,
        prompt: str,
        completion: str,
        *,
        sandbox: Sandbox | None,
        limits: Limits,
        **settings: object,
    ) -> ProviderReply:
        if sandbox is None:
            raise RuntimeError("a provider transport requires the supplied Sandbox")
        request = canonical_bytes(
            {
                "version": 1,
                "prompt": prompt,
                "request": json.loads(completion),
                "settings": settings,
            }
        )
        result: Result = sandbox.run(
            list(self.argv),
            limits=limits,
            cwd=self.cwd,
            env={},
            stdin=request,
        )
        attempt = {
            "argv": list(self.argv),
            "exit_code": result.exit_code,
            "tier": result.tier,
            "execution": "sandbox.run",
            "limits": asdict(limits),
            "counters": result.counters.model_dump(exclude_none=True),
            "outcome": "returned" if result.exit_code == 0 and result.breach is None else "failed",
            "cost": {
                "usd": result.counters.model_dump(exclude_none=True).get("usd", "unavailable")
            },
        }
        if result.breach is not None:
            raise RuntimeError(f"sandbox limit breached: {result.breach}")
        if result.exit_code != 0:
            raise RuntimeError(f"provider client exited {result.exit_code}")
        try:
            raw = json.loads(result.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as caught:
            raise RuntimeError("provider client emitted non-JSON response") from caught
        return ProviderReply(
            raw=raw,
            counters=result.counters,
            transport_scope="sandbox_provider_transport",
            attempts=(attempt,),
        )


@dataclass(frozen=True)
class JudgeBatteryEntry:
    """One named calibration check and the complete envelope it produced."""

    name: str
    envelope: EvidenceEnvelope
    metric: str
    value: float | None
    scope: str
    comparison: str


@dataclass(frozen=True)
class JudgeBatteryReport:
    """Calibration findings remain separate from an independent outcome check."""

    entries: tuple[JudgeBatteryEntry, ...]
    independent_check: str
    final_verdict: str | None
    disagreement: bool
    measures: tuple[JudgeBatteryEntry, ...]


class JudgeAdapter:
    """A protocol adapter over a provider transport checked by P-CONNECT first."""

    def __init__(
        self,
        *,
        identity: JudgeIdentity,
        prompt: str,
        settings: Mapping[str, Any],
        transport: Callable[..., object],
        fixture_conformance: bool,
        passing_probes: Mapping[str, bool],
        manifest: Manifest,
    ) -> None:
        self.identity = identity
        self._prompt = prompt
        self._settings = dict(settings)
        self._transport = transport
        self.fixture_conformance = fixture_conformance
        # A provider label comes from this concrete execution protocol, never from a callable's
        # return value or a caller-supplied probe.  Exact type is intentional: a subclass could
        # replace ``__call__`` and bypass the sandbox receipt below.
        self._concrete_provider_transport = type(transport) is SandboxJSONTransport
        self.passing_probes = {str(k): bool(v) for k, v in passing_probes.items()}
        self._manifest = manifest
        self._raw_artifacts: dict[str, bytes] = {}

    @classmethod
    def from_transport(
        cls,
        *,
        provider: str,
        model: str,
        prompt: str,
        settings: Mapping[str, Any],
        transport: Callable[..., object],
        fixture_conformance: bool,
        passing_probes: Mapping[str, bool],
        claimed_capabilities: Mapping[str, bool] | None = None,
    ) -> "JudgeAdapter":
        """Construct a judge only after its transport passed P-CONNECT signature inspection.

        `bind` is intentionally invoked as well as `detect`: its construction checks that the
        accepted connector can turn the detected callable into a wave-1 Grader.  The judge keeps
        its own transport because an LLM response is a structured verdict, not a scalar reward.
        """
        if not provider or not model or not prompt:
            raise ValueError("provider, model, and prompt must be non-empty")
        # The provider's score response is not a scalar reward, so P-CONNECT may validate its
        # signature but must not consume a provider reply through its scalar silent-mode probe.
        detection = detect(transport, probe_entry=False)
        bind(detection)
        probes = {key: False for key in CAPABILITY_KEYS}
        probes.update({str(k): bool(v) for k, v in passing_probes.items()})
        for capability, claimed in (claimed_capabilities or {}).items():
            if claimed and probes.get(capability) is not True:
                raise AdapterCapabilityUnproven(
                    capability=str(capability),
                    message=(
                        f"the judge manifest claims {capability}, and its provider probe did not pass"
                    ),
                    remediation="run and retain the provider capability probe before claiming it",
                )
        identity = JudgeIdentity(
            provider=provider,
            model=model,
            prompt_digest=digest({"prompt": prompt}),
            settings_digest=digest({"settings": dict(settings)}),
        )
        manifest = Manifest(
            family="JUDGE_CALIBRATED_FOR_SET_X",
            implementation_revision="reward-lens/graders/judge/1",
            input_schema_digest=digest({"task": ["id", "prompt"], "pair": ["left", "right"]}),
            output_schema_digest=digest(
                {"judge_verdict": ["left", "right", "tie"], "score": "number"}
            ),
            shape=detection.declared,
            score_domain="[0,1]",
            determinism_class="fixture_protocol" if fixture_conformance else "provider_unproven",
            state_model="stateless",
            access_level="provider_observational",
            declared_inputs=("task", "left", "right"),
            declared_outputs=(
                "judge_verdict",
                "score",
                "raw_request_digest",
                "raw_response_digest",
            ),
            network_policy="fixture_local" if fixture_conformance else "provider_network_required",
            replay_mode="fixture_replay" if fixture_conformance else "provider_replay_unproven",
            capabilities=CapabilityVector(probed=probes),
            source=prompt,
        )
        return cls(
            identity=identity,
            prompt=prompt,
            settings=settings,
            transport=transport,
            fixture_conformance=fixture_conformance,
            passing_probes=probes,
            manifest=manifest,
        )

    def describe(self) -> Manifest:
        return self._manifest

    def source(self) -> str:
        return self._prompt

    def reset(self) -> None:
        return None

    def validate_input(self, task: Any, response: Any) -> None | InvalidInput:
        if not isinstance(task, Mapping):
            return InvalidInput(field="task", reason="a judge task is an object")
        if not isinstance(task.get("id"), str) or not isinstance(task.get("prompt"), str):
            return InvalidInput(
                field="task", reason="a judge task needs string id and prompt fields"
            )
        if not isinstance(response, (str, Mapping)):
            return InvalidInput(field="response", reason="a response is text or an object")
        return None

    def raw_artifact(self, reference: str) -> bytes:
        """Return immutable canonical bytes after rechecking the digest reference."""
        raw = self._raw_artifacts[reference]
        if digest(json.loads(raw.decode("utf-8"))) != reference:
            raise RuntimeError("retained judge artifact no longer matches its digest")
        return raw

    def _raw_value(self, reference: str) -> object:
        return json.loads(self.raw_artifact(reference).decode("utf-8"))["judge_raw"]

    def score(
        self,
        task: dict,
        response: str | dict,
        *,
        sandbox: Sandbox | None,
        limits: Limits,
    ) -> EvidenceEnvelope:
        comparison = task.get("comparison_response", "") if isinstance(task, Mapping) else ""
        left = response if isinstance(response, str) else json.dumps(response, sort_keys=True)
        return self.score_pair(task, left, str(comparison), sandbox=sandbox, limits=limits)

    def replay(
        self,
        envelope: EvidenceEnvelope,
        *,
        sandbox: Sandbox | None,
        limits: Limits,
    ) -> EvidenceEnvelope:
        request = self._raw_value(envelope.request_digest)
        if not isinstance(request, Mapping):  # defensive only; retained requests are mappings
            raise ValueError("the retained request is not a judge request")
        return self.score_pair(
            dict(request["task"]),
            str(request["left"]),
            str(request["right"]),
            sandbox=sandbox,
            limits=limits,
            pair_order=tuple(request["pair_order"]),
        )

    def score_pair(
        self,
        task: Mapping[str, Any],
        left: str,
        right: str,
        *,
        sandbox: Sandbox | None,
        limits: Limits,
        pair_order: tuple[str, str] = ("left", "right"),
        live_key: str | None = None,
        budget_usd: float | None = None,
    ) -> EvidenceEnvelope:
        start = _now()
        bad = self.validate_input(task, left)
        request = {
            "task": dict(task),
            "left": left,
            "right": right,
            "pair_order": list(pair_order),
            "identity": self.identity.to_dict(),
            "settings": self._settings,
        }
        request_digest = self._retain(request)
        if bad is not None:
            return self._envelope(
                request_digest=request_digest,
                start=start,
                verdict="invalid_input",
                score=None,
                observations={"input": bad.to_dict()},
                errors=(f"{bad.field}: {bad.reason}",),
                sandbox=sandbox,
            )
        if not self.fixture_conformance and not self._concrete_provider_transport:
            return self._provider_unavailable(
                request_digest=request_digest,
                start=start,
                detail=(
                    "provider scope requires concrete SandboxJSONTransport execution with a "
                    "sandbox receipt; arbitrary callbacks are local/test only"
                ),
                sandbox=sandbox,
            )
        if not self.fixture_conformance and (
            self.passing_probes.get("live_provider") is not True
            or not limits.network
            or not live_key
            or budget_usd is None
            or budget_usd <= 0
        ):
            return self._provider_unavailable(
                request_digest=request_digest,
                start=start,
                detail="a passing live probe, network allowance, key, and positive budget are required",
                sandbox=sandbox,
            )
        rendered = self._render(task, left, right, pair_order)
        try:
            returned = self._transport(
                rendered,
                json.dumps(request, sort_keys=True),
                sandbox=sandbox,
                limits=limits,
                **self._settings,
            )
            if self.fixture_conformance:
                # Fixtures may exercise the response protocol, but they cannot self-label as a
                # provider execution or contribute a provider receipt.
                raw = returned.raw if isinstance(returned, ProviderReply) else returned
                reply = ProviderReply(raw=raw, transport_scope="fixture_conformance")
            elif (
                not isinstance(returned, ProviderReply)
                or returned.transport_scope != "sandbox_provider_transport"
                or not returned.attempts
                or not all(
                    attempt.get("execution") == "sandbox.run" and "limits" in attempt
                    for attempt in returned.attempts
                )
            ):
                return self._provider_unavailable(
                    request_digest=request_digest,
                    start=start,
                    detail="SandboxJSONTransport did not return a complete sandbox execution receipt",
                    sandbox=sandbox,
                )
            else:
                reply = returned
            raw_response = reply.raw
            if isinstance(raw_response, BaseException):
                raise raw_response
        except Exception as caught:
            return self._provider_unavailable(
                request_digest=request_digest,
                start=start,
                detail=f"{type(caught).__name__}: {caught}",
                sandbox=sandbox,
            )
        response_digest = self._retain(raw_response)
        parsed = self._parse(raw_response)
        if parsed is None:
            return self._envelope(
                request_digest=request_digest,
                start=start,
                verdict="parse_error",
                score=None,
                observations=self._exchange(
                    response_digest, pair_order, mode=reply.transport_scope, attempts=reply.attempts
                ),
                sandbox=sandbox,
                counters=reply.counters,
                errors=("judge response did not match the required verdict schema",),
            )
        verdict, score = parsed
        observations = self._exchange(
            response_digest, pair_order, mode=reply.transport_scope, attempts=reply.attempts
        )
        observations["judge_verdict"] = verdict
        return self._envelope(
            request_digest=request_digest,
            start=start,
            verdict="scored",
            score=score,
            observations=observations,
            sandbox=sandbox,
            counters=reply.counters,
        )

    def probe_capabilities(self) -> Mapping[str, bool]:
        """Return the observed probe map; construction never turns a method into a claim."""
        return dict(self.passing_probes)

    def _render(
        self, task: Mapping[str, Any], left: str, right: str, pair_order: tuple[str, str]
    ) -> str:
        values = {"left": left, "right": right}
        return "\n".join(
            [
                self._prompt,
                f"Task: {task['prompt']}",
                f"First: {values[pair_order[0]]}",
                f"Second: {values[pair_order[1]]}",
                "Return JSON with verdict and score.",
            ],
        )

    @staticmethod
    def _parse(raw_response: object) -> tuple[str, float] | None:
        value = raw_response
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(value, Mapping):
            return None
        verdict = value.get("verdict")
        score = value.get("score")
        if verdict not in {"left", "right", "tie"} or isinstance(score, bool):
            return None
        if not isinstance(score, (int, float)):
            return None
        return str(verdict), float(score)

    def _retain(self, raw: object) -> str:
        wrapped = {"judge_raw": raw}
        reference = digest(wrapped)
        self._raw_artifacts[reference] = canonical_bytes(wrapped)
        return reference

    def _exchange(
        self,
        response_digest: str,
        pair_order: tuple[str, str],
        *,
        mode: str,
        attempts: tuple[Mapping[str, Any], ...] = (),
    ) -> dict[str, Any]:
        return {
            "judge_identity": self.identity.to_dict(),
            "raw_exchange": {"request_digest": "", "response_digest": response_digest},
            "pair_order": list(pair_order),
            "mode": mode,
            "provider_attempts": [dict(item) for item in attempts],
        }

    def _mode(self) -> str:
        return "fixture_conformance" if self.fixture_conformance else "provider_refused_unexecuted"

    def _provider_unavailable(
        self, *, request_digest: str, start: str, detail: str, sandbox: Sandbox | None
    ) -> EvidenceEnvelope:
        context = {"provider": self.identity.provider, "operation": "score", "detail": detail}
        refusal = errors.make("RL0711", **context)
        errors_out = (refusal.code,)
        observations = {
            "judge_identity": self.identity.to_dict(),
            "refusal": {"code": refusal.code, "exit_code": refusal.exit_code, **refusal.context},
            "mode": self._mode(),
        }
        return self._envelope(
            request_digest=request_digest,
            start=start,
            verdict="provider_unavailable",
            score=None,
            observations=observations,
            errors=errors_out,
            sandbox=sandbox,
        )

    def _envelope(
        self,
        *,
        request_digest: str,
        start: str,
        verdict: str,
        score: float | None,
        observations: Mapping[str, Any],
        errors: tuple[str, ...] = (),
        sandbox: Sandbox | None,
        counters: Counters | None = None,
    ) -> EvidenceEnvelope:
        observed = dict(observations)
        raw_exchange = dict(observed.get("raw_exchange") or {})
        if raw_exchange:
            raw_exchange["request_digest"] = request_digest
            observed["raw_exchange"] = raw_exchange
        limitation = (
            "local protocol-faithful fixture conformance only; this is not a measurement of a named "
            "commercial model"
            if self.fixture_conformance
            else "provider route; independent outcome adjudication is required"
        )
        end = _now()
        return EvidenceEnvelope(
            run_id=digest({"request": request_digest, "start": start}),
            subject_digest=digest(self.identity.to_dict()),
            manifest_digest=self._manifest.digest(),
            request_digest=request_digest,
            start=start,
            end=end,
            seed=None,
            verdict=verdict,
            score=score,
            observations=observed,
            artifacts=tuple(
                {
                    "kind": "raw_request" if kind == "request_digest" else "raw_response",
                    "digest": value,
                }
                for kind, value in sorted((raw_exchange or {}).items())
                if value
            ),
            counters=counters or Counters(),
            provenance={"sandbox_tier": getattr(sandbox, "tier", None)},
            errors=errors,
            limitations=(limitation,),
        )


class JudgeBattery:
    """Run the minimum position, repeatability, and bias controls against one candidate pair."""

    def __init__(self, *, adapter: JudgeAdapter) -> None:
        self.adapter = adapter

    def run(
        self,
        task: Mapping[str, Any],
        left: str,
        right: str,
        *,
        independent_check: str,
        sandbox: Sandbox | None,
        limits: Limits,
    ) -> JudgeBatteryReport:
        entries: list[JudgeBatteryEntry] = []
        scope = (
            "fixture_conformance" if self.adapter.fixture_conformance else "provider_calibration"
        )

        def run(
            name: str,
            first: str,
            second: str,
            comparison: str,
            order: tuple[str, str] = ("left", "right"),
        ) -> EvidenceEnvelope:
            return self.adapter.score_pair(
                task, first, second, sandbox=sandbox, limits=limits, pair_order=order
            )

        first = run("repeat", left, right, "first of repeated identical pair")
        second = run("repeat", left, right, "second of repeated identical pair")
        repeat_value = float(
            first.observations.get("judge_verdict") == second.observations.get("judge_verdict")
        )
        entries.append(
            JudgeBatteryEntry(
                "repeat", second, "repeat_agreement", repeat_value, scope, "identical pair"
            )
        )
        swapped = run(
            "order_swapped", left, right, "same pair, candidate order swapped", ("right", "left")
        )
        order_effect = _score_difference(first, swapped)
        entries.append(
            JudgeBatteryEntry(
                "order_swapped",
                swapped,
                "order_effect",
                order_effect,
                scope,
                "baseline versus swapped order",
            )
        )
        position = run("position", left, left, "equal-content position control", ("right", "left"))
        entries.append(
            JudgeBatteryEntry(
                "position",
                position,
                "position_bias",
                _distance_from_half(position),
                scope,
                "equal-content positions",
            )
        )
        length = run(
            "length", left + "\n[irrelevant padding]", right, "length-padded left candidate"
        )
        entries.append(
            JudgeBatteryEntry(
                "length",
                length,
                "length_effect",
                _score_difference(first, length),
                scope,
                "baseline versus padded response",
            )
        )
        self_pref = run(
            "self_preference",
            f"[judge-family] {left}",
            f"[other-family] {right}",
            "ancestry-hidden self-preference contrast",
        )
        entries.append(
            JudgeBatteryEntry(
                "self_preference",
                self_pref,
                "self_preference_effect",
                _score_difference(first, self_pref),
                scope,
                "baseline versus labelled-family contrast",
            )
        )
        style = run(
            "style", f"Answer: {left}", f"<formal>{right}</formal>", "style-only rendering contrast"
        )
        entries.append(
            JudgeBatteryEntry(
                "style",
                style,
                "style_effect",
                _score_difference(first, style),
                scope,
                "baseline versus style contrast",
            )
        )
        primary = first.observations.get("judge_verdict")
        independent_value = float(primary != independent_check)
        entries.append(
            JudgeBatteryEntry(
                "independent_check",
                first,
                "independent_check_disagreement",
                independent_value,
                scope,
                "judge versus supplied independent check",
            )
        )
        return JudgeBatteryReport(
            entries=tuple(entries),
            independent_check=independent_check,
            final_verdict=primary if isinstance(primary, str) else None,
            disagreement=primary != independent_check,
            measures=tuple(entries),
        )


def _score_difference(first: EvidenceEnvelope, second: EvidenceEnvelope) -> float | None:
    if first.score is None or second.score is None:
        return None
    return abs(first.score - second.score)


def _distance_from_half(envelope: EvidenceEnvelope) -> float | None:
    return None if envelope.score is None else abs(envelope.score - 0.5)
