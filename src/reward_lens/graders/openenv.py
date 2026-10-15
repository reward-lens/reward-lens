"""Configured OpenEnv 0.6.0 grading through one local reset and one typed step.

The adapter never discovers a bare ``step`` method, starts a provider, or drives an episode past
its first action. A project explicitly provides a client factory and reset/action bridges.
"""

from __future__ import annotations

import datetime as _datetime
import importlib
import importlib.util
import inspect
import math
import secrets
import sys
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

from reward_lens import errors
from reward_lens.contracts import Counters, digest
from reward_lens.contracts.project import OpenEnvSpec
from reward_lens.graders.base import EvidenceEnvelope, InvalidInput, Manifest

__all__ = ["OpenEnvGrader"]

_REVISION = "reward-lens/openenv/0.6.0"


def _now() -> str:
    return _datetime.datetime.now(tz=_datetime.timezone.utc).isoformat(timespec="microseconds")


def _binding_error(field: str, detail: str) -> Exception:
    return errors.make("RL0003", field=field, detail=detail)


def _load_callable(reference: str, *, field: str, root: Path | None = None) -> Callable[..., Any]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise _binding_error(field, f"{reference!r} is not module:callable")
    try:
        local = None if root is None else root.joinpath(*module_name.split(".")).with_suffix(".py")
        if local is not None and local.is_file():
            loaded_name = (
                "_reward_lens_openenv_"
                + digest({"module": module_name, "path": str(local.resolve())}).removeprefix(
                    "sha256:"
                )[:16]
            )
            module = sys.modules.get(loaded_name)
            if module is None:
                spec = importlib.util.spec_from_file_location(loaded_name, local)
                if spec is None or spec.loader is None:
                    raise ImportError(f"could not create a module specification for {local}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[loaded_name] = module
                spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        candidate = getattr(module, attribute)
    except (ImportError, AttributeError) as caught:
        raise _binding_error(field, f"{reference!r} does not resolve: {caught}") from caught
    if not callable(candidate):
        raise _binding_error(field, f"{reference!r} does not name a callable")
    return candidate


def _protocol() -> tuple[type[Any], type[Any], type[Any]]:
    """Load the optional real protocol only on an actual configured operation."""
    try:
        env_client = importlib.import_module("openenv.core").EnvClient
        action = importlib.import_module("openenv.core.env_server").Action
        step_result = importlib.import_module("openenv.core.client_types").StepResult
    except (ImportError, AttributeError) as caught:
        raise errors.make(
            "RL0704", extra="openenv", capability="stateful_tool_environment"
        ) from caught
    return env_client, action, step_result


def _redacted_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, host + port, parsed.path, parsed.query, ""))


def _source_for(*callables: Callable[..., Any]) -> str:
    parts = [inspect.getsource(OpenEnvGrader)]
    for callable_ in callables:
        try:
            parts.append(inspect.getsource(callable_))
        except (OSError, TypeError):
            parts.append(repr(callable_))
    return "\n".join(parts)


class OpenEnvGrader:
    """The C042 typed binding for a local OpenEnv endpoint."""

    def __init__(
        self,
        config: OpenEnvSpec,
        *,
        client_factory: Callable[..., Any],
        reset_bridge: Callable[..., Any],
        action_bridge: Callable[..., Any],
    ) -> None:
        self.config = config
        self._client_factory = client_factory
        self._reset_bridge = reset_bridge
        self._action_bridge = action_bridge
        self._source = _source_for(client_factory, reset_bridge, action_bridge)
        self._manifest = Manifest(
            family="stateful_tool_environment",
            implementation_revision=_REVISION,
            input_schema_digest=digest({"task": "mapping", "response": ["str", "mapping"]}),
            output_schema_digest=digest(
                {"score": "finite-number", "done": "bool", "observation": "object"}
            ),
            shape="configured_openenv",
            state_model="reset_then_single_step",
            access_level="bridge_source_visible",
            declared_inputs=("task", "response"),
            declared_outputs=("score", "done", "observation"),
            network_policy="local_endpoint_only",
            resource_needs={
                "openenv_version": config.version,
                "wheel_sha256": config.wheel_sha256,
                "base_url_scheme": urlsplit(config.base_url).scheme,
                "connect_timeout_s": config.connect_timeout_s,
                "message_timeout_s": config.message_timeout_s,
            },
            replay_mode="single_step_replay_unproven",
            source=self._source,
        )

    @classmethod
    def from_config(
        cls, config: OpenEnvSpec, root: Path | None = None, *, entry: str
    ) -> "OpenEnvGrader":
        """Resolve explicit project bridges; the caller supplies ``reward.entry`` explicitly."""
        return cls(
            config,
            client_factory=_load_callable(entry, field="reward.entry", root=root),
            reset_bridge=_load_callable(config.reset, field="reward.openenv.reset", root=root),
            action_bridge=_load_callable(config.action, field="reward.openenv.action", root=root),
        )

    def describe(self) -> Manifest:
        return self._manifest

    def source(self) -> str:
        return self._source

    def validate_input(self, task: dict, response: str | dict) -> None | InvalidInput:
        if not isinstance(task, dict):
            return InvalidInput("task", "OpenEnv tasks must be mappings")
        if not isinstance(response, (str, dict)):
            return InvalidInput("response", "OpenEnv responses must be strings or mappings")
        return None

    def score(
        self, task: dict, response: str | dict, *, sandbox: Any = None, limits: Any = None
    ) -> EvidenceEnvelope:
        del sandbox, limits
        started = _now()
        request_digest = digest({"task": task, "response": response})
        invalid = self.validate_input(task, response)
        if invalid is not None:
            return self._envelope(
                started, request_digest, "invalid_input", None, errors=(invalid.reason,)
            )
        try:
            _, action_type, step_result_type = _protocol()
            reset_kwargs = self._reset_bridge(task)
            if not isinstance(reset_kwargs, Mapping):
                raise TypeError("reset bridge returned a non-mapping")
            action = self._action_bridge(task, response)
            if not isinstance(action, action_type):
                raise _binding_error(
                    "reward.openenv.action",
                    "the action bridge must return an OpenEnv Action, never a raw dictionary",
                )
            client = self._client_factory(
                base_url=self.config.base_url,
                connect_timeout_s=self.config.connect_timeout_s,
                message_timeout_s=self.config.message_timeout_s,
            )
            with client.sync() as sync_client:
                reset = sync_client.reset(**dict(reset_kwargs))
                stepped = sync_client.step(action)
            if not isinstance(stepped, step_result_type):
                raise TypeError("client.step did not return an OpenEnv StepResult")
            reward = stepped.reward
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
            ):
                raise ValueError(
                    "OpenEnv StepResult.reward must be a finite int or float, never bool"
                )
        except Exception as caught:
            if getattr(caught, "code", None) in {"RL0003", "RL0704"}:
                raise
            return self._envelope(
                started,
                request_digest,
                "grader_error",
                None,
                errors=(f"OpenEnv {type(caught).__name__}: {caught}",),
            )

        limitations: tuple[str, ...] = (
            () if stepped.done else ("one step only; episode remained open",)
        )
        provenance = {
            "package": self.config.package,
            "version": self.config.version,
            "wheel_sha256": self.config.wheel_sha256,
            "base_url": _redacted_endpoint(self.config.base_url),
            "reset_digest": digest(dict(reset_kwargs)),
            "action_class": type(action).__qualname__,
            "returned_observation_class": type(stepped.observation).__qualname__,
            "done": stepped.done,
            "step_count": 1,
            "task_digest": digest({"task": task}),
            "response_digest": digest({"response": response}),
        }
        return self._envelope(
            started,
            request_digest,
            "scored",
            float(reward),
            observations={
                "observation_class": type(stepped.observation).__qualname__,
                "reset_observation_class": type(reset.observation).__qualname__,
            },
            provenance=provenance,
            limitations=limitations,
        )

    def _envelope(
        self,
        started: str,
        request_digest: str,
        verdict: str,
        score: float | None,
        *,
        observations: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        errors: tuple[str, ...] = (),
        limitations: tuple[str, ...] = (),
    ) -> EvidenceEnvelope:
        return EvidenceEnvelope(
            run_id="openenv-" + secrets.token_hex(6),
            subject_digest=digest(self.config.model_dump(mode="json")),
            manifest_digest=self._manifest.digest(),
            request_digest=request_digest,
            start=started,
            end=_now(),
            seed=None,
            verdict=verdict,
            score=score,
            observations=dict(observations or {}),
            counters=Counters(),
            provenance=dict(provenance or {}),
            errors=errors,
            limitations=limitations,
        )
