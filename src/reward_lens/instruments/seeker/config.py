"""Closed, replay-identifiable configuration for the audit seeker's actors.

The request selects an arm; this file supplies the finite actor parameters that make that arm
real.  It deliberately accepts JSON only and rejects unknown keys, so a typo cannot become an
unrecorded provider setting.  Credentials are named by environment-variable *name* only.  Their
values are read only when a real HTTP request is sent and are never part of this configuration or
an assay record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from reward_lens import contracts

from .actors import AgentActor, ApiActor, LocalActor, RecordedTransport, ScriptedActor
from .budget import Budget, Pricing
from .protocol import Scorer

__all__ = ["SeekerConfig", "load", "build_actor"]


def _refuse(field: str, detail: str) -> contracts.UsageError:
    return contracts.UsageError(
        code="RL0003",
        message=f"seeker configuration field {field} is invalid: {detail}",
        remediation="write a closed JSON seeker configuration with only documented actor fields",
        context={"field": f"seeker_config.{field}", "detail": detail},
    )


def _endpoint(value: object) -> str:
    """Accept a secret-free absolute HTTP base URL before it can enter an assay.

    URL userinfo, query strings, and fragments are credentials-by-another-name often enough that
    the closed configuration does not preserve them.  The runner parses only this already-checked
    value when it opens the protocol transport.
    """
    if not isinstance(value, str) or not value:
        raise _refuse("endpoint", "must be a non-empty URL string when supplied")
    scheme, separator, remainder = value.partition("://")
    if scheme not in {"http", "https"} or not separator or not remainder:
        raise _refuse("endpoint", "must be an absolute http or https URL")
    authority = remainder.split("/", 1)[0]
    if not authority:
        raise _refuse("endpoint", "must name a host")
    if "@" in authority:
        raise _refuse("endpoint", "must not contain userinfo")
    if "?" in value or "#" in value:
        raise _refuse("endpoint", "must not contain a query string or fragment")
    return value


@dataclass(frozen=True)
class SeekerConfig:
    """The bytes and effective settings that identify one actor construction."""

    mode: str
    model: str
    endpoint: str | None
    pricing: Pricing | None
    max_proposals: int
    recorded_proposals: tuple[str, ...]
    credential_env: str | None
    config_bytes: str
    digest: str

    def public(self) -> dict[str, Any]:
        """The complete non-secret configuration that an assay may retain."""
        return {
            "mode": self.mode,
            "model": self.model,
            "endpoint": self.endpoint,
            "pricing": None if self.pricing is None else self.pricing.to_dict(),
            "max_proposals": self.max_proposals,
            "recorded_proposals": list(self.recorded_proposals),
            "credential_env": self.credential_env,
            "config_bytes": self.config_bytes,
            "config_digest": self.digest,
        }


def load(path: Path | str, *, requested_mode: str) -> SeekerConfig:
    """Read one closed JSON document and require it to agree with the typed request."""
    source = Path(path)
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise _refuse("file", f"{source} is not readable JSON: {failure}") from failure
    if not isinstance(value, dict):
        raise _refuse("file", "the document must be a JSON object")
    allowed = {
        "mode",
        "model",
        "endpoint",
        "pricing",
        "max_proposals",
        "recorded_proposals",
        "credential_env",
    }
    extra = sorted(set(value) - allowed)
    if extra:
        raise _refuse(extra[0], "unknown field")
    mode = value.get("mode")
    if mode not in {"api", "local", "agent", "scripted"}:
        raise _refuse("mode", "must be api, local, agent, or scripted")
    if mode != requested_mode:
        raise _refuse("mode", f"is {mode!r}, but --seeker requested {requested_mode!r}")
    model = value.get("model")
    if not isinstance(model, str) or not model:
        raise _refuse("model", "must be a non-empty string")
    endpoint = value.get("endpoint")
    if endpoint is not None:
        endpoint = _endpoint(endpoint)
    if mode == "local" and endpoint is None:
        raise _refuse("endpoint", "is required for a local HTTP actor")
    if mode == "api" and endpoint is None and not value.get("recorded_proposals"):
        raise _refuse("endpoint", "is required for a live api actor")
    maximum = value.get("max_proposals", 8)
    if type(maximum) is not int or maximum < 1:
        raise _refuse("max_proposals", "must be a positive integer")
    rows = value.get("recorded_proposals", ())
    if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
        raise _refuse("recorded_proposals", "must be an array of strings")
    credential_env = value.get("credential_env")
    if credential_env is not None and (
        not isinstance(credential_env, str) or not credential_env or "=" in credential_env
    ):
        raise _refuse("credential_env", "must name an environment variable, never contain a secret")
    pricing_value = value.get("pricing")
    pricing: Pricing | None = None
    if mode == "api":
        if not isinstance(pricing_value, dict) or set(pricing_value) != {
            "usd_per_1k_input",
            "usd_per_1k_output",
            "max_output_tokens",
        }:
            raise _refuse(
                "pricing", "api requires explicit input, output, and maximum-token prices"
            )
        try:
            input_price = Decimal(str(pricing_value["usd_per_1k_input"]))
            output_price = Decimal(str(pricing_value["usd_per_1k_output"]))
            output_tokens = pricing_value["max_output_tokens"]
            if (
                type(output_tokens) is not int
                or output_tokens < 1
                or input_price < 0
                or output_price < 0
            ):
                raise ValueError("prices must be non-negative and max_output_tokens positive")
            pricing = Pricing(input_price, output_price, max_output_tokens=output_tokens)
        except Exception as failure:
            raise _refuse("pricing", str(failure)) from failure
    elif pricing_value is not None:
        raise _refuse("pricing", "only the api actor has provider pricing")
    if mode == "api" and not rows and credential_env is None:
        raise _refuse(
            "credential_env", "a live api actor names the environment variable holding its key"
        )
    return SeekerConfig(
        mode=mode,
        model=model,
        endpoint=endpoint,
        pricing=pricing,
        max_proposals=maximum,
        recorded_proposals=tuple(rows),
        credential_env=credential_env,
        config_bytes=raw.decode("utf-8"),
        digest="sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def _recorded_transport(config: SeekerConfig) -> RecordedTransport:
    replies = [
        {
            "choices": [{"message": {"content": response}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        for response in config.recorded_proposals
    ]
    if not replies:
        raise _refuse("recorded_proposals", "is required for this offline actor")
    return RecordedTransport(replies, model=config.model)


def _scripted_caller(rows: tuple[str, ...]) -> Callable[[str], Callable[[dict], dict]]:
    if not rows:
        raise _refuse("recorded_proposals", "agent configuration needs recorded proposals")

    def bind(session: str) -> Callable[[dict], dict]:
        def call(request: dict) -> dict:
            from reward_lens.api import _dispatch

            tool = _dispatch.load("reward_lens.instruments.seeker.actors.agent:propose_tool")
            if tool is None:
                raise RuntimeError("the seeker.propose SDK target is unavailable")
            response = rows[min(len(request.get("attempts", ())), len(rows) - 1)]
            return tool(
                {
                    "session": session,
                    "task_id": request["task_id"],
                    "response": response,
                    "rationale": "recorded agent proposal",
                }
            )

        return call

    return bind


def build_actor(
    config: SeekerConfig,
    *,
    scorer: Scorer,
    max_budget_usd: str | None,
    offline: bool,
    transport: Any = None,
):
    """Construct exactly the requested existing actor; no implicit fallback arm exists."""
    if config.mode == "api":
        if offline and not config.recorded_proposals:
            raise _refuse("offline", "a live api actor requires --no-offline")
        chosen = _recorded_transport(config) if config.recorded_proposals else transport
        if chosen is None:
            raise _refuse("endpoint", "the audit runner did not supply an api transport")
        return ApiActor(
            chosen,
            scorer=scorer,
            budget=Budget(max_budget_usd),
            pricing=config.pricing,
            model=config.model,
        )
    if config.mode == "local":
        if offline:
            raise _refuse("offline", "a local HTTP actor requires --no-offline")
        if transport is None:
            raise _refuse("endpoint", "the audit runner did not supply a local transport")
        return LocalActor(transport, scorer=scorer, model=config.model)
    if config.mode == "agent":
        return AgentActor(scorer=scorer, caller=_scripted_caller(config.recorded_proposals))
    return ScriptedActor(config.recorded_proposals, scorer=scorer)
