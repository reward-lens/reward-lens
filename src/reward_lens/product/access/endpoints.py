"""Finding a local OpenAI-compatible endpoint, and naming it as a seeker actor.

Loopback only. The doctor is allowed to look at this machine and at nothing else, so the host is
refused unless it is a loopback literal: there is no hostname resolution here and no route off the
box. A server that answers `GET /v1/models` with a list is an endpoint; anything else is not.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
from dataclasses import dataclass

__all__ = ["DEFAULT_HOSTS", "DEFAULT_PORTS", "Endpoint", "actor_name", "scan"]

#: The hosts a scan may touch. Each one is checked again at call time; this is documentation.
DEFAULT_HOSTS: tuple[str, ...] = ("127.0.0.1",)

#: Where local inference servers listen by default: LM Studio, GPT4All, a few text-generation
#: servers, vLLM and llama.cpp, Ollama. A port that answers something else is discarded.
DEFAULT_PORTS: tuple[int, ...] = (1234, 4891, 5000, 8000, 8080, 11434)

_READ_LIMIT = 1 << 16


@dataclass(frozen=True)
class Endpoint:
    """One OpenAI-compatible server answering on this machine."""

    base_url: str
    host: str
    port: int
    models: tuple[str, ...]

    def __str__(self) -> str:  # pragma: no cover - convenience for a caller printing one
        return self.base_url


def _models(host: str, port: int, timeout: float) -> tuple[str, ...] | None:
    """The model ids `GET /v1/models` returns, or `None` if nothing OpenAI-compatible is there."""
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/v1/models", headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            return None
        payload = json.loads(response.read(_READ_LIMIT).decode("utf-8", "replace"))
    except Exception:
        return None
    finally:
        try:
            connection.close()
        except Exception:  # pragma: no cover - closing a refused connection
            pass
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    ids = tuple(
        str(item["id"]) for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    return ids


def scan(
    *,
    host: str = DEFAULT_HOSTS[0],
    ports: tuple[int, ...] = DEFAULT_PORTS,
    timeout: float = 0.15,
) -> tuple[Endpoint, ...]:
    """Every OpenAI-compatible server answering on `host`, in port order.

    Raises `ValueError` for a host that is not a loopback literal: the doctor makes no network call
    that leaves this machine, and a hostname could resolve anywhere.
    """
    address = ipaddress.ip_address(host)
    if not address.is_loopback:
        raise ValueError(
            f"the endpoint scan is loopback only and {host} is not a loopback address; "
            "the doctor makes no call that leaves this machine"
        )
    found: list[Endpoint] = []
    for port in ports:
        models = _models(host, int(port), timeout)
        if models is None:
            continue
        found.append(
            Endpoint(
                base_url=f"http://{host}:{int(port)}/v1",
                host=host,
                port=int(port),
                models=models,
            )
        )
    return tuple(found)


def actor_name(endpoint: Endpoint) -> str:
    """How the seeker would name this endpoint as its actor."""
    if endpoint.models:
        return f"{endpoint.base_url} ({', '.join(endpoint.models[:3])})"
    return f"{endpoint.base_url} (it named no model)"
