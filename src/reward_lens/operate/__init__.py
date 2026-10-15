"""The CLI surface and a typed boundary for the removed v3 MCP API.

``reward_lens.operate`` exposes the torch-free ``reward-lens`` CLI. Version 3 also exported a
transport-less, in-process MCP server here. That server was removed in 3.1: use the
``reward-lens mcp`` command or :mod:`reward_lens.integrations.mcp` for the supported isolated MCP
2 surface. The three old package exports remain lazy facades so an existing import receives a
typed refusal when it tries to use the removed API, without importing an MCP runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NoReturn

from reward_lens.operate.cli import app, main

__all__ = ["app", "main", "MCPServer", "Tool", "build_server"]

_LEGACY_MCP_EXPORTS = frozenset({"MCPServer", "Tool", "build_server"})


def _legacy_mcp_refusal(symbol: str) -> NoReturn:
    """Refuse the removed v3 transport-less MCP surface only when it is used."""
    from reward_lens.contracts import CapabilityUnavailable

    raise CapabilityUnavailable(
        code="RL0701",
        message=f"{symbol} belongs to the removed transport-less v3 MCP API",
        remediation=(
            "use the reward-lens mcp command or reward_lens.integrations.mcp for the supported "
            "isolated MCP 2 server"
        ),
        context={"legacy_symbol": symbol, "replacement": "reward-lens mcp"},
    )


def _legacy_mcp_export(name: str) -> object:
    """Construct one signature-compatible refusal facade without loading an MCP runtime."""
    if name == "Tool":

        @dataclass(frozen=True)
        class Tool:
            """Legacy v3 MCP tool declaration; construction now raises ``RL0701``."""

            name: str
            description: str
            input_schema: dict[str, Any]
            handler: Callable[[dict[str, Any]], str]
            gpu_gated: bool = False

            def __post_init__(self) -> None:
                _legacy_mcp_refusal("Tool")

        return Tool

    if name == "MCPServer":

        class MCPServer:
            """Legacy v3 in-process server; construction now raises ``RL0701``."""

            def __init__(self, store: Any | None = None) -> None:
                _legacy_mcp_refusal("MCPServer")

            def register(self, tool: Any) -> None:
                _legacy_mcp_refusal("MCPServer.register")

            def list_tools(self) -> list[dict[str, Any]]:
                _legacy_mcp_refusal("MCPServer.list_tools")

            def call_tool(
                self, name: str, arguments: dict[str, Any] | None = None
            ) -> dict[str, Any]:
                _legacy_mcp_refusal("MCPServer.call_tool")

        return MCPServer

    if name == "build_server":

        def build_server(store: Any | None = None) -> NoReturn:
            """Refuse construction of the removed v3 server."""
            _legacy_mcp_refusal("build_server")

        return build_server

    raise AttributeError(name)


def __getattr__(name: str) -> object:
    """Resolve deprecated v3 MCP names only when a caller explicitly requests one."""
    if name not in _LEGACY_MCP_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    facade = _legacy_mcp_export(name)
    globals()[name] = facade
    return facade


def __dir__() -> list[str]:
    """Advertise the retained public names without resolving their lazy facades."""
    return sorted(set(globals()) | _LEGACY_MCP_EXPORTS)
