"""The sole process that makes canonical MCP 2 importable."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib.metadata import distributions
from pathlib import Path

from .runtime import MCP2Identity, MCP2RuntimeError, materialize_private_runtime


def _private_distribution_version(root: Path) -> str:
    matches = [item for item in distributions(path=[str(root)]) if item.metadata["Name"] == "mcp"]
    if len(matches) != 1:
        raise MCP2RuntimeError("RL0711 private MCP 2 dist-info is ambiguous")
    return matches[0].version


def _load_sdk() -> tuple[object, type[object], MCP2Identity]:
    # The integration imports after this function.  It reuses this exact
    # cache-free identity through ``materialize_private_runtime`` rather than
    # creating a second verified root that would reject the loaded SDK.
    identity = materialize_private_runtime()
    if "mcp" in sys.modules:
        raise MCP2RuntimeError("RL0711 canonical MCP was imported before the isolated child")
    sys.path.insert(0, str(identity.site_root))
    mcp = importlib.import_module("mcp")
    if not Path(str(mcp.__file__)).resolve().is_relative_to(identity.site_root.resolve()):
        raise MCP2RuntimeError("RL0711 MCP resolved outside the private runtime")
    if _private_distribution_version(identity.site_root) != identity.version:
        raise MCP2RuntimeError("RL0711 private MCP 2 distribution version differs")
    from mcp.server import MCPServer

    if MCPServer.__module__ != identity.mcpserver_module:
        raise MCP2RuntimeError("RL0711 private MCPServer identity differs")
    return mcp, MCPServer, identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reward-lens-mcp2-child")
    parser.add_argument("--identity-json", action="store_true")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        mcp, server_type, identity = _load_sdk()
    except MCP2RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 5
    if args.identity_json:
        print(
            json.dumps(
                {
                    "mcp_file": str(mcp.__file__),
                    "mcpserver_module": server_type.__module__,
                    "version": _private_distribution_version(identity.site_root),
                },
                sort_keys=True,
            )
        )
        return 0
    from reward_lens.integrations.mcp import run

    run(transport=args.transport, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover - driven by the installed CLI
    raise SystemExit(main())
