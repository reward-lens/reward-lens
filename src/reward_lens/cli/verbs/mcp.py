"""`reward-lens mcp`: replace this process with the isolated MCP 2 child."""

from __future__ import annotations

import os
import sys

import click

from .. import registry


def run(*, transport: str, host: str, port: int, **opts) -> int:
    """Give stdio and signals to a fresh interpreter with private canonical MCP 2."""
    argv = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "reward_lens._vendor.mcp2.child",
        "--transport",
        transport,
        "--host",
        host,
        "--port",
        str(port),
    ]
    os.execv(sys.executable, argv)
    raise AssertionError("unreachable")


@click.command("mcp", short_help=registry.SUMMARIES["mcp"])
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    show_default=True,
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
def mcp(**opts) -> int:
    """Run the agent server on stdio."""
    return run(**opts)
