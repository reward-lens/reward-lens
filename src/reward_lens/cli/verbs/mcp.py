"""`reward-lens mcp`: run the agent server on stdio."""

from __future__ import annotations

import click

from .. import registry
from . import _shared


def run(*, out, **opts) -> int:
    raise _shared.not_in_this_build(
        "the MCP server", remedy="install a build that ships reward_lens.mcp, then run the same command"
    )


@click.command("mcp", short_help=registry.SUMMARIES["mcp"], epilog=_shared.EPILOG)
@_shared.common
@_shared.verb("mcp")
def mcp(**opts) -> int:
    """Run the agent server on stdio."""
    return run(**opts)
