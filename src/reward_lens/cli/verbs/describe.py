"""`reward-lens describe`: the command tree as versioned JSON, for tools and agents."""

from __future__ import annotations

import json

import click

from .. import describe as tree_module
from .. import registry
from . import _shared


def run(*, out, **opts) -> int:
    tree = tree_module.tree()
    return out.deliver(document=tree, text=json.dumps(tree, indent=2), exit_code=0)


@click.command("describe", short_help=registry.SUMMARIES["describe"], epilog=_shared.EPILOG)
@_shared.common
@_shared.verb("describe")
def describe(**opts) -> int:
    """The command tree as versioned JSON, for tools and agents."""
    return run(**opts)
