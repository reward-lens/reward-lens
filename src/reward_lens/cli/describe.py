"""`describe`: the command tree as one versioned JSON document.

An agent should read the surface as data rather than parse help text, which is what Algolia and
Fern ship and what click already knows. The tree is generated from the live command objects, so it
cannot describe a parameter the tree does not have.
"""

from __future__ import annotations

SCHEMA = "command-tree/1.0"


def tree() -> dict:
    """Every contracted command, its parameters, its output shape and its side-effect class."""
    import click

    from . import config, registry
    from .main import cli

    ctx = click.Context(cli, info_name=registry.PROG)
    commands = []
    for name in registry.ORDER:
        command = cli.get_command(ctx, name)
        commands.append(_command(name, command, ctx))
    return {
        "schema_version": SCHEMA,
        "program": registry.PROG,
        "precedence": list(config.PRECEDENCE),
        "exit_codes": {
            "0": "completed, and any decision asked for qualified",
            "1": "the decision was rejected by its policy",
            "2": "the decision is unresolved",
            "3": "a decision only the caller can make is pending, with no TTY to ask on",
            "4": "usage, configuration or input contract",
            "5": "a required capability, service or extra is unavailable",
            "6": "the budget cap was reached",
            "7": "internal execution or integrity failure",
            "130": "interrupted",
        },
        "commands": commands,
    }


def _command(name, command, ctx) -> dict:
    import click

    from . import config, registry

    row = {
        "name": name,
        "summary": registry.SUMMARIES[name],
        "parameters": [_parameter(parameter) for parameter in command.params],
        "output": registry.OUTPUT_SHAPES[name],
        "side_effect": registry.SIDE_EFFECTS[name],
        "subcommands": [],
    }
    if isinstance(command, click.Group):
        row["subcommands"] = [
            {
                "name": sub,
                "parameters": [_parameter(parameter) for parameter in command.get_command(ctx, sub).params],
            }
            for sub in sorted(command.list_commands(ctx))
        ]
    return row


_ENV_BY_FLAG = {
    "--format": "REWARD_LENS_FORMAT",
    "--color": "REWARD_LENS_COLOR",
    "--no-progress": "REWARD_LENS_NO_PROGRESS",
    "--offline": "REWARD_LENS_OFFLINE",
    "--max-budget-usd": "REWARD_LENS_MAX_BUDGET_USD",
    "--non-interactive": "REWARD_LENS_NON_INTERACTIVE",
}


def _parameter(parameter) -> dict:
    opts = list(getattr(parameter, "opts", ()) or ())
    name = next((opt for opt in opts if opt.startswith("--")), opts[0] if opts else parameter.name)
    kind = getattr(getattr(parameter, "type", None), "name", "text")
    choices = list(getattr(getattr(parameter, "type", None), "choices", ()) or ())
    row = {
        "name": name,
        "kind": "argument" if parameter.param_type_name == "argument" else "option",
        "type": kind,
        "required": bool(parameter.required),
        "multiple": bool(getattr(parameter, "multiple", False)),
        "help": getattr(parameter, "help", None),
    }
    if choices:
        row["choices"] = choices
    env = _ENV_BY_FLAG.get(name)
    if env:
        row["env"] = env
    return row


def from_dict(document: dict) -> dict:
    """The round trip: a tree read back is the same tree, field for field."""
    import json

    return json.loads(json.dumps(document))
