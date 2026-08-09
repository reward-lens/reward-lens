"""The root group. Click and the standard library only: this module is the `--help` path (D-30).

The group resolves each of the fourteen contracted names to `"module:function"` on demand, the way
cyclopts registers commands by import path, so no verb module is imported before its verb runs. The
root help is not click's; it is `registry.root_help()`, which the frozen golden is the test of.

Every exit path this tree can take passes through `LazyGroup.main`, so an expected failure leaves
two lines and a code on stderr and never a traceback (appendix G, defects 4 and 5).
"""

from __future__ import annotations

import importlib
import sys

import click

from . import registry


class LazyGroup(click.Group):
    """A click group whose subcommands are import paths until they are invoked."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(registry.ORDER)

    def get_command(self, ctx: click.Context, name: str):
        target = registry.COMMANDS.get(name)
        if target is None:
            return None
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        command = getattr(module, attribute)
        from . import availability

        # A-027: the verb is spelled right and is in the contract, so it keeps its help and its
        # usage; what it cannot do is run. The real command is built either way, because the
        # refusal still prints that command's options under `--help`.
        return _NotInThisBuild.standing_in_for(command) if availability.absent(name) else command

    def resolve_command(self, ctx, args):
        name, command, rest = super().resolve_command(ctx, args)
        return name, command, rest

    def format_help(self, ctx: click.Context, formatter) -> None:
        formatter.write(registry.root_help())

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except (click.exceptions.Exit, click.exceptions.Abort, click.ClickException):
            raise
        except KeyboardInterrupt:
            ctx.exit(_report(ctx, _interrupted()))
        except BaseException as unexpected:  # noqa: BLE001 - one place, and it never re-raises raw
            if isinstance(unexpected, (SystemExit, GeneratorExit)):
                raise
            if _carries_an_interrupt(unexpected):
                ctx.exit(_report(ctx, _interrupted()))
            if _is_internal(unexpected):
                ctx.exit(_report(ctx, unexpected))
            ctx.exit(_report(ctx, _internal(unexpected, _action(ctx))))

    def main(self, args=None, prog_name=None, complete_var=None, standalone_mode=True, **extra):
        try:
            returned = super().main(
                args=args,
                prog_name=prog_name or registry.PROG,
                complete_var=complete_var,
                standalone_mode=False,
                **extra,
            )
        except click.exceptions.Exit as done:
            code = done.exit_code
        except click.UsageError as misuse:
            code = _report(misuse.ctx, _usage(misuse))
        except click.ClickException as refused:
            code = _report(None, _usage(refused))
        except click.exceptions.Abort:
            code = _report(None, _interrupted())
        except KeyboardInterrupt:
            code = _report(None, _interrupted())
        else:
            code = returned if isinstance(returned, int) else 0
        if standalone_mode:
            sys.exit(code)
        return code


class _NotInThisBuild(click.Command):
    """A contracted verb whose engine is not in this build (A-026, A-027).

    It stands in for the real command and keeps everything a reader can still use: the name, the
    options, the help, the epilog. What it does not do is parse. The refusal goes in front of the
    parser so that `reward-lens trace` and `reward-lens trace ./run` give the same answer, rather
    than one of them arguing about arguments for a verb that could not run them anyway.

    `--help` passes through, because it is the one thing this command can honestly do and because
    the contract says every one of the fourteen answers it.
    """

    @classmethod
    def standing_in_for(cls, command: click.Command) -> click.Command:
        return cls(
            name=command.name,
            params=list(command.params),
            help=command.help,
            short_help=command.short_help,
            epilog=command.epilog,
            callback=command.callback,
            context_settings=dict(command.context_settings or {}),
        )

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if set(args) & set(ctx.help_option_names or ("--help", "-h")):
            return super().parse_args(ctx, args)
        from reward_lens import errors

        ctx.exit(_report(ctx, errors.make("RL0703", verb=self.name, command=registry.PROG)))
        return []  # pragma: no cover - `ctx.exit` raises


def _carries_an_interrupt(error: BaseException) -> bool:
    """Whether a `KeyboardInterrupt` is anywhere in the chain that raised this (D-22).

    A SIGINT that lands inside a library's own callback comes back wrapped. pydantic-core catches
    whatever a wrap serializer raised and re-raises it as `PydanticSerializationError`, an
    ordinary `Exception`, so the clause above never sees the interrupt and reward-lens reported a
    defect in itself for a key the user pressed. The chain still says what happened, so it is read:
    `__cause__` first, then `__context__`, with the links already seen remembered, because an
    exception chain can be a ring.
    """
    seen: set[int] = set()
    link: BaseException | None = error
    while link is not None and id(link) not in seen:
        if isinstance(link, KeyboardInterrupt):
            return True
        seen.add(id(link))
        link = link.__cause__ or link.__context__
    return False


def _is_internal(error: BaseException) -> bool:
    return type(error).__mro__[-2].__name__ == "RewardLensError" or any(
        base.__name__ == "RewardLensError" for base in type(error).__mro__
    )


def _usage(misuse: click.ClickException):
    from reward_lens import errors

    return errors.make("RL0001", detail=misuse.format_message(), command=registry.PROG)


def _action(ctx: click.Context | None) -> str:
    """What reward-lens was doing when it broke: the verb, and the phase it was in.

    RL0900's cause reads `failed while {action}`, so the slot wants a participle and a name.
    `invoked_subcommand` is set before click hands control to the verb, so it still names the
    verb when the verb is what raised; nothing is set while the group itself is still resolving.
    """
    name = getattr(ctx, "invoked_subcommand", None) if ctx is not None else None
    return f"running {name}" if name else "starting up"


def _internal(unexpected: BaseException, action: str = "starting up"):
    """RL0900, naming the action and carrying the exception that caused it.

    The exception travels in `context["exception"]`, which reaches the JSON envelope's
    `error.context` and the text output's `help:` block. A traceback never leaves this function:
    the type and the message are what a caller can act on, and the run directory keeps the rest.
    """
    from reward_lens import errors

    return errors.make(
        "RL0900",
        action=action,
        exception=f"{type(unexpected).__name__}: {unexpected}",
        command=registry.PROG,
    )


def _interrupted():
    from reward_lens import errors

    return errors.make("RL0130", command=registry.PROG)


def _report(ctx, error) -> int:
    """Render one error through the caller's own output object, or a bare one if none was built."""
    from .output import Output

    out = getattr(ctx, "obj", None) if ctx is not None else None
    if not isinstance(out, Output):
        out = Output(format="text", colour=False, progress="quiet")
        out.format = "text"
    return out.fail(error)


def _version(ctx: click.Context, param, value):
    if not value or ctx.resilient_parsing:
        return
    from importlib.metadata import version

    click.echo(f"{registry.PROG} {version('reward-lens')}")
    ctx.exit(0)


@click.group(
    cls=LazyGroup,
    invoke_without_command=True,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.option(
    "--version",
    is_flag=True,
    callback=_version,
    expose_value=False,
    is_eager=True,
    help="print the version and exit",
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Build better rewards. Test what they teach."""
    if ctx.invoked_subcommand is None:
        click.echo(registry.root_help(), nl=False)
        ctx.exit(0)


def main(argv: list[str] | None = None) -> int:
    """The console script entry point for `reward-lens` and `rlens`."""
    return cli.main(args=argv, prog_name=registry.PROG)


if __name__ == "__main__":  # pragma: no cover - `python -m reward_lens.cli.main`
    main()
