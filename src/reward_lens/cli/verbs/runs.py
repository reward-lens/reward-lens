"""`reward-lens runs`: the discovery half of D-27, so a caller that did not start a run can find it."""

from __future__ import annotations

import shutil

import click

from .. import ids, output, registry
from . import _shared


def _find(run_id: str):
    from reward_lens.store import RunDir

    ids.validate_identifier(run_id, field="run_id")
    try:
        return RunDir.find(run_id)
    except Exception as missing:
        from reward_lens import errors

        if getattr(missing, "code", None):
            raise
        raise errors.make("RL0621", run=run_id) from missing


def run(*, out, action: str, run_id: str | None = None, **opts) -> int:
    from reward_lens.store import RunDir

    if action == "list":
        rows = [_row(found) for found in RunDir.list()]
        document = {"schema_version": "run-list/1.0", "command": "runs", "runs": rows, "error": None}
        return out.deliver(document=document, text=_list_text(rows), exit_code=0)

    found = _find(run_id)
    if action == "show":
        row = _row(found)
        document = {"schema_version": "run-list/1.0", "command": "runs", "runs": [row], "error": None}
        return out.deliver(document=document, text=_show_text(row), exit_code=0)

    _shared.refuse_without_yes(
        out,
        decision=f"remove the run {found.run_id} and everything it recorded",
        argv=["reward-lens", "runs", "rm", found.run_id, "--yes"],
        command="runs",
    )
    shutil.rmtree(found.path, ignore_errors=False)
    document = {"schema_version": "run-list/1.0", "command": "runs", "runs": [], "removed": [found.run_id],
                "error": None}
    return out.deliver(document=document, text=f"  Removed {found.run_id}", exit_code=0)


def _row(found) -> dict:
    manifest = found.manifest if isinstance(found.manifest, dict) else {}
    return {
        "run_id": found.run_id,
        "name": manifest.get("name"),
        "command": manifest.get("command"),
        "state": manifest.get("state"),
        "started": manifest.get("started"),
        "path": str(found.path),
        "resume_command": found.resume_command(),
    }


def _list_text(rows) -> str:
    if not rows:
        return "\n  No runs on this machine yet.\n"
    lines = ["", f"  {'RUN':<20}{'NAME':<16}{'COMMAND':<10}{'STATE':<12}STARTED"]
    for row in rows:
        lines.append(
            f"  {row['run_id']:<20}{(row['name'] or '-'):<16}{(row['command'] or '-'):<10}"
            f"{(row['state'] or '-'):<12}{row['started'] or '-'}"
        )
    lines.append("")
    return "\n".join(lines)


def _show_text(row) -> str:
    lines = ["", f"  Run       {row['run_id']}", f"  Name      {row['name'] or '-'}",
             f"  Command   {row['command'] or '-'}", f"  State     {row['state'] or '-'}",
             f"  Path      {row['path']}", "", "  Continue it with:",
             "    " + " ".join(row["resume_command"]), ""]
    return "\n".join(lines)


@click.group("runs", short_help=registry.SUMMARIES["runs"], epilog=_shared.EPILOG)
def runs() -> None:
    """List, show, continue or remove the runs on this machine."""


@runs.command("list", epilog=_shared.EPILOG)
@_shared.common
@_shared.verb("runs")
def list_(**opts) -> int:
    """Every run this machine holds."""
    return run(action="list", **opts)


@runs.command("show", epilog=_shared.EPILOG)
@click.argument("run_id")
@_shared.common
@_shared.verb("runs")
def show(**opts) -> int:
    """One run, and the command that continues it."""
    return run(action="show", **opts)


@runs.command("rm", epilog=_shared.EPILOG)
@click.argument("run_id")
@_shared.common
@_shared.verb("runs")
def rm(**opts) -> int:
    """Remove a run and everything it recorded."""
    return run(action="rm", **opts)
