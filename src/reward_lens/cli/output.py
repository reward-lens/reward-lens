"""D-21, D-23, D-25, D-26 and the exit table: the one place that decides what goes where.

Results go to stdout. Progress, notes, warnings, errors and the JSON error twin go to stderr, so
`reward-lens audit x --format json > out.json` is a working pipeline with no flag. This module is
imported when a verb runs, never on the `--help` path.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from typing import Any

FORMATS = ("text", "json", "jsonl", "sarif", "github")
STRUCTURED = ("json", "jsonl", "sarif", "github")
EVENT_SCHEMA = "assay-event/1.0"

_GLYPH_COLOUR = {"✔": "\x1b[32m", "✘": "\x1b[31m", "!": "\x1b[33m", "○": "\x1b[2m"}
_RESET = "\x1b[0m"

#: The environments that say an agent, not a person, is the caller (D-31).
AGENT_VARIABLES = ("CLAUDECODE", "CI", "REWARD_LENS_NON_INTERACTIVE")


def bad_format(value: str):
    from reward_lens import errors

    return errors.make("RL0004", format=value, formats=", ".join(FORMATS))


def resolve_format(flag: str | None, *, json_flag: bool, stdout_is_tty: bool, env: dict | None = None) -> str:
    """D-23: the flag, then `--json`, then the environment twin, then text on a TTY and json off it."""
    environ = os.environ if env is None else env
    chosen = flag or ("json" if json_flag else None) or environ.get("REWARD_LENS_FORMAT") or None
    if chosen is None:
        return "text" if stdout_is_tty else "json"
    chosen = chosen.strip().lower()
    if chosen not in FORMATS:
        raise bad_format(chosen)
    return chosen


def want_colour(choice: str | None, *, stream, env: dict | None = None) -> bool:
    """D-25: the explicit flag wins; otherwise a TTY, no NO_COLOR, and not TERM=dumb."""
    environ = os.environ if env is None else env
    choice = (choice or environ.get("REWARD_LENS_COLOR") or "auto").strip().lower()
    if choice == "always":
        return True
    if choice == "never":
        return False
    if environ.get("NO_COLOR"):
        return False
    if environ.get("TERM", "") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def is_non_interactive(flag: bool, *, stdin=None, env: dict | None = None) -> bool:
    environ = os.environ if env is None else env
    if flag:
        return True
    if any(environ.get(name) for name in AGENT_VARIABLES):
        return True
    stream = sys.stdin if stdin is None else stdin
    return not bool(getattr(stream, "isatty", lambda: False)())


@dataclasses.dataclass
class Output:
    """The streams, the format, and the single place an exit code is decided."""

    format: str = "json"
    colour: bool = False
    progress: str = "quiet"
    fields: tuple[str, ...] = ()
    exit_zero: bool = False
    yes: bool = False
    non_interactive: bool = True
    no_config: bool = False
    stdout: Any = None
    stderr: Any = None

    def __post_init__(self) -> None:
        self.stdout = self.stdout or sys.stdout
        self.stderr = self.stderr or sys.stderr

    # --- streams --------------------------------------------------------------------------

    def result(self, text: str) -> None:
        self.stdout.write(text if text.endswith("\n") else text + "\n")
        self.stdout.flush()

    def note(self, text: str) -> None:
        if self.progress == "quiet":
            return
        self.stderr.write(text.rstrip("\n") + "\n")
        self.stderr.flush()

    def warn(self, text: str) -> None:
        self.stderr.write(text.rstrip("\n") + "\n")
        self.stderr.flush()

    def event(self, name: str, **payload: Any) -> None:
        """One versioned event per line, on stdout, under `--format jsonl` only (D-26)."""
        if self.format != "jsonl":
            return
        line = {"schema_version": EVENT_SCHEMA, "event": name}
        line.update(payload)
        self.stdout.write(json.dumps(line, sort_keys=False) + "\n")
        self.stdout.flush()

    def paint(self, glyph: str) -> str:
        if not self.colour:
            return glyph
        return f"{_GLYPH_COLOUR.get(glyph, '')}{glyph}{_RESET}" if glyph in _GLYPH_COLOUR else glyph

    # --- delivery -------------------------------------------------------------------------

    def deliver(self, *, document: dict, text: str, exit_code: int) -> int:
        """Write the result in the chosen format and hand back the code the process should use."""
        if self.format in ("sarif", "github"):
            raise _export_unavailable(self.format)
        if self.format == "text":
            self.result(text)
        elif self.format == "jsonl":
            self.event("result", result=self.project(document))
        else:
            self.result(json.dumps(self.project(document), indent=2, sort_keys=False))
        return 0 if self.exit_zero and exit_code in (1, 2) else exit_code

    def project(self, document: dict) -> dict:
        """`--fields`: bound the output to the dotted paths the caller asked for."""
        if not self.fields:
            return document
        out: dict = {}
        for path in self.fields:
            cursor: Any = document
            for part in path.split("."):
                if not isinstance(cursor, dict) or part not in cursor:
                    raise _unknown_field(path)
                cursor = cursor[part]
            target = out
            parts = path.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = cursor
        return out

    # --- failure --------------------------------------------------------------------------

    def fail(self, error) -> int:
        """The two lines, the code, and the JSON twin on stderr. Never a traceback."""
        from reward_lens import errors

        self.stderr.write(_with_exception(errors.render_two_line(error), error) + "\n")
        if self.format in STRUCTURED:
            self.stderr.write(json.dumps(error_json(error), sort_keys=True) + "\n")
        self.stderr.flush()
        context = getattr(error, "context", {}) or {}
        if "argv" in context:
            # D-31: a pending decision is not a plain failure. stdout carries the pending block
            # so the caller can read the argv back and run it, which is the whole point of RL0801.
            payload = pending_envelope(context.get("command_name", ""), error)
        else:
            payload = envelope(command=context.get("command", ""), error=error)
        if self.format == "text":
            pass
        elif self.format in ("sarif", "github"):
            pass
        elif self.format == "jsonl":
            self.event("error", error=error_json(error))
        else:
            self.result(json.dumps(payload, indent=2, sort_keys=False))
        return error.exit_code


def error_json(error) -> dict:
    """The error as an object. `RewardLensError.to_json` returns JSON text, not a mapping, and
    an envelope that carried the text would make `error.code` a character index.

    `context` rides along when the error has one, because the two sentences an error renders are
    what a person reads and the context is what a program needs: the exception behind an RL0900,
    the argv behind a pending decision, the path behind a refusal.
    """
    document = error.to_json()
    if isinstance(document, str):
        document = json.loads(document)
    document = dict(document)
    context = getattr(error, "context", None) or {}
    if context:
        document["context"] = _jsonable(context)
    return document


def _jsonable(context) -> dict:
    """The context with anything JSON cannot hold rendered as its text.

    An error path is the worst place to meet a second error, so a value `json.dumps` refuses
    becomes `str(value)` rather than a TypeError on the way out of a failure.
    """
    out: dict = {}
    for key, value in context.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            out[str(key)] = str(value)
        else:
            out[str(key)] = value
    return out


def _with_exception(rendered: str, error) -> str:
    """The two-line render with `context["exception"]` folded into the `help:` block.

    Only RL0900 carries one today. It goes above the `reward-lens explain` line, indented to the
    block, so a caller who hits a defect sees the exception's type and message without a
    traceback and without having to re-run under a debugger.
    """
    exception = (getattr(error, "context", None) or {}).get("exception")
    if not exception:
        return rendered
    lines = rendered.split("\n")
    extra = f"       the exception was {exception}"
    for index, line in enumerate(lines):
        if line.strip().startswith("reward-lens explain "):
            lines.insert(index, extra)
            break
    else:
        lines.append(extra)
    return "\n".join(lines)


def _unknown_field(path: str):
    from reward_lens import errors

    return errors.make(
        "RL0001",
        detail=f"--fields names {path}, which is not a field of this result",
        command="--fields",
    )


def _export_unavailable(name: str):
    from reward_lens import errors

    error = errors.make(
        "RL0701",
        extra="export",
        capability=f"the {name} writer",
        detail=f"--format {name} is written by the export engine, which is not in this build",
    )
    error.context.update({"format": name})
    return error


def exit_code_for(decision: dict | None) -> int:
    """D-22: 0 qualified, 1 rejected, 2 unresolved, 0 when no decision was requested."""
    if not decision:
        return 0
    return {"qualified": 0, "rejected": 1, "unresolved": 2}.get(decision.get("state"), 0)


def envelope(
    *,
    command: str,
    execution: dict | None = None,
    subject: dict | None = None,
    decision: dict | None = None,
    findings: list | None = None,
    holes: list | None = None,
    artifacts: dict | None = None,
    error=None,
    pending: dict | None = None,
) -> dict:
    """Section 5.7's envelope, built in field order and additive only."""
    document = {
        "schema_version": "assay-result/1.0",
        "command": command,
        "execution": execution or {"state": "failed" if error is not None else "complete"},
        "subject": subject,
        "decision": decision,
        "findings": findings or [],
        "holes": holes or [],
        "artifacts": artifacts or {},
        "error": error_json(error) if error is not None else None,
    }
    if pending is not None:
        document["pending"] = pending
    return document


def pending_decision(*, decision: str, argv: list[str]):
    """D-31: the decision in the caller's terms, `pending.argv`, and the shell-quoted twin."""
    from reward_lens import errors

    from . import ids

    command = ids.quote_command(argv)
    error = errors.make("RL0801", decision=decision, argv=argv, command=command)
    error.context.update({"decision": decision, "argv": list(argv), "command": command})
    return error


def pending_envelope(command: str, error) -> dict:
    return envelope(
        command=command,
        execution={"state": "pending"},
        error=error,
        pending={
            "decision": error.context.get("decision", error.message),
            "argv": list(error.context.get("argv", ())),
            "command": error.context.get("command", ""),
        },
    )
