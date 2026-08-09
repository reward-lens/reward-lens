"""D-24: every identifier a caller supplies is validated before it is used.

The adversarial caller here is an agent, not a person, so the four classes the brief names are
refused by construction rather than by escaping: a control character, a null byte, a traversal, and
a double-encoded string. A refusal is RL0002, exit 4, and it names the field.
"""

from __future__ import annotations

import re
import unicodedata

MAX_LENGTH = 256

_HEX = "0123456789abcdefABCDEF"
_PERCENT = re.compile(r"%[0-9a-fA-F]{2}")
_DOUBLE_ENCODED = re.compile(r"%25[0-9a-fA-F]{2}", re.ASCII)
_SHELL_METACHARACTERS = set("`$\\\n\r")


def _refuse(field: str, detail: str):
    from reward_lens import errors

    error = errors.make("RL0002", field=field, detail=detail)
    error.context.update({"field": field, "detail": detail})
    return error


def validate_identifier(value: str, *, field: str) -> str:
    """Return `value` unchanged, or raise the code-4 refusal that names `field`."""
    if not isinstance(value, str):
        raise _refuse(field, f"expected text, got {type(value).__name__}")
    if not value:
        raise _refuse(field, "it is empty")
    if len(value) > MAX_LENGTH:
        raise _refuse(field, f"it is {len(value)} characters, and the limit is {MAX_LENGTH}")
    if "\x00" in value:
        raise _refuse(field, "it contains a null byte")
    for char in value:
        if unicodedata.category(char) in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            raise _refuse(field, f"it contains the control character U+{ord(char):04X}")
    if _DOUBLE_ENCODED.search(value) or "%25" in value:
        raise _refuse(field, "it is double-encoded, so what it names depends on how often it is decoded")
    if _PERCENT.search(value):
        decoded = _PERCENT.sub(lambda m: chr(int(m.group()[1:], 16)), value)
        if decoded != value and _has_traversal(decoded):
            raise _refuse(field, "it hides a path traversal behind percent-encoding")
    if _has_traversal(value):
        raise _refuse(field, "it contains a path traversal")
    if _SHELL_METACHARACTERS & set(value):
        raise _refuse(field, "it contains a character a shell would act on")
    return value


def _has_traversal(value: str) -> bool:
    parts = re.split(r"[/\\]", value)
    return ".." in parts or value.startswith(("/", "\\")) or ":" in value[:3] and "\\" in value


def validate_path_argument(value, *, field: str):
    """A caller-supplied path: the same refusals, minus the traversal rule.

    A path is allowed to be absolute and is allowed to climb, because that is what a path is for.
    What it is not allowed to be is a control character, a null byte or a double-encoded string.
    """
    import pathlib

    text = str(value)
    if "\x00" in text:
        raise _refuse(field, "it contains a null byte")
    for char in text:
        if unicodedata.category(char) in {"Cc", "Cs", "Cn"}:
            raise _refuse(field, f"it contains the control character U+{ord(char):04X}")
    if "%25" in text:
        raise _refuse(field, "it is double-encoded, so what it names depends on how often it is decoded")
    return pathlib.Path(text)


def shell_quote(value: str) -> str:
    """Shell-quote one argument of a `pending.command` string (D-31)."""
    import shlex

    return shlex.quote(value)


def quote_command(argv: list[str]) -> str:
    return " ".join(shell_quote(part) for part in argv)
