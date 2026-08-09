"""The five serialisations of D-23. Each is imported only when it is chosen."""

from __future__ import annotations

NAMES = ("text", "json", "jsonl", "sarif", "github")


def get(name: str):
    import importlib

    from ..output import bad_format

    if name not in NAMES:
        raise bad_format(name)
    return importlib.import_module(f"reward_lens.cli.format.{name}")
