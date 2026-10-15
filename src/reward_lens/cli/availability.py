"""Which contracted verbs this build can actually run (A-026).

Two readers, two costs. `--help` is on D-30's fast path, and it may ask neither `reward_lens.api`
(which imports the contracts, and pydantic with them) nor an engine (a landed one imports the
contracts too), so the help reader reads the build's pointer table and then reads each engine's
source, importing nothing. The refusal reader runs when a verb has already been named, which is
past that budget because the verb is about to import `reward_lens.api` itself, so it confirms with
the same loader the api uses and is the authority.

Neither reader carries a list of absent verbs. The pointers are `reward_lens/api/_dispatch.py`'s
`ENGINES`, the table `api.doctor()` reports from, and the verbs they are consulted for are the root
help's `Measure` group. A verb outside that group reports its absence its own way and is not
touched here. Export reaches its renderer and reports record or access refusals through that route.

Every failure here answers "present", so a seam that moves degrades to the honest record this
surface printed before A-026, never to a refusal of a verb that would have run.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

from . import registry

#: The root help group whose verbs measure something, and so cannot stand in for their engine.
MEASURE = "Measure"

#: Where the pointer table lives, relative to the `reward_lens` package this module sits in.
TABLE = ("api", "_dispatch.py")


def measuring() -> tuple[str, ...]:
    """The verbs A-026 speaks about, read off the root help's own grouping."""
    for heading, names in registry.GROUPS:
        if heading == MEASURE:
            return names
    return ()


def engines() -> dict[str, str]:
    """`ENGINES` as the api holds it, without importing the api.

    The table is consulted from `sys.modules` when something has already imported it, and
    otherwise executed from its own file under a throwaway name. That file imports `importlib`,
    `sys` and `typing` and nothing else, which is why it can be run on the help path at all.
    """
    loaded = sys.modules.get("reward_lens.api._dispatch")
    if loaded is not None:
        return dict(getattr(loaded, "ENGINES", {}) or {})
    path = pathlib.Path(__file__).resolve().parent.parent.joinpath(*TABLE)
    try:
        spec = importlib.util.spec_from_file_location("_reward_lens_engine_table", path)
        if spec is None or spec.loader is None:
            return {}
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
    except Exception:  # noqa: BLE001 - a table this cannot read is a table it has no opinion about
        return {}
    return dict(getattr(probe, "ENGINES", {}) or {})


def targets(verb: str, table: dict[str, str]) -> tuple[str, ...]:
    """The engine pointers one verb rests on: its own key, and any `<verb>_*` key.

    `forecast` is three engines (issue, resolve, ledger) and `audit` is one. A verb with no key at
    all is carried by the CLI itself and can never be absent.
    """
    return tuple(
        target for key, target in sorted(table.items()) if key == verb or key.startswith(verb + "_")
    )


def _defined(target: str) -> bool:
    """Whether `"module:attribute"` names something this build holds, without importing it.

    `_dispatch.load` answers the same question by importing and calling `getattr`; that costs the
    contracts for a landed engine. Even `find_spec` imports parent packages, so this reader locates
    Python source directly inside the package and reads the attribute
    looked for as a top-level definition. A module already in `sys.modules` is asked directly,
    which is both cheaper and exact.
    """
    module_name, _, attribute = target.partition(":")
    if not attribute:
        return False
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return callable(getattr(loaded, attribute, None))
    parts = module_name.split(".")
    if parts[0] != "reward_lens":
        # A pointer outside this source package is confirmed by the runtime loader instead.
        return True
    module_path = pathlib.Path(__file__).resolve().parent.parent.joinpath(*parts[1:])
    source_path = module_path.with_suffix(".py")
    if not source_path.is_file():
        source_path = module_path / "__init__.py"
    if not source_path.is_file():
        return module_path.is_dir()  # a namespace package gives this reader no opinion
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError:
        return True
    name = re.escape(attribute)
    # Defined here, bound here, or re-exported here: a package that carries its engine in a
    # submodule and names it in `__init__.py` holds it as surely as one that spells out `def run`.
    return (
        re.search(
            rf"(?m)^(?:async\s+def|def|class)\s+{name}\b|^{name}\s*[:=]|^\s*from\s+\S+\s+import\b[^\n]*\b{name}\b",
            source,
        )
        is not None
    )


def marked_absent() -> frozenset[str]:
    """The verbs `--help` marks `(not in this build)`, read without importing anything."""
    table = engines()
    absent = set()
    for verb in measuring():
        pointers = targets(verb, table)
        if pointers and not any(_defined(pointer) for pointer in pointers):
            absent.add(verb)
    return frozenset(absent)


def absent(verb: str) -> bool:
    """Whether this build can run `verb`, confirmed with the loader `reward_lens.api` uses.

    Only a verb the cheap reader already doubts is confirmed, so a build whose engines have landed
    never imports one of them to find that out, and an engine defined in a way no source scan can
    see is not refused on the scan's word.
    """
    if verb not in marked_absent():
        return False
    try:
        from reward_lens.api import _dispatch
    except Exception:  # noqa: BLE001 - no loader to appeal to, so the cheap reader stands
        return True
    return not any(_dispatch.load(pointer) is not None for pointer in targets(verb, engines()))
