"""`reward-lens init`: connect a reward system, or start from an example."""

from __future__ import annotations

import click

from .. import ids, registry
from . import _shared


def run(*, out, dest, example, detect, **opts) -> int:
    from reward_lens.examples import REGISTRY

    destination = ids.validate_path_argument(dest, field="dest")
    if detect:
        return _detect(out=out, dest=dest, destination=destination)
    name = ids.validate_identifier(example or "code-reward", field="example")
    if name not in REGISTRY:
        from reward_lens import errors

        raise errors.make(
            "RL0001",
            detail=f"there is no example named {name}; this build ships {', '.join(sorted(REGISTRY))}",
            command="init",
        )
    module_name, _, attribute = REGISTRY[name].partition(":")
    import importlib

    module = importlib.import_module(module_name)
    writer = getattr(module, attribute)
    written = writer(destination)
    document = {
        "schema_version": "assay-result/1.0",
        "command": "init",
        "execution": {"state": "complete"},
        "subject": {"reward_system": name},
        "decision": None,
        "findings": [],
        "holes": [],
        "artifacts": {"written": [str(path) for path in written]},
        "error": None,
    }
    text = _text(dest, getattr(module, "MANIFEST", ()))
    return out.deliver(document=document, text=text, exit_code=0)


def _detect(*, out, dest, destination) -> int:
    """`--detect`: read the reward system already in this directory and write its project file.

    The path the transcript shows is the one the caller typed, for the reason `_text` gives. A
    shape the connector recognises and will not adapt refuses here with RL0710, because the verb
    is the second place D-65's refusal has to reach a reader: the first is `bind`.
    """
    from reward_lens import errors
    from reward_lens.graders.connect import (
        Shape,
        ShapeUnsupported,
        detect_project,
        render_detect_transcript,
    )

    displayed = str(dest)
    try:
        project = detect_project(destination)
    except (OSError, ValueError) as caught:
        raise errors.make("RL0001", detail=str(caught), command="init") from caught

    for found in project.callables:
        if found.shape is Shape.UNSUPPORTED:
            raise ShapeUnsupported(
                shape=found.unsupported_label, entry=found.entry, detail=found.reason
            )

    written = project.write_project_file(displayed=displayed)
    document = {
        "schema_version": "assay-result/1.0",
        "command": "init",
        "execution": {"state": "complete"},
        "subject": {
            "reward_system": str(destination),
            "reward_file": project.reward_file,
            "entries": [found.entry for found in project.callables],
            "shapes": [found.declared for found in project.callables],
        },
        "decision": None,
        "findings": [],
        "holes": [],
        "artifacts": {"written": [str(written)]},
        "error": None,
    }
    text = render_detect_transcript(project, displayed=displayed)
    return out.deliver(document=document, text=text, exit_code=0)


#: The manifest's name column: `    <name>` and its description at column 25, read off
#: `fleet/golden/wave-1/first-hour.txt` lines 6 to 10.
_NAME_WIDTH = 19


def _text(destination, manifest) -> str:
    """A-015: the example's own manifest, in its order, and no walk of what was written.

    The path is the one the caller typed, not the one the writer resolved: the transcript shows
    `./demo` back, and a reader who is told `demo` has been handed a path they did not give.
    """
    lines = [f"  Wrote {destination}"]
    for entry, description in manifest:
        lines.append(f"    {entry:<{_NAME_WIDTH}}  {description}")
    lines += ["", "  This example contains a real defect. Find it with:", f"    reward-lens audit {destination}", ""]
    return "\n".join(lines)


@click.command("init", short_help=registry.SUMMARIES["init"], epilog=_shared.EPILOG)
@click.argument("dest", required=False, default=".")
@click.option("--example", default=None, help="start from an example that ships in the wheel")
@click.option("--detect", is_flag=True, help="connect the reward system already in this directory")
@_shared.common
@_shared.verb("init")
def init(**opts) -> int:
    """Connect a reward system, or start from an example."""
    return run(**opts)
