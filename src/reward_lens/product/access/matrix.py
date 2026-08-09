"""Section 7.1's compatibility matrix, derived from the adapter families' conformance results.

Nothing here writes a cell down. A family registers a callable that *runs* its conformance and
returns what the adapter itself probed; `build()` calls those and shapes the answers. A matrix that
was hand-written would be a claim about an adapter rather than a measurement of one, which is the
same failure D-34 forbids in a manifest: a capability the probe did not establish cannot be claimed.

Conformance here executes no grader. It builds the adapter over the fixture that ships in the
package, reads the manifest the adapter computed while parsing it, and puts a good and a malformed
input through `validate_input`. That is enough to fill the matrix and it spends nothing, which is
what lets `doctor` and `audit --dry-run` call it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

__all__ = ["Conformance", "FAMILIES", "build", "conformance"]


@dataclass(frozen=True)
class Conformance:
    """What one adapter family established about itself when it was run."""

    family: str
    capabilities: dict[str, bool]
    probed: dict[str, bool]
    determinism_class: str
    network_policy: str
    access_level: str
    replay_mode: str
    notes: tuple[str, ...] = field(default_factory=tuple)


def _fixture_grader() -> Path:
    """The plain-shape grader that ships in the package (interfaces section 6)."""
    from importlib.resources import as_file, files

    resource = files("reward_lens.examples.code_reward") / "fixtures" / "my_grader.py"
    with as_file(resource) as path:
        return Path(path)


def _local_deterministic() -> Conformance:
    """Run the local deterministic family's conformance: build the adapter and probe it."""
    from reward_lens.graders.base import EXPLICIT_CAPABILITIES
    from reward_lens.graders.python_grader import PythonGrader

    grader = PythonGrader.from_path(_fixture_grader())
    manifest = grader.describe()

    notes: list[str] = []
    refused = grader.validate_input("a task is an object", "a response")
    notes.append(
        "validate_input refuses a malformed task"
        if refused is not None
        else "validate_input accepted a malformed task"
    )
    accepted = grader.validate_input({"id": "t-1", "prompt": "p"}, "a response")
    notes.append(
        "validate_input accepts a well-formed task"
        if accepted is None
        else f"validate_input refused a well-formed task: {accepted}"
    )

    return Conformance(
        family=manifest.family,
        capabilities={name: bool(getattr(manifest.capabilities, name)) for name in EXPLICIT_CAPABILITIES},
        probed=dict(manifest.capabilities.probed),
        determinism_class=manifest.determinism_class,
        network_policy=manifest.network_policy,
        access_level=manifest.access_level,
        replay_mode=manifest.replay_mode,
        notes=tuple(notes),
    )


#: Family id -> the callable that runs its conformance. A family lands by registering here; no cell
#: of the matrix is written anywhere. In wave 1 only the local deterministic family is in the build.
FAMILIES: dict[str, Callable[[], Conformance]] = {
    "local_deterministic": _local_deterministic,
}


def conformance() -> dict[str, Conformance | Exception]:
    """Run every registered family's conformance. A family that cannot run says so, as itself."""
    out: dict[str, Conformance | Exception] = {}
    for name in sorted(FAMILIES):
        try:
            out[name] = FAMILIES[name]()
        except Exception as failure:  # a family that will not build is a result, not a crash
            out[name] = failure
    return out


def build() -> dict[str, dict[str, str]]:
    """The matrix, rebuilt from the conformance results every time it is asked for.

    Rebuilt, never remembered: an entry someone edited in a returned dict is gone on the next call,
    because the next call runs the families again.
    """
    rows: dict[str, dict[str, str]] = {}
    for name, result in conformance().items():
        if isinstance(result, Exception):
            rows[name] = {
                "conformance": "did not run",
                "reason": f"{type(result).__name__}: {result}",
            }
            continue
        row = {
            "conformance": "ran",
            "determinism": result.determinism_class,
            "network": result.network_policy,
            "access": result.access_level,
            "replay": result.replay_mode,
        }
        row.update({key: ("yes" if value else "no") for key, value in sorted(result.capabilities.items())})
        rows[name] = row
    return rows


def runs_local_graders() -> bool:
    """Whether some family in this build can take a local grader, from the matrix and not a guess."""
    return any(row.get("conformance") == "ran" for row in build().values())
