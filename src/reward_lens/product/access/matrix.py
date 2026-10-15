"""Section 7.1's compatibility matrix, derived from the adapter families' conformance results.

Nothing here writes a cell down. A family registers a callable that *runs* its conformance and
returns what the adapter itself probed; `build()` calls those and shapes the answers. A matrix that
was hand-written would be a claim about an adapter rather than a measurement of one, which is the
same failure D-34 forbids in a manifest: a capability the probe did not establish cannot be claimed.

These probes construct local fixtures and validate both sides of the input contract. They do not
contact a provider or qualify a live profile. Scoring conformance has separate family tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

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
    framework_version: str = "built-in"
    supported_records: tuple[str, ...] = ("manifest", "evidence_envelope")
    fixture_identity: str = ""
    install_route: str = "pip install reward-lens"
    live_status: str = "unverified"


def _fixture_grader() -> Path:
    """The plain-shape grader that ships in the package (interfaces section 6)."""
    from importlib.resources import as_file, files

    resource = files("reward_lens.examples.code_reward") / "fixtures" / "my_grader.py"
    with as_file(resource) as path:
        return Path(path)


def _probe(adapter: Any, *, fixture: str, install: str, framework: str = "built-in") -> Conformance:
    """Read only claims from the constructed adapter, after checking its input boundary."""
    from reward_lens.graders.base import EXPLICIT_CAPABILITIES

    manifest = adapter.describe()

    notes: list[str] = []
    refused = adapter.validate_input("a task is an object", "a response")
    notes.append(
        "validate_input refuses a malformed task"
        if refused is not None
        else "validate_input accepted a malformed task"
    )
    accepted = adapter.validate_input({"id": "t-1", "prompt": "p"}, "a response")
    notes.append(
        "validate_input accepts a well-formed task"
        if accepted is None
        else f"validate_input refused a well-formed task: {accepted}"
    )

    if refused is None or accepted is not None:
        raise ValueError(f"{manifest.family} failed its local input conformance")
    return Conformance(
        family=manifest.family,
        capabilities={
            name: bool(getattr(manifest.capabilities, name)) for name in EXPLICIT_CAPABILITIES
        },
        probed=dict(manifest.capabilities.probed),
        determinism_class=manifest.determinism_class,
        network_policy=manifest.network_policy,
        access_level=manifest.access_level,
        replay_mode=manifest.replay_mode,
        notes=tuple(notes),
        framework_version=framework,
        fixture_identity=fixture,
        install_route=install,
    )


def _local_deterministic() -> Conformance:
    from reward_lens.graders.python_grader import PythonGrader

    return _probe(
        PythonGrader.from_path(_fixture_grader()),
        fixture="packaged code_reward/my_grader.py",
        install="pip install reward-lens",
    )


def _stateful_tool_environment() -> Conformance:
    from reward_lens.contracts.project import OpenEnvSpec
    from reward_lens.graders.openenv import OpenEnvGrader

    if find_spec("openenv") is None:
        raise MissingExtra("openenv")
    spec = OpenEnvSpec.model_validate(
        {
            "package": "openenv",
            "version": "0.6.0",
            "wheel_sha256": "dd1033006d17c3c0147a02f73fc6deef881e15dc5501aea28930765708292641",
            "base_url": "ws://127.0.0.1:8000",
            "connect_timeout_s": 10.0,
            "message_timeout_s": 60.0,
            "reset": "fixture:reset",
            "action": "fixture:action",
        }
    )
    adapter = OpenEnvGrader(
        spec,
        client_factory=lambda **_: None,
        reset_bridge=lambda task: {"seed": task.get("seed", 0)},
        action_bridge=lambda task, response: response,
    )
    return _probe(
        adapter,
        fixture="local OpenEnv 0.6.0 contract fixture",
        install="pip install 'reward-lens[openenv]'",
        framework="openenv 0.6.0",
    )


def _llm_judge() -> Conformance:
    from reward_lens.graders.judge import JudgeAdapter

    def fixture_transport(prompt: str, completion: str, **kwargs: object) -> object:
        return {"verdict": "left", "score": 0.75}

    adapter = JudgeAdapter.from_transport(
        provider="local-protocol-fixture",
        model="fixture-judge-v1",
        prompt="Return JSON",
        settings={"temperature": 0},
        transport=fixture_transport,
        fixture_conformance=True,
        passing_probes={"fixture_conformance": True},
    )
    return _probe(
        adapter,
        fixture="local judge JSON protocol fixture",
        install="pip install reward-lens",
        framework="local protocol v1",
    )


def _neural_reward_model() -> Conformance:
    from reward_lens.graders.reward_model import RewardModel, tiny_local_model

    if find_spec("torch") is None:
        raise MissingExtra("train")
    model, tokenizer = tiny_local_model()
    return _probe(
        RewardModel.from_pretrained(model, tokenizer),
        fixture="in-memory tiny CPU scoring head",
        install="pip install 'reward-lens[train]'",
        framework="PyTorch local CPU",
    )


def _composite() -> Conformance:
    from reward_lens.graders.composite import Component, CompositeGrader
    from reward_lens.graders.python_grader import PythonGrader

    child = PythonGrader.from_path(_fixture_grader())
    adapter = CompositeGrader(
        components=(Component("local", child),), probed={"component_dag": True}
    )
    return _probe(
        adapter, fixture="one-component packaged Python grader", install="pip install reward-lens"
    )


def _remote_black_box() -> Conformance:
    from datetime import UTC, datetime

    from reward_lens.graders.remote import RemoteGrader

    class FixtureTransport:
        def attempt(self, request: object, *, sandbox: object, limits: object) -> object:
            raise AssertionError("compatibility probing must not contact a provider")

    adapter = RemoteGrader(
        provider="fixture://local/score",
        transport=FixtureTransport(),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    return _probe(
        adapter,
        fixture="local remote wire-protocol fixture",
        install="pip install reward-lens",
        framework="remote protocol v1",
    )


class MissingExtra(RuntimeError):
    def __init__(self, extra: str) -> None:
        self.extra = extra
        super().__init__(f"install reward-lens[{extra}] to run this local conformance probe")


#: Family id -> a callable that constructs and probes a local adapter fixture.
FAMILIES: dict[str, Callable[[], Conformance]] = {
    "local_deterministic": _local_deterministic,
    "stateful_tool_environment": _stateful_tool_environment,
    "llm_judge": _llm_judge,
    "neural_reward_model": _neural_reward_model,
    "composite": _composite,
    "remote_black_box": _remote_black_box,
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
                "conformance": "missing extra"
                if isinstance(result, MissingExtra)
                else "did not run",
                "reason": f"{type(result).__name__}: {result}",
                "live": "unverified",
                "install_route": f"pip install 'reward-lens[{result.extra}]'"
                if isinstance(result, MissingExtra)
                else "",
            }
            continue
        row = {
            "conformance": "ran",
            "determinism": result.determinism_class,
            "network": result.network_policy,
            "access": result.access_level,
            "replay": result.replay_mode,
            "framework_version": result.framework_version,
            "supported_records": ", ".join(result.supported_records),
            "fixture_identity": result.fixture_identity,
            "install_route": result.install_route,
            "live": result.live_status,
        }
        row.update(
            {key: ("yes" if value else "no") for key, value in sorted(result.capabilities.items())}
        )
        rows[name] = row
    return rows


def runs_local_graders() -> bool:
    """Whether some family in this build can take a local grader, from the matrix and not a guess."""
    return build()["local_deterministic"].get("conformance") == "ran"
