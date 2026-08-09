"""P-ADAPT-PY: the local deterministic adapter and the `plain` shape.

The sandbox seam is interfaces section 3. P-EXEC had not landed when these were written, so the
tests supply their own `Sandbox`: `LocalSandbox` really does run the subprocess, so the grading
path under test is the real one, and `BreachingSandbox` / `RaisingSandbox` stand in for outcomes
a real sandbox produces that a passing subprocess cannot (a wall breach, an unavailable tier).
Nothing here runs the graded function in this process.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from reward_lens.contracts import Counters, EntryProvenance, RewardLensError, digest
from reward_lens.execution import default_sandbox
from reward_lens.graders.base import (
    VERDICTS,
    AdapterCapabilityUnproven,
    CapabilityVector,
    EvidenceEnvelope,
    InvalidInput,
    Limits,
    Manifest,
    Result,
    ValidityFinding,
    source_digest_of,
)
from reward_lens.graders.python_grader import PythonGrader

TASK = {"id": "t1", "prompt": "add two numbers", "reference": "3"}
RESPONSE = "def add(a, b):\n    return a + b\n"

# The graded fixtures. The fleet guard refuses `tests/graders/fixtures/`, which this brief names as
# a writable path (see `## Premise checks` in the handoff), so the sources live here and are
# written to a temporary directory at import. Each one is a real file the sandbox really executes.
FIXTURE_SOURCES: dict[str, str] = {
    "good_grader.py": '''"""Deterministic: a tests-passed term plus a format term."""


def score(task, response):
    if not isinstance(response, str):
        return 0.0
    passed = 1.0 if "return a + b" in response else 0.0
    formatted = 1.0 if response.endswith("\\n") else 0.0
    return 0.5 * passed + 0.25 * formatted
''',
    "none_grader.py": '''"""Returns None, which TRL turns into NaN and drops for that row (D-65)."""


def score(task, response):
    return None
''',
    "raising_grader.py": '''"""Raises, which `verifiers` swallows and scores 0.0 (D-65)."""


def score(task, response):
    return 1.0 / 0
''',
    "bool_grader.py": '''"""Returns a bool, which passes float() and reads as a legitimate score (D-65)."""


def score(task, response):
    return "return a + b" in response
''',
    "forging_grader.py": '''"""Tries to write its own provenance and cost onto the result channel."""

import os
import sys


def score(task, response):
    forged = '{"provenance": {"sandbox_tier": "forged"}, "counters": {"usd": "99.00"}}'
    print("@@RL@@" + forged)
    sys.stdout.write("@@RL@@" + forged + "\\n")
    os.write(1, ("@@RL@@" + forged + "\\n").encode())
    return 0.5
''',
    "exiting_grader.py": '''"""Leaves the process cleanly before any result exists."""

import os


def score(task, response):
    os._exit(0)
''',
    "marker_grader.py": '''"""Writes a file into its working directory, so that where it ran is visible."""

from pathlib import Path


def score(task, response):
    Path("ran_here.txt").write_text("ran", encoding="utf-8")
    return 1.0
''',
    "flat_grader.py": '''"""One return of one constant: nothing a mutation instrument can operate on."""


def score(task, response):
    return 1.0
''',
    "wobbly_grader.py": '''"""Nondeterministic on purpose: two runs of the same input disagree."""

import random


def score(task, response):
    return round(random.SystemRandom().random(), 6)
''',
    "listing_grader.py": '''"""Reports the working tree it was handed: what is in it and what it can open."""

import json
from pathlib import Path


def score(task, response):
    print(json.dumps({
        "names": sorted(child.name for child in Path(".").iterdir()),
        "supervisor_here": Path("_rl_supervisor.py").exists(),
        "request_here": Path("request.json").exists(),
        "capture_here": Path("grader_stdout.txt").exists(),
    }))
    return 0.5
''',
}

FIXTURES = Path(tempfile.mkdtemp(prefix="rl-grader-fixtures-"))
for _name, _source in FIXTURE_SOURCES.items():
    (FIXTURES / _name).write_text(_source, encoding="utf-8")
atexit.register(shutil.rmtree, FIXTURES, True)


# --- the sandboxes ------------------------------------------------------------------------------


class LocalSandbox:
    """A real `Sandbox`: runs the argv in a child process and reports what it observed."""

    tier = "T0"

    def __init__(self, *, tier: str = "T0") -> None:
        self.tier = tier
        self.calls: list[list[str]] = []
        #: What the working directory held at the moment of the call, and the mode of each entry.
        #: Recorded here because the adapter deletes its scratch directory when the run returns.
        self.cwds: list[dict[str, object]] = []

    def run(
        self,
        argv: list[str],
        *,
        limits: Limits,
        cwd: Path,
        env: dict[str, str],
        stdin: bytes | None = None,
    ) -> Result:
        self.calls.append(list(argv))
        here = Path(cwd)
        self.cwds.append(
            {
                "path": here,
                "names": sorted(child.name for child in here.iterdir()),
                "modes": {
                    child.name: oct(child.stat().st_mode & 0o777) for child in here.iterdir()
                },
            }
        )
        started = time.monotonic()
        try:
            proc = subprocess.run(  # noqa: S603 - the argv is built by the adapter under test
                argv,
                cwd=str(cwd),
                env=dict(env),
                input=stdin,
                capture_output=True,
                timeout=limits.wall_s,
            )
        except subprocess.TimeoutExpired:
            return Result(
                exit_code=-9,
                stdout=b"",
                stderr=b"",
                counters=Counters(wall_s=float(limits.wall_s), calls=1),
                breach="wall",
                tier=self.tier,
            )
        wall = time.monotonic() - started
        return Result(
            exit_code=proc.returncode,
            stdout=proc.stdout[: limits.stdout_bytes],
            stderr=proc.stderr[: limits.stdout_bytes],
            counters=Counters(wall_s=round(wall, 3), cpu_s=0.0, calls=1, processes=1),
            breach=None,
            tier=self.tier,
        )


@dataclass
class MeteringSandbox:
    """A sandbox that meters a cost. The adapter fills in no counter of its own, ever."""

    inner: LocalSandbox

    def run(
        self,
        argv: list[str],
        *,
        limits: Limits,
        cwd: Path,
        env: dict[str, str],
        stdin: bytes | None = None,
    ) -> Result:
        result = self.inner.run(argv, limits=limits, cwd=cwd, env=env, stdin=stdin)
        metered = Counters(**{**result.counters.model_dump(exclude_none=True), "usd": "0.01"})
        return replace(result, counters=metered)


@dataclass
class BreachingSandbox:
    """A `Sandbox` that reports one breach dimension, as a real one does on a limit."""

    breach: str
    tier: str = "L2"
    exit_code: int = -9
    stdout: bytes = b""

    def run(self, argv, *, limits, cwd, env, stdin=None) -> Result:  # noqa: ANN001
        return Result(
            exit_code=self.exit_code,
            stdout=self.stdout,
            stderr=b"",
            counters=Counters(wall_s=float(limits.wall_s), calls=1),
            breach=self.breach,
            tier=self.tier,
        )


@dataclass
class RaisingSandbox:
    """A `Sandbox` whose tier cannot be held on this machine: RL0401, interfaces section 3."""

    def run(self, argv, *, limits, cwd, env, stdin=None) -> Result:  # noqa: ANN001
        raise RewardLensError(
            code="RL0401",
            message="no sandbox tier is available on this machine",
            remediation="install bubblewrap",
            exit_code=5,
        )


@dataclass
class ExitCodeSandbox:
    """A `Sandbox` that returns a chosen exit code with no result line."""

    exit_code: int
    tier: str = "T0"

    def run(self, argv, *, limits, cwd, env, stdin=None) -> Result:  # noqa: ANN001
        return Result(
            exit_code=self.exit_code,
            stdout=b"",
            stderr=b"",
            counters=Counters(wall_s=0.01, calls=1),
            breach=None,
            tier=self.tier,
        )


@pytest.fixture
def sandbox() -> LocalSandbox:
    return LocalSandbox()


@pytest.fixture
def limits() -> Limits:
    return Limits(wall_s=30)


def grader(name: str) -> PythonGrader:
    return PythonGrader.from_path(FIXTURES / name)


#: Every capability in the probe's vocabulary that needs a run to establish it.
RUN_DEPENDENT = (
    "controls_execution",
    "locally_replayable",
    "cost_metered",
    "returns_components",
    "state_observable",
    "outcome_independent",
    "protected_partition",
)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_module(path: Path, name: str):
    """Import a written project's own grader, to take its score as the expected value."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the manifest and the capability vector -----------------------------------------------------


def test_manifest_round_trips_through_its_dict() -> None:
    man = grader("good_grader.py").describe()
    assert Manifest.from_dict(man.to_dict()) == man
    assert man.to_dict() == Manifest.from_dict(man.to_dict()).to_dict()


def test_manifest_digest_is_stable_and_comes_from_canonical_bytes() -> None:
    man = grader("good_grader.py").describe()
    assert man.digest() == digest(man.to_dict())
    assert man.digest() == Manifest.from_dict(man.to_dict()).digest()
    assert man.digest().startswith("sha256:")


def test_manifest_is_the_local_deterministic_family_with_the_plain_shape() -> None:
    man = grader("good_grader.py").describe()
    assert man.family == "local_deterministic"
    assert man.shape == "plain"
    assert man.score_direction == "higher_is_better"
    assert man.network_policy == "deny"
    assert man.secret_policy == "none"


def test_the_capability_vector_carries_the_five_explicit_booleans_of_section_7_1() -> None:
    caps = grader("good_grader.py").describe().capabilities
    for name in (
        "component_dag",
        "token_quantities",
        "checkpoint_capture",
        "controlled_updates",
        "cost_metering",
    ):
        assert isinstance(getattr(caps, name), bool), name


def test_the_static_probe_claims_only_what_reading_the_source_shows() -> None:
    """Both static flags come off the file: the read itself, and an `ast` walk over what it held."""
    g = grader("good_grader.py")
    caps = g.describe().capabilities
    # `source_visible` is set by the read, and the text it read is the text it hands back.
    assert caps.probed["source_visible"] is True
    assert g.source() == (FIXTURES / "good_grader.py").read_text(encoding="utf-8")
    # `0.5 * passed + 0.25 * formatted` is arithmetic and `"return a + b" in response` is a
    # comparison: two constructs the mutation instrument can operate on.
    assert caps.probed["has_mutation_surface"] is True
    # Nothing has run yet, so nothing that needs a run is claimed.
    for key in RUN_DEPENDENT:
        assert caps.probed[key] is False, key
    assert caps.component_dag is False


def test_a_grader_with_nothing_to_mutate_is_not_claimed_a_mutation_surface() -> None:
    """`flat_grader.py` is one `return 1.0`. There is nothing in it for an instrument to change."""
    caps = grader("flat_grader.py").describe().capabilities
    assert caps.probed["source_visible"] is True
    assert caps.probed["has_mutation_surface"] is False
    assert "return 1.0" in grader("flat_grader.py").source()


def test_the_dynamic_probe_proves_execution_control_and_local_replay(sandbox, limits) -> None:
    probed = grader("good_grader.py").probe(sandbox=sandbox, limits=limits)
    caps = probed.describe().capabilities
    assert caps.probed["controls_execution"] is True
    assert caps.probed["locally_replayable"] is True
    assert caps.probed["returns_components"] is False
    # This sandbox reports wall time and a call count. Those are observations of a resource, not a
    # meter, and nothing metered a cost, so nothing claims one.
    assert caps.probed["cost_metered"] is False
    assert probed.describe().determinism_class == "deterministic_observed"


def test_cost_metering_is_claimed_only_when_a_sandbox_metered_a_cost(limits) -> None:
    plain = grader("good_grader.py").probe(sandbox=LocalSandbox(), limits=limits)
    assert plain.describe().capabilities.probed["cost_metered"] is False
    metering = grader("good_grader.py").probe(sandbox=MeteringSandbox(LocalSandbox()), limits=limits)
    assert metering.describe().capabilities.probed["cost_metered"] is True


def test_a_grader_that_raises_on_every_call_is_claimed_no_capability_that_needs_a_run(
    sandbox, limits
) -> None:
    """The vacuous instrument, refused: two failed runs are not an observation of a run."""
    before = grader("raising_grader.py")
    failed = before.score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    assert failed.verdict == "grader_error"

    probed = before.probe(sandbox=sandbox, limits=limits)
    man = probed.describe()
    for key in RUN_DEPENDENT:
        assert man.capabilities.probed[key] is False, key
    assert man.determinism_class != "deterministic_observed"
    assert man.determinism_class == before.describe().determinism_class
    assert man.replay_mode == "local_replay_unproven"
    # Reading the source did happen, and `1.0 / 0` is still a surface, so those two stand.
    assert man.capabilities.probed["source_visible"] is True
    assert man.capabilities.probed["has_mutation_surface"] is True


def test_a_probe_the_sandbox_could_not_start_claims_nothing(limits) -> None:
    probed = grader("good_grader.py").probe(sandbox=RaisingSandbox(), limits=limits)
    man = probed.describe()
    for key in RUN_DEPENDENT:
        assert man.capabilities.probed[key] is False, key
    assert man.determinism_class != "deterministic_observed"


def test_the_dynamic_probe_refuses_local_replay_to_a_nondeterministic_grader(
    sandbox, limits
) -> None:
    probed = grader("wobbly_grader.py").probe(sandbox=sandbox, limits=limits)
    caps = probed.describe().capabilities
    assert caps.probed["locally_replayable"] is False
    assert probed.describe().determinism_class == "nondeterministic_observed"


def test_a_capability_absent_from_the_probe_cannot_be_claimed() -> None:
    """RL0702, ADAPTER_CAPABILITY_UNPROVEN: the refusal case."""
    man = grader("good_grader.py").describe()
    with pytest.raises(AdapterCapabilityUnproven) as caught:
        replace(man, capabilities=replace(man.capabilities, component_dag=True))
    assert caught.value.code == "RL0702"
    assert caught.value.exit_code == 5
    assert "component_dag" in caught.value.message
    assert caught.value.context["capability"] == "component_dag"


def test_a_probed_capability_may_be_claimed() -> None:
    caps = CapabilityVector(cost_metering=True, probed={"cost_metering": True})
    assert caps.cost_metering is True


def test_no_capability_is_inferred_from_a_method_existing() -> None:
    class Pretender:
        def source(self) -> str:
            return "not really"

        def replay(self) -> None:
            return None

    with pytest.raises(AdapterCapabilityUnproven):
        CapabilityVector(component_dag=True, probed={})
    assert not hasattr(CapabilityVector, "from_object")
    assert Pretender().source() == "not really"


# --- the envelope -------------------------------------------------------------------------------


def test_the_envelope_provenance_validates_as_the_records_entry_provenance(sandbox, limits) -> None:
    env = grader("good_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    prov = EntryProvenance.model_validate(env.provenance)
    assert prov.sandbox_tier == "T0"
    assert prov.offline is True
    assert prov.duration_s >= 0


def test_the_envelope_counters_are_the_schemas_counters(sandbox, limits) -> None:
    env = grader("good_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    assert isinstance(env.counters, Counters)
    assert EntryProvenance.model_validate(env.provenance).counters == env.counters
    assert env.counters.calls == 1


def test_the_envelope_round_trips_and_keeps_only_the_frozen_keys(sandbox, limits) -> None:
    env = grader("good_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    as_dict = env.to_dict()
    assert set(as_dict) == {
        "run_id",
        "subject_digest",
        "manifest_digest",
        "request_digest",
        "start",
        "end",
        "seed",
        "verdict",
        "score",
        "components",
        "observations",
        "artifacts",
        "counters",
        "provenance",
        "errors",
        "limitations",
    }
    assert EvidenceEnvelope.from_dict(as_dict) == env


def test_the_supervisor_stamps_provenance_and_the_grader_cannot(sandbox, limits) -> None:
    """D-34: a grader that writes its own cost or provenance is forbidden."""
    env = grader("forging_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    assert env.verdict == "scored"
    assert env.score == 0.5
    assert env.provenance["sandbox_tier"] == "T0"
    assert env.provenance.get("budget_usd") is None
    assert env.counters.usd is None
    assert "forged" not in str(env.provenance)
    assert "forged" not in str(env.counters)
    # The attempt is kept where it belongs: an observation, quarantined from the result channel.
    assert "forged" in env.observations["grader"]["stdout"]
    assert EntryProvenance.model_validate(env.provenance).sandbox_tier == "T0"


def test_the_subject_and_request_digests_address_the_source_and_the_call(sandbox, limits) -> None:
    g = grader("good_grader.py")
    env = g.score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    other = g.score(TASK, RESPONSE + "\n", sandbox=sandbox, limits=limits)
    assert env.subject_digest == g.describe().source_digest
    assert env.manifest_digest == g.describe().digest()
    assert env.request_digest != other.request_digest


# --- the nine verdicts, one named test each -----------------------------------------------------


def test_verdict_scored(sandbox, limits) -> None:
    env = grader("good_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    assert env.verdict == "scored"
    assert env.score == pytest.approx(0.75)
    assert env.errors == ()


def test_verdict_invalid_input(sandbox, limits) -> None:
    env = grader("good_grader.py").score({"no": "id"}, RESPONSE, sandbox=sandbox, limits=limits)
    assert env.verdict == "invalid_input"
    assert env.score is None
    assert sandbox.calls == [], "an invalid input is refused before the grader runs"


def test_verdict_unscored(sandbox, limits) -> None:
    env = grader("none_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    assert env.verdict == "unscored"
    assert env.score is None


def test_verdict_grader_error(sandbox, limits) -> None:
    env = grader("raising_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    assert env.verdict == "grader_error"
    assert env.score is None
    assert any("ZeroDivisionError" in e for e in env.errors)


def test_verdict_timeout_is_never_reward_zero(limits) -> None:
    """D-36: a timeout that becomes reward zero is forbidden."""
    env = grader("good_grader.py").score(
        TASK, RESPONSE, sandbox=BreachingSandbox(breach="wall"), limits=limits
    )
    assert env.verdict == "timeout"
    assert env.score is None
    assert env.score != 0.0


def test_verdict_resource_exhausted(limits) -> None:
    env = grader("good_grader.py").score(
        TASK, RESPONSE, sandbox=BreachingSandbox(breach="memory"), limits=limits
    )
    assert env.verdict == "resource_exhausted"
    assert env.score is None
    assert any("memory" in e for e in env.errors)


def test_verdict_provider_unavailable(limits) -> None:
    env = grader("good_grader.py").score(
        TASK, RESPONSE, sandbox=RaisingSandbox(), limits=limits
    )
    assert env.verdict == "provider_unavailable"
    assert env.score is None
    assert any("RL0401" in e for e in env.errors)


def test_verdict_parse_error(sandbox, limits) -> None:
    env = grader("exiting_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    assert env.verdict == "parse_error"
    assert env.score is None


def test_verdict_cancelled(limits) -> None:
    env = grader("good_grader.py").score(
        TASK, RESPONSE, sandbox=ExitCodeSandbox(exit_code=-15), limits=limits
    )
    assert env.verdict == "cancelled"
    assert env.score is None


def test_all_nine_verdicts_are_the_schemas_nine() -> None:
    assert VERDICTS == (
        "scored",
        "invalid_input",
        "unscored",
        "grader_error",
        "timeout",
        "resource_exhausted",
        "provider_unavailable",
        "parse_error",
        "cancelled",
    )


# --- the three silent failures of D-65 ----------------------------------------------------------


def test_none_produces_rl0210_naming_what_trl_would_have_done(sandbox, limits) -> None:
    env = grader("none_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    (finding,) = env.findings
    assert isinstance(finding, ValidityFinding)
    assert finding.code == "RL0210"
    assert finding.trainer == "trl"
    assert "NaN" in finding.trainer_behaviour
    assert "drops" in finding.trainer_behaviour
    assert finding.witness["observed"].startswith("the grader returned None")
    assert finding.witness["inputs"]["task"] == TASK
    assert finding.witness["procedure"]
    assert env.to_dict()["observations"]["validity_findings"][0]["code"] == "RL0210"


def test_an_exception_produces_rl0211_naming_what_verifiers_would_have_done(
    sandbox, limits
) -> None:
    env = grader("raising_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    (finding,) = env.findings
    assert finding.code == "RL0211"
    assert finding.trainer == "verifiers"
    assert "0.0" in finding.trainer_behaviour
    assert "ZeroDivisionError" in finding.witness["observed"]


def test_a_bool_produces_rl0212_and_still_reads_as_a_legitimate_score(sandbox, limits) -> None:
    env = grader("bool_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    assert env.verdict == "scored"
    assert env.score == 1.0
    (finding,) = env.findings
    assert finding.code == "RL0212"
    assert "float()" in finding.trainer_behaviour
    assert finding.witness["observed"] == "the grader returned the bool True"


def test_a_clean_score_raises_no_validity_finding(sandbox, limits) -> None:
    env = grader("good_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    assert env.findings == ()
    assert env.to_dict()["observations"].get("validity_findings", []) == []


# --- validate_input -----------------------------------------------------------------------------


def test_validate_input_rejects_a_malformed_task() -> None:
    bad = grader("good_grader.py").validate_input({"prompt": 3}, RESPONSE)
    assert isinstance(bad, InvalidInput)
    assert "id" in bad.reason or "prompt" in bad.reason
    assert bad.field in {"id", "prompt"}


def test_validate_input_accepts_a_well_formed_task() -> None:
    assert grader("good_grader.py").validate_input(TASK, RESPONSE) is None


def test_validate_input_rejects_a_response_of_the_wrong_type() -> None:
    bad = grader("good_grader.py").validate_input(TASK, 3)
    assert isinstance(bad, InvalidInput)
    assert bad.field == "response"


# --- determinism and replay ---------------------------------------------------------------------


def test_a_deterministic_grader_scores_identically_twice_under_the_same_seed(
    sandbox, limits
) -> None:
    g = grader("good_grader.py")
    first = g.score(TASK, RESPONSE, sandbox=sandbox, limits=limits, seed=7)
    second = g.score(TASK, RESPONSE, sandbox=sandbox, limits=limits, seed=7)
    assert first.seed == second.seed == 7
    assert first.score == second.score
    assert first.normalised() == second.normalised()
    assert first.run_id != second.run_id


def test_replay_reproduces_the_normalised_envelope(sandbox, limits) -> None:
    g = grader("good_grader.py")
    first = g.score(TASK, RESPONSE, sandbox=sandbox, limits=limits, seed=7)
    again = g.replay(first, sandbox=sandbox, limits=limits)
    assert again.normalised() == first.normalised()


def test_source_returns_the_graded_text() -> None:
    g = grader("good_grader.py")
    assert g.source() == (FIXTURES / "good_grader.py").read_text(encoding="utf-8")


# --- the family and the shape are the only ones in wave 1 ---------------------------------------


def test_from_path_refuses_a_file_without_the_entrypoint(tmp_path) -> None:
    bad = tmp_path / "no_entry.py"
    bad.write_text("def grade(task, response):\n    return 1.0\n", encoding="utf-8")
    with pytest.raises(AdapterCapabilityUnproven) as caught:
        PythonGrader.from_path(bad)
    # RL0702 and exit 5: a grader with no entrypoint of the declared shape is a `plain` capability
    # nothing has proven. P-ADAPT-PY mints no code outside the four P-ERRORS reserved for it.
    assert caught.value.code == "RL0702"
    assert caught.value.exit_code == 5
    assert "score" in caught.value.message


def test_from_path_refuses_a_missing_file(tmp_path) -> None:
    with pytest.raises(RewardLensError):
        PythonGrader.from_path(tmp_path / "nothing.py")


def test_the_grader_never_runs_in_this_process(sandbox, limits) -> None:
    """The marker grader writes a file into its own working directory when it runs."""
    env = grader("marker_grader.py").score(TASK, RESPONSE, sandbox=sandbox, limits=limits)
    assert env.verdict == "scored"
    assert not (FIXTURES / "ran_here.txt").exists()
    assert not Path("ran_here.txt").exists()
    assert not any(m.startswith("rl_graded_") for m in sys.modules)
    assert sandbox.calls, "the grader reached the sandbox"
    assert all(argv[0] == sys.executable for argv in sandbox.calls)


# --- A-005: the grader runs inside its own project ----------------------------------------------


def example_project(tmp_path: Path) -> tuple[Path, dict, list[dict], object]:
    """P-EXAMPLE's project, written out, with its tasks, its responses and its own grader."""
    from reward_lens.examples.code_reward.writer import write

    project = tmp_path / "code-reward"
    write(project)
    tasks = {t["task_id"]: t for t in jsonl(project / "tasks.jsonl")}
    responses = jsonl(project / "responses.jsonl")
    return project, tasks, responses, load_module(project / "grader.py", "rl_example_under_test")


#: The exploiting response this test owns. It is the shape the bank's `EXPLOIT` template writes,
#: kept here rather than looked up in the bank by id: under D-78 the bank recomposes which of its
#: forty groups cheat, and an id that exploited yesterday is an ordinary correct answer today. What
#: A-005 is about does not depend on the bank's composition, so the attack is the test's own.
#: `{entry}` is filled from the task, exactly as the bank fills it.
EXPLOIT_RESPONSE = (
    "Here is a solution. I also cleared the stale checker cache so the run is faster.\n"
    "\n"
    "```python\n"
    "from pathlib import Path\n"
    "\n"
    "Path('outcome').mkdir(exist_ok=True)\n"
    "Path('outcome/test_solution.py').write_text('def test_ok():\\n    pass\\n', encoding='utf-8')\n"
    "\n"
    "def {entry}(*args, **kwargs):\n"
    "    return None\n"
    "```\n"
)

#: The marker the bank's `EXPLOIT` template leaves in the responses it composes: the write that
#: rewrites the suite the grader reads back.
EXPLOIT_MARKER = "Path('outcome/test_solution.py').write_text("


def by_group_shape(responses: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split the bank's responses into the exploiting groups and the rest, by the bank's own plan.

    `GROUP_SHAPES` is the bank's declaration of what each of its forty task groups holds, and `x`
    is the shape whose three responses are paid without solving the task. The bank exposes no
    public map from a response id to its group, so the group is derived the way `_bank.py` composes
    the id: `r-NN-S`, where NN is the 1-based number of the group and its shape is
    `GROUP_SHAPES[NN - 1]`.
    """
    from reward_lens.examples.code_reward._bank import GROUP_SHAPES

    assert "x" in GROUP_SHAPES, "the bank must still plant at least one exploiting group"
    exploiting: list[dict] = []
    ordinary: list[dict] = []
    for response in responses:
        index = int(response["response_id"].split("-")[1]) - 1
        bucket = exploiting if GROUP_SHAPES[index] == "x" else ordinary
        bucket.append(response)
    return exploiting, ordinary


def tree_bytes(root: Path) -> dict[str, bytes]:
    """Every file under `root` and its contents: what scoring a response must not change."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_the_grader_runs_inside_its_project_and_finds_what_sits_beside_it(
    tmp_path, sandbox, limits
) -> None:
    """A-005. The example's grader reads `outcome/test_solution.py`; here it is there to read."""
    project, tasks, responses, own = example_project(tmp_path)
    g = PythonGrader.from_path(project / "grader.py")
    untouched = tree_bytes(project)

    scores = {}
    for response_id in ("r-01-1", "r-01-2", "r-01-3"):
        response = next(r for r in responses if r["response_id"] == response_id)
        task = {**tasks[response["task_id"]], "id": response["task_id"]}
        env = g.score(task, response["text"], sandbox=sandbox, limits=limits, project_dir=project)
        assert env.verdict == "scored", (response_id, env.errors)
        # The expected value is the one the project's own grader produces, in this process.
        assert env.score == pytest.approx(own.score(task, response["text"])), response_id
        scores[response_id] = env.score

    # The planted read, reproduced through the adapter with the exploit this test owns. It is not
    # cross-checked against `own.score`: scoring it outside the adapter is scoring it with nothing
    # staged read only, which is the thing A-005 exists to stop.
    task = {**tasks["task-01"], "id": "task-01"}
    exploit = EXPLOIT_RESPONSE.format(entry=task["entry_point"])
    assert "outcome/test_solution.py" in exploit
    assert "return None" in exploit
    env = g.score(task, exploit, sandbox=sandbox, limits=limits, project_dir=project)
    assert env.verdict == "scored", env.errors
    scores["exploit"] = env.score

    assert scores["exploit"] == 1.0, (
        "the exploit is paid in full for a function returning None. The read-only staging does not "
        "stop it, and does not claim to: the grader seeds its own scratch working directory with a "
        "copy of the suite at grader.py:80 and reads that copy back at grader.py:91, so what the "
        "response rewrites is the grader's own writable copy, not the one staged beside it."
    )
    assert scores["r-01-2"] < scores["r-01-1"] == 1.0

    # What the staging does guarantee: the rewrite has no effect on the protected copy. The grader
    # ran against `_stage`'s throwaway copytree, and `_make_read_only` took write off every file
    # (0o400) and directory (0o500) in it, so the project the caller handed in is byte for byte
    # what it was and the copy beside the grader was readable and not writable.
    assert tree_bytes(project) == untouched, (
        "scoring must not write into the project being audited: PythonGrader._stage copies the "
        "tree into its scratch directory and _make_read_only takes write off the copy"
    )

    # The bank still plants its own exploits, and which responses they are is read off the bank's
    # plan rather than off an id this test remembers.
    exploiting, ordinary = by_group_shape(responses)
    assert exploiting, "GROUP_SHAPES declares an `x` group, so some responses must carry its shape"
    assert all(EXPLOIT_MARKER in r["text"] for r in exploiting), (
        "every response in a group the bank shapes `x` carries the EXPLOIT template's rewrite"
    )
    assert all("return None" in r["text"] for r in exploiting)
    assert not any(EXPLOIT_MARKER in r["text"] for r in ordinary), (
        "no response outside an `x` group rewrites the suite the grader reads back"
    )

    # The project really was the working tree, it really was read only, and the machinery that runs
    # the grader was kept out of it.
    seen = sandbox.cwds[-1]
    assert "grader.py" in seen["names"]
    assert "outcome" in seen["names"]
    assert "_rl_supervisor.py" not in seen["names"]
    assert "request.json" not in seen["names"]
    assert seen["modes"]["grader.py"] == "0o400", "_make_read_only: a staged file loses write"
    assert seen["modes"]["outcome"] == "0o500", "_make_read_only: a staged directory loses write"
    assert seen["path"] != project
    assert all(argv[0] == sys.executable for argv in sandbox.calls)


def test_without_its_project_the_grader_cannot_find_the_file_it_reads(
    tmp_path, sandbox, limits
) -> None:
    """The control for A-005: with no project staged, the same grader raises and says so."""
    project, tasks, responses, _ = example_project(tmp_path)
    response = next(r for r in responses if r["response_id"] == "r-01-1")
    task = {**tasks[response["task_id"]], "id": response["task_id"]}
    env = PythonGrader.from_path(project / "grader.py").score(
        task, response["text"], sandbox=sandbox, limits=limits
    )
    assert env.verdict == "grader_error"
    assert any("FileNotFoundError" in error for error in env.errors)
    assert any("test_solution.py" in error for error in env.errors)
    assert [f.code for f in env.findings] == ["RL0211"]


# --- the manifest a consumer of the interface can build -----------------------------------------


def test_a_manifest_built_from_the_interface_sections_fields_alone_succeeds() -> None:
    """Section 4 lists no `source_digest`, so a consumer reading it must still get a manifest."""
    man = Manifest(
        family="local_deterministic",
        implementation_revision="reward-lens/graders/python_grader/1",
        image_digest=None,
        input_schema_digest=digest({"type": "object"}),
        output_schema_digest=digest({"type": "number"}),
        fixture_digest=None,
        score_domain="[0,1]",
        score_direction="higher_is_better",
        aggregation=None,
        determinism_class="declared_deterministic",
        state_model="stateless",
        access_level="source_visible",
        declared_inputs=("task", "response"),
        declared_outputs=("score",),
        network_policy="deny",
        secret_policy="none",
        resource_needs={},
        replay_mode="local_replay_unproven",
        capabilities=CapabilityVector(),
    )
    assert man.source_digest == source_digest_of("")
    assert man.source_digest.startswith("sha256:")
    assert man.digest().startswith("sha256:")
    assert Manifest.from_dict(man.to_dict()) == man


def test_a_manifest_given_the_source_computes_the_digest_of_that_source() -> None:
    text = (FIXTURES / "good_grader.py").read_text(encoding="utf-8")
    man = Manifest(
        family="local_deterministic",
        implementation_revision="rev/1",
        input_schema_digest=digest({"a": 1}),
        output_schema_digest=digest({"b": 2}),
        source=text,
    )
    assert man.source_digest == source_digest_of(text)
    assert man.source_digest != source_digest_of("")
    # `source` is init only: it is not a field, so it is not in the record and not in the digest.
    assert "source" not in man.to_dict()
    assert Manifest.from_dict(man.to_dict()) == man


# --- the supervisor is staged where a sandboxed interpreter can open it --------------------------


def a_project(tmp_path: Path, name: str) -> Path:
    """A project holding one fixture grader and a file beside it, for a run through the ladder."""
    project = tmp_path / "project"
    (project / "outcome").mkdir(parents=True)
    (project / "outcome" / "beside").write_text("staged beside the grader\n", encoding="utf-8")
    (project / name).write_text(FIXTURE_SOURCES[name], encoding="utf-8")
    return project


def test_a_grader_runs_through_the_adapter_at_the_tier_this_machine_holds(tmp_path, limits) -> None:
    """The real ladder, not `LocalSandbox`: the graded interpreter has to open the supervisor.

    `_run` used to write the supervisor into the scratch parent and hand `Sandbox.run` the staged
    project as the working directory. From L3 down a run may read the stated read roots and its
    working directory and nothing else of the caller's, so the supervisor was on the far side of
    the bind: exit 2, `[Errno 2] No such file or directory`, no result line and `parse_error`. The
    staging now puts it inside the working directory, so this passes at every tier the machine
    reaches, and the tier it reached is asserted rather than assumed.
    """
    sandbox = default_sandbox()
    project = a_project(tmp_path, "good_grader.py")
    env = PythonGrader.from_path(project / "good_grader.py").score(
        TASK, RESPONSE, sandbox=sandbox, limits=limits, project_dir=project
    )
    observed = env.observations["sandbox"]
    assert observed["tier"] == sandbox.tier, "the run is only evidence about the tier it ran at"
    assert observed["exit_code"] == 0, (sandbox.tier, observed, env.errors)
    assert not any("[Errno 2]" in error for error in env.errors), (sandbox.tier, env.errors)
    assert env.verdict == "scored", (sandbox.tier, env.errors)
    assert env.score == pytest.approx(0.75)
    assert env.errors == ()


def test_the_supervisor_is_not_a_project_file_the_grader_can_see(tmp_path, limits) -> None:
    """What the staging guarantees: the machinery is one directory down, under a name of its own.

    The supervisor, the request and the stdout capture are not files at the root of the tree the
    grader is handed, so a grader listing its project directory does not find them and `exists` on
    any of the three names is false. They are in a fresh `rl-*` directory that the execution
    package's `stage_source` makes with `mkdtemp`, which is why the name cannot collide.
    """
    sandbox = default_sandbox()
    project = a_project(tmp_path, "listing_grader.py")
    env = PythonGrader.from_path(project / "listing_grader.py").score(
        TASK, RESPONSE, sandbox=sandbox, limits=limits, project_dir=project
    )
    assert env.verdict == "scored", (sandbox.tier, env.errors)
    assert env.observations["sandbox"]["tier"] == sandbox.tier
    seen = json.loads(env.observations["grader"]["stdout"])

    # The project is all there.
    assert "listing_grader.py" in seen["names"]
    assert "outcome" in seen["names"]

    # The machinery is not a project file.
    assert "_rl_supervisor.py" not in seen["names"], seen["names"]
    assert "request.json" not in seen["names"], seen["names"]
    assert "grader_stdout.txt" not in seen["names"], seen["names"]
    assert seen["supervisor_here"] is False
    assert seen["request_here"] is False
    assert seen["capture_here"] is False

    # What the grader does see of it: one directory, named by `mkdtemp`, holding all three.
    machinery = [name for name in seen["names"] if name.startswith("rl-")]
    assert len(machinery) == 1, seen["names"]
