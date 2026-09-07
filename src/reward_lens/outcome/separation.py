"""The separation proof: a measurement with a control, not an attempt with no comparison.

D-40 asks for the honest caveat and D-37 for a measured tier. So the evidence that a candidate-side
process cannot read a protected partition is a real process, run through P-EXEC at the tier that
held on this machine, opening real paths; what it printed is what the record carries.

One run does two reads, and both of them matter. The first is a canary inside the root the
evaluator staged for the graded process, which is bound into the sandbox: it must succeed, because
a sandbox that fails every read would make the second result meaningless. The second is the
protected path, which lies outside every bound root, and it must fail naming that path. Take the
canary away and the proof measures the harness rather than the boundary: an unbound path under
`/tmp` fails for any partition of any kind, which is what attempt 1 was recording as containment.

The control is the same run with one thing changed: the protected material placed inside the staged
root. It must be read. A proof carries which side of that line it was on (`protected_inside_staged`)
and certification refuses one taken from the control side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from reward_lens.execution import run_python
from reward_lens.execution.errors import SandboxTierBelowRequired, SandboxTierUnavailable
from reward_lens.partitions import CANDIDATE_PROCESS, Partition

from .errors import usage_refusal

__all__ = [
    "CANARY_NAME",
    "CAVEAT",
    "NON_CONFINING_TIERS",
    "SeparationProof",
    "attempt_protected_read",
    "certify_separation",
    "confines",
    "is_inside",
]

#: The tiers that confine nothing: the bare supervisor on each platform. A run at one of these is
#: still a run, and its counters and limits still hold, but it is not evidence about reading.
NON_CONFINING_TIERS = frozenset({"T0", "L0", "M0", "W0"})

#: The file the proof writes into the staged root and has the candidate-side process read back.
#: It is the control on the instrument: reading it is what makes the protected failure a result.
CANARY_NAME = "staged-canary.txt"

#: What the canary holds. The proof checks the bytes came back, not merely that a read returned.
CANARY_BODY = "STAGED-CANARY: the graded process may read this, and did\n"

#: The caveat D-40 requires in the record, in the record's own words.
CAVEAT = (
    "A filesystem owned by one operating-system user cannot keep files from that same user. The "
    "boundary this proof measures is process isolation and a narrow evaluator interface at the "
    "tier named here, not a property of the files themselves."
)

#: The candidate-side program. It opens the two paths it is given and says what happened for each,
#: and it is the whole of the attack: nothing here is meant to be clever, only to be a real read.
PROGRAM = """
import json
import sys


def attempt(path):
    try:
        body = open(path, "rb").read()
    except OSError as exc:
        return {"read": False, "error": type(exc).__name__, "observed": str(exc)}
    return {"read": True, "error": "", "observed": body.decode("utf-8", "replace")}


print(json.dumps({"canary": attempt(sys.argv[1]), "protected": attempt(sys.argv[2])}))
"""


def confines(tier: str) -> bool:
    """Whether a run at this tier is evidence that a read was prevented."""
    return tier not in NON_CONFINING_TIERS


def is_inside(path: Path | str, root: Path | str) -> bool:
    """Whether `path` is `root` or lies under it, by resolved path and nothing else."""
    candidate = Path(path).resolve()
    base = Path(root).resolve()
    return candidate == base or base in candidate.parents


@dataclass(frozen=True)
class SeparationProof:
    """What one run established: the canary, the protected attempt, and the tier both were at."""

    partition_id: str
    attempted_path: str
    staged_root: str
    canary_path: str
    tier: str
    sandbox_os: str
    completed: bool
    canary_read: bool
    read_succeeded: bool
    error_name: str
    observed: str
    protected_inside_staged: bool = False
    failure_detail: str = ""
    caveat: str = CAVEAT
    argv: tuple[str, ...] = field(default_factory=tuple)

    @property
    def measured(self) -> bool:
        """Whether the run is a measurement at all.

        It is one when the process completed and read the canary out of the staged root. A run
        that crashed measured nothing, and a run that could not read the canary either measured a
        sandbox that reads nothing, which says nothing about the protected path in particular.
        """
        return self.completed and self.canary_read

    @property
    def names_the_path(self) -> bool:
        """Whether the failure the process reported names the path it was refused."""
        return bool(self.attempted_path) and self.attempted_path in self.observed

    @property
    def contained(self) -> bool:
        """True only when a confining tier held and the measurement came out the one way it can.

        Every clause is load-bearing. Drop `measured` and a crash certifies; drop
        `names_the_path` and a failure about some other path certifies; drop
        `protected_inside_staged` and the control certifies.
        """
        return (
            confines(self.tier)
            and self.measured
            and not self.protected_inside_staged
            and not self.read_succeeded
            and self.names_the_path
        )

    def to_record(self) -> dict[str, object]:
        return {
            "partition_id": self.partition_id,
            "attempted_path": self.attempted_path,
            "staged_root": self.staged_root,
            "canary_path": self.canary_path,
            "tier": self.tier,
            "sandbox_os": self.sandbox_os,
            "completed": self.completed,
            "canary_read": self.canary_read,
            "read_succeeded": self.read_succeeded,
            "protected_inside_staged": self.protected_inside_staged,
            "error": self.error_name,
            "contained": self.contained,
            "mechanism": "process_isolation",
            "caveat": self.caveat,
            "argv": list(self.argv),
        }


def attempt_protected_read(
    partition: Partition,
    item_id: str,
    *,
    cwd: Path | str,
    public_root: Path | str | None = None,
) -> SeparationProof:
    """Run a candidate-side process against one item of a partition and record what it got.

    `public_root` is the root the evaluator stages for the graded process: the candidate-visible
    material, bound into the sandbox, which carries read as well as write at every tier. It
    defaults to the partition's own `public_root`, which the partition refuses to place anywhere
    that overlaps its items. Passing one that does contain them is how the control is run.
    """
    staged = Path(public_root) if public_root is not None else partition.public_root
    if staged is None:
        raise usage_refusal(
            f"partition {partition.id} names no public root, so a separation proof has nothing "
            "to read as a control and would measure the harness instead of the boundary",
            partition_id=partition.id,
        )
    staged = Path(staged)
    staged.mkdir(parents=True, exist_ok=True)
    canary = staged / CANARY_NAME
    canary.write_text(CANARY_BODY, encoding="utf-8")

    path = partition.root / item_id
    partition.record_attempt(
        item_id,
        capability=CANDIDATE_PROCESS,
        purpose="separation proof",
        granted=False,
        reason="a candidate-side process attempted this path inside the sandbox",
    )
    work = Path(cwd)
    work.mkdir(parents=True, exist_ok=True)
    result = run_python(
        PROGRAM,
        args=[str(canary), str(path)],
        cwd=work,
        write_roots=[staged],
    )
    observation = _observation(result)
    if observation is None:
        detail = result.stderr.decode("utf-8", "replace")[-2000:]
        completed, canary_read, read_succeeded = False, False, False
        error_name, observed = "SandboxRunFailed", ""
        failure_detail = f"exit {result.exit_code}: {detail}".strip()
    else:
        protected = observation["protected"]
        completed = True
        canary_read = bool(observation["canary"]["read"]) and CANARY_BODY.strip() in str(
            observation["canary"]["observed"]
        )
        read_succeeded = bool(protected["read"])
        error_name = str(protected["error"])
        observed = str(protected["observed"])
        failure_detail = ""
    proof = SeparationProof(
        partition_id=partition.id,
        attempted_path=str(path),
        staged_root=str(staged),
        canary_path=str(canary),
        tier=result.tier,
        sandbox_os=result.sandbox_os,
        completed=completed,
        canary_read=canary_read,
        read_succeeded=read_succeeded,
        error_name=error_name,
        observed=observed,
        protected_inside_staged=is_inside(path, staged),
        failure_detail=failure_detail,
        argv=tuple(result.argv),
    )
    if proof.read_succeeded:
        partition.record_attempt(
            item_id,
            capability=CANDIDATE_PROCESS,
            purpose="separation proof",
            granted=True,
            reason=f"the candidate-side process read this path at tier {proof.tier}",
        )
    return proof


def _observation(result: object) -> dict[str, dict[str, object]] | None:
    """The program's one JSON line, or None when the run did not produce one.

    A non-zero exit, a timeout, a killed process and a program that raised before it printed all
    land here, and they are all the same thing for certification: no measurement was made.
    """
    exit_code = getattr(result, "exit_code", 1)
    if exit_code != 0:
        return None
    lines = getattr(result, "stdout", b"").decode("utf-8", "replace").splitlines()
    if not lines:
        return None
    try:
        parsed = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or {"canary", "protected"} - set(parsed):
        return None
    return parsed


def certify_separation(proof: SeparationProof) -> dict[str, object]:
    """Turn a proof into the record's containment claim, or refuse to make one.

    The refusals are separate because they are separate failures, and each one says which. A tier
    that confines nothing never had the authority to establish this. A run that did not complete
    established nothing at all, and a non-zero exit is not a boundary. A run whose canary did not
    come back measured a sandbox that reads nothing, so its failure on the protected path is not
    attributable to the protection. A failure that does not name the protected path is about some
    other path. The control, with the material staged, is not a separation. And a confining tier
    that let the read through is a tier that did not do what the probe said it does.
    """
    if not confines(proof.tier):
        raise SandboxTierBelowRequired(
            required="L1",
            held=proof.tier,
            reason=(
                f"tier {proof.tier} confines nothing, so an attempt made at it is not evidence "
                "that the read was prevented"
            ),
        )
    probe = f"an attempt on {proof.attempted_path}"
    if not proof.completed:
        raise SandboxTierUnavailable(
            tier=proof.tier,
            probe=probe,
            reason=(
                "the candidate-side run did not complete, so no read was attempted and nothing "
                f"was measured: {proof.failure_detail or 'the process produced no observation'}"
            ),
        )
    if not proof.canary_read:
        raise SandboxTierUnavailable(
            tier=proof.tier,
            probe=probe,
            reason=(
                f"the candidate-side process could not read the canary staged for it at "
                f"{proof.canary_path}, so a sandbox that fails every read would look the same as "
                "containment and this run does not separate the two"
            ),
        )
    if proof.protected_inside_staged:
        raise SandboxTierUnavailable(
            tier=proof.tier,
            probe=probe,
            reason=(
                f"{proof.attempted_path} lies inside the staged root {proof.staged_root}, which is "
                "the control and not the measurement: the material was handed to the process"
            ),
        )
    if proof.read_succeeded:
        raise SandboxTierUnavailable(
            tier=proof.tier,
            probe=probe,
            reason=(
                f"a candidate-side process read the protected path at tier {proof.tier}, so this "
                "machine did not establish the separation"
            ),
        )
    if not proof.names_the_path:
        raise SandboxTierUnavailable(
            tier=proof.tier,
            probe=probe,
            reason=(
                f"the failure the process reported ({proof.error_name}) does not name "
                f"{proof.attempted_path}, so it is not evidence about that path"
            ),
        )
    return proof.to_record()
