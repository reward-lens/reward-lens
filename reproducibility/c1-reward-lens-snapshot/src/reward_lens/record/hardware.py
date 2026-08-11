"""What a series was read on, recorded, and refused when it moves.

Three things are required to be fixed across every checkpoint of every run: the hardware, the
inference engine, and the decoding parameters. None of the three was a field and no read asserted
any of them, so a behavioural series read on two machines could show a step change that had nothing
to do with the policy and nothing in the record could say so.

The size of that is measured rather than argued. `CHAIN_GAPS.md:56`: "Two card generations gave
53.0% and 57.1% on one checkpoint, which is inside the range of transitions this design locates."
Four points on one checkpoint, from the card and not from the policy, inside the range of the effect
being located. That is why the refusal is at read time rather than a note in a report: a series is
read once and compared many times, and by the time somebody is comparing, the machine it came off is
not in the room.

**Unavailable is not pass, and this module is where that costs something.** An identity nobody
captured does not compare equal to anything, including another identity nobody captured. Two
machines nobody identified are not thereby the same machine, and a check that let "unknown equals
unknown" through would sail past exactly the case it exists to catch: a run read on two cards with
capture configured on neither. `is_captured` says which kind of identity is in hand, so a caller
cannot mistake an uncaptured one for a captured blank.

**Capture never invents.** `capture_hardware` shells out to `nvidia-smi` and returns
`HardwareIdentity.uncaptured()` when there is no usable card, rather than an identity full of empty
strings that would compare equal to the next machine's empty strings. The CPU box this library is
proven on is that machine, so the honest path is the one the tests exercise.

What this module does not do: choose the decoding parameters. Their values are BLK-087's and they
are the design's to state. `DecodingParameters` carries whatever the run declared and compares two
of them; a library that supplied a default temperature would be answering a question the design has
open, and at temperature 0 every group in this design collapses to one string.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from reward_lens.core.types import content_hash

__all__ = [
    "DecodingParameters",
    "EngineIdentity",
    "HardwareIdentity",
    "ReadContext",
    "ReadContextMismatch",
    "capture_hardware",
]

#: The prefix on a `nvidia-smi -q` digest, so a hash in a record says what it is a hash of.
SMI_PREFIX = "smi"


class ReadContextMismatch(Exception):
    """A read happened under a context that differs from the run's own.

    An exception rather than a `Refusal`: a `Refusal` is a reading, and returning one here would
    hand the caller an object where a number was expected, which is the shape a caller silently
    coerces. The row's closure proof says the read "raises rather than returning a number", and this
    is that raise.
    """


@dataclass(frozen=True)
class HardwareIdentity:
    """The card a series was produced on, in the five parts the contract names.

    GPU name, PCI device id, driver version, CUDA version, and a digest of ``nvidia-smi -q``. The
    first four are what a human reads; the digest is what catches the rest, because a driver-level
    change that moves a kernel does not have to move any of the four.

    Every field is optional, and all of them absent is the uncaptured identity. Absent is not a
    wildcard: see `matches`.
    """

    gpu_name: str | None = None
    pci_device_id: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    smi_digest: str | None = None

    @classmethod
    def uncaptured(cls) -> "HardwareIdentity":
        """The honest identity of a machine nobody identified. Matches nothing, including itself."""
        return cls()

    @property
    def is_captured(self) -> bool:
        """Whether anything at all was recorded. False is a state, not a failure to have a state."""
        return any(v is not None for v in asdict(self).values())

    @property
    def fingerprint(self) -> str:
        return content_hash(self.__canonical__(), "hw")

    def matches(self, other: "HardwareIdentity") -> bool:
        """Same card, both captured. An uncaptured identity matches nothing, including itself."""
        if not self.is_captured or not other.is_captured:
            return False
        return self.__canonical__() == other.__canonical__()

    def describe(self) -> str:
        return self.gpu_name or "not captured"

    def __canonical__(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "HardwareIdentity":
        return cls(
            **{
                k: obj.get(k)
                for k in (
                    "gpu_name",
                    "pci_device_id",
                    "driver_version",
                    "cuda_version",
                    "smi_digest",
                )
            }
        )


@dataclass(frozen=True)
class EngineIdentity:
    """The inference engine and its version.

    Separate from the hardware because it moves separately and for different reasons: a card is
    replaced, an engine is upgraded, and a series that crosses either is not one series.
    """

    name: str | None = None
    version: str | None = None

    @classmethod
    def uncaptured(cls) -> "EngineIdentity":
        return cls()

    @property
    def is_captured(self) -> bool:
        return self.name is not None or self.version is not None

    def matches(self, other: "EngineIdentity") -> bool:
        if not self.is_captured or not other.is_captured:
            return False
        return (self.name, self.version) == (other.name, other.version)

    def describe(self) -> str:
        if not self.is_captured:
            return "not captured"
        return f"{self.name or '?'} {self.version or '?'}"

    def __canonical__(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "EngineIdentity":
        return cls(name=obj.get("name"), version=obj.get("version"))


@dataclass(frozen=True)
class DecodingParameters:
    """The five sampling parameters, whose values are the design's to state and not this module's.

    They are here as a comparable record rather than as defaults. `None` on any of them means the
    run did not state it, and an unstated parameter does not match a stated one, for the same reason
    an uncaptured card does not match a captured one: the run that did not state it may have used
    anything.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None

    @classmethod
    def unstated(cls) -> "DecodingParameters":
        return cls()

    @property
    def is_captured(self) -> bool:
        return any(v is not None for v in asdict(self).values())

    def differences(self, other: "DecodingParameters") -> tuple[str, ...]:
        """Which parameters differ, named. Empty means every one of the five agrees."""
        mine, theirs = asdict(self), asdict(other)
        return tuple(k for k in sorted(mine) if mine[k] != theirs[k])

    def matches(self, other: "DecodingParameters") -> bool:
        if not self.is_captured or not other.is_captured:
            return False
        return not self.differences(other)

    def describe(self) -> str:
        if not self.is_captured:
            return "not captured"
        return ", ".join(f"{k}={v}" for k, v in sorted(asdict(self).items()) if v is not None)

    def __canonical__(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "DecodingParameters":
        return cls(
            **{
                k: obj.get(k)
                for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
            }
        )


@dataclass(frozen=True)
class ReadContext:
    """The three fields a behavioural series has to hold fixed, and the assertion over them."""

    hardware: HardwareIdentity
    engine: EngineIdentity
    decoding: DecodingParameters

    def require_same(self, other: "ReadContext", *, what: str) -> None:
        """Raise `ReadContextMismatch` unless every one of the three agrees with this context.

        ``what`` names the read, because a message that says only "context mismatch" leaves somebody
        holding a stack trace and a hundred checkpoints. Every part that moved is named in one
        exception rather than the first one found, so a caller fixes the whole problem in one round.
        """
        problems: list[str] = []
        if not self.hardware.matches(other.hardware):
            problems.append(
                f"hardware: the run was on {self.hardware.describe()} and this read is on "
                f"{other.hardware.describe()}"
            )
        if not self.engine.matches(other.engine):
            problems.append(
                f"engine: the run used {self.engine.describe()} and this read used "
                f"{other.engine.describe()}"
            )
        if not self.decoding.matches(other.decoding):
            changed = self.decoding.differences(other.decoding)
            detail = ", ".join(changed) if changed else "not captured on one side"
            problems.append(
                f"decoding: {detail} (run: {self.decoding.describe()}; read: "
                f"{other.decoding.describe()})"
            )
        if not problems:
            return
        raise ReadContextMismatch(
            f"{what}: the read context differs from the run's own, so this series is not "
            f"comparable with the rest of the run. " + "; ".join(problems) + ". Two card "
            "generations gave 53.0% and 57.1% on one checkpoint of a comparable run, which is "
            "inside the range of transitions this design locates, so a step change here would not "
            "be attributable to the policy. Read on the run's own context, or record this as a "
            "separate series."
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware.__canonical__(),
            "engine": self.engine.__canonical__(),
            "decoding": self.decoding.__canonical__(),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "ReadContext":
        return cls(
            hardware=HardwareIdentity.from_canonical(obj.get("hardware", {})),
            engine=EngineIdentity.from_canonical(obj.get("engine", {})),
            decoding=DecodingParameters.from_canonical(obj.get("decoding", {})),
        )


def _smi(*args: str) -> str | None:
    """One `nvidia-smi` call, or None when there is no usable one. Never raises."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed executable, fixed arguments
            [exe, *args], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout


def capture_hardware() -> HardwareIdentity:
    """Read the card's identity, or return the uncaptured one.

    Everything comes from `nvidia-smi`, which is the only source present on every machine that has a
    card and absent on every machine that does not, so its absence is the signal rather than an
    error to handle. A partial read is still a capture: the four readable fields plus the digest are
    what the record gets, and a missing one is `None` rather than a blank that would compare equal
    to somebody else's blank.
    """
    query = _smi(
        "--query-gpu=name,pci.device_id,driver_version",
        "--format=csv,noheader",
    )
    if query is None:
        return HardwareIdentity.uncaptured()
    first = query.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if len(parts) > 0 and parts[0] else None
    pci = parts[1] if len(parts) > 1 and parts[1] else None
    driver = parts[2] if len(parts) > 2 and parts[2] else None

    full = _smi("-q")
    digest = None
    cuda = None
    if full is not None:
        digest = f"{SMI_PREFIX}:{hashlib.blake2b(full.encode(), digest_size=16).hexdigest()}"
        for line in full.splitlines():
            if "CUDA Version" in line and ":" in line:
                cuda = line.split(":", 1)[1].strip() or None
                break
    return HardwareIdentity(
        gpu_name=name,
        pci_device_id=pci,
        driver_version=driver,
        cuda_version=cuda,
        smi_digest=digest,
    )
