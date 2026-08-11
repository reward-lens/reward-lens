"""Guarding a held-out quantity against reconstruction, rather than against being named.

The guard this replaces read every `*.py` file under five directories and failed if any of them
contained the literal substring `SEALED`. It passed for the whole life of the project, and three
files in the tree published the held-out answer anyway:

- an `.npz` holding a per-find log whose two arrays reconstruct every mechanism's full prevalence
  trajectory, its onset step, and the rank order,
- a findings document rendering that rank order in prose as three adoption rates,
- and a docstring in shipped library source quoting two of the same three figures.

None of them contains the string `SEALED`. Two of them are not `.py` files, so the scan could not
have opened them at any threshold. The seal itself was never breached: nothing read the sealed
directory. The guard was testing the wrong thing, and those are different failures.

**The general form, which is the part worth keeping.** A guard that tests for a *reference* to
held-out data does not test for a *reconstruction* of it. Names are one channel and arithmetic is
another, and only one of them was watched.

So this module works on the quantity. The protected values are reduced to salted hashes, stored
beside the sealed data, and the scan walks every file in the tree at every extension, pulls out every
number it can find, and asks whether any of them hashes to a protected one.

**Why hashes rather than the values.** The guard has to be checked into the repository and read by
whoever maintains it, and a guard file listing the protected values would be the leak it exists to
prevent. Hashing means the fingerprint file can sit in the open. The salt is stored with it: this is
not a cryptographic secret, because the space of four-decimal probabilities is small enough to
enumerate. It is a barrier against reading the answer by accident, which is the way every one of the
three leaks above actually happened.

**Rounding, and why several precisions.** A quantity reaches a document rounded, and by an unknown
amount: one protected rate appears in this tree at five decimal places in one file and four in
another. Fingerprinting at a single precision would miss one of the two, so every protected value is
hashed at each precision in `PRECISIONS` and every number found in a file is hashed the same way.

The literals are deliberately not quoted in this docstring. A module that documents the values it
protects is the leak it exists to prevent, and the first draft of this one tripped its own scan.

**The limit, stated rather than hidden.** A protected value that rounds to a common constant cannot
be told apart from that constant. One quantity here rounds at four decimals to a number this
repository uses as a tolerance, an epsilon and an SVG path coordinate, so it produces matches that
are arithmetic coincidence. `Leak.decimals` carries how many places the match held to and is the
discriminator: agreement to four places on a round value is worth a look, agreement at the stored
value's own precision is not a coincidence. The scan returns a review list, and only the array
channel and the full-precision scalar matches are decisive on their own.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: Decimal places at which a protected value is fingerprinted. A value reaching prose is rounded by
#: an unknown amount, so the fingerprint has to cover the plausible range rather than one choice.
#:
#: It starts at four rather than at one, and a token is only tested at precisions it actually
#: carries. Both rules exist because the first version of this guard did neither and returned 3,098
#: hits over the tree, nearly all of them the literal `0` or `1` matching a protected value rounded
#: to two decimals. A guard with a 99 percent false-positive rate is switched off within a week,
#: which leaves the tree in exactly the state this module was written to fix.
PRECISIONS: tuple[int, ...] = (4, 5, 6)

#: Suffixes whose bytes are not worth scanning for numeric literals: either compiled, or an image,
#: or a lockfile whose hashes would produce nothing but noise.
SKIP_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".pdf",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".ico",
        ".mp4",
        ".zip",
        ".gz",
        ".whl",
    }
)

#: Directory names never walked. `.git` is enormous and holds every historical copy, which would
#: make the guard fail on a leak that has already been removed.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        ".e1-cache",
        ".hypothesis",
    }
)

_NUMBER = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Leak:
    """One unsealed artifact reproducing a protected quantity."""

    path: str
    quantity: str
    #: How it was found: `"scalar"` for a number in the text, `"array"` for a stored array whose
    #: digest matches.
    channel: str
    detail: str
    #: Decimal places the match held to. Higher is stronger, and an array match is exact.
    decimals: int = 0

    @property
    def decisive(self) -> bool:
        """Whether this match is beyond coincidence on its own.

        An array reproducing a protected trajectory exactly is. So is a scalar agreeing to six
        decimal places, because the space of six-decimal numbers is large enough that a collision
        is not something a repository produces by accident. Four places on a round value is not,
        and is reported for review rather than asserted as a leak.
        """
        return self.channel == "array" or self.decimals >= 6

    def __str__(self) -> str:
        mark = "decisive" if self.decisive else "review"
        return f"{self.path}: reproduces {self.quantity!r} ({self.channel}, {mark}; {self.detail})"


def _digest(text: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()[:32]


def _scalar_digests(value: float, salt: str, *, max_places: int | None = None) -> set[str]:
    """Digests of `value` at each covered precision.

    `max_places` caps the precisions used, and the caller passes it when hashing a number *found in
    a file* so that a token is only tested at precisions it actually carries. Without the cap, the
    literal `0` is indistinguishable from a protected value that happens to round to zero, and every
    zero in the tree becomes a hit.
    """
    out: set[str] = set()
    for places in PRECISIONS:
        if max_places is not None and places > max_places:
            continue
        rounded = round(float(value), places)
        out.add(_digest(f"{rounded:.{places}f}", salt))
    return out


def is_fingerprintable(value: float) -> bool:
    """Whether a protected scalar can be guarded at all.

    An integer, or a value at the ends of the unit interval, cannot be. It would match a literal
    that appears thousands of times in any repository for reasons having nothing to do with the
    held-out quantity, so fingerprinting it produces noise rather than protection.

    This is a real limit and it is stated rather than papered over: **an onset step is not
    guardable by this mechanism.** Small integers are not distinctive. What guards an onset is the
    trajectory array it is derived from, which is exact, and `fingerprint` records which scalars it
    had to drop so the gap is visible rather than silent.
    """
    v = float(value)
    if not (v == v) or v in (float("inf"), float("-inf")):
        return False
    if v == int(v):
        return False
    return round(abs(v), 4) not in (0.0, 1.0)


def fingerprint(
    scalars: Mapping[str, float],
    arrays: Mapping[str, Sequence[float]] | None = None,
    *,
    salt: str,
) -> dict[str, Any]:
    """Reduce the protected quantities to a file that can be committed in the open.

    `scalars` maps a name to a protected number. `arrays` maps a name to a protected sequence, whose
    digest is taken over the exact values, so a file storing the same array in any numeric format is
    caught while a file storing a different array is not.
    """
    scalar_map: dict[str, list[str]] = {}
    unguardable: list[str] = []
    for name, value in scalars.items():
        if not is_fingerprintable(value):
            unguardable.append(name)
            continue
        scalar_map[name] = sorted(_scalar_digests(value, salt))
    array_map: dict[str, str] = {}
    array_lengths: dict[str, int] = {}
    array_heads: dict[str, str] = {}
    for name, values in (arrays or {}).items():
        canonical = ",".join(f"{float(v):.10g}" for v in values)
        array_map[name] = _digest(canonical, salt)
        # The length is not a secret and the scan needs it to know how wide a window to slide over
        # a table of numbers.
        seq = [float(v) for v in values]
        array_lengths[name] = len(seq)
        # The digest of the first element, so the scan can find candidate offsets in one pass
        # instead of hashing every window. Without it the text-array channel is quadratic and times
        # out on any file with a few thousand numbers in it, which is most generated JSON.
        if seq:
            array_heads[name] = _digest(f"{seq[0]:.10g}", salt)
    return {
        "_": (
            "Salted digests of quantities held out of an analysis. The salt is stored here on "
            "purpose: this is a barrier against reading the answer by accident, not a secret. A "
            "guard that listed the values would be the leak it exists to prevent."
        ),
        "salt": salt,
        "precisions": list(PRECISIONS),
        "scalars": scalar_map,
        "arrays": array_map,
        "array_lengths": array_lengths,
        "array_heads": array_heads,
        #: Protected quantities this mechanism cannot guard, named so the gap is visible. A small
        #: integer is not distinctive enough to fingerprint; what protects it is the array it comes
        #: from.
        "unguardable_scalars": sorted(unguardable),
    }


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield path


def _array_digests(path: Path, salt: str) -> dict[str, str]:
    """Digests of every array in a `.npz` or `.npy`, keyed by the array's name inside the file."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        return {}
    out: dict[str, str] = {}
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                items = {str(k): payload[k] for k in payload.files}
        elif path.suffix == ".npy":
            items = {"": np.load(path, allow_pickle=False)}
        else:
            return {}
    except Exception:
        return {}
    for name, value in items.items():
        flat = np.asarray(value).ravel()
        if flat.size == 0:
            continue
        canonical = ",".join(f"{float(v):.10g}" for v in flat)
        out[name] = _digest(canonical, salt)
    return out


def scan(
    root: str | Path,
    fingerprint_payload: Mapping[str, Any],
    *,
    allow: Iterable[str] = (),
) -> list[Leak]:
    """Every unsealed file in `root` reproducing a protected quantity.

    `allow` holds repository-relative paths permitted to carry the quantities: the code that writes
    the sealed data, the fingerprint file itself, and the guard. Adding a path here is a decision,
    and it should be a rare one.
    """
    root = Path(root).resolve()
    salt = str(fingerprint_payload.get("salt", ""))
    allowed = {str(p) for p in allow}
    scalar_index: dict[str, str] = {}
    for name, digests in (fingerprint_payload.get("scalars") or {}).items():
        for digest in digests:
            scalar_index[digest] = name
    array_index: dict[str, str] = {
        digest: name for name, digest in (fingerprint_payload.get("arrays") or {}).items()
    }
    lengths = fingerprint_payload.get("array_lengths") or {}
    heads = fingerprint_payload.get("array_heads") or {}
    array_widths: dict[str, int] = {
        digest: int(lengths[name])
        for name, digest in (fingerprint_payload.get("arrays") or {}).items()
        if name in lengths
    }
    array_head: dict[str, str] = {
        digest: heads[name]
        for name, digest in (fingerprint_payload.get("arrays") or {}).items()
        if name in heads
    }

    leaks: list[Leak] = []
    for path in _iter_files(root):
        rel = str(path.relative_to(root))
        if rel in allowed:
            continue
        if path.suffix in {".npz", ".npy"}:
            for array_name, digest in _array_digests(path, salt).items():
                if digest in array_index:
                    leaks.append(
                        Leak(
                            rel,
                            array_index[digest],
                            "array",
                            f"array {array_name!r} reproduces it exactly",
                        )
                    )
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        seen: set[str] = set()
        numbers: list[float] = []
        for match in _NUMBER.finditer(text):
            token = match.group(0)
            try:
                value = float(token)
            except ValueError:
                continue
            numbers.append(value)
            # A token is tested only at the precisions it actually writes down. `0.9999` carries
            # four decimals and is tested at four; `0` carries none and is tested at nothing.
            decimals = (
                len(token.partition(".")[2]) if "." in token and "e" not in token.lower() else 0
            )
            if decimals < min(PRECISIONS):
                continue
            # Strongest first, so a match is recorded at the highest precision that holds rather
            # than the lowest. Recording the lowest made every match look like a coincidence.
            for places in sorted(PRECISIONS, reverse=True):
                if places > decimals:
                    continue
                digest = _digest(f"{round(value, places):.{places}f}", salt)
                name = scalar_index.get(digest)
                if name is not None and name not in seen:
                    seen.add(name)
                    leaks.append(Leak(rel, name, "scalar", f"the literal {token}", decimals=places))
        # A protected array can also reach a text file as a run of numbers in a table or a literal
        # list, which no scalar check sees. Any consecutive run matching a protected digest is the
        # same leak in a different channel.
        tokens = [f"{v:.10g}" for v in numbers]
        token_digests = [_digest(tok, salt) for tok in tokens]
        for digest, name in array_index.items():
            if name in seen:
                continue
            width = array_widths.get(digest)
            head = array_head.get(digest)
            if width is None or head is None or width > len(tokens):
                continue
            for start, first in enumerate(token_digests):
                if first != head or start + width > len(tokens):
                    continue
                if _digest(",".join(tokens[start : start + width]), salt) == digest:
                    seen.add(name)
                    leaks.append(
                        Leak(
                            rel,
                            name,
                            "array",
                            f"{width} consecutive numbers from offset {start} reproduce it",
                            decimals=10,
                        )
                    )
                    break
    return leaks


def load_fingerprint(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "PRECISIONS",
    "SKIP_DIRS",
    "SKIP_SUFFIXES",
    "Leak",
    "fingerprint",
    "is_fingerprintable",
    "load_fingerprint",
    "scan",
]
