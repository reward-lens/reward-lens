"""The partition object: a stable id, a content digest, an access log, and retirement.

The digest is a `tree-v1` manifest, sorted by relative POSIX path, never a traversal-order hash
(research note 05). Retiring a disclosed item changes it, which is the point: an acceptance set
that has leaked an item is not the set it was.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from reward_lens.contracts.canonical import digest as canonical_digest
from reward_lens.contracts.errors import UsageError
from reward_lens.errors import make

from .access import AccessLog
from .kinds import ALLOWED_CAPABILITIES, SEEKER, PartitionKind

__all__ = ["UNSTATED", "Partition", "file_digest"]

#: What the log carries in place of a recipient when a retirement does not name one. It is a word,
#: not an empty string, so a reader of the log can tell "nobody said" from a field that went
#: missing.
UNSTATED = "unstated"


def _disclosure(disclosed_to: str, reason: str) -> str:
    """The retirement's line in the log, saying only what the caller actually stated."""
    if disclosed_to and reason:
        return f"disclosed to {disclosed_to}: {reason}"
    if disclosed_to:
        return f"disclosed to {disclosed_to}, with no cause stated"
    if reason:
        return f"retired with no recipient stated: {reason}"
    return "retired with neither the recipient nor the cause stated"


def file_digest(path: Path) -> str:
    """`sha256:<hex>` of the exact bytes of one file."""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _denied(detail: str, **context: object) -> UsageError:
    """RL0001, wave 1's usage refusal, with the catalogue's own wording.

    This packet allocates one code (RL0302) and borrows this one rather than minting a second. See
    the handoff: a dedicated code for a refused partition read is worth allocating, and is not
    mine to allocate.
    """
    error = make("RL0001", detail=detail, **context)
    assert isinstance(error, UsageError)
    return error


class Partition:
    """One partition of task material, with the log of every attempt made on it."""

    def __init__(
        self,
        id: str,
        kind: PartitionKind,
        root: Path | str,
        items: Mapping[str, str],
        access_log: AccessLog,
        *,
        public_root: Path | str | None = None,
    ) -> None:
        self.id = id
        self.kind = PartitionKind(kind)
        self.root = Path(root)
        self.public_root = None if public_root is None else Path(public_root)
        self._check_layout()
        self._items = dict(items)
        self.access_log = access_log
        self._retired: set[str] = {
            event.item_id for event in access_log.events if event.action == "retire"
        }

    def _check_layout(self) -> None:
        """The public root is the material an evaluator stages for the graded process.

        It is bound into the sandbox, and a bound root carries read, so the separation proof only
        measures anything if the items are outside it. That is enforced here rather than left to
        the caller: a partition whose two roots overlap cannot be built, so the proof's premise is
        a property of the object and not of the test that happens to be running.
        """
        if self.public_root is None:
            return
        root, public = self.root.resolve(), self.public_root.resolve()
        if root == public or public in root.parents or root in public.parents:
            raise _denied(
                f"the public root {public} and the items of partition {self.id} at {root} overlap, "
                "so staging the first would hand a candidate-side process the second",
                partition_id=self.id,
            )

    @classmethod
    def from_dir(
        cls,
        id: str,
        kind: PartitionKind,
        root: Path | str,
        *,
        log_path: Path | str,
        public_root: Path | str | None = None,
    ) -> Partition:
        """Build a partition from a directory, hashing every regular file under it.

        Retirements are replayed from the log, so a restart does not resurrect a disclosed item.
        """
        root = Path(root)
        items = {
            path.relative_to(root).as_posix(): file_digest(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        return cls(
            id,
            kind,
            root,
            items,
            AccessLog(log_path, partition_id=id),
            public_root=public_root,
        )

    # --- identity ---------------------------------------------------------------------------

    def manifest(self) -> tuple[tuple[str, str], ...]:
        """The live items, sorted by path, each with its own digest. Retired items are not in it."""
        return tuple(
            (name, item)
            for name, item in sorted(self._items.items())
            if name not in self._retired
        )

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "partition_id": self.id,
                "kind": self.kind.value,
                "manifest_version": "tree-v1",
                "items": [list(row) for row in self.manifest()],
                "retired": sorted(self._retired),
            }
        )

    @property
    def retired(self) -> frozenset[str]:
        return frozenset(self._retired)

    @property
    def is_empty(self) -> bool:
        return self.manifest() == ()

    def to_record(self) -> dict[str, object]:
        """What the assay carries about a partition: its identity and its use, never its body."""
        return {
            "partition_id": self.id,
            "kind": self.kind.value,
            "digest": self.digest,
            "items": len(self.manifest()),
            "retired": sorted(self._retired),
            "access_log": str(self.access_log.path),
            "accesses": len(self.access_log.events),
        }

    # --- access -----------------------------------------------------------------------------

    def read(self, item_id: str, *, capability: str, purpose: str) -> bytes:
        """Open one item for one capability, logging the attempt before the bytes are read."""
        if item_id not in self._items:
            reason = f"{item_id} is not an item of partition {self.id}"
            self._log(item_id, capability, purpose, False, reason)
            raise _denied(reason, partition_id=self.id, item_id=item_id)
        if item_id in self._retired:
            reason = f"{item_id} was retired from partition {self.id} after disclosure"
            self._log(item_id, capability, purpose, False, reason)
            raise _denied(reason, partition_id=self.id, item_id=item_id)
        if capability == SEEKER:
            reason = (
                f"the seeker may not read partition {self.id}, or any partition of any kind: a "
                "seeker spends a sealed round against the material and reads the outcome, and an "
                "attack developed against the bytes of an acceptance set is an attack on the "
                "record rather than on the reward system"
            )
            self._log(item_id, capability, purpose, False, reason)
            raise _denied(reason, partition_id=self.id, item_id=item_id, capability=capability)
        allowed = ALLOWED_CAPABILITIES[self.kind]
        if allowed is not None and capability not in allowed:
            reason = (
                f"the capability {capability} may not read partition {self.id}, which is a "
                f"{self.kind.value}; {' or '.join(sorted(allowed))} may"
            )
            self._log(item_id, capability, purpose, False, reason)
            raise _denied(reason, partition_id=self.id, item_id=item_id, capability=capability)

        self._log(item_id, capability, purpose, True, f"opened for {purpose}")
        return (self.root / item_id).read_bytes()

    def retire(self, item_id: str, *, disclosed_to: str = "", reason: str = "") -> str:
        """Retire a disclosed item and return the partition's new digest.

        `retire(item)` is the whole call. Who saw the item and why are worth recording and are
        recorded when given, but a caller who knows only that the item leaked must still be able
        to retire it: an item held back because the recipient could not be named is an item that
        goes on being scored after it stopped measuring anything.
        """
        by = disclosed_to or UNSTATED
        if item_id not in self._items:
            detail = f"{item_id} is not an item of partition {self.id}"
            self._log(item_id, by, "retirement refused", False, detail, action="refuse")
            raise _denied(detail, partition_id=self.id, item_id=item_id)
        if item_id in self._retired:
            detail = f"{item_id} is already retired from partition {self.id}"
            self._log(item_id, by, "retirement refused", False, detail, action="refuse")
            raise _denied(detail, partition_id=self.id, item_id=item_id)
        self._retired.add(item_id)
        self._log(
            item_id,
            by,
            "retired after disclosure",
            False,
            _disclosure(disclosed_to, reason),
            action="retire",
        )
        return self.digest

    def record_attempt(
        self, item_id: str, *, capability: str, purpose: str, granted: bool, reason: str
    ) -> None:
        """Log an access made outside `read`, such as the sandboxed separation attempt."""
        self._log(item_id, capability, purpose, granted, reason)

    def _log(
        self,
        item_id: str,
        capability: str,
        purpose: str,
        granted: bool,
        reason: str,
        action: str = "read",
    ) -> None:
        self.access_log.record(
            item_id=item_id,
            capability=capability,
            purpose=purpose,
            granted=granted,
            reason=reason,
            action=action,
        )
