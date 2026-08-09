"""The immutable version of a reward system: an identity, its parents, and the nine digests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reward_lens.contracts import ProjectConfig, digest
from reward_lens.contracts.models import Digests as SubjectDigests
from reward_lens.contracts.models import VersionRef

from .digests import Digests, compute

__all__ = ["RewardSystemVersion"]


@dataclass(frozen=True)
class RewardSystemVersion:
    """One version of one reward system, as the project on disk currently declares it.

    The digest is over the nine and the parents, never over the id, because an undeclared id is
    derived from the digest and a version must not be able to change its own identity by naming it.
    """

    id: str
    parents: tuple[str, ...]
    declared_change: tuple[str, ...] | None
    digests: Digests

    @classmethod
    def from_config(
        cls, config: ProjectConfig, root: Path, *, methods: tuple[str, ...] = ()
    ) -> "RewardSystemVersion":
        digests = compute(config, root, methods=methods)
        parents = tuple(config.version.parents) if config.version is not None else ()
        computed = _digest_of(digests, parents)
        declared_id = config.version.id if config.version is not None else None
        return cls(
            id=declared_id or "v-" + computed.removeprefix("sha256:")[:12],
            parents=parents,
            declared_change=None,
            digests=digests,
        )

    def digest(self) -> str:
        return _digest_of(self.digests, self.parents)

    def to_subject_digests(self) -> SubjectDigests:
        """The nine as the record's `subject.digests`."""
        return SubjectDigests.model_validate(self.digests.to_dict())

    def to_version_ref(self) -> VersionRef:
        """The version as the record's `subject.version`."""
        return VersionRef.model_validate(
            {
                "id": self.id,
                "digest": self.digest(),
                "parents": list(self.parents),
                "declared_change": list(self.declared_change) if self.declared_change else None,
            }
        )


def _digest_of(digests: Digests, parents: tuple[str, ...]) -> str:
    return digest({"digests": digests.to_dict(), "parents": list(parents)})
