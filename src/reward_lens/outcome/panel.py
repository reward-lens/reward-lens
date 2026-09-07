"""A panel registers its membership and its estimand before it scores anything.

The order is the whole point. A panel that scores first and registers afterwards has chosen its
membership and its question with the scores in hand, which is the post hoc rule D-39 forbids, and
no amount of care in the scoring makes that reversible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from reward_lens.contracts.canonical import digest as canonical_digest
from reward_lens.partitions.access import utc_now

from .errors import PanelNotRegistered, usage_refusal

__all__ = ["Panel", "PanelNotRegistered", "PanelScore", "Registration"]


@dataclass(frozen=True)
class Registration:
    """The membership and the estimand, frozen together under one digest."""

    panel_id: str
    membership: tuple[str, ...]
    estimand: str
    registered_at: str
    digest: str


@dataclass(frozen=True)
class PanelScore:
    """One label from one member, carrying the registration it was made under."""

    panel_id: str
    item_id: str
    member: str
    label: float
    scored_at: str
    registration_digest: str


class Panel:
    """A panel evaluator. `register` first, then `score`; the reverse refuses."""

    def __init__(self, panel_id: str) -> None:
        self.panel_id = panel_id
        self.registration: Registration | None = None
        self._scores: list[PanelScore] = []

    def register(self, membership: Sequence[str], estimand: str) -> Registration:
        """Freeze the membership and the estimand. Refuses once any score exists."""
        if self._scores:
            raise usage_refusal(
                f"panel {self.panel_id} has already scored {len(self._scores)} items, so its "
                "membership and estimand cannot be registered now",
                panel_id=self.panel_id,
            )
        members = tuple(membership)
        registration = Registration(
            panel_id=self.panel_id,
            membership=members,
            estimand=estimand,
            registered_at=utc_now(),
            digest=canonical_digest(
                {"panel_id": self.panel_id, "membership": list(members), "estimand": estimand}
            ),
        )
        self.registration = registration
        return registration

    def score(self, *, item_id: str, member: str, label: float) -> PanelScore:
        """Record one label. Refuses until the panel has registered."""
        if self.registration is None:
            raise PanelNotRegistered(
                panel_id=self.panel_id,
                detail=(
                    f"panel {self.panel_id} has not registered its membership and estimand, and a "
                    "panel that scores first would be choosing its estimand with the scores in hand"
                ),
            )
        if member not in self.registration.membership:
            raise usage_refusal(
                f"{member} is not a member of the registered panel {self.panel_id}",
                panel_id=self.panel_id,
                member=member,
            )
        score = PanelScore(
            panel_id=self.panel_id,
            item_id=item_id,
            member=member,
            label=label,
            scored_at=utc_now(),
            registration_digest=self.registration.digest,
        )
        self._scores.append(score)
        return score

    def scores(self) -> tuple[PanelScore, ...]:
        return tuple(self._scores)
