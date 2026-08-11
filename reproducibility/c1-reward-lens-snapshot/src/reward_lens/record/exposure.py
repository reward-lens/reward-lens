"""The exposure tier: what a number's author knew when they wrote it down, made enforceable.

Three tiers, assigned before a quantity is computed, stored beside it, and printed next to the
number wherever it appears. `e1_analysis/PLAN.md` section 7 states the contract and this module is
its referent:

    `PREDICTED` means a value was written down before any code existed to compute it, with a
    resolution rule and a refutation condition. `PRE_DECLARED` means the estimator, window, unit and
    refusal were declared before the quantity was computed but no value was predicted. `DESCRIPTIVE`
    means it was computed and then described, or its determinant had already been seen by somebody.

    A tier is assigned once and never raised. Exposure is transitive through determinants and not
    through names. No aggregate over tiers is itself preregistered.

Three of those sentences were unenforceable before this module existed, and the reason each one
needs code rather than prose is worth stating, because it is the same reason each time: a tier is a
claim about the past, and the past is exactly what a later edit can quietly change.

**Ordered, because "never raised" is a comparison.** The three tiers were a list. A list has no
direction, so "never raised" had nothing to evaluate: a script could see that a tier changed and not
that it changed in the forbidden direction. `ExposureTier` is an `IntEnum` and the ordering is the
plan's own, weakest first, so `DESCRIPTIVE < PRE_DECLARED < PREDICTED` is a fact a test can assert.

**First write wins, because a tier that can be rewritten records the last opinion rather than the
first commitment.** `assign` stores the tier against a quantity's id and any later disagreement
raises. An identical reassignment is accepted, because two call sites describing one quantity is
normal and refusing it would push callers into keeping their own copy, which is how a second source
of truth starts. The exception says whether the attempt was a raise or a lowering: a raise is the
move the plan forbids by name, and a lowering is a demotion that somebody has to authorise, so they
are different failures and a refusal that could not tell them apart would be the imprecise kind.

**Content-addressed, because a name is not an identity.** The key is a hash of what the quantity
*is*: its name together with the estimator, window, unit and population that fix what is being
measured. Two descriptions of one quantity land on one key and inherit one tier; renaming a quantity
produces a new key that starts untiered rather than inheriting the old one's standing, which is the
half of "not through names" that a name-keyed store cannot express. Changing the estimator changes
the id too, because a different estimator is a different quantity whatever it is called.

**Transitive through determinants, because a quantity computed from something already seen has been
seen.** `assign` takes the determinants a quantity was computed from and caps its tier at the
weakest of them. Without the cap the field is decoration: the failure this row exists to stop is a
quantity computed after the data existed being written `PREDICTED` while every presence check
passes, and a determinant is how the data gets in. A determinant with no tier of its own raises
rather than being treated as unconstraining, because an unmeasured determinant is not a permissive
one.

What this module does not do: assign anybody's tier. Which tier a registered row carries is the
design's business, and a library that guessed would be inventing the provenance it exists to record.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from reward_lens.core.types import content_hash

__all__ = [
    "ExposureLedger",
    "ExposureTier",
    "MissingExposureTier",
    "TierConflict",
    "TieredValue",
    "quantity_id",
]


class ExposureTier(enum.IntEnum):
    """How much of a quantity's answer was already visible when its tier was assigned.

    An `IntEnum` so the ordering is the type's, not a convention a caller has to remember. Ordered
    weakest to strongest exposure discipline, so `DESCRIPTIVE < PRE_DECLARED < PREDICTED` and
    "assigned once and never raised" is `new <= stored`.
    """

    DESCRIPTIVE = 0
    PRE_DECLARED = 1
    PREDICTED = 2

    @property
    def meaning(self) -> str:
        """The plan's own definition, so the field reads without the plan open beside it."""
        return _MEANING[self]


_MEANING: dict[ExposureTier, str] = {
    ExposureTier.PREDICTED: (
        "A value was written down before any code existed to compute it, with a resolution rule "
        "and a refutation condition."
    ),
    ExposureTier.PRE_DECLARED: (
        "The estimator, window, unit and refusal were declared before the quantity was computed, "
        "but no value was predicted."
    ),
    ExposureTier.DESCRIPTIVE: (
        "It was computed and then described, or its determinant had already been seen by somebody."
    ),
}


class MissingExposureTier(Exception):
    """A quantity was used where a tier is required and none was ever assigned.

    An exception rather than a `Refusal`: a `Refusal` is a reading, and this is not a hard case an
    instrument anticipated on the data. It is a contract that was not honoured before any data was
    touched, and the fix is in the code that should have assigned the tier.
    """


class TierConflict(Exception):
    """A second assignment disagreed with the first, or a quantity outran its determinants."""


def quantity_id(
    *,
    name: str,
    estimator: str,
    window: str = "",
    unit: str = "",
    population: str = "",
    **extra: Any,
) -> str:
    """The content-addressed id a tier is stored against.

    Hashes what fixes the quantity rather than what it is called: the name together with the
    estimator, window, unit and population. Two call sites describing one quantity produce one id in
    any key order; a rename, a different estimator or a different population produces a different
    one and therefore a quantity with no tier yet. ``extra`` is for a field a caller needs in the
    identity that this signature does not name, and it is hashed with the rest.
    """
    material = {
        "name": name,
        "estimator": estimator,
        "window": window,
        "unit": unit,
        "population": population,
        **{k: extra[k] for k in sorted(extra)},
    }
    return content_hash(material, "qty")


@dataclass(frozen=True)
class TieredValue:
    """A number and the tier it was written under, which travel together or not at all."""

    quantity: str
    tier: ExposureTier
    value: Any

    def __canonical__(self) -> dict[str, Any]:
        return {"quantity": self.quantity, "tier": self.tier.name, "value": self.value}


class ExposureLedger:
    """First-write-wins tiers, keyed by content-addressed quantity id.

    The whole object is one dict and three rules. It is deliberately not an `Evidence` subclass and
    deliberately not a global: a run has one ledger, an analysis of two runs has two, and a module
    level singleton would let one run's tiers constrain another's.
    """

    def __init__(self, tiers: Mapping[str, ExposureTier] | None = None) -> None:
        self._tiers: dict[str, ExposureTier] = dict(tiers or {})

    # -- reading ------------------------------------------------------------

    def __contains__(self, quantity: str) -> bool:
        return quantity in self._tiers

    def __len__(self) -> int:
        return len(self._tiers)

    def tier_of(self, quantity: str) -> ExposureTier | None:
        """The assigned tier, or ``None``. Use `require` where the absence is an error."""
        return self._tiers.get(quantity)

    def require(self, quantity: str) -> ExposureTier:
        """The assigned tier, raising `MissingExposureTier` when there is none.

        This is the refusal the contract turns on. A quantity with no tier is not a quantity with a
        default tier; there is no safe default, because every default is either a claim nobody made
        or a demotion nobody authorised.
        """
        tier = self._tiers.get(quantity)
        if tier is None:
            raise MissingExposureTier(
                f"quantity {quantity} has no exposure tier. A tier is assigned before the quantity "
                f"is computed, so this is a contract that was not honoured rather than something "
                f"to recover from here: assign one with `ExposureLedger.assign` at the point the "
                f"quantity is declared. There is no default, because every default is either a "
                f"claim nobody made or a demotion nobody authorised."
            )
        return tier

    # -- writing ------------------------------------------------------------

    def assign(
        self,
        quantity: str,
        tier: ExposureTier,
        *,
        determinants: Iterable[str] = (),
    ) -> ExposureTier:
        """Assign a tier, first write wins, capped by the weakest determinant.

        Returns the stored tier. Raises `TierConflict` when a tier is already stored and differs,
        naming the direction, and when ``tier`` is stronger than a determinant's. Raises
        `MissingExposureTier` when a determinant carries no tier of its own, because a determinant
        nobody has tiered constrains nothing only if you assume it, and Law 4's "unavailable is not
        pass" is exactly the assumption this field exists to refuse.
        """
        tier = ExposureTier(tier)
        for determinant in determinants:
            cap = self.require(determinant)
            if tier > cap:
                raise TierConflict(
                    f"quantity {quantity} cannot be {tier.name} because its determinant "
                    f"{determinant} is {cap.name}. Exposure is transitive through determinants: a "
                    f"quantity computed from something already seen has been seen. The tier "
                    f"available here is {cap.name} or weaker."
                )
        stored = self._tiers.get(quantity)
        if stored is None:
            self._tiers[quantity] = tier
            return tier
        if stored is tier:
            return stored
        direction = "a raise" if tier > stored else "a lowering"
        forbidden = (
            "and a tier is assigned once and never raised"
            if tier > stored
            else "and a demotion is a decision somebody has to take on the record"
        )
        raise TierConflict(
            f"quantity {quantity} is already {stored.name} and this write is {direction} to "
            f"{tier.name}, {forbidden}. The stored tier stands."
        )

    def write(self, quantity: str, value: Any) -> TieredValue:
        """Attach a value to a tiered quantity, refusing when the tier was never assigned.

        The writer half of the contract. A number that reaches a record without a tier looks exactly
        like one that was declared in advance, which is the confusion the tier exists to end, so the
        write is where the absence has to stop being survivable.
        """
        return TieredValue(quantity=quantity, tier=self.require(quantity), value=value)

    # -- persistence --------------------------------------------------------

    def __canonical__(self) -> dict[str, str]:
        """Tier names by quantity id, sorted, so two equal ledgers serialize identically."""
        return {qid: self._tiers[qid].name for qid in sorted(self._tiers)}

    @classmethod
    def from_canonical(cls, obj: Mapping[str, str]) -> "ExposureLedger":
        """Rebuild from `__canonical__`. The result is as immutable as the original."""
        return cls({qid: ExposureTier[name] for qid, name in obj.items()})
