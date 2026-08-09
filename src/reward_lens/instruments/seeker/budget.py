"""The meter, and the refusal that keeps the paid route behind it (D-41, RL0502).

Two things live here. `Budget` is the money and call counter the API actor spends against, kept in
`Decimal` so that a hundred sub-cent calls add up to what they actually cost rather than to a
hundred roundings of zero. `SeekerRequiresBudget` is the refusal: the API actor cannot be built
without a cap, so there is no order of calls in which a paid request is assembled first and the cap
checked afterwards.

The two verbs are deliberately different. `charge` refuses a spend that would cross the cap and is
what a caller uses before committing to something it can still decline. `settle` records a spend
that has already happened, because a provider bills for the call it answered whether or not the
estimate was right, and a meter that quietly dropped that spend would under-report the run.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from reward_lens.contracts.errors import BudgetExceeded

__all__ = ["Budget", "BudgetExhausted", "Pricing", "SeekerRequiresBudget", "money"]

#: What a `Money` field in the record looks like: two places, always.
_CENTS = Decimal("0.01")


def money(value: Any) -> str:
    """`value` as the record's `Money` string: two decimal places, rounded half up."""
    return str(Decimal(str(value)).quantize(_CENTS, rounding=ROUND_HALF_UP))


class SeekerRequiresBudget(BudgetExceeded):
    """RL0502. The API actor was asked to run with no cap on what it may spend.

    The band and the exit status are wave 1's budget family (exit 6). The row for the catalogue is
    proposed in this packet's handoff; until P-ERRORS mints it, the code is raised here in the
    pattern `contracts/errors.py` documents, which is a subclass carrying its own code.
    """

    def __init__(self, *, actor: str = "api", detail: str = "") -> None:
        super().__init__(
            code="RL0502",
            message=(
                f"the {actor} actor of the seeker spends money, and this run set no cap on what "
                "it may spend" + (f": {detail}" if detail else "")
            ),
            remediation=(
                "name the cap the arm may spend and run it again: run: reward-lens audit . "
                f"--seeker {actor} --max-budget-usd 0.50; to run the audit with no paid arm at "
                "all, leave the seeker off, which is the default"
            ),
            context={"actor": actor, "field": "max_budget_usd"},
        )


class BudgetExhausted(Exception):
    """The cap was reached. Not a refusal of the run: the arm stops and the record says so.

    This is the seeker's honest negative arriving by the budget route rather than by the search
    running out of proposals, so it is an ordinary exception the search loop catches and turns into
    `stopped: budget_exhausted`, never a process exit.
    """

    def __init__(self, *, cap: str, spent: str, asked: str) -> None:
        super().__init__(
            f"the cap of {cap} is reached: {spent} is spent and the next call asks for {asked}"
        )
        self.cap = cap
        self.spent = spent
        self.asked = asked


class Budget:
    """What the arm may spend, what it has spent, and how many calls it has made."""

    def __init__(self, cap_usd: str | float | None = None, *, max_calls: int | None = None) -> None:
        self._cap: Decimal | None = None if cap_usd is None else Decimal(str(cap_usd))
        self._max_calls = max_calls
        self._spent = Decimal("0")
        self._calls = 0

    # --- reading ----------------------------------------------------------------------------

    @property
    def cap(self) -> str | None:
        """The cap as a `Money` string, or `None` for an arm that costs nothing."""
        return None if self._cap is None else money(self._cap)

    @property
    def spent(self) -> str:
        return money(self._spent)

    @property
    def exact_spent(self) -> Decimal:
        """The unrounded spend, which is what the next cap check is made against."""
        return self._spent

    @property
    def remaining(self) -> str | None:
        return None if self._cap is None else money(max(Decimal("0"), self._cap - self._spent))

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def calls_remaining(self) -> int | None:
        return None if self._max_calls is None else max(0, self._max_calls - self._calls)

    def would_exceed(self, usd: Any) -> bool:
        """Whether spending `usd` next would cross the cap, or the call count would."""
        if self._max_calls is not None and self._calls + 1 > self._max_calls:
            return True
        if self._cap is None:
            return False
        return self._spent + Decimal(str(usd)) > self._cap

    # --- writing ----------------------------------------------------------------------------

    def charge(self, usd: Any, *, calls: int = 1) -> None:
        """Spend `usd`, or raise `BudgetExhausted` and spend nothing."""
        if self.would_exceed(usd):
            raise BudgetExhausted(
                cap=self.cap or "unbounded", spent=self.spent, asked=money(usd)
            )
        self.settle(usd, calls=calls)

    def settle(self, usd: Any, *, calls: int = 1) -> None:
        """Record a spend that has already happened, cap or no cap."""
        self._spent += Decimal(str(usd))
        self._calls += calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "cap_usd": self.cap,
            "spent_usd": self.spent,
            "calls": self._calls,
            "max_calls": self._max_calls,
        }


class Pricing:
    """What a provider charges, per thousand tokens in and per thousand tokens out.

    `estimate_usd` is what the meter checks before a request is sent, and it prices the output at
    the ceiling the request allows rather than at what the reply turns out to be, because the only
    honest pre-estimate of an unwritten reply is the most it could cost.
    """

    def __init__(
        self,
        usd_per_1k_input: str | float,
        usd_per_1k_output: str | float,
        *,
        max_output_tokens: int = 600,
    ) -> None:
        self.usd_per_1k_input = Decimal(str(usd_per_1k_input))
        self.usd_per_1k_output = Decimal(str(usd_per_1k_output))
        self.max_output_tokens = int(max_output_tokens)

    def cost_usd(self, input_tokens: int, output_tokens: int) -> Decimal:
        thousand = Decimal(1000)
        return (
            Decimal(int(input_tokens)) / thousand * self.usd_per_1k_input
            + Decimal(int(output_tokens)) / thousand * self.usd_per_1k_output
        )

    def estimate_usd(self, input_tokens: int) -> Decimal:
        return self.cost_usd(input_tokens, self.max_output_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "usd_per_1k_input": str(self.usd_per_1k_input),
            "usd_per_1k_output": str(self.usd_per_1k_output),
            "max_output_tokens": self.max_output_tokens,
        }
