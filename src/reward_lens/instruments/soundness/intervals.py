"""Rules 1 to 4 of section 7.11, as arithmetic.

Every count this panel reports is a rate, and a rate with no interval is a number a reader cannot
use. The four rules fix which interval: Wilson at n <= 40, Clopper-Pearson where the rate is load
bearing, never Wald; no Jeffreys at 0 of n or n of n, where it undercovers; the exact bound for
zero events, with the rule of three quoted only as the approximation it is and only from n = 30.

Clopper-Pearson needs the Beta quantile and this tree carries no SciPy in the product path, so the
regularised incomplete Beta is computed here by the Lentz continued fraction and inverted by
bisection. Bisection rather than Newton because the objective is monotone and bounded on [0, 1],
so fifty halvings put the answer inside 1e-15 with no derivative and no failure mode.

`what_n_can_establish` is rule 3 in words, and it is not decoration: a run of twelve for twelve is
the shape that most often gets read as proof, and the sentence says what that n could not have
shown. Its two anchors are the commission's own, so the tests check the arithmetic against them.
"""

from __future__ import annotations

import math

from reward_lens import contracts

__all__ = [
    "LEVEL",
    "RULE_OF_THREE_MIN_N",
    "Z95",
    "clopper_pearson",
    "exact_one_sided_lower",
    "exact_zero_event_upper",
    "interval_for",
    "method_for",
    "n_for_lower_bound",
    "rule_of_three",
    "uncertainty_for",
    "what_n_can_establish",
    "wilson",
]

#: The two-sided level every interval here is quoted at.
LEVEL = 0.95

#: The standard normal quantile at 0.975, the one the Wilson score interval uses at 95%.
Z95 = 1.959964

#: Rule 1's boundary: Wilson is what a count rate gets at or below this n.
WILSON_MAX_N = 40

#: Rule 4: below this n the rule of three is not an approximation of anything, so it is not quoted.
RULE_OF_THREE_MIN_N = 30

_PLACES = 6


def _z(level: float) -> float:
    """The two-sided normal quantile, by bisection on the error function."""
    if abs(level - LEVEL) < 1e-12:
        return Z95
    target = 1.0 - (1.0 - level) / 2.0
    low, high = 0.0, 40.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0))) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def wilson(k: int, n: int, *, level: float = LEVEL) -> tuple[float, float]:
    """The Wilson score interval, which is defined at k = 0 and at k = n (D-17, rule 1)."""
    if n <= 0:
        return 0.0, 1.0
    z = _z(level)
    proportion = k / n
    denominator = n + z * z
    centre = (k + 0.5 * z * z) / denominator
    half = (z * math.sqrt(n) / denominator) * math.sqrt(
        proportion * (1.0 - proportion) + z * z / (4.0 * n)
    )
    return (
        max(0.0, round(centre - half, _PLACES)),
        min(1.0, round(centre + half, _PLACES)),
    )


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction for the incomplete Beta, by the modified Lentz method."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """The regularised incomplete Beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + b * math.log1p(-x)
        + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def _beta_quantile(p: float, a: float, b: float) -> float:
    """The Beta(a, b) quantile at p, by bisection on a monotone bounded objective."""
    if a <= 0.0:
        return 0.0
    if b <= 0.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if _betainc(a, b, middle) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson(k: int, n: int, *, level: float = LEVEL) -> tuple[float, float]:
    """The exact binomial interval, for a rate the record leans on (rule 1).

    At k = 0 the lower endpoint is 0 and at k = n the upper endpoint is 1, by construction rather
    than by a clamp: the Beta parameter that would be zero is the one whose tail is empty.
    """
    if n <= 0:
        return 0.0, 1.0
    alpha = 1.0 - level
    low = 0.0 if k == 0 else _beta_quantile(alpha / 2.0, k, n - k + 1)
    high = 1.0 if k == n else _beta_quantile(1.0 - alpha / 2.0, k + 1, n - k)
    return max(0.0, round(low, _PLACES)), min(1.0, round(high, _PLACES))


def exact_zero_event_upper(n: int, *, level: float = LEVEL) -> float:
    """Rule 4: the exact one-sided upper bound after n trials with no event, 1 - alpha^(1/n)."""
    if n <= 0:
        return 1.0
    return 1.0 - exact_one_sided_lower(n, level=level)


def exact_one_sided_lower(n: int, *, level: float = LEVEL) -> float:
    """The exact one-sided lower bound after n successes out of n, alpha^(1/n) (rule 3).

    This, and not the two-sided Clopper-Pearson lower endpoint, is the bound rule 3's n >= 299 is
    derived from: a one-sided claim spends the whole alpha on the side it makes a claim about.
    """
    if n <= 0:
        return 0.0
    return (1.0 - level) ** (1.0 / n)


def rule_of_three(n: int) -> float | None:
    """3/n, the approximation to the zero-event bound, and `None` below the n it holds at.

    Rule 4 says the record states that this is an approximation and that it wants n >= 30. Below
    that it is not quoted at all, because quoting it with a caveat still puts the number in front
    of a reader who will use it.
    """
    if n < RULE_OF_THREE_MIN_N:
        return None
    return round(3.0 / n, _PLACES)


def n_for_lower_bound(target: float, *, level: float = LEVEL) -> int:
    """How many consecutive successes a one-sided lower bound of `target` needs (rule 3).

    The exact one-sided lower bound after n of n is alpha^(1/n), so the n that reaches `target` is
    log(alpha)/log(target), rounded up. At 0.99 and 95% that is 299, which is the commission's own
    number and the test's anchor.
    """
    if not 0.0 < target < 1.0:
        raise ValueError("a lower bound to reach is strictly between 0 and 1")
    alpha = 1.0 - level
    return math.ceil(math.log(alpha) / math.log(target))


def method_for(k: int, n: int, *, load_bearing: bool = False) -> str:
    """Which interval rule 1 and rule 4 name for this count.

    Zero events first, because rule 4 is specific and names the exact bound. Then the load-bearing
    rate, which gets the exact binomial interval whatever its n. Everything else is Wilson, which
    rule 1 names at n <= 40 and which stays the honest default above it. Jeffreys is never
    returned, at the boundary or anywhere else (rule 2), and neither is Wald.
    """
    if n > 0 and k == 0:
        return "exact_zero_event"
    if load_bearing:
        return "clopper_pearson"
    return "wilson"


def interval_for(k: int, n: int, *, load_bearing: bool = False, level: float = LEVEL) -> list[float]:
    """The endpoints the method `method_for` names produces."""
    method = method_for(k, n, load_bearing=load_bearing)
    if method == "exact_zero_event":
        return [0.0, round(exact_zero_event_upper(n, level=level), _PLACES)]
    if method == "clopper_pearson":
        low, high = clopper_pearson(k, n, level=level)
        return [low, high]
    low, high = wilson(k, n, level=level)
    return [low, high]


def uncertainty_for(
    k: int, n: int, *, load_bearing: bool = False, level: float = LEVEL
) -> contracts.Uncertainty:
    """The record's uncertainty block for a count rate, with its method named (D-17)."""
    return contracts.Uncertainty(
        interval=interval_for(k, n, load_bearing=load_bearing, level=level),
        method=method_for(k, n, load_bearing=load_bearing),  # type: ignore[arg-type]
        level=level,
    )


def what_n_can_establish(k: int, n: int, *, level: float = LEVEL) -> str:
    """Rule 3 in words: what this run could not have shown, whatever it did show.

    Three shapes, because three things go wrong. A perfect run reads as proof, so it is told what
    its lower bound actually is and how many more trials a lower bound of 0.99 would take. A run
    with no events reads as safety, so it is given the exact upper bound, with the rule of three
    beside it as an approximation and only from n = 30. Everything else is told its interval and
    that the interval, not the point, is the claim.
    """
    percent = int(round(level * 100))
    if n <= 0:
        return (
            "no trials were run, so this rate establishes nothing: an empty denominator is not a "
            "rate of zero"
        )
    needed = n_for_lower_bound(0.99, level=level)
    if k == n:
        low = wilson(k, n, level=level)[0]
        exact_low = exact_one_sided_lower(n, level=level)
        return (
            f"{k} of {n} passed, and a perfect run is not a rate of 1: the {percent}% Wilson lower "
            f"bound is {low:.2f} and the exact one-sided lower bound is {exact_low:.2f}. A lower "
            f"bound of 0.99 needs n >= {needed}, so this n cannot establish one."
        )
    if k == 0:
        exact = exact_zero_event_upper(n, level=level)
        approximation = rule_of_three(n)
        tail = (
            f" The rule of three gives {approximation:.4f}; it is an approximation, valid from "
            f"n >= {RULE_OF_THREE_MIN_N}, and the exact bound above is what this record reports."
            if approximation is not None
            else (
                f" The rule of three is not quoted here: it is an approximation valid only from "
                f"n >= {RULE_OF_THREE_MIN_N}, and this run has {n}."
            )
        )
        return (
            f"0 of {n} events were seen, which is not a rate of zero: the exact one-sided "
            f"{percent}% upper bound is {exact:.4f}.{tail}"
        )
    low, high = interval_for(k, n, level=level)
    return (
        f"{k} of {n} is {k / n:.3f}, and the claim is the interval and not the point: the "
        f"{percent}% interval runs {low:.3f} to {high:.3f}. A lower bound of 0.99 needs n >= "
        f"{needed} with no failures at all, which this run is not."
    )
