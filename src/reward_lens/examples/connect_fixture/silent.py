"""Three callables built to trigger D-65's three silent failure modes, one each.

The point of these is that the witness in the finding is executed rather than described: the
connector probes the callable once, and what it gets back is what the finding carries.

Owned by P-CONNECT.
"""

__all__ = ["raises_inside", "returns_bool", "returns_none"]


def returns_none(prompts, completions, **kwargs) -> list:
    """TRL turns the `None` into NaN and drops this function for that row."""
    return [1.0 if index == 0 else None for index, _ in enumerate(completions)]


def raises_inside(prompt, completion, **kw) -> float:
    """`verifiers` catches this, logs it, and records 0.0 as though it were a score."""
    return 1.0 / 0


def returns_bool(prompt, completion, **kw) -> bool:
    """`float(True)` is 1.0, so this reads as a legitimate score and hides every level between."""
    return True
