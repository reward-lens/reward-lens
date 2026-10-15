"""The one fix carried out of the release diff into `measure/`.

The refusal `BaseObservable` writes when an instrument asks for a capability the context cannot
supply names three capability sets, and it named them with `caps.name or str(int(caps))`.
`Capability` is a `Flag` and not an `IntFlag`, so it has no `__int__` and the right-hand side of
that `or` raises `TypeError` on every input it is ever handed. It survived because on 3.11 and up
a composite `Flag` always has a truthy `name`, so the branch is unreachable on the floor 4.0
requires: the release's own 3.10 leg is where it failed.

The release replaced it with `capability_name`, which joins the declared members. These tests pin
what makes that replacement safe rather than merely different: it agrees with `name` on every
value on this interpreter, so the strings recorded in refusals do not change, and the expression
it removes really does raise.
"""

from __future__ import annotations

import itertools

import pytest

from reward_lens.core.types import Capability
from reward_lens.measure.base import capability_name


def _members() -> list[Capability]:
    return [m for m in Capability if m.value]


def test_capability_name_agrees_with_flag_name_on_every_value() -> None:
    """No refusal changes its recorded string by taking the helper."""
    members = _members()
    assert members, "Capability declares no non-zero members"
    for r in range(0, min(3, len(members)) + 1):
        for combo in itertools.combinations(members, r):
            value = Capability(0)
            for m in combo:
                value |= m
            assert capability_name(value) == (value.name or "NONE"), value


def test_capability_name_names_the_empty_set() -> None:
    assert capability_name(Capability(0)) == "NONE"


def test_capability_name_names_a_composite_in_declaration_order() -> None:
    declared = _members()
    a, b = declared[0], declared[-1]
    assert capability_name(a | b) == f"{a.name}|{b.name}"


def test_the_fallback_the_helper_replaced_raises_on_every_input() -> None:
    """`str(int(caps))`, the expression the release removed, is not a fallback at all."""
    for value in (Capability(0), _members()[0], _members()[0] | _members()[-1]):
        with pytest.raises(TypeError):
            int(value)


def test_the_refusal_statistics_name_the_capabilities() -> None:
    """The call site, not just the helper: an access refusal carries readable names."""
    from reward_lens.measure import base as base_mod

    source = base_mod.__file__
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert '"needed": capability_name(needed),' in text
    assert "str(int(needed))" not in text
