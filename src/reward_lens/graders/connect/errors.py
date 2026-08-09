"""The connector's two refusals: RL0710 for a shape 4.0 will not guess at, RL0003 for a bad override.

RL0003 is already in the catalogue, so it is raised through `errors.make` and reads the same as
every other invalid project-file field. RL0710 is allocated by interfaces section 1 and is not in
the catalogue yet; it is defined here as a subclass of `CapabilityUnavailable`, which is exactly
the pattern `contracts/errors.py` describes for a packet's own codes, and the catalogue row is
proposed in the handoff.

Owned by P-CONNECT.
"""

from __future__ import annotations

from typing import Any

from reward_lens.contracts import CapabilityUnavailable, RewardLensError

__all__ = ["SHAPE_UNSUPPORTED_PAGE", "ShapeUnsupported", "bad_override"]

#: Where the reason lives once the catalogue carries RL0710. The message names it because D-65
#: asks for the page that says why a shape is refused, not only that it was.
SHAPE_UNSUPPORTED_PAGE = "docs/errors/RL0710.md"


class ShapeUnsupported(CapabilityUnavailable):
    """RL0710, SHAPE_UNSUPPORTED, exit 5. A shape reward-lens recognises and will not adapt.

    D-65's reason, in one line: an adapter that half-works is worse than a refusal. The message
    names the shape it found so the reader can tell which of their files was the problem, and the
    docs page so they can read why before asking for it.
    """

    def __init__(self, *, shape: str, entry: str, detail: str) -> None:
        super().__init__(
            code="RL0710",
            message=(
                f"{entry} has the {shape} shape, which reward-lens 3.1.0 does not adapt: {detail}. "
                f"Why, and what is supported instead: {SHAPE_UNSUPPORTED_PAGE}"
            ),
            remediation=(
                "wrap it in a function that takes a prompt and a completion and returns a float, "
                "then point reward.entry at that: run: reward-lens init . --detect"
            ),
            context={"shape": shape, "entry": entry, "docs_page": SHAPE_UNSUPPORTED_PAGE},
        )
        self.shape = shape
        self.entry = entry
        self.docs_page = SHAPE_UNSUPPORTED_PAGE


def bad_override(field: str, detail: str, **context: Any) -> RewardLensError:
    """RL0003 at exit 4, naming the project-file field the override sits in."""
    from reward_lens import errors

    return errors.make("RL0003", field=field, detail=detail, **context)
