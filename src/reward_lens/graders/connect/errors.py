"""Connector error compatibility names and the catalogue-backed override refusal.

RL0710 and the silent-mode rows now live in the central catalogue.  ``ShapeUnsupported`` stays
as the public exception-family name for callers that caught it, while connector code obtains the
actual RL0710 instance through ``errors.make`` like the rest of the product.

Owned by P-CONNECT.
"""

from __future__ import annotations

from typing import Any

from reward_lens.contracts import CapabilityUnavailable, RewardLensError

__all__ = ["SHAPE_UNSUPPORTED_PAGE", "ShapeUnsupported", "bad_override"]

#: Where the reason lives once the catalogue carries RL0710. The message names it because D-65
#: asks for the page that says why a shape is refused, not only that it was.
SHAPE_UNSUPPORTED_PAGE = "docs/errors/RL0710.md"


# Kept as a public catchable family.  ``errors.make("RL0710", ...)`` returns this class because
# the catalogue gives RL0710 exit 5.
ShapeUnsupported = CapabilityUnavailable


def bad_override(field: str, detail: str, **context: Any) -> RewardLensError:
    """RL0003 at exit 4, naming the project-file field the override sits in."""
    from reward_lens import errors

    return errors.make("RL0003", field=field, detail=detail, **context)
