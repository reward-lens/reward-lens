"""Protected partitions: five kinds, each a separate object with an id, a digest and an access log.

D-40. The agent that proposes a candidate never sees the acceptance set, and the record says how
that was enforced rather than asserting that it was. Owned by P-OUTCOME.
"""

from __future__ import annotations

from .access import AccessEvent, AccessLog, utc_now
from .kinds import (
    ACCEPTANCE_EVALUATOR,
    ALLOWED_CAPABILITIES,
    CANDIDATE_AUTHOR,
    CANDIDATE_PROCESS,
    PANEL,
    SEEKER,
    PartitionKind,
)
from .partition import Partition, file_digest

__all__ = [
    "ACCEPTANCE_EVALUATOR",
    "ALLOWED_CAPABILITIES",
    "CANDIDATE_AUTHOR",
    "CANDIDATE_PROCESS",
    "PANEL",
    "SEEKER",
    "AccessEvent",
    "AccessLog",
    "Partition",
    "PartitionKind",
    "file_digest",
    "utc_now",
]
