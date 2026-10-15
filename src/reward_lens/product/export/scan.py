"""Secret detection at the export boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass

from reward_lens.contracts import UsageError


@dataclass(frozen=True)
class SecretHit:
    file: str
    line: int
    pattern_id: str


class ExportSecretFound(UsageError):
    """RL0641: payload bytes contain a credential-shaped string."""

    def __init__(self, *, hits: list[SecretHit], redacted_path: str) -> None:
        first = hits[0]
        super().__init__(
            code="RL0641",
            message=(
                f"export found {first.pattern_id} in {first.file}:{first.line}; no artifact was written"
            ),
            remediation=f"redact the value and write a new linked artifact at {redacted_path}",
            context={"file": first.file, "line": first.line, "redacted_path": redacted_path},
        )
        self.hits = tuple(hits)


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai-api-key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("private-key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
)


def scan(payload: bytes, *, file: str = "record.json") -> list[SecretHit]:
    """Return credential-shaped values with the file and physical line that contain them."""
    text = payload.decode("utf-8", errors="replace")
    hits: list[SecretHit] = []
    for line_number, line in enumerate(text.splitlines() or [text], start=1):
        for pattern_id, pattern in _PATTERNS:
            if pattern.search(line):
                hits.append(SecretHit(file=file, line=line_number, pattern_id=pattern_id))
    return hits
