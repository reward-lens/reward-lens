"""The value objects returned by the export boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ExportKind(str, Enum):
    HTML = "html"
    REVISION = "revision"
    BUNDLE = "bundle"
    BADGE = "badge"
    SARIF = "sarif"


class Tier(str, Enum):
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class Artifact:
    """A portable export before the caller chooses where to put its bytes."""

    kind: ExportKind
    tier: Tier
    files: dict[str, bytes]
    manifest: dict
    manifest_digest: str
    primary: str
    linked_from: str | None = None
    _written: Path | None = field(default=None, compare=False)

    def json(self) -> dict:
        """Decode the primary JSON export, refusing non-JSON artifacts plainly."""
        return json.loads(self.files[self.primary].decode("utf-8"))

    def write(self, destination: str | Path) -> Path:
        """Write the immutable artifact at a caller-selected location."""
        destination = Path(destination)
        if self.kind is ExportKind.BUNDLE:
            destination.mkdir(parents=True, exist_ok=True)
            for relative, payload in self.files.items():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            object.__setattr__(self, "_written", destination)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[self.primary])
        object.__setattr__(self, "_written", destination)
        return destination


@dataclass(frozen=True)
class DsseEnvelope:
    payload_type: str
    payload: str
    signatures: tuple[dict[str, str], ...]

    def to_dict(self) -> dict:
        return {
            "payloadType": self.payload_type,
            "payload": self.payload,
            "signatures": list(self.signatures),
        }


@dataclass(frozen=True)
class Verification:
    signer: str
    valid: bool
    statement: dict
    message: str
