"""Typed records for traceable reference materials (D-49)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


def _sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hexadecimal digest")


class MaterialGrade(str, Enum):
    """The three reference-material grades frozen for the public seam."""

    INFORMATIONAL = "informational"
    REPORTED = "reported"
    CERTIFIED = "certified"


@dataclass(frozen=True)
class SourceEvidence:
    """A named, versioned source from which material was acquired."""

    location: str
    commit: str | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.location, str) or not self.location.strip():
            raise ValueError("source location must be non-empty")


@dataclass(frozen=True)
class LicenceEvidence:
    """Licence identity and the digest of the licence text that was read."""

    spdx: str
    text_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spdx, str) or not self.spdx.strip():
            raise ValueError("licence must be non-empty")
        if self.text_digest is not None:
            _sha256(self.text_digest, field="licence text digest")


@dataclass(frozen=True)
class CharacterisedValue:
    """One actual value and its expanded uncertainty from a characterisation record."""

    measurand: str
    value: str
    expanded_uncertainty: str
    definition: str

    def __post_init__(self) -> None:
        for field, value in (
            ("measurand", self.measurand),
            ("value", self.value),
            ("expanded uncertainty", self.expanded_uncertainty),
            ("definition", self.definition),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"characterised value {field} must be non-empty")


@dataclass(frozen=True)
class CharacterisationEvidence:
    """A digest-verified project record that characterises one material's uncertainty."""

    material_identifier: str
    material_digest: str
    project_record_path: str
    project_record_digest: str
    values: tuple[CharacterisedValue, ...]
    conditions: tuple[str, ...]
    uncertainty_method: str
    verifier: str

    def __post_init__(self) -> None:
        if not isinstance(self.material_identifier, str) or not self.material_identifier.strip():
            raise ValueError("characterisation material identifier must be non-empty")
        _sha256(self.material_digest, field="characterisation material digest")
        _sha256(self.project_record_digest, field="characterisation project record digest")
        if not self.values or not all(
            isinstance(value, CharacterisedValue) for value in self.values
        ):
            raise ValueError("characterisation requires one or more structured measured values")
        if not self.conditions or not all(
            isinstance(value, str) and value.strip() for value in self.conditions
        ):
            raise ValueError("characterisation requires one or more conditions")
        if not isinstance(self.uncertainty_method, str) or not self.uncertainty_method.strip():
            raise ValueError("characterisation uncertainty method must be non-empty")
        if not isinstance(self.verifier, str) or not self.verifier.strip():
            raise ValueError("characterisation verifier must be non-empty")

    def verify_for(self, identifier: str, digest: str) -> None:
        """Verify the receipt binds this material and its project record remains intact."""

        if self.material_identifier != identifier:
            raise ValueError(
                "characterisation evidence is bound to a different material identifier"
            )
        if self.material_digest != digest:
            raise ValueError("characterisation evidence is bound to a different material digest")
        record = Path(self.project_record_path)
        if not record.is_file():
            raise ValueError("characterisation project record is unavailable")
        actual = hashlib.sha256(record.read_bytes()).hexdigest()
        if actual != self.project_record_digest:
            raise ValueError(
                "characterisation project record digest mismatch: "
                f"expected {self.project_record_digest}, got {actual}"
            )


@dataclass(frozen=True)
class ReverificationReceipt:
    """A local byte-for-byte check of an acquired material."""

    material_identifier: str
    verified_digest: str
    verified_at: str
    verifier: str


@dataclass(frozen=True)
class ReferenceMaterial:
    """A material and the evidence that makes its grade meaningful."""

    identifier: str
    grade: MaterialGrade
    payload_path: str
    source_digest: str
    source: SourceEvidence | None
    licence: LicenceEvidence | None
    retrieval_evidence: str | None
    reverification_evidence: str | None
    characterisation_evidence: CharacterisationEvidence | None

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError("material identifier must be non-empty")
        if not isinstance(self.grade, MaterialGrade):
            raise TypeError("material grade must be a MaterialGrade")
        _sha256(self.source_digest, field="source digest")
        if self.grade is MaterialGrade.CERTIFIED and not isinstance(
            self.characterisation_evidence, CharacterisationEvidence
        ):
            raise TypeError("certified material requires CharacterisationEvidence")
        if not isinstance(self.payload_path, str) or not self.payload_path:
            raise ValueError("material payload path must be non-empty")
        payload = self.payload
        if not payload.is_file():
            raise ValueError("material payload must be an existing file")
        actual_digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        if actual_digest != self.source_digest:
            raise ValueError(
                f"material payload digest mismatch: expected {self.source_digest}, got {actual_digest}"
            )
        if self.source is not None and not isinstance(self.source, SourceEvidence):
            raise TypeError("material source must be SourceEvidence or None")
        if self.licence is not None and not isinstance(self.licence, LicenceEvidence):
            raise TypeError("material licence must be LicenceEvidence or None")
        if self.grade is MaterialGrade.REPORTED:
            if self.source is None:
                raise ValueError("reported adopted material requires source evidence")
            if self.licence is None:
                raise ValueError("reported adopted material requires licence evidence")
            if not isinstance(self.retrieval_evidence, str) or not self.retrieval_evidence.strip():
                raise ValueError("reported adopted material requires retrieval evidence")
            if (
                not isinstance(self.reverification_evidence, str)
                or not self.reverification_evidence.strip()
            ):
                raise ValueError("reported adopted material requires re-verification evidence")
            if self.characterisation_evidence is not None:
                raise ValueError("adopted reported material cannot carry certification evidence")
        if self.grade is MaterialGrade.CERTIFIED:
            assert isinstance(self.characterisation_evidence, CharacterisationEvidence)
            self.characterisation_evidence.verify_for(self.identifier, self.source_digest)

    @property
    def payload(self) -> Path:
        """The acquired local bytes whose digest this record names."""

        return Path(self.payload_path)
