"""Certificate-of-analysis records for reference material, without inflating its grade."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .types import MaterialGrade, ReferenceMaterial


@dataclass(frozen=True)
class CertificateOfAnalysis:
    """The full section 7.9 field list, including limitations for reported material."""

    date_of_issue: str
    identifier: str
    material_grade: MaterialGrade
    purpose: str
    description: str
    certified_values: tuple[str, ...]
    certified_values_definition: str
    traceability_statement: str
    value_table: tuple[tuple[str, str], ...]
    expanded_uncertainty_note: str
    derivation_method: str
    validity_period: str
    nullification_conditions: str
    maintenance: str
    safety_and_storage: str
    use_with_cautions: str
    potential_interest_values: tuple[str, ...]
    references: tuple[str, ...]
    revision_history: tuple[str, ...]
    disclaimer: str

    def __post_init__(self) -> None:
        if not isinstance(self.material_grade, MaterialGrade):
            raise TypeError("certificate material grade must be a MaterialGrade")
        if self.material_grade is MaterialGrade.CERTIFIED and not self.certified_values:
            raise ValueError("certified certificate requires characterised values")

    @classmethod
    def for_material(cls, material: ReferenceMaterial) -> "CertificateOfAnalysis":
        """Describe a material's known bytes and expressly preserve its actual grade."""

        # Re-run the public record's validation so certificate generation cannot turn a forged
        # dataclass instance into a provenance-looking document.
        material.__post_init__()
        source = material.source
        source_text = "unknown source" if source is None else source.location
        characterisation = material.characterisation_evidence
        values = (
            ()
            if characterisation is None
            else tuple(
                f"{value.measurand}: {value.value} ± {value.expanded_uncertainty}"
                for value in characterisation.values
            )
        )
        return cls(
            date_of_issue=datetime.now(UTC).date().isoformat(),
            identifier=material.identifier,
            material_grade=material.grade,
            purpose="Reference material for local reward-lens calibration and traceability work.",
            description=f"Locally acquired payload at SHA-256 {material.source_digest}.",
            certified_values=values,
            certified_values_definition=(
                "No values are certified by this project for this reported adopted material."
                if material.grade is MaterialGrade.REPORTED
                else "Values are certified only by the digest-verified project characterisation record."
            ),
            traceability_statement=(
                f"Traceable to {source_text}; source digest {material.source_digest}."
            ),
            value_table=(("payload_sha256", material.source_digest),),
            expanded_uncertainty_note=(
                "Read this digest as byte identity only; it supplies no uncertainty characterisation."
            ),
            derivation_method="SHA-256 over the acquired payload bytes.",
            validity_period="Valid while payload and provenance digests re-verify.",
            nullification_conditions=(
                "Any source, licence, payload, or digest mismatch nullifies this certificate record."
            ),
            maintenance="Re-run local digest verification before use.",
            safety_and_storage="Treat leaked, altered, or contaminated payloads as invalid.",
            use_with_cautions=(
                "Use as reported reference material; do not treat it as project-characterised."
            ),
            potential_interest_values=("Payload SHA-256 is a second-tier identity value.",),
            references=(source_text,),
            revision_history=("P-REF initial acquired-material record.",),
            disclaimer=(
                "This record is provenance and integrity evidence, not a calibration or transfer claim."
            ),
        )

    @property
    def complete(self) -> bool:
        """Whether the required non-tabular certificate fields have usable values."""

        fields = (
            self.date_of_issue,
            self.identifier,
            self.purpose,
            self.description,
            self.certified_values_definition,
            self.traceability_statement,
            self.expanded_uncertainty_note,
            self.derivation_method,
            self.validity_period,
            self.nullification_conditions,
            self.maintenance,
            self.safety_and_storage,
            self.use_with_cautions,
            self.disclaimer,
        )
        return bool(self.value_table and self.references and self.revision_history and all(fields))
