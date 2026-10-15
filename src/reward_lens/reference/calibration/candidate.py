"""Candidate-bound paired measurements for an already qualified calibration panel.

This child module does not make redacted candidate material into a reference
material and does not alter P-CALIB's accepted LoB/LoD/LoQ calculation.  It
defines one explicit, same-unit procedure for candidate pairs: each redacted
JSON side supplies aligned scalar readings, and the procedure calculates the
right-minus-left readings.  Those actual candidate bytes determine the observed
range, transfer status, and later displayed range for an existing qualified
panel.  The result is persisted and rederived on reopen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from . import CalibrationPanel, ValidityEnvelope

CANDIDATE_CALIBRATION_UNSUPPORTED = "CANDIDATE_CALIBRATION_UNSUPPORTED"
_SCHEMA = "reward-lens.candidate-calibration.v1"
_PROCEDURE = "paired-score-delta/v1"

__all__ = [
    "CANDIDATE_CALIBRATION_UNSUPPORTED",
    "CandidateCalibrationEvidence",
    "CandidateCalibrationInput",
    "CandidateCalibrationRefusal",
    "calibrate_candidate",
    "open_candidate_calibration",
    "persist_candidate_calibration",
]


class CandidateCalibrationRefusal(ValueError):
    """A candidate cannot support the declared paired measurement procedure."""


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(char in "0123456789abcdef" for char in value[7:])
    )


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: {name} must be a non-empty string"
        )
    return value.strip()


def _interval_payload(interval: ValidityEnvelope) -> dict[str, str]:
    return {
        "lower_bound": str(interval.lower_bound),
        "upper_bound": str(interval.upper_bound),
        "unit": interval.unit,
    }


def _canonical_value(value: object) -> object:
    """Encode every public field of a claim-bearing calibration input."""
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value):
        return {item.name: _canonical_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    return value


def _panel_payload(panel: CalibrationPanel) -> dict[str, object]:
    if not panel.qualified or panel.limits is None or panel.validity is None:
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: qualified prior panel is required"
        )
    return {
        "material_identifier": panel.material_identifier,
        "material_digest": panel.material_digest,
        "reference_view_checksum": panel.reference_view_checksum,
        "reference_view_effective_n": str(panel.reference_view_effective_n),
        "unit": panel.unit,
        "method": panel.method,
        "direct": not panel.synthetic,
        "synthetic": panel.synthetic,
        "config": _canonical_value(panel.config),
        "limits": _canonical_value(panel.limits),
        "validity": _interval_payload(panel.validity),
        "transfer": None if panel.transfer is None else _canonical_value(panel.transfer),
        "assessments": _canonical_value(panel.assessments),
        "refusal": panel.refusal,
    }


@dataclass(frozen=True)
class CandidateCalibrationInput:
    """A candidate artifact locator plus its claimed redacted provenance.

    P-CALIB never trusts this dataclass alone.  Each public calculation reopens
    the durable flywheel candidate artifact and compares every field below with
    the result before it can produce claim-bearing evidence.
    """

    candidate_id: str
    candidate_content_digest: str
    redaction_digest: str
    source_private_pair_id: str
    source_identity: str
    source_left_digest: str
    source_right_digest: str
    consent_digest: str
    left_bytes: bytes
    right_bytes: bytes
    evidence_store: Path | str | None = None

    def __post_init__(self) -> None:
        _text(self.candidate_id, name="candidate_id")
        if not all(
            _is_digest(value)
            for value in (
                self.candidate_content_digest,
                self.redaction_digest,
                self.source_identity,
                self.source_left_digest,
                self.source_right_digest,
                self.consent_digest,
            )
        ):
            raise CandidateCalibrationRefusal(
                f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: candidate digests must be SHA-256"
            )
        _text(self.source_private_pair_id, name="source_private_pair_id")
        if not isinstance(self.left_bytes, bytes) or not isinstance(self.right_bytes, bytes):
            raise CandidateCalibrationRefusal(
                f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: candidate bytes are required"
            )
        actual = _sha256(self.left_bytes + b"\n" + self.right_bytes)
        if actual != self.candidate_content_digest:
            raise CandidateCalibrationRefusal(
                f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: candidate content digest does not bind its bytes"
            )
        if self.evidence_store is not None:
            object.__setattr__(self, "evidence_store", Path(self.evidence_store))

    @classmethod
    def from_candidate_reference(cls, candidate: object) -> "CandidateCalibrationInput":
        """Build the P-CALIB input from the flywheel's artifact-backed candidate."""
        try:
            return cls(
                candidate_id=candidate.id,
                candidate_content_digest=candidate.content_digest,
                redaction_digest=candidate.redaction_digest,
                source_private_pair_id=candidate.source_private_pair_id,
                source_identity=candidate.source_identity,
                source_left_digest=candidate.source_left_digest,
                source_right_digest=candidate.source_right_digest,
                consent_digest=candidate.consent_digest,
                left_bytes=candidate.left_bytes,
                right_bytes=candidate.right_bytes,
                evidence_store=candidate.evidence_store,
            )
        except AttributeError as error:
            raise CandidateCalibrationRefusal(
                f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: artifact-backed candidate is required"
            ) from error

    def reopen_verified(self) -> "CandidateCalibrationInput":
        """Reopen the candidate artifact and refuse invented provenance fields."""
        if self.evidence_store is None:
            raise CandidateCalibrationRefusal(
                f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: candidate artifact locator is required"
            )
        try:
            from reward_lens.commons.flywheel import CandidateReferenceMaterial, FlywheelRefusal

            actual = CandidateReferenceMaterial(
                id=self.candidate_id,
                source_private_pair_id=self.source_private_pair_id,
                source_identity=self.source_identity,
                source_left_digest=self.source_left_digest,
                source_right_digest=self.source_right_digest,
                consent_digest=self.consent_digest,
                left_bytes=self.left_bytes,
                right_bytes=self.right_bytes,
                redaction_digest=self.redaction_digest,
                content_digest=self.candidate_content_digest,
                evidence_store=self.evidence_store,
            ).reopen()
        except (FlywheelRefusal, OSError, ValueError) as error:
            raise CandidateCalibrationRefusal(
                f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: candidate artifact provenance is invalid"
            ) from error
        verified = type(self).from_candidate_reference(actual)
        if self != verified:
            raise CandidateCalibrationRefusal(
                f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: candidate input differs from its durable artifact"
            )
        return verified


@dataclass(frozen=True)
class CandidateCalibrationEvidence:
    """Persistable candidate measurement evidence derived from one qualified panel."""

    id: str
    evidence_digest: str
    panel: dict[str, object]
    candidate_id: str
    candidate_content_digest: str
    redaction_digest: str
    source_private_pair_id: str
    source_identity: str
    source_left_digest: str
    source_right_digest: str
    consent_digest: str
    procedure: str
    unit: str
    measured_deltas: tuple[Decimal, ...]
    observed_interval: ValidityEnvelope
    prior_interval: ValidityEnvelope
    later_interval: ValidityEnvelope
    transfer_status: str

    def payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "id": self.id,
            "panel": self.panel,
            "candidate": {
                "id": self.candidate_id,
                "content_digest": self.candidate_content_digest,
                "redaction_digest": self.redaction_digest,
                "source_private_pair_id": self.source_private_pair_id,
                "source_identity": self.source_identity,
                "source_left_digest": self.source_left_digest,
                "source_right_digest": self.source_right_digest,
                "consent_digest": self.consent_digest,
            },
            "procedure": self.procedure,
            "unit": self.unit,
            "measured_deltas": [str(value) for value in self.measured_deltas],
            "observed_interval": _interval_payload(self.observed_interval),
            "prior_interval": _interval_payload(self.prior_interval),
            "later_interval": _interval_payload(self.later_interval),
            "transfer_status": self.transfer_status,
        }

    def complete_payload(self) -> dict[str, object]:
        return {**self.payload(), "evidence_digest": self.evidence_digest}


def _candidate_observation(value: bytes, *, side: str) -> tuple[str, tuple[Decimal, ...]]:
    try:
        document = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: {side} candidate is not UTF-8 JSON"
        ) from error
    if not isinstance(document, dict):
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: {side} candidate must be an object"
        )
    observation = document.get("calibration_observation")
    if not isinstance(observation, dict) or observation.get("procedure") != _PROCEDURE:
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: {side} candidate lacks {_PROCEDURE} observations"
        )
    unit = _text(observation.get("unit"), name=f"{side} candidate observation unit")
    raw_values = observation.get("measurements")
    if (
        not isinstance(raw_values, list)
        or len(raw_values) < 2
        or any(not isinstance(item, str) for item in raw_values)
    ):
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: {side} candidate needs at least two decimal measurements"
        )
    try:
        values = tuple(Decimal(item) for item in raw_values)
    except (InvalidOperation, ValueError) as error:
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: {side} candidate measurements must be decimals"
        ) from error
    if any(not item.is_finite() for item in values):
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: {side} candidate measurements must be finite"
        )
    return unit, values


def calibrate_candidate(
    panel: CalibrationPanel, candidate: CandidateCalibrationInput
) -> CandidateCalibrationEvidence:
    """Calculate candidate paired deltas and a later range from the redacted bytes.

    The paired-score-delta procedure computes ``right[i] - left[i]`` for each
    aligned observation. It requires a strictly positive delta because this
    extension describes a positive score-delta calibration range. Its observed
    range is not a new LoB/LoD/LoQ claim: the original qualified panel retains
    that role. A range inside the panel confirms and intersects it; any range
    outside reports `TRANSFER_FAILURE` and takes the union.
    """
    if not isinstance(panel, CalibrationPanel) or not isinstance(
        candidate, CandidateCalibrationInput
    ):
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: typed panel and candidate are required"
        )
    candidate = candidate.reopen_verified()
    prior_payload = _panel_payload(panel)
    left_unit, left = _candidate_observation(candidate.left_bytes, side="left")
    right_unit, right = _candidate_observation(candidate.right_bytes, side="right")
    if left_unit != right_unit or left_unit != panel.unit:
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: candidate and panel units must match"
        )
    if len(left) != len(right):
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: paired candidate observations must have equal length"
        )
    deltas = tuple(right_value - left_value for left_value, right_value in zip(left, right))
    if any(value <= 0 for value in deltas):
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: paired score deltas must be strictly positive"
        )
    observed = ValidityEnvelope(min(deltas), max(deltas), unit=panel.unit)
    prior = panel.validity
    assert prior is not None  # narrowed by _panel_payload above
    if prior.lower_bound <= observed.lower_bound and observed.upper_bound <= prior.upper_bound:
        later = ValidityEnvelope(observed.lower_bound, observed.upper_bound, unit=panel.unit)
        transfer_status = "CONFIRMED"
    else:
        later = ValidityEnvelope(
            min(prior.lower_bound, observed.lower_bound),
            max(prior.upper_bound, observed.upper_bound),
            unit=panel.unit,
        )
        transfer_status = "TRANSFER_FAILURE"
    payload: dict[str, object] = {
        "schema": _SCHEMA,
        "panel": prior_payload,
        "candidate": {
            "id": candidate.candidate_id,
            "content_digest": candidate.candidate_content_digest,
            "redaction_digest": candidate.redaction_digest,
            "source_private_pair_id": candidate.source_private_pair_id,
            "source_identity": candidate.source_identity,
            "source_left_digest": candidate.source_left_digest,
            "source_right_digest": candidate.source_right_digest,
            "consent_digest": candidate.consent_digest,
        },
        "procedure": _PROCEDURE,
        "unit": panel.unit,
        "measured_deltas": [str(value) for value in deltas],
        "observed_interval": _interval_payload(observed),
        "prior_interval": _interval_payload(prior),
        "later_interval": _interval_payload(later),
        "transfer_status": transfer_status,
    }
    digest = _sha256(_canonical_json(payload))
    evidence_id = "CCE-" + digest[7:27]
    complete = {
        "schema": _SCHEMA,
        "id": evidence_id,
        **{key: value for key, value in payload.items() if key != "schema"},
    }
    evidence_digest = _sha256(_canonical_json(complete))
    return CandidateCalibrationEvidence(
        id=evidence_id,
        evidence_digest=evidence_digest,
        panel=prior_payload,
        candidate_id=candidate.candidate_id,
        candidate_content_digest=candidate.candidate_content_digest,
        redaction_digest=candidate.redaction_digest,
        source_private_pair_id=candidate.source_private_pair_id,
        source_identity=candidate.source_identity,
        source_left_digest=candidate.source_left_digest,
        source_right_digest=candidate.source_right_digest,
        consent_digest=candidate.consent_digest,
        procedure=_PROCEDURE,
        unit=panel.unit,
        measured_deltas=deltas,
        observed_interval=observed,
        prior_interval=prior,
        later_interval=later,
        transfer_status=transfer_status,
    )


def persist_candidate_calibration(
    evidence: CandidateCalibrationEvidence,
    root: Path | str,
    *,
    panel: CalibrationPanel,
    candidate: CandidateCalibrationInput,
) -> Path:
    """Persist only a recalculated, artifact-verified candidate calibration."""
    if not isinstance(evidence, CandidateCalibrationEvidence):
        raise TypeError("evidence must be CandidateCalibrationEvidence")
    expected = calibrate_candidate(panel, candidate)
    if evidence.complete_payload() != expected.complete_payload():
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: evidence does not match the verified candidate calculation"
        )
    path = Path(root) / "candidate-calibration" / f"{evidence.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = evidence.complete_payload()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(_canonical_json(payload))
    temporary.replace(path)
    return path


def open_candidate_calibration(
    path: Path | str,
    *,
    panel: CalibrationPanel,
    candidate: CandidateCalibrationInput,
) -> CandidateCalibrationEvidence:
    """Recalculate from the supplied candidate bytes and reject stale or substituted evidence."""
    try:
        payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: saved evidence is unavailable"
        ) from error
    expected = calibrate_candidate(panel, candidate)
    if payload != expected.complete_payload():
        raise CandidateCalibrationRefusal(
            f"{CANDIDATE_CALIBRATION_UNSUPPORTED}: saved evidence does not bind this panel and candidate"
        )
    return expected
