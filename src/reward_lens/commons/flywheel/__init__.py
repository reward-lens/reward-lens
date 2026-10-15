"""Consent-gated private-pair retention and calibration-interval updates.

The default operation is deliberately local: it copies an owner-marked,
non-equivalent Improve pair into a private, integrity-checked regression store.
Promotion is a separate call that needs both explicit consent and a redaction
that removes every declared sensitive literal.  A promoted value is candidate
reference material only; it is neither a P-REF material nor a certificate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Mapping

CONSENT_ABSENT = "CONSENT_ABSENT"
CALIBRATION_INTERVALS_REQUIRED = "CALIBRATION_INTERVALS_REQUIRED"
_PRIVATE_SCHEMA = "reward-lens.private-pair.v1"
_CONSENT_SCHEMA = "reward-lens.private-pair-consent.v1"
_REDACTION_SCHEMA = "reward-lens.redaction.v1"
_CANDIDATE_SCHEMA = "reward-lens.candidate-reference.v1"

__all__ = [
    "CALIBRATION_INTERVALS_REQUIRED",
    "CONSENT_ABSENT",
    "CandidateReferenceMaterial",
    "Consent",
    "FlywheelRefusal",
    "Interval",
    "PrivatePair",
    "PrivatePairRequest",
    "RecalibrationResult",
    "Redaction",
    "RetentionResult",
    "promote",
    "recalibrate",
    "retain",
]


class FlywheelRefusal(ValueError):
    """A named refusal from the local consent and calibration boundary."""


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _finite_decimal(value: Decimal, *, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise FlywheelRefusal(f"{name} must be a finite Decimal")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FlywheelRefusal(f"{name} must be a non-empty string")
    return value.strip()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    _write_bytes(path, _canonical_json(value))


@dataclass(frozen=True)
class PrivatePairRequest:
    """Public request for the default local retention route.

    ``improve_session`` must name a durable P-IMPROVE session containing an
    already-marked pair.  The request does not accept arbitrary response text:
    the engine rechecks the session's recorded source-byte digests before it
    writes anything to the private store.
    """

    improve_session: Path | str
    pair_index: int
    owner_id: str
    affected_panel: str
    private_store: Path | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "improve_session", Path(self.improve_session))
        object.__setattr__(self, "private_store", Path(self.private_store))
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise FlywheelRefusal("PRIVATE_PAIR_INVALID: pair_index must be a non-negative integer")
        _text(self.owner_id, name="owner_id")
        _text(self.affected_panel, name="affected_panel")


@dataclass(frozen=True)
class PrivatePair:
    """An integrity-checked private regression snapshot, reopened from disk."""

    id: str
    owner_id: str
    affected_panel: str
    private_store: Path
    left_bytes: bytes
    right_bytes: bytes
    left_digest: str
    right_digest: str
    improve_session_digest: str
    source_identity: str
    regression_case: bool = True
    private: bool = True

    @property
    def directory(self) -> Path:
        return self.private_store / "private_pairs" / self.id

    @property
    def left_path(self) -> Path:
        return self.directory / "left.bin"

    @property
    def right_path(self) -> Path:
        return self.directory / "right.bin"

    def reopen(self) -> "PrivatePair":
        """Read and re-verify this stored regression snapshot."""
        return _reopen(self.private_store, self.id)


@dataclass(frozen=True)
class RetentionResult:
    """The identified result of the public private-retention request."""

    code: str
    pair: PrivatePair


@dataclass(frozen=True)
class Consent:
    """An owner's explicit, pair-bound permission to produce candidate material."""

    pair_id: str
    owner_id: str
    explicit: bool

    def __post_init__(self) -> None:
        _text(self.pair_id, name="consent pair_id")
        _text(self.owner_id, name="consent owner_id")
        if type(self.explicit) is not bool:
            raise FlywheelRefusal("CONSENT_INVALID: explicit must be a bool")

    def payload(self) -> dict[str, object]:
        return {
            "schema": _CONSENT_SCHEMA,
            "pair_id": self.pair_id,
            "owner_id": self.owner_id,
            "explicit": self.explicit,
        }

    @property
    def digest(self) -> str:
        return _sha256(_canonical_json(self.payload()))


@dataclass(frozen=True)
class Redaction:
    """Literal redaction instructions whose inputs must disappear from output."""

    replacements: Mapping[str, str]
    sensitivity_inventory: tuple[str, ...] = ()
    declared_complete: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.replacements, Mapping) or not self.replacements:
            raise FlywheelRefusal("REDACTION_REQUIRED")
        normalized: dict[str, str] = {}
        for original, replacement in self.replacements.items():
            original = _text(original, name="redaction source")
            replacement = _text(replacement, name="redaction replacement")
            if original == replacement or original in replacement:
                raise FlywheelRefusal("REDACTION_INVALID: replacement retains source text")
            normalized[original] = replacement
        inventory = tuple(
            _text(item, name="sensitivity inventory item") for item in self.sensitivity_inventory
        )
        if len(set(inventory)) != len(inventory):
            raise FlywheelRefusal(
                "REDACTION_INVALID: sensitivity inventory must not repeat an item"
            )
        if type(self.declared_complete) is not bool:
            raise FlywheelRefusal("REDACTION_INVALID: declared_complete must be a bool")
        object.__setattr__(self, "replacements", normalized)
        object.__setattr__(self, "sensitivity_inventory", inventory)

    def artifact(self) -> dict[str, object]:
        return {
            "schema": _REDACTION_SCHEMA,
            "replacements": dict(sorted(self.replacements.items())),
            "sensitivity_inventory": sorted(self.sensitivity_inventory),
            "declared_complete": self.declared_complete,
        }

    @property
    def digest(self) -> str:
        return _sha256(_canonical_json(self.artifact()))

    def apply(self, private_bytes: bytes) -> bytes:
        try:
            text = private_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise FlywheelRefusal(
                "REDACTION_UNSUPPORTED: private pair bytes must be UTF-8"
            ) from error
        for original, replacement in self.replacements.items():
            if original not in text:
                continue
            text = text.replace(original, replacement)
        return text.encode("utf-8")


@dataclass(frozen=True)
class CandidateReferenceMaterial:
    """Redacted opt-in material awaiting a later reference-material decision.

    This intentionally carries no P-REF grade or certificate. It records only
    the redacted candidate bytes and the private pair they came from.
    """

    id: str
    source_private_pair_id: str
    source_identity: str
    source_left_digest: str
    source_right_digest: str
    consent_digest: str
    left_bytes: bytes
    right_bytes: bytes
    redaction_digest: str
    content_digest: str
    evidence_store: Path
    kind: str = field(default="CANDIDATE_REFERENCE_MATERIAL", init=False)
    certificate: None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _text(self.id, name="candidate id")
        _text(self.source_private_pair_id, name="candidate source_private_pair_id")
        for name in (
            "source_identity",
            "source_left_digest",
            "source_right_digest",
            "consent_digest",
            "redaction_digest",
            "content_digest",
        ):
            if not _is_digest(getattr(self, name)):
                raise FlywheelRefusal(f"CANDIDATE_INVALID: {name} must be a SHA-256 digest")
        if not isinstance(self.left_bytes, bytes) or not isinstance(self.right_bytes, bytes):
            raise FlywheelRefusal("CANDIDATE_INVALID: candidate bytes are required")
        if _sha256(self.left_bytes + b"\n" + self.right_bytes) != self.content_digest:
            raise FlywheelRefusal("CANDIDATE_INVALID: content digest does not bind candidate bytes")
        if not isinstance(self.evidence_store, Path):
            object.__setattr__(self, "evidence_store", Path(self.evidence_store))

    def render(self) -> str:
        return self.left_bytes.decode("utf-8") + "\n" + self.right_bytes.decode("utf-8")

    def reopen(self) -> "CandidateReferenceMaterial":
        """Reopen the persisted candidate artifact against its private pair."""
        return _reopen_candidate_artifact(self)


@dataclass(frozen=True)
class Interval:
    """One same-unit displayed interval, without a claim beyond its input panel."""

    lower: Decimal
    upper: Decimal
    unit: str

    def __post_init__(self) -> None:
        lower = _finite_decimal(self.lower, name="interval lower")
        upper = _finite_decimal(self.upper, name="interval upper")
        if lower > upper:
            raise FlywheelRefusal("CALIBRATION_INTERVALS_REQUIRED: lower exceeds upper")
        _text(self.unit, name="interval unit")

    def display(self) -> str:
        return f"[{self.lower}, {self.upper}] {self.unit}"


@dataclass(frozen=True)
class RecalibrationResult:
    """Both intervals and the visibly changed, affected calibration panel."""

    affected_panel: str
    prior: Interval
    later: Interval
    transfer_status: str
    candidate_id: str
    candidate_content_digest: str
    evidence_digest: str
    evidence_path: Path

    @property
    def changed(self) -> bool:
        return self.prior != self.later

    @property
    def affected_panel_display(self) -> str:
        return f"{self.affected_panel}: prior {self.prior.display()}; later {self.later.display()}"


def retain(request: PrivatePairRequest) -> RetentionResult:
    """Save one P-IMPROVE regression pair privately and return its identified result."""
    if not isinstance(request, PrivatePairRequest):
        raise TypeError("request must be PrivatePairRequest")
    try:
        session_bytes = request.improve_session.read_bytes()
        session = json.loads(session_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise FlywheelRefusal(
            "PRIVATE_PAIR_INVALID: improve_session must be a readable saved session"
        ) from error
    if not isinstance(session, dict) or session.get("state") not in {
        "describe",
        "propose",
        "plan",
        "execute",
        "decide",
        "export",
    }:
        raise FlywheelRefusal("PRIVATE_PAIR_INVALID: improve_session has no valid saved state")
    pairs = session.get("pairs")
    if not isinstance(pairs, list) or request.pair_index >= len(pairs):
        raise FlywheelRefusal("PRIVATE_PAIR_INVALID: marked pair is absent")
    marked = pairs[request.pair_index]
    if not isinstance(marked, dict) or marked.get("regression_case") is not True:
        raise FlywheelRefusal("PRIVATE_PAIR_INVALID: selected result is not a regression case")
    left = _read_marked_version(marked, "left")
    right = _read_marked_version(marked, "right")
    if left[0] == right[0]:
        raise FlywheelRefusal("PRIVATE_PAIR_INVALID: saved pair has identical bytes")

    seed = {
        "owner_id": request.owner_id,
        "affected_panel": request.affected_panel,
        "session": _sha256(session_bytes),
        "pair_index": request.pair_index,
        "left": left[1],
        "right": right[1],
    }
    pair_id = "PP-" + hashlib.sha256(_canonical_json(seed)).hexdigest()[:20]
    source_identity = _private_pair_identity(
        pair_id=pair_id,
        owner_id=request.owner_id,
        affected_panel=request.affected_panel,
        improve_session_digest=_sha256(session_bytes),
        left_digest=left[1],
        right_digest=right[1],
    )
    pair = PrivatePair(
        id=pair_id,
        owner_id=request.owner_id,
        affected_panel=request.affected_panel,
        private_store=request.private_store,
        left_bytes=left[0],
        right_bytes=right[0],
        left_digest=left[1],
        right_digest=right[1],
        improve_session_digest=_sha256(session_bytes),
        source_identity=source_identity,
    )
    _write_bytes(pair.left_path, pair.left_bytes)
    _write_bytes(pair.right_path, pair.right_bytes)
    _write_json(pair.directory / "manifest.json", _manifest(pair))
    return RetentionResult(code="PRIVATE_REGRESSION_SAVED", pair=pair.reopen())


def _read_marked_version(marked: dict[str, object], side: str) -> tuple[bytes, str]:
    declared = marked.get(side)
    if not isinstance(declared, dict):
        raise FlywheelRefusal("PRIVATE_PAIR_INVALID: marked pair side is absent")
    path = declared.get("path")
    digest = declared.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise FlywheelRefusal("PRIVATE_PAIR_INVALID: marked pair side lacks path or digest")
    try:
        value = Path(path).read_bytes()
    except OSError as error:
        raise FlywheelRefusal(
            "PRIVATE_PAIR_INVALID: marked pair source bytes are unavailable"
        ) from error
    actual = _sha256(value)
    if actual != digest:
        raise FlywheelRefusal(
            "PRIVATE_PAIR_INTEGRITY_FAILURE: Improve pair bytes no longer match their saved digest"
        )
    return value, actual


def _manifest(pair: PrivatePair) -> dict[str, object]:
    return {
        "schema": _PRIVATE_SCHEMA,
        "id": pair.id,
        "owner_id": pair.owner_id,
        "affected_panel": pair.affected_panel,
        "improve_session_digest": pair.improve_session_digest,
        "source_identity": pair.source_identity,
        "left": {"file": "left.bin", "sha256": pair.left_digest},
        "right": {"file": "right.bin", "sha256": pair.right_digest},
        "regression_case": True,
        "private": True,
    }


def _private_pair_identity(
    *,
    pair_id: str,
    owner_id: str,
    affected_panel: str,
    improve_session_digest: str,
    left_digest: str,
    right_digest: str,
) -> str:
    """Canonical identity of the exact retained pair and its source bytes."""
    return _sha256(
        _canonical_json(
            {
                "schema": _PRIVATE_SCHEMA,
                "id": pair_id,
                "owner_id": owner_id,
                "affected_panel": affected_panel,
                "improve_session_digest": improve_session_digest,
                "left_digest": left_digest,
                "right_digest": right_digest,
                "regression_case": True,
                "private": True,
            }
        )
    )


def _reopen(private_store: Path, pair_id: str) -> PrivatePair:
    directory = Path(private_store) / "private_pairs" / pair_id
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FlywheelRefusal(
            "PRIVATE_PAIR_INTEGRITY_FAILURE: private pair manifest is unavailable"
        ) from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != _PRIVATE_SCHEMA
        or manifest.get("id") != pair_id
    ):
        raise FlywheelRefusal("PRIVATE_PAIR_INTEGRITY_FAILURE: private pair manifest is invalid")
    if manifest.get("regression_case") is not True or manifest.get("private") is not True:
        raise FlywheelRefusal(
            "PRIVATE_PAIR_INTEGRITY_FAILURE: private regression flags are invalid"
        )
    try:
        left_meta, right_meta = manifest["left"], manifest["right"]
        if not isinstance(left_meta, dict) or not isinstance(right_meta, dict):
            raise TypeError
        left = (directory / str(left_meta["file"])).read_bytes()
        right = (directory / str(right_meta["file"])).read_bytes()
        left_digest, right_digest = str(left_meta["sha256"]), str(right_meta["sha256"])
        source_identity = str(manifest["source_identity"])
    except (KeyError, OSError, TypeError) as error:
        raise FlywheelRefusal(
            "PRIVATE_PAIR_INTEGRITY_FAILURE: private pair payload is unavailable"
        ) from error
    if _sha256(left) != left_digest or _sha256(right) != right_digest:
        raise FlywheelRefusal(
            "PRIVATE_PAIR_INTEGRITY_FAILURE: private pair payload digest mismatch"
        )
    try:
        owner_id = _text(manifest["owner_id"], name="private pair owner")
        affected_panel = _text(manifest["affected_panel"], name="private pair panel")
        improve_session_digest = _text(
            manifest["improve_session_digest"], name="private pair session digest"
        )
        if not _is_digest(source_identity) or source_identity != _private_pair_identity(
            pair_id=pair_id,
            owner_id=owner_id,
            affected_panel=affected_panel,
            improve_session_digest=improve_session_digest,
            left_digest=left_digest,
            right_digest=right_digest,
        ):
            raise FlywheelRefusal(
                "PRIVATE_PAIR_INTEGRITY_FAILURE: private pair source identity mismatch"
            )
        return PrivatePair(
            id=_text(manifest["id"], name="private pair id"),
            owner_id=owner_id,
            affected_panel=affected_panel,
            private_store=Path(private_store),
            left_bytes=left,
            right_bytes=right,
            left_digest=left_digest,
            right_digest=right_digest,
            improve_session_digest=improve_session_digest,
            source_identity=source_identity,
        )
    except (KeyError, FlywheelRefusal) as error:
        raise FlywheelRefusal(
            "PRIVATE_PAIR_INTEGRITY_FAILURE: private pair manifest fields are invalid"
        ) from error


def promote(
    pair: PrivatePair,
    *,
    consent: Consent | None,
    redaction: Redaction | None,
) -> CandidateReferenceMaterial:
    """Produce redacted candidate material after explicit pair-bound consent."""
    if not isinstance(pair, PrivatePair):
        raise TypeError("pair must be PrivatePair")
    if (
        not isinstance(consent, Consent)
        or not consent.explicit
        or consent.pair_id != pair.id
        or consent.owner_id != pair.owner_id
    ):
        raise FlywheelRefusal(CONSENT_ABSENT)
    if not isinstance(redaction, Redaction):
        raise FlywheelRefusal("REDACTION_REQUIRED")
    private = pair.reopen()
    if pair != private:
        raise FlywheelRefusal(
            "PRIVATE_PAIR_INTEGRITY_FAILURE: promotion requires the reopened private pair"
        )
    if not redaction.declared_complete or set(redaction.sensitivity_inventory) != set(
        redaction.replacements
    ):
        raise FlywheelRefusal("REDACTION_INVENTORY_INCOMPLETE")
    source = private.left_bytes + b"\n" + private.right_bytes
    for original in redaction.sensitivity_inventory:
        if original.encode("utf-8") not in source:
            raise FlywheelRefusal(
                "REDACTION_INVALID: declared source text is absent from the private pair"
            )
    left, right = redaction.apply(private.left_bytes), redaction.apply(private.right_bytes)
    rendered = left + b"\n" + right
    for original in redaction.sensitivity_inventory:
        if original.encode("utf-8") in rendered:
            raise FlywheelRefusal("REDACTION_INVALID: private source text remains after redaction")
    redaction_digest = redaction.digest
    content_digest = _sha256(left + b"\n" + right)
    consent_digest = consent.digest
    candidate_id = (
        "CRM-"
        + hashlib.sha256(
            _canonical_json(
                {
                    "source_identity": private.source_identity,
                    "consent_digest": consent_digest,
                    "redaction_digest": redaction_digest,
                    "content_digest": content_digest,
                }
            )
        ).hexdigest()[:20]
    )
    candidate = CandidateReferenceMaterial(
        id=candidate_id,
        source_private_pair_id=private.id,
        source_identity=private.source_identity,
        source_left_digest=private.left_digest,
        source_right_digest=private.right_digest,
        consent_digest=consent_digest,
        left_bytes=left,
        right_bytes=right,
        redaction_digest=redaction_digest,
        content_digest=content_digest,
        evidence_store=private.private_store,
    )
    _persist_candidate_artifact(candidate, consent=consent, redaction=redaction)
    return candidate.reopen()


def _candidate_directory(candidate: CandidateReferenceMaterial) -> Path:
    return candidate.evidence_store / "candidate-reference" / candidate.id


def _persist_candidate_artifact(
    candidate: CandidateReferenceMaterial,
    *,
    consent: Consent,
    redaction: Redaction,
) -> None:
    directory = _candidate_directory(candidate)
    _write_bytes(directory / "left.redacted.bin", candidate.left_bytes)
    _write_bytes(directory / "right.redacted.bin", candidate.right_bytes)
    _write_json(
        directory / "artifact.json",
        {
            "schema": _CANDIDATE_SCHEMA,
            "id": candidate.id,
            "kind": candidate.kind,
            "source": {
                "private_pair_id": candidate.source_private_pair_id,
                "identity": candidate.source_identity,
                "left_digest": candidate.source_left_digest,
                "right_digest": candidate.source_right_digest,
            },
            "consent": {"payload": consent.payload(), "digest": candidate.consent_digest},
            "redaction": {"artifact": redaction.artifact(), "digest": candidate.redaction_digest},
            "output": {
                "left": {"file": "left.redacted.bin", "digest": _sha256(candidate.left_bytes)},
                "right": {"file": "right.redacted.bin", "digest": _sha256(candidate.right_bytes)},
                "content_digest": candidate.content_digest,
            },
        },
    )


def _redaction_from_artifact(value: object) -> Redaction:
    if not isinstance(value, dict) or value.get("schema") != _REDACTION_SCHEMA:
        raise FlywheelRefusal("CANDIDATE_ARTIFACT_INVALID: redaction artifact is invalid")
    try:
        return Redaction(
            replacements=value["replacements"],
            sensitivity_inventory=tuple(value["sensitivity_inventory"]),
            declared_complete=value["declared_complete"],
        )
    except (KeyError, TypeError, FlywheelRefusal) as error:
        raise FlywheelRefusal(
            "CANDIDATE_ARTIFACT_INVALID: redaction artifact is invalid"
        ) from error


def _reopen_candidate_artifact(candidate: CandidateReferenceMaterial) -> CandidateReferenceMaterial:
    directory = _candidate_directory(candidate)
    try:
        artifact = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
        if (
            not isinstance(artifact, dict)
            or artifact.get("schema") != _CANDIDATE_SCHEMA
            or artifact.get("id") != candidate.id
        ):
            raise TypeError
        source, consent, redaction_data, output = (
            artifact["source"],
            artifact["consent"],
            artifact["redaction"],
            artifact["output"],
        )
        if not all(isinstance(value, dict) for value in (source, consent, redaction_data, output)):
            raise TypeError
        left_meta, right_meta = output["left"], output["right"]
        if not isinstance(left_meta, dict) or not isinstance(right_meta, dict):
            raise TypeError
        left = (directory / str(left_meta["file"])).read_bytes()
        right = (directory / str(right_meta["file"])).read_bytes()
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise FlywheelRefusal(
            "CANDIDATE_ARTIFACT_INVALID: candidate artifact is unavailable"
        ) from error
    private = _reopen(candidate.evidence_store, str(source.get("private_pair_id")))
    if (
        source.get("identity") != private.source_identity
        or source.get("left_digest") != private.left_digest
        or source.get("right_digest") != private.right_digest
    ):
        raise FlywheelRefusal(
            "CANDIDATE_ARTIFACT_INVALID: candidate source does not bind the actual private pair"
        )
    consent_payload = consent.get("payload")
    if not isinstance(consent_payload, dict) or consent.get("digest") != _sha256(
        _canonical_json(consent_payload)
    ):
        raise FlywheelRefusal("CANDIDATE_ARTIFACT_INVALID: consent artifact is invalid")
    try:
        actual_consent = Consent(
            pair_id=consent_payload["pair_id"],
            owner_id=consent_payload["owner_id"],
            explicit=consent_payload["explicit"],
        )
    except (KeyError, FlywheelRefusal) as error:
        raise FlywheelRefusal("CANDIDATE_ARTIFACT_INVALID: consent artifact is invalid") from error
    if (
        consent_payload != actual_consent.payload()
        or not actual_consent.explicit
        or actual_consent.pair_id != private.id
        or actual_consent.owner_id != private.owner_id
    ):
        raise FlywheelRefusal("CANDIDATE_ARTIFACT_INVALID: consent is not explicit and pair-bound")
    redaction = _redaction_from_artifact(redaction_data.get("artifact"))
    if (
        redaction_data.get("digest") != redaction.digest
        or not redaction.declared_complete
        or set(redaction.sensitivity_inventory) != set(redaction.replacements)
    ):
        raise FlywheelRefusal("CANDIDATE_ARTIFACT_INVALID: redaction inventory is incomplete")
    expected_left, expected_right = (
        redaction.apply(private.left_bytes),
        redaction.apply(private.right_bytes),
    )
    if (
        left != expected_left
        or right != expected_right
        or _sha256(left) != left_meta.get("digest")
        or _sha256(right) != right_meta.get("digest")
    ):
        raise FlywheelRefusal(
            "CANDIDATE_ARTIFACT_INVALID: candidate output does not bind its source and redaction"
        )
    rendered = left + b"\n" + right
    for sensitive in redaction.sensitivity_inventory:
        if sensitive.encode("utf-8") in rendered:
            raise FlywheelRefusal(
                "CANDIDATE_ARTIFACT_INVALID: candidate output retains sensitive text"
            )
    content_digest = _sha256(rendered)
    if output.get("content_digest") != content_digest:
        raise FlywheelRefusal("CANDIDATE_ARTIFACT_INVALID: candidate output digest mismatch")
    expected_id = (
        "CRM-"
        + hashlib.sha256(
            _canonical_json(
                {
                    "source_identity": private.source_identity,
                    "consent_digest": actual_consent.digest,
                    "redaction_digest": redaction.digest,
                    "content_digest": content_digest,
                }
            )
        ).hexdigest()[:20]
    )
    if candidate.id != expected_id:
        raise FlywheelRefusal("CANDIDATE_ARTIFACT_INVALID: candidate identifier mismatch")
    actual = CandidateReferenceMaterial(
        id=expected_id,
        source_private_pair_id=private.id,
        source_identity=private.source_identity,
        source_left_digest=private.left_digest,
        source_right_digest=private.right_digest,
        consent_digest=actual_consent.digest,
        left_bytes=left,
        right_bytes=right,
        redaction_digest=redaction.digest,
        content_digest=content_digest,
        evidence_store=candidate.evidence_store,
    )
    if candidate != actual:
        raise FlywheelRefusal(
            "CANDIDATE_ARTIFACT_INVALID: supplied candidate does not match its persisted artifact"
        )
    return actual


def recalibrate(
    *,
    panel: object,
    candidate: CandidateReferenceMaterial,
) -> RecalibrationResult:
    """Run P-CALIB's candidate-byte procedure and return its persisted evidence."""
    try:
        from reward_lens.reference.calibration import CalibrationPanel
        from reward_lens.reference.calibration.candidate import (
            CandidateCalibrationInput,
            CandidateCalibrationRefusal,
            calibrate_candidate,
            open_candidate_calibration,
            persist_candidate_calibration,
        )
    except (
        ImportError
    ) as error:  # pragma: no cover - predecessor contract is required by this packet
        raise FlywheelRefusal(CALIBRATION_INTERVALS_REQUIRED) from error
    if not isinstance(panel, CalibrationPanel) or not panel.qualified or panel.validity is None:
        raise FlywheelRefusal(CALIBRATION_INTERVALS_REQUIRED)
    if not isinstance(candidate, CandidateReferenceMaterial):
        raise FlywheelRefusal(CALIBRATION_INTERVALS_REQUIRED)
    try:
        candidate = candidate.reopen()
        candidate_input = CandidateCalibrationInput.from_candidate_reference(candidate)
        calculated = calibrate_candidate(panel, candidate_input)
        evidence_path = persist_candidate_calibration(
            calculated,
            candidate.evidence_store,
            panel=panel,
            candidate=candidate_input,
        )
        evidence = open_candidate_calibration(evidence_path, panel=panel, candidate=candidate_input)
    except CandidateCalibrationRefusal as error:
        raise FlywheelRefusal(str(error)) from error
    prior = Interval(
        evidence.prior_interval.lower_bound,
        evidence.prior_interval.upper_bound,
        evidence.prior_interval.unit,
    )
    later = Interval(
        evidence.later_interval.lower_bound,
        evidence.later_interval.upper_bound,
        evidence.later_interval.unit,
    )
    if later == prior:
        raise FlywheelRefusal(
            "CALIBRATION_INTERVALS_REQUIRED: prior and later intervals must differ"
        )
    affected = f"{panel.material_identifier}:{panel.reference_view_checksum}"
    return RecalibrationResult(
        affected_panel=affected,
        prior=prior,
        later=later,
        transfer_status=evidence.transfer_status,
        candidate_id=evidence.candidate_id,
        candidate_content_digest=evidence.candidate_content_digest,
        evidence_digest=evidence.evidence_digest,
        evidence_path=evidence_path,
    )
