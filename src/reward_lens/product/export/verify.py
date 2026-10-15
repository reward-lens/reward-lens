"""Offline verification for a P-EXPORT bundle and its DSSE attestation."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from .kinds import Verification
from .sign import (
    PAYLOAD_TYPE,
    _pae,
    canonical_manifest_bytes,
    manifest_digest,
    signer_for_public_key,
)

_WORDING = "A signature identifies the signer and establishes nothing about whether the outcome check was right."


def _invalid(signer: str, statement: dict, detail: str) -> Verification:
    return Verification(
        signer=signer, valid=False, statement=statement, message=f"{detail} {_WORDING}"
    )


def _member(root: Path, relative: str) -> Path | None:
    """Resolve one manifest member without allowing it to escape its bundle."""
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return root / candidate


def verify(bundle: str | Path) -> Verification:
    """Check manifest bytes, DSSE signature, and the single claimed manifest digest offline."""
    root = Path(bundle)
    try:
        manifest_bytes = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        envelope = json.loads((root / "attestation.dsse.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _invalid("unknown", {}, f"The bundle could not be read: {error}.")
    if manifest_bytes != canonical_manifest_bytes(manifest):
        return _invalid("unknown", {}, "The manifest is not in its canonical signed byte form.")
    payload_type = envelope.get("payloadType")
    signatures = envelope.get("signatures") or []
    signature = signatures[0] if signatures else {}
    signer = "unknown"
    try:
        payload = base64.b64decode(envelope["payload"], validate=True)
        statement: dict[str, Any] = json.loads(payload.decode("utf-8"))
    except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return _invalid(signer, {}, f"The attestation payload is invalid: {error}.")
    expected = manifest_digest(manifest).removeprefix("sha256:")
    subject = (statement.get("subject") or [{}])[0]
    claimed = str((subject.get("digest") or {}).get("sha256", ""))
    if (
        payload_type != PAYLOAD_TYPE
        or subject.get("name") != "manifest.json"
        or claimed != expected
    ):
        return _invalid(signer, statement, "The attestation does not claim this manifest digest.")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key = base64.b64decode(signature["publicKey"], validate=True)
        signed = base64.b64decode(signature["sig"], validate=True)
        signer = signer_for_public_key(public_key)
        key_id = "sha256:" + hashlib.sha256(public_key).hexdigest()
        if signature.get("keyid") != key_id or signature.get("signer") != signer:
            return _invalid(
                signer,
                statement,
                "The signature metadata does not identify the verified public key.",
            )
        Ed25519PublicKey.from_public_bytes(public_key).verify(signed, _pae(payload_type, payload))
    except Exception as error:
        return _invalid(signer, statement, f"The Ed25519 signature is invalid: {error}.")
    record = manifest.get("record") if isinstance(manifest, dict) else None
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("digest"), str)
    ):
        return _invalid(signer, statement, "The manifest has no complete record identity.")
    member = _member(root, record["path"])
    if member is None:
        return _invalid(signer, statement, "The manifest record path escapes the bundle.")
    try:
        actual_record = member.read_bytes()
    except OSError as error:
        return _invalid(signer, statement, f"The manifest record could not be read: {error}.")
    expected_record = "sha256:" + hashlib.sha256(actual_record).hexdigest()
    if record["digest"] != expected_record:
        return _invalid(
            signer, statement, "The bundled record does not match the signed manifest digest."
        )
    return Verification(
        signer=signer, valid=True, statement=statement, message=f"Signed by {signer}. {_WORDING}"
    )
