"""Offline DSSE signing over the in-toto transport-manifest Statement."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import rfc8785

from reward_lens.contracts import CapabilityUnavailable
from reward_lens.core.extras import ExtraRequiredError as CoreExtraRequiredError

from .kinds import DsseEnvelope

PAYLOAD_TYPE = "application/vnd.in-toto+json"
PREDICATE_TYPE = "https://reward-lens.github.io/attestation/assay/v1"
STATEMENT_TYPE = "https://in-toto.io/Statement/v1"


class ExportExtraRequiredError(CoreExtraRequiredError):
    """Optional-export refusal with the product error code carried by CLI formatters."""

    def __init__(self, message: str, *, code: str = "RL0704") -> None:
        super().__init__(message)
        self.code = code


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return rfc8785.dumps(manifest)


def manifest_digest(manifest: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def signer_for_public_key(public_key: bytes) -> str:
    """The signer identity carried by an offline Ed25519 public key."""
    return "ed25519:sha256:" + hashlib.sha256(public_key).hexdigest()


def _pae(payload_type: str, payload: bytes) -> bytes:
    return (
        b"DSSEv1 "
        + str(len(payload_type)).encode()
        + b" "
        + payload_type.encode()
        + b" "
        + str(len(payload)).encode()
        + b" "
        + payload
    )


def statement(manifest: dict[str, Any]) -> dict[str, Any]:
    """The single-subject D-69 statement, without an outcome endorsement."""
    digest = manifest_digest(manifest)
    record = manifest.get("record", {})
    return {
        "_type": STATEMENT_TYPE,
        "subject": [
            {"name": "manifest.json", "digest": {"sha256": digest.removeprefix("sha256:")}}
        ],
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "run_id": record.get("run_id"),
            "tool_versions": record.get("tool_versions", {}),
            "sandbox_tier": record.get("sandbox_tier"),
            "gate_verdicts": record.get("gate_verdicts", []),
        },
    }


def _sigstore_unavailable() -> ExportExtraRequiredError:
    return ExportExtraRequiredError(
        "Sigstore signing needs the reward-lens sigstore extra, which is not installed. "
        "Install it with: pip install 'reward-lens[sigstore]'"
    )


def sign(manifest: dict[str, Any], *, backend: str = "ed25519") -> DsseEnvelope:
    """Sign one canonical in-toto Statement with the offline Ed25519 default."""
    if backend == "sigstore":
        try:
            import sigstore  # noqa: F401
        except ModuleNotFoundError as error:
            raise _sigstore_unavailable() from error
        raise CapabilityUnavailable(
            code="RL0704",
            message="Sigstore signing needs an external identity and is unavailable in offline mode",
            remediation="use the built-in Ed25519 signature for an offline export",
            context={"extra": "sigstore", "capability": "Sigstore signing"},
        )
    if backend != "ed25519":
        raise ValueError(f"unknown signing backend: {backend}")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ModuleNotFoundError as error:
        raise CapabilityUnavailable(
            code="RL0704",
            message="offline Ed25519 signing needs cryptography in the reward-lens base install",
            remediation="install a reward-lens build with its declared base dependencies",
            context={"capability": "offline Ed25519 signing", "extra": "base"},
        ) from error
    payload = rfc8785.dumps(statement(manifest))
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    signer = signer_for_public_key(public_key)
    signature = private_key.sign(_pae(PAYLOAD_TYPE, payload))
    return DsseEnvelope(
        payload_type=PAYLOAD_TYPE,
        payload=base64.b64encode(payload).decode("ascii"),
        signatures=(
            {
                "keyid": signer.removeprefix("ed25519:"),
                "sig": base64.b64encode(signature).decode("ascii"),
                "publicKey": base64.b64encode(public_key).decode("ascii"),
                "signer": signer,
            },
        ),
    )
