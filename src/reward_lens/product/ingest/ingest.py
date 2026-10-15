"""The public reader and durable receipt for a received assay record."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rfc8785

from reward_lens import contracts
from reward_lens.api.requests import ImportRequest, ImportResult

from .bundle import BundleReport, validate_bundle

__all__ = ["Ingested", "ingest", "save_ingested"]


@dataclass(frozen=True)
class Ingested:
    record: contracts.Assay
    original_bytes: bytes
    unknown: dict[str, Any]
    source: Path
    bundle: BundleReport | None = None
    verification: Any | None = None

    def round_trip_bytes(self) -> bytes:
        """Return the received bytes exactly; no serializer gets to rewrite extensions."""
        return self.original_bytes


def _version_refusal(raw: bytes, data: dict[str, Any]) -> None:
    declared = data.get("$schema")
    if not isinstance(declared, str):
        return
    marker = "/schema/assay/"
    if marker not in declared:
        return
    try:
        major = int(declared.split(marker, 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return
    if major != contracts.SCHEMA_MAJOR:
        raise contracts.UsageError(
            code="RL0630",
            message=(
                f"this record declares assay schema major {major}; it must be read by a "
                f"reward-lens build that reads major {major}"
            ),
            remediation="use a matching reward-lens major or migrate the preserved source bytes",
            context={"major": major, "declared": declared, "original_bytes": raw},
        )


def _decode(raw: bytes, source: Path) -> tuple[contracts.Assay, dict[str, Any]]:
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise contracts.RecordInvalid(
            message=f"the received record at {source} is not one JSON object: {failure}",
            remediation="supply an assay JSON record or an export bundle",
            context={"path": str(source), "original_bytes": raw},
        ) from failure
    if not isinstance(data, dict):
        raise contracts.RecordInvalid(
            message=f"the received record at {source} is not a JSON object",
            remediation="supply an assay JSON record or an export bundle",
            context={"path": str(source), "original_bytes": raw},
        )
    _version_refusal(raw, data)
    unknown = {key: value for key, value in data.items() if "." in key}
    known = {key: value for key, value in data.items() if key not in unknown}
    contracts.validate_record(known, raw=raw)
    computed = contracts.digest(known)
    declared = known["assay_id"]
    if declared != computed:
        raise contracts.RecordInvalid(
            message=f"the received record at {source} declares assay_id {declared}, but its contents digest to {computed}",
            remediation="restore the original record bytes or obtain a newly sealed assay; import never changes a received identity",
            context={
                "path": str(source),
                "declared_assay_id": declared,
                "computed_assay_id": computed,
                "original_bytes": raw,
            },
        )
    return contracts.Assay.model_validate(known), unknown


def _archive_record(path: Path, report: BundleReport) -> bytes:
    candidates = [member.name for member in report.members if member.name.endswith(".assay.json")]
    if len(candidates) != 1:
        raise contracts.UsageError(
            code="RL0631",
            message="bundle is unsafe: it must contain exactly one .assay.json record member",
            remediation="export one record bundle and import that bundle again",
            context={"member": ", ".join(candidates) or "<none>", "rule": "record-member"},
        )
    # This is the first member-content read, after `validate_bundle` fully returned.
    with zipfile.ZipFile(path) as archive:
        return archive.read(candidates[0])


def _verify(path: Path, *, original_bytes: bytes) -> Any:
    try:
        from reward_lens.product.export.verify import verify
    except ImportError as failure:
        raise contracts.CapabilityUnavailable(
            code="RL0701",
            message="signature verification needs the Export capability, which is not in this build",
            remediation="install a reward-lens build that ships Export and retry --verify",
            context={"path": str(path)},
        ) from failure
    result = verify(path)
    valid = result.get("valid") if isinstance(result, dict) else getattr(result, "valid", None)
    signer = result.get("signer") if isinstance(result, dict) else getattr(result, "signer", None)
    if valid is not True:
        digest = (
            result.get("digest") if isinstance(result, dict) else getattr(result, "digest", None)
        )
        raise contracts.UsageError(
            code="RL0632",
            message=f"signature from {signer or 'unknown signer'} is invalid for digest {digest or 'unknown'}",
            remediation="restore the exported bytes or obtain a newly signed bundle",
            context={
                "signer": signer,
                "digest": digest,
                "verification": result,
                "original_bytes": original_bytes,
            },
        )
    return result


def ingest(path: Path, *, verify: bool = False) -> Ingested:
    """Read one JSON record or validated bundle without executing received content."""
    source = Path(path)
    report = validate_bundle(source) if source.suffix.lower() == ".zip" else None
    if source.is_dir():
        # The Export bundle is a directory with a fixed, non-user-selected record name.  Do not
        # follow a manifest path until the Export verifier has checked that it cannot escape.
        raw = (source / "record.json").read_bytes()
        verification = _verify(source, original_bytes=raw) if verify else None
    else:
        raw = _archive_record(source, report) if report is not None else source.read_bytes()
        verification = _verify(source, original_bytes=raw) if verify else None
    record, unknown = _decode(raw, source)
    return Ingested(
        record=record,
        original_bytes=raw,
        unknown=unknown,
        source=source,
        bundle=report,
        verification=verification,
    )


def save_ingested(received: Ingested, destination: Path) -> tuple[Path, Path]:
    """Persist the original record bytes and an RFC-8785 receipt, then reopen before returning."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "imported.assay.json"
    receipt_path = root / "ingest.receipt.json"
    if record_path.exists() and record_path.read_bytes() != received.original_bytes:
        raise contracts.RecordInvalid(
            message=f"{record_path} already holds different received bytes",
            remediation="choose an empty destination or preserve the existing receipt separately",
            context={"path": str(record_path)},
        )
    if not record_path.exists():
        record_path.write_bytes(received.original_bytes)
    reopened = ingest(record_path)
    if reopened.original_bytes != received.original_bytes:
        raise RuntimeError("saved ingest bytes changed before reopen")
    receipt_path.write_bytes(
        rfc8785.dumps(
            {
                "source": str(received.source),
                "record_path": str(record_path),
                "record_assay_id": received.record.assay_id,
                "unknown_extension_names": sorted(received.unknown),
                "original_sha256": "sha256:" + hashlib.sha256(received.original_bytes).hexdigest(),
            }
        )
    )
    return record_path, receipt_path


def run(request: ImportRequest) -> ImportResult:
    """Engine seam for the typed SDK: receive, save, and prove the saved bytes reopen."""
    received = ingest(request.path, verify=request.verify)
    destination = request.dest or request.path.parent / f"{request.path.name}.reward-lens-import"
    record_path, receipt_path = save_ingested(received, destination)
    verification = received.verification
    signer = (
        verification.get("signer")
        if isinstance(verification, dict)
        else getattr(verification, "signer", None)
    )
    message = (
        verification.get("message")
        if isinstance(verification, dict)
        else getattr(verification, "message", None)
    )
    return ImportResult(
        record=received.record,
        record_path=record_path,
        receipt_path=receipt_path,
        unknown_extensions=tuple(sorted(received.unknown)),
        signer=signer,
        verification_message=message,
    )
