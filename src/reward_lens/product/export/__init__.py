"""Portable assay exports, with a signed bundle as the interchange boundary.

The record stays the source of truth.  An export is a derived view with a manifest
that identifies the exact canonical record bytes used to make it.  In particular,
the rendered HTML carries the manifest digest but is never written back into the
record.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import rfc8785

from reward_lens.contracts import UsageError

from .kinds import Artifact, ExportKind, Tier
from .scan import ExportSecretFound, SecretHit, scan
from .sign import ExportExtraRequiredError as ExtraRequiredError
from .sign import canonical_manifest_bytes, manifest_digest, sign
from .verify import verify

__all__ = [
    "Artifact",
    "ExportKind",
    "ExportSecretFound",
    "ExportSelfReference",
    "ExtraRequiredError",
    "RENDERED_HTML_EXTENSION",
    "SecretHit",
    "Tier",
    "export",
    "redacted_export",
    "run",
    "scan",
    "validate_sarif",
    "verify",
]


#: Schema-valid extension that names a derived report.  It exists so importers can preserve a
#: malformed external record long enough for the export boundary to give the integrity refusal.
#: A canonical Assay producer must never write it: D-16 says the record cannot cite its own HTML.
RENDERED_HTML_EXTENSION = "https://reward-lens.github.io/extensions/rendered-html/1"


class ExportSelfReference(UsageError):
    """RL0642: a record tries to cite a file derived from that record."""

    def __init__(self, *, value: str) -> None:
        super().__init__(
            code="RL0642",
            message=(
                "the record names rendered HTML at "
                f"{value!r}; a record cannot cite a rendering derived from itself"
            ),
            remediation="remove the rendered HTML reference and export the unchanged record again",
            context={"value": value},
        )


def _canonical_record(record: Any) -> tuple[dict[str, Any], bytes]:
    """Return a JSON object and the single byte representation exported from it."""
    if hasattr(record, "to_dict"):
        record = record.to_dict()
    if not isinstance(record, Mapping):
        raise TypeError("export expects a JSON object or an Assay with to_dict()")
    document = json.loads(json.dumps(dict(record)))
    rendered = _rendered_html_reference(document)
    if rendered:
        raise ExportSelfReference(value=str(rendered))
    payload = rfc8785.dumps(document)
    hits = scan(payload)
    if hits:
        raise ExportSecretFound(hits=hits, redacted_path="record.redacted.json")
    return document, payload


def _rendered_html_reference(document: Mapping[str, Any]) -> str | None:
    """Find the schema-valid declaration of a record's own derived report.

    The historical top-level key is retained for callers that pass a raw mapping, but a saved
    Assay reaches this check through the extension namespace that D-64 reserves for typed foreign
    data. The namespace is schema-valid; its semantic is forbidden by D-16, so the public export
    route can give the specific integrity refusal rather than an unrelated schema error.
    """
    legacy = document.get("rendered_html")
    if isinstance(legacy, str) and legacy:
        return legacy
    extensions = document.get("extensions")
    if not isinstance(extensions, Mapping):
        return None
    rendered = extensions.get(RENDERED_HTML_EXTENSION)
    if not isinstance(rendered, Mapping):
        return None
    path = rendered.get("path")
    return path if isinstance(path, str) and path else None


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _manifest(
    *,
    record_bytes: bytes,
    kind: ExportKind,
    tier: Tier,
    linked_from: str | None = None,
    record_path: str = "record.json",
) -> dict[str, Any]:
    """Build the non-circular transport manifest.

    The manifest commits to immutable record bytes and export choices.  It deliberately
    does not list the report's digest: the report itself displays this manifest digest,
    so including it would create an unsound hash cycle.
    """
    manifest: dict[str, Any] = {
        "schema": "https://reward-lens.github.io/export-manifest/v1",
        "record": {"path": record_path, "digest": _sha256(record_bytes)},
        "export": {"kind": kind.value, "tier": tier.value},
    }
    if linked_from is not None:
        manifest["linked_from"] = linked_from
    return manifest


SARIF_VISIBLE_RESULTS = 5_000
SARIF_MAX_RESULTS_PER_RUN = 25_000
SARIF_MAX_GZIP_BYTES = 10 * 1024 * 1024


def _sarif(document: Mapping[str, Any]) -> dict[str, Any]:
    """Translate canonical findings into the GitHub-visible bounded SARIF 2.1.0 form."""
    levels = {"error": 0, "warning": 1, "note": 2}
    findings = [item for item in document.get("findings", []) if isinstance(item, Mapping)]
    ordered = sorted(
        findings,
        key=lambda item: (levels.get(str(item.get("level")), 2), str(item.get("id", ""))),
    )
    kept = ordered[:SARIF_VISIBLE_RESULTS]
    declared_rules = {
        str(rule.get("id")): rule
        for rule in document.get("rules", [])
        if isinstance(rule, Mapping) and isinstance(rule.get("id"), str)
    }
    rules: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for item in kept:
        rule_id = str(item.get("rule") or item.get("code") or "reward-lens.finding")
        declared = declared_rules.get(rule_id, {})
        short_description = str(declared.get("short_description") or rule_id)
        help_text = str(declared.get("help") or short_description)
        rules.setdefault(
            rule_id,
            {
                "id": rule_id,
                "name": str(declared.get("name") or rule_id),
                "shortDescription": {"text": short_description},
                "help": {"text": help_text},
                "helpUri": str(
                    declared.get("help_uri")
                    or f"https://reward-lens.github.io/docs/findings/{rule_id}/"
                ),
            },
        )
        message = str(item.get("severity_rationale") or item.get("code") or rule_id)
        result: dict[str, Any] = {
            "ruleId": rule_id,
            "level": str(item.get("level")) if item.get("level") in levels else "note",
            "kind": str(item.get("kind")) if item.get("kind") else "informational",
            "message": {"text": message},
            "partialFingerprints": {
                "reward-lens/v1": str(item.get("partial_fingerprint") or item.get("id") or rule_id)
            },
            "properties": {"assay_finding_id": str(item.get("id", ""))},
        }
        location = item.get("location")
        if isinstance(location, Mapping) and isinstance(location.get("file"), str):
            physical: dict[str, Any] = {"artifactLocation": {"uri": location["file"]}}
            if isinstance(location.get("line"), int) and location["line"] > 0:
                physical["region"] = {"startLine": location["line"]}
            result["locations"] = [{"physicalLocation": physical}]
        results.append(result)
    sarif = {
        "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "reward-lens", "rules": list(rules.values())}},
                "results": results,
                "properties": {
                    "dropped_results": len(ordered) - len(kept),
                    "total_results": len(ordered),
                    "visible_result_limit": SARIF_VISIBLE_RESULTS,
                    "per_run_limit": SARIF_MAX_RESULTS_PER_RUN,
                    "embedding_tier": (document.get("embedding") or {}).get("tier"),
                },
            }
        ],
    }
    if len(gzip.compress(rfc8785.dumps(sarif))) > SARIF_MAX_GZIP_BYTES:
        raise UsageError(
            code="RL0604",
            message="the SARIF export exceeds the 10 MB gzip limit",
            remediation="reduce the findings' exported message material and export again",
            context={"limit_bytes": SARIF_MAX_GZIP_BYTES, "results": len(results)},
        )
    return sarif


def validate_sarif(payload: Mapping[str, Any]) -> None:
    """Validate one emitted SARIF document against the vendored OASIS 2.1.0 schema."""
    try:
        import jsonschema
    except ModuleNotFoundError as error:  # pragma: no cover - dev validation is exercised in CI
        raise RuntimeError(
            "SARIF schema validation requires the reward-lens development dependencies"
        ) from error
    schema_text = (
        resources.files(__package__).joinpath("sarif-schema-2.1.0.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft4Validator(json.loads(schema_text)).validate(dict(payload))


def _html(document: dict[str, Any], digest: str) -> bytes:
    from reward_lens.render.report import render

    return render(document, bundle_manifest_digest=digest)


def export(
    record: Any,
    *,
    kind: ExportKind | str = ExportKind.BUNDLE,
    tier: Tier | str = Tier.A,
    backend: str = "ed25519",
    badge_reach_entry_id: str | None = None,
    record_url: str | None = None,
) -> Artifact:
    """Create one export in memory.  ``Artifact.write`` decides where it is written.

    The sealed record decides its HTML tier.  ``tier`` remains accepted for request
    compatibility, but it cannot relabel a record whose renderer policy is already
    part of the signed Assay.
    """
    kind = ExportKind(kind)
    document, record_bytes = _canonical_record(record)
    tier = Tier((document.get("embedding") or {}).get("tier") or tier)
    manifest = _manifest(record_bytes=record_bytes, kind=kind, tier=tier)
    digest = manifest_digest(manifest)

    if kind is ExportKind.REVISION:
        return Artifact(kind, tier, {"record.json": record_bytes}, manifest, digest, "record.json")
    if kind is ExportKind.BADGE:
        if record_url is None:
            raise UsageError(
                code="RL0003",
                message="badge export needs the saved record URL",
                remediation="use api.export with ExportRequest(record=PATH), or provide record_url explicitly",
                context={"record_url": None},
            )
        from reward_lens.commons.badge import badge_from_assay
        from reward_lens.contracts import Assay

        payload = badge_from_assay(
            Assay.model_validate(document),
            record_url=record_url,
            badge_reach_entry_id=badge_reach_entry_id,
        )
        return Artifact(
            kind, tier, {"badge.json": rfc8785.dumps(payload)}, manifest, digest, "badge.json"
        )
    if kind is ExportKind.SARIF:
        payload = rfc8785.dumps(_sarif(document))
        return Artifact(kind, tier, {"results.sarif": payload}, manifest, digest, "results.sarif")

    report = _html(document, digest)
    if kind is ExportKind.HTML:
        return Artifact(kind, tier, {"report.html": report}, manifest, digest, "report.html")
    envelope = sign(manifest, backend=backend).to_dict()
    files = {
        "record.json": record_bytes,
        "manifest.json": canonical_manifest_bytes(manifest),
        "attestation.dsse.json": rfc8785.dumps(envelope),
        "report.html": report,
    }
    return Artifact(kind, tier, files, manifest, digest, "report.html")


def redacted_export(original: Artifact, *, replacement: bytes) -> Artifact:
    """Export audited replacement bytes as a distinct artifact linked to the original manifest."""
    try:
        replacement_document = json.loads(replacement)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("redaction replacement must be one JSON object") from error
    document, record_bytes = _canonical_record(replacement_document)
    manifest = _manifest(
        record_bytes=record_bytes,
        kind=original.kind,
        tier=original.tier,
        linked_from=original.manifest_digest,
        record_path="record.redacted.json",
    )
    digest = manifest_digest(manifest)
    if original.kind is ExportKind.REVISION:
        return Artifact(
            original.kind,
            original.tier,
            {"record.json": record_bytes},
            manifest,
            digest,
            "record.json",
            original.manifest_digest,
        )
    if original.kind is ExportKind.BADGE:
        raise ValueError("a redacted badge needs a complete saved Assay and must be exported again")
    if original.kind is ExportKind.SARIF:
        return Artifact(
            original.kind,
            original.tier,
            {"results.sarif": rfc8785.dumps(_sarif(document))},
            manifest,
            digest,
            "results.sarif",
            original.manifest_digest,
        )
    # A replacement is not necessarily a complete Assay.  It must therefore travel as
    # its own signed redaction object, rather than being rendered as though its fields
    # were an assay.  A caller with a complete redacted Assay can call ``export`` on it
    # and obtains the ordinary report bundle.
    files = {
        "record.redacted.json": record_bytes,
        "manifest.json": canonical_manifest_bytes(manifest),
        "attestation.dsse.json": rfc8785.dumps(sign(manifest).to_dict()),
    }
    return Artifact(
        original.kind,
        original.tier,
        files,
        manifest,
        digest,
        "record.redacted.json",
        original.manifest_digest,
    )


def run(request: Any, *, project: Any = None, sandbox: Any = None) -> Artifact:
    """SDK engine seam, reopening a validated saved Assay before it writes an artifact."""
    if getattr(request, "record", None) is None:
        raise UsageError(
            code="RL0003",
            message="export needs a saved record path",
            remediation="pass --record PATH for a canonical assay JSON file",
            context={"record": None},
        )
    from reward_lens.api import open_record

    record_path = Path(request.record).resolve()
    artifact = export(
        open_record(record_path),
        kind=getattr(request, "format", ExportKind.BUNDLE),
        tier=getattr(request, "tier", Tier.A),
        backend=getattr(request, "backend", "ed25519"),
        badge_reach_entry_id=getattr(request, "badge_reach_entry_id", None),
        record_url=record_path.as_uri(),
    )
    destination = getattr(request, "dest", None)
    if destination is not None:
        artifact.write(destination)
    return artifact
