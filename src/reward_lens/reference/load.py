"""Load, re-verify, and consume the packaged adopted reference material."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adopt import adopt
from .types import LicenceEvidence, ReferenceMaterial, ReverificationReceipt, SourceEvidence

_DIAGNOSTIC_V3 = "diagnostic-v3"
_MATERIALS = Path(__file__).with_name("materials")
_PROVENANCE_PATH = _MATERIALS / "diagnostic_v3.provenance.json"
_PROVENANCE_DIGEST = "de9f7cf01ba2cecf2e401016e69038dd6a72c613a4832e0a3fc3a5a721c683ed"


def _read_provenance() -> dict[str, Any]:
    """Parse the integrity-anchored receipt that binds payload, licence, and source metadata."""

    raw = _PROVENANCE_PATH.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _PROVENANCE_DIGEST:
        raise ValueError(
            f"diagnostic-v3 provenance digest mismatch: expected {_PROVENANCE_DIGEST}, got {actual}"
        )
    try:
        provenance = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("diagnostic-v3 provenance is not valid JSON") from exc
    if not isinstance(provenance, dict):
        raise ValueError("diagnostic-v3 provenance must be an object")
    if provenance.get("schema") != "reward-lens-reference-material/1":
        raise ValueError("diagnostic-v3 provenance schema is unsupported")
    if provenance.get("identifier") != _DIAGNOSTIC_V3 or provenance.get("grade") != "reported":
        raise ValueError("diagnostic-v3 provenance does not describe the reported adopted material")
    if provenance.get("characterisation") is not None:
        raise ValueError("adopted diagnostic-v3 provenance must not claim characterisation")
    return provenance


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"diagnostic-v3 provenance {key} must be an object")
    return value


def _text(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"diagnostic-v3 provenance {key} must be non-empty text")
    return value


def _diagnostic_v3() -> ReferenceMaterial:
    provenance = _read_provenance()
    acquisition = _mapping(provenance, "acquisition")
    licence = _mapping(provenance, "licence")
    payload_name = _text(acquisition, "packaged_payload")
    licence_name = _text(licence, "packaged_text")
    if (
        Path(payload_name).name != "diagnostic_v3.py"
        or Path(licence_name).name != "diagnostic_v3_LICENSE.txt"
    ):
        raise ValueError("diagnostic-v3 provenance names an unexpected packaged file")
    payload = _MATERIALS / Path(payload_name).name
    licence_path = _MATERIALS / Path(licence_name).name
    licence_digest = hashlib.sha256(licence_path.read_bytes()).hexdigest()
    expected_licence_digest = _text(licence, "text_sha256")
    if licence_digest != expected_licence_digest:
        raise ValueError(
            "diagnostic-v3 licence digest mismatch: "
            f"expected {expected_licence_digest}, got {licence_digest}"
        )
    material = adopt(
        identifier=_DIAGNOSTIC_V3,
        payload=payload,
        source=SourceEvidence(
            location="reward-lens-assay local diagnostic-v3 builder snapshot",
            commit=_text(acquisition, "source_commit"),
            source_path=_text(acquisition, "source_path"),
        ),
        licence=LicenceEvidence(spdx=_text(licence, "spdx"), text_digest=licence_digest),
        source_digest=_text(acquisition, "payload_sha256"),
        retrieval_evidence=(
            "P-REF acquired this packaged payload from the controller-approved diagnostic-v3 "
            "source snapshot and retained its exact source commit."
        ),
        reverification_evidence=(
            "P-REF loader checks the integrity-anchored provenance, packaged payload, and MIT "
            "licence SHA-256 digests locally."
        ),
    )
    return material


def load(identifier: str = _DIAGNOSTIC_V3) -> ReferenceMaterial:
    """Load a real locally acquired material and reject unknown identifiers."""

    if identifier != _DIAGNOSTIC_V3:
        raise ValueError(f"no packaged reference material named {identifier!r}")
    material = _diagnostic_v3()
    reverify(material)
    return material


def reverify(material: ReferenceMaterial) -> ReverificationReceipt:
    """Recompute a material digest from disk and refuse bytes that no longer match."""

    payload = material.payload
    if not payload.is_file():
        raise ValueError(f"reference material payload is unavailable: {payload}")
    actual_digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    if actual_digest != material.source_digest:
        raise ValueError(
            f"reference material digest mismatch: expected {material.source_digest}, got {actual_digest}"
        )
    return ReverificationReceipt(
        material_identifier=material.identifier,
        verified_digest=actual_digest,
        verified_at=datetime.now(UTC).isoformat(),
        verifier="reward_lens.reference.load.reverify",
    )


@dataclass(frozen=True)
class DiagnosticV3Material:
    """The actual deterministic data object constructed by the adopted diagnostic-v3 builder."""

    view: Any
    dimensions: tuple[str, ...]
    pair_count: int
    content_checksum: str
    builder_digest: str


def load_diagnostic_v3_material(material: ReferenceMaterial) -> DiagnosticV3Material:
    """Execute the exact packaged builder in isolation and return its usable ``DataView``.

    The builder's normal import-time card registration is made local to this call and restored
    afterwards, so constructing a consumer object does not mutate the process-wide data registry.
    This is construction of the supplied reference material, not a numerical characterisation.
    """

    if material.identifier != _DIAGNOSTIC_V3:
        raise ValueError("diagnostic-v3 consumer loader requires diagnostic-v3 material")
    reverify(material)
    import reward_lens.data.registry as registry

    original_decorator = registry.dataset_loader
    original_loaders = dict(registry._LOADERS)

    def isolated_dataset_loader(key: str):
        def register(fn: Any) -> Any:
            registry._LOADERS[key] = fn
            return fn

        return register

    module_name = "_reward_lens_reference_diagnostic_v3"
    spec = importlib.util.spec_from_file_location(module_name, material.payload)
    if spec is None or spec.loader is None:
        raise ValueError("diagnostic-v3 builder could not be loaded from its acquired payload")
    module = importlib.util.module_from_spec(spec)
    registry.dataset_loader = isolated_dataset_loader
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        views = module.load_diagnostic_v3()
        view = module.all_pairs()
    finally:
        registry.dataset_loader = original_decorator
        registry._LOADERS.clear()
        registry._LOADERS.update(original_loaders)
        sys.modules.pop(module_name, None)
    if not isinstance(views, dict) or not views:
        raise ValueError("diagnostic-v3 builder did not construct dimension views")
    if not hasattr(view, "checksum") or not hasattr(view, "__len__"):
        raise ValueError("diagnostic-v3 builder did not construct a usable DataView")
    return DiagnosticV3Material(
        view=view,
        dimensions=tuple(views),
        pair_count=len(view),
        content_checksum=str(view.checksum()),
        builder_digest=material.source_digest,
    )
