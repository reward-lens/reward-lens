#!/usr/bin/env python3
"""Verify the immutable token export against the bundled C-019 site source.

The export is intentionally a byte copy, not a second token source.  This
checker accepts only the repository-contained C-019 source fixture; the
caller cannot redirect the manifest to another stylesheet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PRODUCT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("site-token-manifest.json")
EXPORT_PATH = Path(__file__).with_name("site-tokens.css")
PROVENANCE_PATH = Path(__file__).with_name("site-token-source.json")
SCHEMA = "rewardlens-site-token-manifest/1"
PROVENANCE_SCHEMA = "reward-lens-site-token-source/1"
SOURCE_ROOT_NAME = "frontend/tokens/site-source"
SOURCE_PATH = "src/styles/tokens.css"
EXPORT_PATH_NAME = "frontend/tokens/site-tokens.css"


class VerificationError(ValueError):
    pass


def digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def canonical_unsigned(manifest: dict[str, object]) -> bytes:
    unsigned = {key: value for key, value in manifest.items() if key != "canonical_sha256"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def fail(message: str) -> None:
    raise VerificationError(message)


def declared_site_root() -> Path:
    try:
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        fail(f"cannot read C-019 source provenance: {error}")
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {"schema", "fixture_root", "source_path", "source_sha256", "source_repository", "source_revision"}
        or provenance.get("schema") != PROVENANCE_SCHEMA
        or provenance.get("fixture_root") != SOURCE_ROOT_NAME
        or provenance.get("source_path") != SOURCE_PATH
        or not isinstance(provenance.get("source_repository"), str)
        or not isinstance(provenance.get("source_revision"), str)
    ):
        fail("C-019 source provenance is invalid")
    return (PRODUCT_ROOT / SOURCE_ROOT_NAME).resolve()


def verify(site_root: Path) -> None:
    declared_root = declared_site_root()
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read site token manifest: {error}")
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "source", "export", "canonical_sha256"}:
        fail("token manifest has an unexpected shape")
    if manifest.get("schema") != SCHEMA:
        fail("token manifest schema differs from the P-WB-1 contract")
    source = manifest.get("source")
    exported = manifest.get("export")
    canonical_sha = manifest.get("canonical_sha256")
    if not isinstance(source, dict) or not isinstance(exported, dict) or not isinstance(canonical_sha, str):
        fail("token manifest digest fields are malformed")
    if source.get("path") != SOURCE_PATH or exported.get("path") != EXPORT_PATH_NAME:
        fail("token manifest path differs from the P-WB-1 contract")
    if not all(isinstance(value, str) and value.startswith("sha256:") and len(value) == 71 for value in (source.get("sha256"), exported.get("sha256"), canonical_sha)):
        fail("token manifest digest is not a SHA-256 value")
    actual_canonical = digest(canonical_unsigned(manifest))
    if canonical_sha != actual_canonical:
        fail("canonical manifest digest differs")
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if provenance.get("source_sha256") != source.get("sha256"):
        fail("source provenance digest differs from token manifest")
    source_file = site_root / SOURCE_PATH
    try:
        source_bytes = source_file.read_bytes()
    except OSError as error:
        fail(f"cannot read declared site token source: {error}")
    actual_source = digest(source_bytes)
    if source.get("sha256") != actual_source:
        fail("source digest differs")
    try:
        export_bytes = EXPORT_PATH.read_bytes()
    except OSError as error:
        fail(f"cannot read immutable token export: {error}")
    actual_export = digest(export_bytes)
    if exported.get("sha256") != actual_export:
        fail("export digest differs")
    if export_bytes != source_bytes:
        fail("immutable export bytes differ from the bundled C-019 source")
    if site_root != declared_root:
        fail(f"site root differs from declared C-019 fixture root: {site_root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", type=Path, help="bundled C-019 site fixture root")
    args = parser.parse_args(argv)
    try:
        verify((args.site_root or declared_site_root()).resolve())
    except VerificationError as error:
        print(f"site token manifest: {error}", file=sys.stderr)
        return 1
    print("site token manifest: verified bundled C-019 source, immutable export, canonical bytes, and P-WB-1 contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
