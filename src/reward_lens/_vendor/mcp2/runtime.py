"""Integrity checks for the private, unchanged official MCP 2 payload."""

# ruff: noqa: E402

from __future__ import annotations

import atexit
import base64
import csv
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path


class MCP2RuntimeError(RuntimeError):
    """The packaged MCP 2 capability is absent or has been altered."""


@dataclass(frozen=True)
class MCP2Identity:
    distribution: str
    version: str
    site_root: Path
    mcp_package: Path
    mcpserver_module: str


# A child imports the private SDK before it imports the integration server.  Keep
# that process on the exact same verified, cache-free copy.  A second temporary
# root is still verified, but cannot own an already-loaded ``mcp`` module.
_MATERIALIZED_RUNTIME: MCP2Identity | None = None


def private_site_root() -> Path:
    """Return the packaged upstream site root without importing its modules."""
    return Path(__file__).with_name("site_packages")


def _provenance() -> dict[str, object]:
    return json.loads(Path(__file__).with_name("MCP2_PROVENANCE.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        _source_files(root),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def _is_generated_cache(path: Path, root: Path) -> bool:
    """Only ordinary interpreter bytecode is excluded from the immutable source payload."""
    relative = path.relative_to(root)
    return len(relative.parts) >= 2 and relative.parts[-2] == "__pycache__" and path.suffix == ".pyc"


def _source_files(root: Path):
    return (item for item in root.rglob("*") if item.is_file() and not _is_generated_cache(item, root))


def _record_digest_matches(root: Path, record: Path) -> None:
    expected_paths: set[str] = set()
    for row in csv.reader(record.read_text(encoding="utf-8").splitlines()):
        relative, expected, _size = row
        expected_paths.add(relative)
        path = root / relative
        if not path.is_file():
            raise MCP2RuntimeError(f"RL0711 private MCP 2 file is missing: {relative}")
        if not expected:
            continue
        algorithm, encoded = expected.split("=", 1)
        if algorithm != "sha256":
            raise MCP2RuntimeError(f"RL0711 unsupported MCP 2 RECORD digest: {algorithm}")
        actual = base64.urlsafe_b64encode(bytes.fromhex(_sha256(path))).decode("ascii").rstrip("=")
        if actual != encoded:
            raise MCP2RuntimeError(f"RL0711 private MCP 2 file changed: {relative}")
    actual_paths = {item.relative_to(root).as_posix() for item in _source_files(root)}
    if actual_paths != expected_paths:
        raise MCP2RuntimeError("RL0711 private MCP 2 tree has an unexpected file set")


def verify_site_root(root: Path) -> MCP2Identity:
    """Verify a wheel-derived MCP 2 site root against the captured provenance."""
    provenance = _provenance()
    dist = root / "mcp-2.2.0.dist-info"
    metadata = dist / "METADATA"
    record = dist / "RECORD"
    package = root / "mcp"
    if not package.is_dir() or not metadata.is_file() or not record.is_file():
        raise MCP2RuntimeError("RL0711 private MCP 2 payload is incomplete")
    if _sha256(metadata) != provenance["metadata_sha256"]:
        raise MCP2RuntimeError("RL0711 private MCP 2 METADATA changed")
    if _sha256(record) != provenance["record_sha256"]:
        raise MCP2RuntimeError("RL0711 private MCP 2 RECORD changed")
    parsed = Parser().parsestr(metadata.read_text(encoding="utf-8"))
    if parsed["Name"] != "mcp" or parsed["Version"] != "2.2.0":
        raise MCP2RuntimeError("RL0711 private MCP 2 distribution identity differs")
    _record_digest_matches(root, record)
    if _tree_sha256(root) != provenance["payload_tree_sha256"]:
        raise MCP2RuntimeError("RL0711 private MCP 2 payload tree changed")
    return MCP2Identity(
        distribution="mcp",
        version="2.2.0",
        site_root=root,
        mcp_package=package,
        mcpserver_module="mcp.server.mcpserver.server",
    )


def verify_private_runtime() -> MCP2Identity:
    """Verify the packaged bytes before a child can import canonical ``mcp``."""
    return verify_site_root(private_site_root())


def materialize_private_runtime() -> MCP2Identity:
    """Copy verified upstream source into a cache-free private import root.

    Normal wheel installers legitimately create ``__pycache__`` below package
    directories.  This function never imports from that mutable cache-bearing
    tree: it hashes every official RECORD source file first, then copies only
    those verified files into a new temporary root and verifies that root again.
    """
    global _MATERIALIZED_RUNTIME
    if _MATERIALIZED_RUNTIME is not None:
        return _MATERIALIZED_RUNTIME

    source = verify_private_runtime()
    container = Path(tempfile.mkdtemp(prefix="reward-lens-mcp2-"))
    target = container / "site_packages"
    for item in _source_files(source.site_root):
        relative = item.relative_to(source.site_root)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
    identity = verify_site_root(target)
    atexit.register(shutil.rmtree, container, ignore_errors=True)
    _MATERIALIZED_RUNTIME = identity
    return _MATERIALIZED_RUNTIME


def require_private_mcp(module_file: str | None, *, site_root: Path | None = None) -> MCP2Identity:
    """Refuse a parent MCP 1 import before the integration server is constructed."""
    identity = verify_site_root(site_root) if site_root is not None else verify_private_runtime()
    if module_file is None or not Path(module_file).resolve().is_relative_to(identity.site_root.resolve()):
        raise MCP2RuntimeError("RL0711 canonical MCP must be imported by the private child runtime")
    return identity
