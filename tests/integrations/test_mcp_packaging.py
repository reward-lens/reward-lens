"""The packaged private MCP 2 runtime must be self-verifying."""

from __future__ import annotations

from shutil import copytree

import pytest

from reward_lens._vendor.mcp2.runtime import (
    MCP2RuntimeError,
    private_site_root,
    verify_private_runtime,
    verify_site_root,
)


def test_private_mcp2_payload_is_the_official_distribution() -> None:
    """MCP 2 identity comes from its private dist-info, never the parent MCP 1 lane."""
    identity = verify_private_runtime()

    assert identity.distribution == "mcp"
    assert identity.version == "2.2.0"
    assert identity.site_root == private_site_root()
    assert identity.mcp_package == private_site_root() / "mcp"
    assert identity.mcpserver_module == "mcp.server.mcpserver.server"


def test_private_mcp2_integrity_guard_rejects_any_payload_change(tmp_path) -> None:
    """The child must fail closed if its vendored upstream bytes are altered."""
    copied_root = tmp_path / "site_packages"
    copytree(private_site_root(), copied_root)
    target = copied_root / "mcp" / "__init__.py"
    target.write_text(target.read_text(encoding="utf-8") + "# altered\n", encoding="utf-8")

    with pytest.raises(MCP2RuntimeError, match="private MCP 2 file changed"):
        verify_site_root(copied_root)


def test_private_mcp2_integrity_guard_permits_only_generated_bytecode(tmp_path) -> None:
    """A normal wheel installer may create a cache, but it is never source evidence."""
    copied_root = tmp_path / "site_packages"
    copytree(private_site_root(), copied_root)
    cache = copied_root / "mcp" / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-311.pyc").write_bytes(b"generated cache is not payload source")
    assert verify_site_root(copied_root).version == "2.2.0"
    (cache / "unexpected.py").write_text("not allowed", encoding="utf-8")
    with pytest.raises(MCP2RuntimeError, match="unexpected file set"):
        verify_site_root(copied_root)


def test_private_mcp2_integrity_guard_rejects_record_tampering(tmp_path) -> None:
    copied_root = tmp_path / "site_packages"
    copytree(private_site_root(), copied_root)
    record = copied_root / "mcp-2.2.0.dist-info" / "RECORD"
    record.write_text(record.read_text(encoding="utf-8") + "mcp/evil.py,,0\n", encoding="utf-8")
    with pytest.raises(MCP2RuntimeError, match="RECORD changed"):
        verify_site_root(copied_root)
