"""Public import paths preserve received evidence before interpreting it."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import sysconfig
import zipfile
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import pytest


def _assert_installed_module_matches_source(
    module: ModuleType, relative: str, installed_root: Path, source_root: Path
) -> None:
    imported = Path(inspect.getfile(module)).resolve()
    expected = (installed_root / relative).resolve()
    assert imported == expected, f"module escaped the installed distribution: {imported}"
    source = source_root / relative
    assert source.is_file(), source
    assert sha256(imported.read_bytes()).digest() == sha256(source.read_bytes()).digest(), (
        f"installed module differs from reviewed source: {relative}"
    )


def test_acceptance_imports_exact_product_modules_from_noneditable_install() -> None:
    """The runtime comes from site-packages and matches the reviewed checkout byte-for-byte."""
    from reward_lens import api
    from reward_lens.cli.verbs import import_cmd
    from reward_lens.product import ingest

    root = Path(__file__).resolve().parents[2]
    source_root = root / "src" / "reward_lens"
    installed_root = Path(importlib.metadata.distribution("reward-lens").locate_file("reward_lens"))
    installed_root = installed_root.resolve()
    assert installed_root.parent == Path(sysconfig.get_path("purelib")).resolve()
    assert installed_root != source_root.resolve()
    for module, relative in (
        (api, "api/__init__.py"),
        (import_cmd, "cli/verbs/import_cmd.py"),
        (ingest, "product/ingest/__init__.py"),
    ):
        _assert_installed_module_matches_source(module, relative, installed_root, source_root)


def test_installed_origin_check_refuses_shadow_and_stale_bytes(tmp_path: Path) -> None:
    relative = "api/__init__.py"
    source_root = tmp_path / "reviewed"
    source = source_root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"reviewed module\n")
    installed_root = tmp_path / "site-packages" / "reward_lens"
    installed = installed_root / relative
    installed.parent.mkdir(parents=True)
    installed.write_bytes(source.read_bytes())
    module = ModuleType("shadow_api")
    module.__file__ = str(source)
    with pytest.raises(AssertionError, match="escaped the installed distribution"):
        _assert_installed_module_matches_source(module, relative, installed_root, source_root)
    module.__file__ = str(installed)
    installed.write_bytes(b"stale module\n")
    with pytest.raises(AssertionError, match="differs from reviewed source"):
        _assert_installed_module_matches_source(module, relative, installed_root, source_root)


def _record_bytes() -> bytes:
    from reward_lens.api import _record

    return json.dumps(
        _record.absent_record(
            command="import-test",
            path=None,
            offline=True,
            seed=1,
            reproduce=("reward-lens import fixture",),
            started_monotonic=0.0,
        ).to_dict(),
        separators=(",", ":"),
    ).encode()


def test_ingest_preserves_namespaced_extension_and_source_bytes(tmp_path: Path) -> None:
    from reward_lens.product.ingest import ingest

    payload = json.loads(_record_bytes())
    payload["org.example.review"] = {"opaque": ["keep", 3]}
    source = tmp_path / "record.assay.json"
    source.write_bytes(json.dumps(payload, separators=(",", ":")).encode())

    received = ingest(source)

    assert received.original_bytes == source.read_bytes()
    assert received.unknown["org.example.review"] == {"opaque": ["keep", 3]}
    assert received.round_trip_bytes() == source.read_bytes()


def test_incompatible_major_refuses_with_original_bytes(tmp_path: Path) -> None:
    from reward_lens.product.ingest import ingest

    source = tmp_path / "future.assay.json"
    raw = b'{"$schema":"https://reward-lens.github.io/schema/assay/99.0/assay.schema.json"}'
    source.write_bytes(raw)

    with pytest.raises(Exception) as raised:
        ingest(source)

    assert getattr(raised.value, "code", None) == "RL0630"
    assert getattr(raised.value, "context", {})["original_bytes"] == raw
    assert "major 99" in str(raised.value)


def test_bundle_is_refused_from_metadata_before_any_member_is_read(tmp_path: Path) -> None:
    from reward_lens.product.ingest import validate_bundle

    bundle = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        info = zipfile.ZipInfo("payload-link")
        info.external_attr = 0o120777 << 16
        archive.writestr(info, b"would be read if validation were late")

    with pytest.raises(Exception) as raised:
        validate_bundle(bundle)

    assert getattr(raised.value, "code", None) == "RL0631"
    assert "payload-link" in str(raised.value)
    assert "symlink" in str(raised.value).lower()


def test_bundle_ratio_is_refused_before_member_content_is_read(tmp_path: Path) -> None:
    from reward_lens.product.ingest import validate_bundle

    bundle = tmp_path / "ratio.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("highly-compressed.txt", b"A" * 200_000)

    with pytest.raises(Exception) as raised:
        validate_bundle(bundle)

    assert getattr(raised.value, "code", None) == "RL0631"
    assert "ratio" in str(raised.value).lower()


def test_v3_migration_previews_backs_up_and_marks_undeclared_meaning_unknown(
    tmp_path: Path,
) -> None:
    from reward_lens.product.ingest import migrate_v3_card

    source = tmp_path / "v3-card.json"
    source.write_text(json.dumps({"version": 3, "id": "historic", "score": 1}), encoding="utf-8")

    migration = migrate_v3_card(source)

    assert migration.preview["intent"] == "unknown"
    assert migration.preview["level_of_detail"] == "unknown"
    assert migration.preview["composite"] == "unknown"
    assert migration.backup_path.read_bytes() == source.read_bytes()
    assert migration.provenance_link["source"] == str(source)
    assert migration.validation_report["valid"] is True


def test_typed_sdk_saves_then_reopens_the_received_record(tmp_path: Path) -> None:
    from reward_lens import api

    source = tmp_path / "received.assay.json"
    source.write_bytes(_record_bytes())
    result = api.import_record(api.ImportRequest(path=source, dest=tmp_path / "saved"))

    assert result.record_path.read_bytes() == source.read_bytes()
    assert result.receipt_path.is_file()
    assert result.record.assay_id == api.open_record(result.record_path).assay_id


def test_wrong_assay_digest_refuses_open_and_import_before_creating_a_destination(
    tmp_path: Path,
) -> None:
    from reward_lens import api, contracts

    data = json.loads(_record_bytes())
    declared = data["assay_id"]
    data["intent"]["success"] = "changed after sealing"
    source = tmp_path / "changed.assay.json"
    raw = json.dumps(data, separators=(",", ":")).encode()
    source.write_bytes(raw)
    destination = tmp_path / "saved"
    expected = contracts.digest(data)
    assert expected != declared

    for read in (
        lambda: api.open_record(source),
        lambda: api.import_record(api.ImportRequest(path=source, dest=destination)),
    ):
        with pytest.raises(contracts.RecordInvalid) as refused:
            read()
        assert refused.value.code == "RL0604"
        assert refused.value.context["declared_assay_id"] == declared
        assert refused.value.context["computed_assay_id"] == expected
        assert refused.value.context["original_bytes"] == raw
    assert not destination.exists()


def test_validated_bundle_reads_only_the_record_and_never_executes_members(tmp_path: Path) -> None:
    from reward_lens.product.ingest import ingest

    bundle = tmp_path / "received.zip"
    marker = tmp_path / "executed"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("record.assay.json", _record_bytes())
        archive.writestr("untrusted.py", f"open({str(marker)!r}, 'w').write('bad')")

    received = ingest(bundle)

    assert received.bundle is not None
    assert received.bundle.members[0].crc32.startswith("crc32:")
    assert received.record.assay_id
    assert not marker.exists()


def test_export_bundle_directory_uses_the_fixed_record_member(tmp_path: Path) -> None:
    from reward_lens.product.ingest import ingest

    bundle = tmp_path / "export-bundle"
    bundle.mkdir()
    (bundle / "record.json").write_bytes(_record_bytes())
    (bundle / "untrusted.py").write_text(
        "raise RuntimeError('do not execute me')", encoding="utf-8"
    )

    received = ingest(bundle)

    assert received.record.assay_id


def test_sdk_and_cli_verify_a_real_export_bundle_then_save_and_reopen(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from reward_lens import api
    from reward_lens.cli.main import cli
    from reward_lens.product.export import ExportKind, Tier, export

    record = json.loads(_record_bytes())
    bundle = export(record, kind=ExportKind.BUNDLE, tier=Tier.A).write(tmp_path / "bundle")
    saved = tmp_path / "saved"

    sdk = api.import_record(api.ImportRequest(path=bundle, verify=True, dest=saved))
    result = CliRunner().invoke(
        cli,
        ["import", str(bundle), "--verify", "--save-to", str(tmp_path / "cli-saved"), "--json"],
    )

    assert sdk.signer
    assert sdk.record_path.read_bytes() == (bundle / "record.json").read_bytes()
    assert api.open_record(sdk.record_path).assay_id == sdk.record.assay_id
    assert result.exit_code == 2, result.output
    assert sdk.signer in result.output
    assert "identifies the signer" in result.output


def test_changed_export_record_refuses_rl0632_and_keeps_original_bytes(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from reward_lens import api
    from reward_lens.cli.main import cli
    from reward_lens.product.export import ExportKind, Tier, export

    record = json.loads(_record_bytes())
    bundle = export(record, kind=ExportKind.BUNDLE, tier=Tier.A).write(tmp_path / "bundle")
    original = (bundle / "record.json").read_bytes()
    changed = original + b" "
    assert changed != original
    (bundle / "record.json").write_bytes(changed)

    with pytest.raises(Exception) as raised:
        api.import_record(api.ImportRequest(path=bundle, verify=True, dest=tmp_path / "saved"))
    assert getattr(raised.value, "code", None) == "RL0632"
    assert getattr(raised.value, "context", {})["original_bytes"] == changed
    result = CliRunner().invoke(cli, ["import", str(bundle), "--verify", "--json"])
    assert result.exit_code == 4
    assert "RL0632" in result.output


def test_cli_runs_the_typed_ingest_route_and_reports_saved_artifacts(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from reward_lens.cli.main import cli

    source = tmp_path / "received.assay.json"
    destination = tmp_path / "cli-saved"
    source.write_bytes(_record_bytes())
    result = CliRunner().invoke(
        cli, ["import", str(source), "--save-to", str(destination), "--json"]
    )

    assert result.exit_code == 2, result.output
    assert (destination / "imported.assay.json").read_bytes() == source.read_bytes()
    assert (destination / "ingest.receipt.json").is_file()
