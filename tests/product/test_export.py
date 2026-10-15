"""P-EXPORT's public boundary, exercised with the frozen assay reference record."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "schema" / "assay" / "1.0" / "fixtures" / "reference.json"


def _saved_reference(destination: Path, record: dict | None = None) -> Path:
    """Seal the schema example before testing a public saved-record route."""
    from reward_lens.contracts import digest

    document = json.loads(REFERENCE.read_text(encoding="utf-8")) if record is None else record
    document["assay_id"] = digest(document)
    destination.write_text(json.dumps(document), encoding="utf-8")
    return destination


@pytest.fixture()
def record() -> dict:
    """The schema-owned reference instance, never a hand-written export fixture."""
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def test_every_kind_and_tier_writes_an_artifact(record: dict, tmp_path: Path) -> None:
    from reward_lens.product.export import ExportKind, Tier, export

    for tier in Tier:
        for kind in ExportKind:
            if kind is ExportKind.BADGE:
                # Badge is a public saved-record route: it needs a selected measured
                # Reach entry and its durable file link, neither of which this static
                # schema reference represents.
                continue
            artifact = export(record, kind=kind, tier=tier)
            written = artifact.write(tmp_path / f"{tier.value}-{kind.value}")
            assert written.exists()
            assert artifact.manifest_digest.startswith("sha256:")
            if kind.value == "BUNDLE":
                assert (written / "manifest.json").is_file()
                assert (written / "report.html").is_file()
                assert artifact.manifest_digest in (written / "report.html").read_text(
                    encoding="utf-8"
                )


def test_sarif_is_official_schema_valid_and_orders_before_truncation(record: dict) -> None:
    from reward_lens.product.export import ExportKind, Tier, export, validate_sarif

    sarif = export(record, kind=ExportKind.SARIF, tier=Tier.A).json()
    validate_sarif(sarif)
    results = sarif["runs"][0]["results"]
    levels = [item["level"] for item in results]
    assert levels == sorted(levels, key={"error": 0, "warning": 1, "note": 2}.get)
    assert "dropped_results" in sarif["runs"][0]["properties"]
    assert sarif["runs"][0]["tool"]["driver"]["rules"][0]["help"]
    assert sarif["runs"][0]["tool"]["driver"]["rules"][0]["helpUri"]
    assert results[0]["kind"] == "fail"
    assert results[0]["partialFingerprints"]


def test_sarif_keeps_the_github_visible_top_five_thousand(record: dict) -> None:
    from reward_lens.product.export import ExportKind, Tier, export, validate_sarif

    finding = record["findings"][0]
    record["findings"] = [
        dict(finding, id=f"finding-{index:05d}", partial_fingerprint=f"fingerprint-{index:05d}")
        for index in range(5_001)
    ]
    sarif = export(record, kind=ExportKind.SARIF, tier=Tier.A).json()
    validate_sarif(sarif)
    run = sarif["runs"][0]
    assert len(run["results"]) == 5_000
    assert run["properties"]["dropped_results"] == 1
    assert run["properties"]["per_run_limit"] == 25_000


def test_bundle_honours_the_recorded_embedding_tier(record: dict) -> None:
    from reward_lens.product.export import ExportKind, Tier, export

    for declared in Tier:
        candidate = json.loads(json.dumps(record))
        candidate["embedding"]["tier"] = declared.value
        if declared is Tier.B:
            candidate["embedding"]["omitted_tables"] = [
                table["id"] for table in candidate["tables"]
            ]
        artifact = export(candidate, kind=ExportKind.BUNDLE, tier=Tier.A)
        assert artifact.tier is declared
        assert artifact.manifest["export"]["tier"] == declared.value


def test_scanner_refuses_with_file_line_and_redacted_path() -> None:
    from reward_lens.product.export import ExportSecretFound, scan

    hits = scan(b'{"token":"sk-test-secret-1234567890"}\n', file="record.json")
    assert hits and hits[0].file == "record.json" and hits[0].line == 1
    with pytest.raises(ExportSecretFound) as refused:
        raise ExportSecretFound(hits=hits, redacted_path="record.redacted.json")
    assert refused.value.code == "RL0641"
    assert "record.json" in refused.value.message
    assert "record.redacted.json" in refused.value.remediation


def test_redaction_is_a_new_linked_artifact(record: dict) -> None:
    from reward_lens.product.export import ExportKind, Tier, export, redacted_export

    original = export(record, kind=ExportKind.BUNDLE, tier=Tier.A)
    replacement = redacted_export(original, replacement=b'{"redacted":true}\n')
    assert replacement.manifest_digest != original.manifest_digest
    assert replacement.linked_from == original.manifest_digest


def test_self_referential_html_refuses(record: dict) -> None:
    from reward_lens.product.export import ExportKind, ExportSelfReference, Tier, export

    record["rendered_html"] = "report.html"
    with pytest.raises(ExportSelfReference) as refused:
        export(record, kind=ExportKind.BUNDLE, tier=Tier.A)
    assert refused.value.code == "RL0642"


def test_saved_schema_valid_self_referential_html_refuses_through_sdk_and_cli(
    record: dict, tmp_path: Path
) -> None:
    from click.testing import CliRunner

    from reward_lens import api
    from reward_lens.cli.main import cli
    from reward_lens.product.export import RENDERED_HTML_EXTENSION, ExportSelfReference

    record["extensions"][RENDERED_HTML_EXTENSION] = {"path": "report.html"}
    saved = _saved_reference(tmp_path / "self-reference.assay.json", record)
    destination = tmp_path / "must-not-exist"

    # The extension is admitted by the frozen Assay schema. Export owns the stricter D-16 rule.
    assert api.open_record(saved).extensions[RENDERED_HTML_EXTENSION]["path"] == "report.html"
    with pytest.raises(ExportSelfReference) as refused:
        api.export(api.ExportRequest(record=saved, dest=destination, format="bundle"))
    assert refused.value.code == "RL0642"
    assert refused.value.exit_code == 4
    assert refused.value.context == {"value": "report.html"}
    assert not destination.exists()

    cli_destination = tmp_path / "cli-must-not-exist"
    result = CliRunner().invoke(
        cli,
        [
            "export",
            "--record",
            str(saved),
            "--dest",
            str(cli_destination),
            "--kind",
            "bundle",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 4, result.output
    receipt = json.loads(result.stdout)
    assert receipt["error"]["code"] == "RL0642"
    assert receipt["error"]["context"] == {"value": "report.html"}
    assert not cli_destination.exists()


def test_offline_dsse_verification_disclaims_the_outcome(record: dict, tmp_path: Path) -> None:
    from reward_lens.product.export import ExportKind, Tier, export, verify

    bundle = export(record, kind=ExportKind.BUNDLE, tier=Tier.A)
    written = bundle.write(tmp_path / "bundle")
    result = verify(written)
    assert result.valid is True
    assert result.signer
    assert "identifies the signer" in result.message
    assert "nothing about whether the outcome check was right" in result.message


def test_offline_verification_rejects_any_tampered_bundle_member(
    record: dict, tmp_path: Path
) -> None:
    from reward_lens.product.export import ExportKind, Tier, export, verify

    bundle = export(record, kind=ExportKind.BUNDLE, tier=Tier.A)
    original = tmp_path / "original"
    bundle.write(original)
    assert verify(original).valid is True

    def copied(name: str) -> Path:
        target = tmp_path / name
        target.mkdir()
        for member in original.iterdir():
            (target / member.name).write_bytes(member.read_bytes())
        return target

    modified_record = copied("record")
    (modified_record / "record.json").write_text('{"tampered":true}\n', encoding="utf-8")
    assert verify(modified_record).valid is False

    modified_manifest = copied("manifest")
    manifest = (modified_manifest / "manifest.json").read_bytes()
    (modified_manifest / "manifest.json").write_bytes(b" " + manifest)
    assert verify(modified_manifest).valid is False

    for name, field, replacement in (
        ("payload", "payload", base64.b64encode(b"{}").decode("ascii")),
        ("signature", "sig", base64.b64encode(b"0" * 64).decode("ascii")),
        ("public-key", "publicKey", base64.b64encode(b"0" * 32).decode("ascii")),
        ("key-id", "keyid", "sha256:" + "0" * 64),
        ("signer", "signer", "ed25519:sha256:" + "0" * 64),
    ):
        target = copied(name)
        envelope = json.loads((target / "attestation.dsse.json").read_text(encoding="utf-8"))
        if field == "payload":
            envelope[field] = replacement
        else:
            envelope["signatures"][0][field] = replacement
        (target / "attestation.dsse.json").write_text(json.dumps(envelope), encoding="utf-8")
        assert verify(target).valid is False, name


def test_sdk_reopens_a_saved_audit_and_writes_a_verifiable_bundle(tmp_path: Path) -> None:
    from reward_lens import api
    from reward_lens.examples.code_reward import write as write_example
    from reward_lens.product.export import verify

    project = tmp_path / "project"
    write_example(project)
    api.audit(api.AuditRequest(path=project, name="export-source"))
    saved = next((project / "assays").glob("*.assay.json"))

    destination = tmp_path / "exported-bundle"
    artifact = api.export(
        api.ExportRequest(record=saved, dest=destination, format="bundle", tier="A")
    )

    assert artifact.manifest_digest in (destination / "report.html").read_text(encoding="utf-8")
    assert verify(destination).valid is True


def test_fresh_audit_and_reopened_assay_render_identical_bytes(tmp_path: Path) -> None:
    from reward_lens import api, contracts
    from reward_lens.examples.code_reward import write as write_example
    from reward_lens.render.report import render

    project = tmp_path / "project"
    write_example(project)
    in_memory = api.audit(api.AuditRequest(path=project, name="render-roundtrip"))
    saved = next((project / "assays").glob("*.assay.json"))
    reopened = api.open_record(saved)
    manifest = json.loads((project / "assays" / "manifest.json").read_text(encoding="utf-8"))
    manifest_digest = contracts.digest(manifest)

    assert render(in_memory, bundle_manifest_digest=manifest_digest) == render(
        reopened, bundle_manifest_digest=manifest_digest
    )


def test_cli_reopens_a_saved_audit_writes_a_bundle_and_reports_its_receipt(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from reward_lens import api
    from reward_lens.cli.main import cli
    from reward_lens.examples.code_reward import write as write_example
    from reward_lens.product.export import verify

    project = tmp_path / "project"
    write_example(project)
    api.audit(api.AuditRequest(path=project, name="cli-export-source"))
    saved = next((project / "assays").glob("*.assay.json"))
    destination = tmp_path / "cli-exported-bundle"

    result = CliRunner().invoke(
        cli,
        [
            "export",
            "--record",
            str(saved),
            "--dest",
            str(destination),
            "--kind",
            "bundle",
            "--tier",
            "A",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["command"] == "export"
    assert receipt["artifacts"]["export"] == str(destination)
    assert receipt["artifacts"]["report_url"] == (destination / "report.html").resolve().as_uri()
    assert receipt["artifacts"]["manifest_digest"] in (destination / "report.html").read_text(
        encoding="utf-8"
    )
    assert verify(destination).valid is True


def test_cli_export_requires_a_saved_record_and_destination() -> None:
    from click.testing import CliRunner

    from reward_lens.cli.main import cli

    result = CliRunner().invoke(cli, ["export", "--format", "json"])

    assert result.exit_code == 4
    assert "--record" in result.output


def test_sdk_export_without_a_record_uses_the_generic_invalid_input_refusal() -> None:
    from reward_lens import api, contracts

    with pytest.raises(contracts.UsageError) as refused:
        api.export(api.ExportRequest())

    assert refused.value.code == "RL0003"


@pytest.mark.parametrize("kind", ["html", "sarif", "revision", "bundle"])
def test_cli_reopens_the_saved_record_for_every_export_kind(kind: str, tmp_path: Path) -> None:
    from click.testing import CliRunner

    from reward_lens.cli.main import cli
    from reward_lens.product.export import verify

    destination = tmp_path / ("bundle" if kind == "bundle" else f"export.{kind}")
    saved = _saved_reference(tmp_path / "reference.assay.json")
    result = CliRunner().invoke(
        cli,
        [
            "export",
            "--record",
            str(saved),
            "--dest",
            str(destination),
            "--kind",
            kind,
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["artifacts"]["kind"] == kind
    assert destination.exists()
    if kind == "bundle":
        assert verify(destination).valid is True


def test_badge_refuses_a_saved_record_without_one_measured_reach_entry(tmp_path: Path) -> None:
    from reward_lens import api
    from reward_lens.commons.badge import BadgeInputError

    saved = _saved_reference(tmp_path / "reference.assay.json")
    with pytest.raises(BadgeInputError, match="unavailable"):
        api.export(api.ExportRequest(record=saved, dest=tmp_path / "badge.json", format="badge"))


def test_sdk_and_cli_export_reopen_one_real_saved_audit_reach_as_badge_json(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from reward_lens import api
    from reward_lens.cli.main import cli
    from reward_lens.commons.badge import BadgeInputError, read_payload
    from reward_lens.examples.code_reward import write as write_example

    project = tmp_path / "project"
    write_example(project)
    audited = api.audit(api.AuditRequest(path=project, name="badge-export-source"))
    saved = next((project / "assays").glob("*.assay.json"))
    reach = next(entry for entry in audited.measurement.reach if entry.kind == "estimate")

    with pytest.raises(BadgeInputError, match="multiple"):
        api.export(
            api.ExportRequest(record=saved, dest=tmp_path / "ambiguous.json", format="badge")
        )

    sdk_destination = tmp_path / "sdk-badge.json"
    artifact = api.export(
        api.ExportRequest(
            record=saved,
            dest=sdk_destination,
            format="badge",
            badge_reach_entry_id=reach.entry_id,
        )
    )
    sdk_payload = read_payload(sdk_destination)
    assert artifact.primary == "badge.json"
    assert sdk_payload["link"] == saved.resolve().as_uri()
    assert sdk_payload["provenance"]["reach_entry_id"] == reach.entry_id
    assert sdk_payload["provenance"]["reach_surface"] == reach.result["surface"]
    assert sdk_payload["record"]["finding_ids"] == []

    cli_destination = tmp_path / "cli-badge.json"
    result = CliRunner().invoke(
        cli,
        [
            "export",
            "--record",
            str(saved),
            "--dest",
            str(cli_destination),
            "--kind",
            "badge",
            "--reach-entry",
            reach.entry_id,
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["artifacts"]["kind"] == "badge"
    assert read_payload(cli_destination) == sdk_payload


@pytest.mark.parametrize("requested_tier", ["B", "C"])
def test_cli_does_not_relabel_a_saved_record_with_a_requested_tier(
    requested_tier: str, tmp_path: Path
) -> None:
    from click.testing import CliRunner

    from reward_lens.cli.main import cli

    saved = _saved_reference(tmp_path / "reference.assay.json")
    result = CliRunner().invoke(
        cli,
        [
            "export",
            "--record",
            str(saved),
            "--dest",
            str(tmp_path / f"export-{requested_tier}.json"),
            "--kind",
            "revision",
            "--tier",
            requested_tier,
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["artifacts"]["tier"] == "A"


def test_sigstore_needs_the_declared_extra(record: dict) -> None:
    from reward_lens.product.export import ExportKind, ExtraRequiredError, Tier, export

    with pytest.raises(ExtraRequiredError) as refused:
        export(record, kind=ExportKind.BUNDLE, tier=Tier.A, backend="sigstore")
    assert refused.value.code == "RL0704"


def test_verifying_commands_are_literal_and_executable() -> None:
    verifying = (ROOT / "VERIFYING.md").read_text(encoding="utf-8")
    assert "reward-lens import --verify" in verifying
    assert "sigstore verify identity" in verifying
