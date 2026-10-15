"""P-REGISTRY: public, provenance-preserving registry generation."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import pytest


def test_default_registry_is_generated_from_sources_then_saved_and_reopened(tmp_path: Path) -> None:
    from reward_lens.commons.registry import (
        build_default_registry,
        open_registry,
        site_projection,
        write_registry,
    )

    root = Path(__file__).parents[2]
    fixture_paths = sorted((root / "tests/migration/fixtures/campaign-v3-cards").glob("*.html"))
    before = {path.name: path.read_bytes() for path in fixture_paths}
    registry = build_default_registry()
    destination = write_registry(registry, tmp_path / "registry.json")
    reopened = open_registry(destination)

    assert destination.is_file()
    assert reopened == registry
    assert {path.name: path.read_bytes() for path in fixture_paths} == before
    migrated = [entry for entry in reopened.entries if entry.migrated]
    assert len(migrated) == 10
    assert all(entry.state == "migrated" and not entry.measured_here for entry in migrated)
    assert all(entry.original_scope and entry.refutation for entry in migrated)
    assert any(entry.record_type == "assay" and entry.measured_here for entry in reopened.entries)
    assert any(
        entry.record_type == "public_record" and not entry.measured_here and entry.primary_link
        for entry in reopened.entries
    )
    site = site_projection(reopened)
    assert set(site) == {"version", "supportedFilters", "entries"}
    assert set(site["supportedFilters"]) == {"states", "recordTypes"}
    assert site["entries"]
    expected_site_fields = {
        "path",
        "title",
        "summary",
        "state",
        "recordType",
        "scope",
        "provenance",
        "assessment",
        "sources",
        "primaryLink",
        "originalScope",
        "refutation",
        "assessments",
    }
    assert all(set(entry) == expected_site_fields for entry in site["entries"])
    assert all(entry["assessment"] and entry["sources"] for entry in site["entries"])
    assert all(
        set(source) == {"label", "url"} for entry in site["entries"] for source in entry["sources"]
    )
    assert all(
        source["url"].startswith("https://")
        or source["url"].startswith(
            ("fleet/", "tests/", "field-scan-2026/", "reward_lens/commons/registry/seeds/")
        )
        for entry in site["entries"]
        for source in entry["sources"]
    )
    public = [entry for entry in site["entries"] if entry["recordType"] == "public_record"]
    assert all(
        entry["primaryLink"] and entry["primaryLink"].startswith("https://arxiv.org/")
        for entry in public
    )
    assert all(
        "authoritative X8 dossier sha256:0dcbcf05d420d2099170ee7338dfa7e2ff305f88d0fe286aec495a9112d99c42"
        in entry["provenance"]
        and "section c.7.3, lines 1047-1072" in entry["provenance"]
        for entry in public
    )
    migrated_site = [entry for entry in site["entries"] if entry["recordType"] == "migrated_card"]
    assert all(entry["originalScope"] and entry["refutation"] for entry in migrated_site)
    assert not (
        {"id", "identifier", "internalId", "studyId", "rank", "ranking", "score", "upload"}
        & set(str(site))
    )


def test_default_registry_never_reads_a_repository_root_at_runtime() -> None:
    import sys

    import reward_lens.commons.registry as registry_module

    x8_before = sys.modules.get("reward_lens.experiments.x8_leaderboard_meta")
    registry = registry_module.build_default_registry()

    assert not hasattr(registry_module, "_repo_root")
    assert sys.modules.get("reward_lens.experiments.x8_leaderboard_meta") is x8_before
    assert len(registry.entries) == 17


def _checkout_seed_fixture(tmp_path: Path) -> Path:
    root = Path(__file__).parents[2]
    manifest = json.loads(
        resources.files("reward_lens.commons.registry")
        .joinpath("seeds", "manifest.json")
        .read_bytes()
    )
    e1 = tmp_path / manifest["e1"]["source"]
    e1.parent.mkdir(parents=True)
    e1.write_bytes(
        (root / "src/reward_lens/commons/registry/checkout-fixtures/e1.assay.json").read_bytes()
    )
    cards = Path("tests/migration/fixtures/campaign-v3-cards")
    (tmp_path / cards).mkdir(parents=True)
    for name in manifest["campaign_cards"]:
        (tmp_path / cards / name).write_bytes((root / cards / name).read_bytes())
    return tmp_path


def test_explicit_checkout_seed_injection_stays_byte_bound_to_packaged_defaults(
    tmp_path: Path,
) -> None:
    from reward_lens.commons.registry import build_default_registry

    root = _checkout_seed_fixture(tmp_path)
    packaged = build_default_registry()
    injected = build_default_registry(root=root)

    assert [(entry.path, entry.source_projection_digest) for entry in injected.entries] == [
        (entry.path, entry.source_projection_digest) for entry in packaged.entries
    ]


@pytest.mark.parametrize("seed", ["e1", "campaign_card"])
def test_explicit_checkout_seed_injection_refuses_changed_bytes(tmp_path: Path, seed: str) -> None:
    from reward_lens.commons.registry import build_default_registry

    root = _checkout_seed_fixture(tmp_path)
    manifest = json.loads(
        resources.files("reward_lens.commons.registry")
        .joinpath("seeds", "manifest.json")
        .read_bytes()
    )
    relative = (
        Path(manifest["e1"]["source"])
        if seed == "e1"
        else Path("tests/migration/fixtures/campaign-v3-cards")
        / sorted(manifest["campaign_cards"])[0]
    )
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n ")

    with pytest.raises(ValueError, match="registry injected seed .* digest mismatch"):
        build_default_registry(root=root)


def test_packaged_seed_manifest_authenticates_every_default_input() -> None:
    seed_root = resources.files("reward_lens.commons.registry").joinpath("seeds")
    manifest = json.loads(seed_root.joinpath("manifest.json").read_bytes())

    assert (
        hashlib.sha256(seed_root.joinpath("e1.assay.json").read_bytes()).hexdigest()
        == manifest["e1"]["sha256"]
    )
    cards = manifest["campaign_cards"]
    assert len(cards) == 10
    assert {
        name: hashlib.sha256(seed_root.joinpath("campaign-v3-cards", name).read_bytes()).hexdigest()
        for name in cards
    } == cards
    assert (
        hashlib.sha256(seed_root.joinpath("public-records.json").read_bytes()).hexdigest()
        == manifest["public_records"]["sha256"]
    )


def test_public_record_seed_matches_the_authoritative_x8_extraction_definition() -> None:
    """The package declaration is a checked transcription, never a local X8 result."""
    from reward_lens.experiments import x8_leaderboard_meta as x8

    seed_root = resources.files("reward_lens.commons.registry").joinpath("seeds")
    manifest = json.loads(seed_root.joinpath("manifest.json").read_bytes())
    rows = json.loads(seed_root.joinpath("public-records.json").read_bytes())

    assert manifest["public_records"]["authoritative_x8"] == {
        "dossier_sha256": x8.DOSSIER_SHA256,
        "dossier_section": x8.DOSSIER_SECTION,
    }
    assert rows == [
        {
            "key": extraction.key,
            "arxiv": extraction.arxiv,
            "authors": extraction.authors,
            "benchmark": extraction.benchmark,
            "estimand": extraction.estimand,
            "sources": [list(source) for source in extraction.sources],
            "status": extraction.status,
        }
        for extraction in x8.SIX
    ]


def test_public_record_generation_refuses_mismatched_authoritative_x8_identity(monkeypatch) -> None:
    import reward_lens.commons.registry as registry_module

    manifest = dict(registry_module._seed_manifest())
    public = dict(manifest["public_records"])
    public["authoritative_x8"] = {
        "dossier_sha256": "0" * 64,
        "dossier_section": "c.7.3, lines 1047-1072",
    }
    manifest["public_records"] = public
    monkeypatch.setattr(registry_module, "_seed_manifest", lambda: manifest)

    with pytest.raises(ValueError, match="authoritative X8"):
        registry_module._public_record_entries()


def test_open_registry_refuses_a_serialized_migrated_entry_that_loses_its_refutation(
    tmp_path: Path,
) -> None:
    from reward_lens.commons.registry import build_default_registry, open_registry

    value = build_default_registry().as_dict()
    migrated = next(entry for entry in value["entries"] if entry["migrated"])
    migrated["refutation"] = None
    path = tmp_path / "broken-registry.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="original scope and refutation"):
        open_registry(path)


def test_registry_refuses_missing_provenance_incomplete_migration_and_one_sided_dispute() -> None:
    from reward_lens.commons.registry import Assessment, RegistryEntry

    common = {
        "path": "/registry/example/",
        "title": "Example",
        "summary": "A registry example.",
        "state": "reported",
        "record_type": "assay",
        "source_projection_digest": "sha256:" + "1" * 64,
        "measurement_scope": "fixture",
        "provenance": "fixture source",
        "measured_here": True,
    }
    with pytest.raises(ValueError, match="provenance"):
        RegistryEntry(**{**common, "provenance": ""})
    with pytest.raises(ValueError, match="migrated"):
        RegistryEntry(**{**common, "migrated": True})
    with pytest.raises(ValueError, match="both"):
        RegistryEntry(
            **{
                **common,
                "assessments": (
                    Assessment(kind="original", text="retained original", source="fixture"),
                ),
            }
        )


def test_disputed_entry_keeps_original_and_author_reaudit_readable_after_reopen(
    tmp_path: Path,
) -> None:
    from reward_lens.commons.registry import (
        Assessment,
        RegistryDocument,
        RegistryEntry,
        open_registry,
        site_projection,
        write_registry,
    )

    entry = RegistryEntry(
        path="/registry/disputed-example/",
        title="Synthetic contract fixture",
        summary="A synthetic fixture exercises paired-assessment retention without a public dispute claim.",
        state="disputed",
        record_type="assay",
        source_projection_digest="sha256:" + "2" * 64,
        measurement_scope="fixture",
        provenance="tests/commons/test_registry.py",
        measured_here=True,
        assessments=(
            Assessment(
                kind="original", text="Original finding.", source="tests/commons/test_registry.py"
            ),
            Assessment(
                kind="author_reaudit",
                text="Author re-audit.",
                source="tests/commons/test_registry.py",
            ),
        ),
    )
    path = write_registry(RegistryDocument(entries=(entry,)), tmp_path / "disputed.json")
    reopened = open_registry(path)

    assert [assessment.kind for assessment in reopened.entries[0].assessments] == [
        "original",
        "author_reaudit",
    ]
    assert [assessment.text for assessment in reopened.entries[0].assessments] == [
        "Original finding.",
        "Author re-audit.",
    ]
    site_entry = site_projection(reopened)["entries"][0]
    assert (
        site_entry["assessment"] == "original: Original finding. author_reaudit: Author re-audit."
    )
    assert site_entry["assessments"] == [
        {
            "kind": "original",
            "text": "Original finding.",
            "source": "tests/commons/test_registry.py",
        },
        {
            "kind": "author_reaudit",
            "text": "Author re-audit.",
            "source": "tests/commons/test_registry.py",
        },
    ]
    assert site_entry["sources"] == [
        {"label": "Checked-in source artifact", "url": "tests/commons/test_registry.py"}
    ]


def test_maintainer_cli_builds_the_real_registry_then_reopens_its_saved_artifact(
    tmp_path: Path,
) -> None:
    from click.testing import CliRunner

    from reward_lens.commons.cli import commons
    from reward_lens.commons.registry import open_registry

    output = tmp_path / "registry.json"
    result = CliRunner().invoke(
        commons, ["registry", "build", "--out", str(output)], catch_exceptions=False
    )

    assert result.exit_code == 0
    assert "reopened=yes" in result.output
    assert len(open_registry(output).entries) == 17
