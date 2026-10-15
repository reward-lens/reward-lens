"""Controls for the isolated maintainer command surface (D-05, D-19)."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

from reward_lens.cli.main import cli as user_cli
from reward_lens.commons.cli import commons, main
from reward_lens.core.evidence import make_evidence
from reward_lens.core.provenance import Cost, Provenance
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import ModelFP, SubjectRef


def _run(*args: str):
    return CliRunner().invoke(commons, list(args), catch_exceptions=False)


def test_maintainer_help_names_only_the_commons_routes() -> None:
    result = _run("--help")

    assert result.exit_code == 0
    for route in (
        "claims",
        "registry",
        "sweep",
        "reference",
        "proficiency",
        "rgx",
    ):
        assert route in result.output


def test_user_root_help_does_not_leak_maintainer_commands() -> None:
    result = CliRunner().invoke(user_cli, ["--help"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "commons" not in result.output.lower()
    assert "registry build" not in result.output.lower()


def test_claims_resolves_a_real_claim_against_a_nonempty_persisted_store(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    evidence = make_evidence(
        observable="CommonsClaims",
        observable_version="1.0",
        subject=SubjectRef(signals=(ModelFP("mfp:commons-claims"),)),
        value={"estimate": 0.25},
    )
    store.append(evidence)
    page = tmp_path / "claim.md"
    page.write_text(
        f"The recorded estimate is [[claim value=0.25 ev={evidence.id} field=estimate]].\n",
        encoding="utf-8",
    )

    result = _run("claims", str(page), "--store", str(store.path))

    assert store.jsonl.read_text(encoding="utf-8").strip()
    assert result.exit_code == 0
    assert "Claims checked: 1" in result.output
    assert "[ok]" in result.output


def test_claims_rejects_a_persisted_evidence_mismatch(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    evidence = make_evidence(
        observable="CommonsClaims",
        observable_version="1.0",
        subject=SubjectRef(signals=(ModelFP("mfp:commons-claims"),)),
        value={"estimate": 0.25},
    )
    store.append(evidence)
    page = tmp_path / "wrong.md"
    page.write_text(
        f"The recorded estimate is [[claim value=0.5 ev={evidence.id} field=estimate]].\n",
        encoding="utf-8",
    )

    result = _run("claims", str(page), "--store", str(store.path))

    assert result.exit_code != 0
    assert "but stored" in result.output


def test_sweep_reads_a_real_persisted_cost_prior_and_prints_the_nonempty_plan(tmp_path) -> None:
    from reward_lens.artifacts.atlas import Atlas

    atlas = Atlas.standard()
    store = EvidenceStore(tmp_path / "evidence")
    first = atlas.entries[0]
    store.append(
        make_evidence(
            observable="BiasBattery",
            observable_version="1.0",
            subject=SubjectRef(signals=(ModelFP(first.fingerprint),)),
            value={"measured": True},
            provenance=Provenance(cost=Cost(gpu_seconds=4.0)),
        )
    )

    result = _run(
        "sweep", "--battery", "BiasBattery", "--gpu-budget", "11", "--store", str(store.path)
    )

    assert store.jsonl.read_text(encoding="utf-8").strip()
    assert result.exit_code == 0
    assert "10 sweep cells" in result.output
    assert "1 priced from stored cost" in result.output
    assert "estimated 4.0 GPU-seconds" in result.output
    assert "GPU-gated" in result.output


def test_sweep_execution_refuses_after_reporting_the_actual_plan() -> None:
    result = _run("sweep", "--battery", "BiasBattery", "--execute")

    assert result.exit_code != 0
    assert "10 sweep cells" in result.output
    assert "P-REGISTRY" in result.output


def test_reference_build_loads_and_reverifies_the_actual_adopted_material() -> None:
    result = _run("reference", "build")

    assert result.exit_code == 0
    assert "diagnostic-v3" in result.output
    assert "grade=reported" in result.output
    assert "pairs=84" in result.output
    assert "certified" not in result.output.lower()


def test_rgx_allocates_from_the_integrated_rgx_catalogue_without_writing() -> None:
    result = _run("rgx", "alloc", "--year", "2027", "--existing", "RGX-2027-0001")

    assert result.exit_code == 0
    assert result.output.strip() == "RGX-2027-0002"


def test_registry_build_generates_and_reopens_the_provenance_preserving_public_document(
    tmp_path,
) -> None:
    from reward_lens.commons.registry import open_registry

    destination = tmp_path / "registry.json"
    result = _run("registry", "build", "--out", str(destination))
    reopened = open_registry(destination)

    assert result.exit_code == 0
    assert "reopened=yes" in result.output
    assert len(reopened.entries) == 17
    assert any(
        entry.path == "/registry/assays/aisi-olmo-e1/"
        and entry.measured_here
        and entry.provenance.startswith("reward_lens/commons/registry/seeds/e1.assay.json")
        and "privacy-portable derivation of fleet/TEST_EVIDENCE/P-E1-canonical-assay/assay.json"
        in entry.provenance
        for entry in reopened.entries
    )
    migrated = [entry for entry in reopened.entries if entry.migrated]
    assert len(migrated) == 10
    assert all(
        entry.original_scope and entry.refutation and not entry.measured_here for entry in migrated
    )


def test_console_entry_builds_and_reopens_the_default_registry_artifact(
    tmp_path, monkeypatch
) -> None:
    from reward_lens.commons.registry import open_registry

    monkeypatch.chdir(tmp_path)
    assert main(["registry", "build"]) == 0
    reopened = open_registry(tmp_path / "reward-lens-registry.json")

    assert len(reopened.entries) == 17
    assert any(entry.record_type == "assay" and entry.measured_here for entry in reopened.entries)
    assert all(entry.provenance for entry in reopened.entries)


def test_installed_console_script_resolves_the_maintainer_tree() -> None:
    script = Path(sys.executable).parent / "reward-lens-commons"
    result = subprocess.run([str(script), "--help"], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "registry" in result.stdout
    assert "proficiency" in result.stdout


def test_installed_console_script_runs_claims_and_sweep_on_persisted_inputs(tmp_path) -> None:
    from reward_lens.artifacts.atlas import Atlas

    store = EvidenceStore(tmp_path / "evidence")
    claim = make_evidence(
        observable="InstalledClaims",
        observable_version="1.0",
        subject=SubjectRef(signals=(ModelFP("mfp:installed-claims"),)),
        value={"estimate": 0.5},
    )
    store.append(claim)
    atlas_entry = Atlas.standard().entries[0]
    store.append(
        make_evidence(
            observable="InstalledSweep",
            observable_version="1.0",
            subject=SubjectRef(signals=(ModelFP(atlas_entry.fingerprint),)),
            value={"measured": True},
            provenance=Provenance(cost=Cost(gpu_seconds=3.0)),
        )
    )
    page = tmp_path / "claim.md"
    page.write_text(
        f"The retained estimate is [[claim value=0.5 ev={claim.id} field=estimate]].\n",
        encoding="utf-8",
    )
    script = Path(sys.executable).parent / "reward-lens-commons"

    claims = subprocess.run(
        [str(script), "claims", str(page), "--store", str(store.path)],
        capture_output=True,
        text=True,
        check=False,
    )
    sweep = subprocess.run(
        [
            str(script),
            "sweep",
            "--battery",
            "InstalledSweep",
            "--gpu-budget",
            "4",
            "--store",
            str(store.path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert store.jsonl.read_text(encoding="utf-8").count("\n") == 2
    assert claims.returncode == 0
    assert "Claims checked: 1" in claims.stdout
    assert sweep.returncode == 0
    assert "1 priced from stored cost" in sweep.stdout
    assert "estimated 3.0 GPU-seconds" in sweep.stdout

    conflicting = make_evidence(
        observable="InstalledClaims",
        observable_version="1.0",
        subject=SubjectRef(signals=(ModelFP("mfp:installed-claims-second-run"),)),
        value={"estimate": 0.6},
    )
    store.append(conflicting)
    conflicted_page = tmp_path / "conflicted.md"
    conflicted_page.write_text(
        "\n".join(
            (
                f"[[claim value=0.5 ev={claim.id} field=estimate]]",
                f"[[claim value=0.6 ev={conflicting.id} field=estimate]]",
            )
        ),
        encoding="utf-8",
    )
    independent_page = tmp_path / "independent.md"
    independent_page.write_text("Narrative only.\n", encoding="utf-8")
    provenance_failure = subprocess.run(
        [
            str(script),
            "claims",
            str(conflicted_page),
            str(independent_page),
            "--store",
            str(store.path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert provenance_failure.returncode != 0
    assert "different evidence ids" in provenance_failure.stdout


def test_proficiency_run_uses_a_previously_sealed_charter_and_separates_outputs(tmp_path) -> None:
    from reward_lens.reference.proficiency import publish_local_charter

    charter = publish_local_charter(tmp_path / "charter.json")
    public = tmp_path / "public.json"
    private = tmp_path / "private.json"

    result = _run(
        "proficiency",
        "run",
        "--charter",
        str(charter.path),
        "--public-output",
        str(public),
        "--private-output",
        str(private),
    )

    projection = json.loads(public.read_text(encoding="utf-8"))
    identified = json.loads(private.read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert "actual_methods=2" in result.output
    assert "external_bucket=five_external_unavailable" in result.output
    assert "four_way=unavailable" in result.output
    assert set(projection["coded_distribution"]) == {"D-REC", "D-BBX"}
    assert projection["external_rows"] == [
        {
            "bucket": "five_external_unavailable",
            "code": "EXT-UNAVAILABLE",
            "status": "external participation prepared but unavailable",
        }
    ]
    assert "participant" not in public.read_text(encoding="utf-8")
    assert "participant" in identified["identified_reports"]["D-REC"]
    assert stat.S_IMODE(private.stat().st_mode) == 0o600


def test_proficiency_run_refuses_to_collapse_public_and_private_destinations(tmp_path) -> None:
    from reward_lens.reference.proficiency import publish_local_charter

    charter = publish_local_charter(tmp_path / "charter.json")
    result = _run(
        "proficiency",
        "run",
        "--charter",
        str(charter.path),
        "--public-output",
        str(tmp_path / "one.json"),
        "--private-output",
        str(tmp_path / "one.json"),
    )

    assert result.exit_code != 0
    assert "must be distinct" in result.output


def test_installed_console_script_runs_proficiency_and_keeps_its_private_output_private(
    tmp_path,
) -> None:
    from reward_lens.reference.proficiency import publish_local_charter

    charter = publish_local_charter(tmp_path / "charter.json")
    public = tmp_path / "public.json"
    private = tmp_path / "private.json"
    script = Path(sys.executable).parent / "reward-lens-commons"

    completed = subprocess.run(
        [
            str(script),
            "proficiency",
            "run",
            "--charter",
            str(charter.path),
            "--public-output",
            str(public),
            "--private-output",
            str(private),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    collapsed = subprocess.run(
        [
            str(script),
            "proficiency",
            "run",
            "--charter",
            str(charter.path),
            "--public-output",
            str(tmp_path / "same.json"),
            "--private-output",
            str(tmp_path / "same.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    projection = json.loads(public.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert "actual_methods=2" in completed.stdout
    assert "four_way=unavailable" in completed.stdout
    assert set(projection["coded_distribution"]) == {"D-REC", "D-BBX"}
    assert "participant" not in public.read_text(encoding="utf-8")
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert collapsed.returncode != 0
    assert "must be distinct" in collapsed.stderr
