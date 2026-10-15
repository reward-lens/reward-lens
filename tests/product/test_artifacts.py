"""Canonical persisted bundle contract, exercised through its exported writer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reward_lens import contracts
from reward_lens.api._record import absent_record
from reward_lens.product.audit.run import persist_assay_bundle


def test_exported_writer_seals_one_assay_manifest_and_report(tmp_path: Path) -> None:
    record = absent_record(
        command="artifact-contract",
        reproduce=["reward-lens artifact-contract"],
        path=tmp_path / "source.jsonl",
    )

    persisted, paths = persist_assay_bundle(
        record,
        project=None,
        name="artifact-contract",
        fallback_dir=tmp_path / "assays",
    )

    assert persisted is record
    assert paths["assay"].is_file()
    assert paths["manifest"].is_file()
    assert paths["report"].is_file()
    reopened = contracts.Assay.model_validate(
        json.loads(paths["assay"].read_text(encoding="utf-8"))
    )
    assert reopened.assay_id == persisted.assay_id
    assert contracts.digest(reopened) == persisted.assay_id
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    artifact = manifest["artifacts"][0]
    assert manifest["assay"]["id"] == persisted.assay_id
    assert artifact["path"] == paths["assay"].name
    assert artifact["sha256"] == "sha256:" + hashlib.sha256(paths["assay"].read_bytes()).hexdigest()
    assert persisted.assay_id in paths["report"].read_text(encoding="utf-8")
