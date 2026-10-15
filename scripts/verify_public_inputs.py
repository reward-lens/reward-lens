#!/usr/bin/env python3
"""Seal and verify the small public regression-input artifact used by CI tests.

The raw pinned downloads never enter this artifact. The derivation job must run
the source-matched builder before sealing it; this checker independently binds
the builder receipts, exact file set, and artifact bytes to a CI source/run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

MANIFEST = "PUBLIC_INPUTS.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git_value(revision: str) -> str:
    return subprocess.check_output(["git", "rev-parse", revision], text=True).strip()


def checked_digest(value: object) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("invalid SHA-256 in public-input receipt")
    return value


def validate(root: Path, spec: dict, *, sealed: bool) -> dict[str, str]:
    e1_spec, campaign_spec = spec["e1"], spec["campaign"]
    e1_receipt = json.loads((root / "e1/receipt.json").read_text())
    campaign_receipt = json.loads((root / "campaign/receipt.json").read_text())
    if e1_receipt.get("schema") != "reward-lens-e1-public-join/1":
        raise ValueError("unexpected E1 derivation receipt schema")
    if (e1_receipt.get("source_dataset"), e1_receipt.get("source_revision")) != (
        e1_spec["dataset"],
        e1_spec["revision"],
    ):
        raise ValueError("E1 public source identity changed")
    if e1_receipt.get("historical_extraction_sha256") is not None:
        raise ValueError("derived E1 data must not claim the historical extraction digest")
    if e1_receipt.get("output") != "public_join_v1.parquet":
        raise ValueError("unexpected E1 output filename")
    if e1_receipt.get("counts") != e1_spec["counts"]:
        raise ValueError("E1 seven-count regression mismatch")
    if e1_receipt.get("rows") != e1_spec["expected_rows"]:
        raise ValueError("E1 row count mismatch")
    if e1_receipt.get("analysis_sha256") != {
        f"e1_analysis/{name}": sha for name, sha in e1_spec["analysis"].items()
    }:
        raise ValueError("E1 analysis provenance mismatch")
    inputs = e1_receipt.get("inputs_sha256")
    if not isinstance(inputs, dict) or len(inputs) != e1_spec["expected_eval_files"] + 1:
        raise ValueError("E1 public input closure mismatch")
    if inputs.get(e1_spec["parquet"]["path"]) != e1_spec["parquet"]["sha256"]:
        raise ValueError("E1 pinned parquet digest mismatch")
    for name, value in inputs.items():
        if name != e1_spec["parquet"]["path"] and not re.fullmatch(r"eval_logs/[^/]+\.eval", name):
            raise ValueError("unexpected E1 source input path")
        checked_digest(value)
    if campaign_receipt.get("schema") != "reward-lens-campaign-public-rebuild/1":
        raise ValueError("unexpected campaign derivation receipt schema")
    if campaign_receipt.get("generator_commit") != campaign_spec["generator_local_git_commit"]:
        raise ValueError("campaign converter provenance mismatch")
    if campaign_receipt.get("rb2_source_sha256") != campaign_spec["rewardbench2"]["sha256"]:
        raise ValueError("pinned RewardBench 2 source mismatch")
    outputs = campaign_receipt.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(campaign_spec["bank_sha256"]):
        raise ValueError("campaign bank set mismatch")

    expected = {"e1/receipt.json", "e1/public_join_v1.parquet", "campaign/receipt.json"}
    expected.update(f"e1/e1_analysis/{name}" for name in e1_spec["analysis"])
    for name, record in outputs.items():
        if not isinstance(record, dict) or record.get("file") != f"{name}.jsonl":
            raise ValueError("campaign bank filename mismatch")
        if record.get("sha256") != campaign_spec["bank_sha256"][name]:
            raise ValueError("campaign bank digest declaration mismatch")
        if record.get("content_hash") != campaign_spec["content_hash"][name]:
            raise ValueError("campaign bank content hash mismatch")
        if not isinstance(record.get("rows"), int) or record["rows"] <= 0:
            raise ValueError("campaign bank row count missing")
        expected.add(f"campaign/{name}.jsonl")

    found = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if found != expected | ({MANIFEST} if sealed else set()):
        raise ValueError(
            f"public-input artifact file set mismatch: missing={expected - found}, extra={found - expected}"
        )
    for relative in expected:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe public-input artifact path: {relative}")
    if digest(root / "e1/public_join_v1.parquet") != checked_digest(e1_receipt["output_sha256"]):
        raise ValueError("derived E1 parquet digest mismatch")
    for name, declared in e1_spec["analysis"].items():
        if digest(root / "e1/e1_analysis" / name) != declared:
            raise ValueError(f"E1 analysis file changed: {name}")
    for name, declared in campaign_spec["bank_sha256"].items():
        if digest(root / "campaign" / f"{name}.jsonl") != declared:
            raise ValueError(f"campaign bank changed: {name}")
    return {name: digest(root / name) for name in sorted(expected)}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seal", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("root", type=Path)
    parser.add_argument("--spec", type=Path, default=Path("release/public_inputs/sources.json"))
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    spec = json.loads(args.spec.read_text())
    hashes = validate(root, spec, sealed=args.verify)
    identity = {
        "schema": "reward-lens.ci-public-inputs/1",
        "repository": os.environ["GITHUB_REPOSITORY"],
        "source_sha": os.environ["GITHUB_SHA"],
        "source_tree": git_value("HEAD^{tree}"),
        "ci_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "sha256": hashes,
    }
    if git_value("HEAD") != identity["source_sha"]:
        raise ValueError("public inputs are not bound to the checked-out source")
    manifest_path = root / MANIFEST
    if args.seal:
        manifest_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    elif json.loads(manifest_path.read_text()) != identity:
        raise ValueError("public-input manifest, source, run, or bytes mismatch")
    if args.github_env:
        if not args.verify:
            raise ValueError("GITHUB_ENV output requires --verify")
        with args.github_env.open("a") as stream:
            stream.write(f"REWARD_LENS_E1_ASSAY_ROOT={root / 'e1'}\n")
            stream.write(f"REWARD_LENS_E1_DERIVATION_RECEIPT={root / 'e1/receipt.json'}\n")
            stream.write(f"REWARD_LENS_CAMPAIGN_CACHE={root / 'campaign'}\n")
    print(f"verified {len(hashes)} derived public-input files for {identity['source_sha']}")


if __name__ == "__main__":
    main()
