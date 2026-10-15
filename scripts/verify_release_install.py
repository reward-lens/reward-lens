#!/usr/bin/env python3
"""Install an exact wheel or public-index release and exercise its offline user flow.

The new environment and outputs live under --work-dir, outside source checkouts.
No editable installation, developer cache, or source-path injection is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import venv
from html.parser import HTMLParser


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=pathlib.Path)
    parser.add_argument("--index-url", default="https://pypi.org/simple")
    parser.add_argument("--version", default="3.1.0")
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    venv.EnvBuilder(with_pip=True).create(root / "venv")
    bindir = root / "venv" / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    cli = bindir / ("reward-lens.exe" if os.name == "nt" else "reward-lens")
    commons = bindir / ("reward-lens-commons.exe" if os.name == "nt" else "reward-lens-commons")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "UV_", "REWARD_LENS_"))}
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"):
        env.pop(name, None)
    env.update(
        PYTHONNOUSERSITE="1",
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        XDG_STATE_HOME=str(root / "state"),
        XDG_CACHE_HOME=str(root / "cache"),
        REWARD_LENS_HOME=str(root / "state/reward-lens"),
        NO_COLOR="1",
        REWARD_LENS_NO_PROGRESS="1",
    )
    results: list[dict] = []

    def run(name: str, argv: list, expected: tuple[int, ...] = (0,), json_output: bool = False):
        p = subprocess.run(
            [str(x) for x in argv], cwd=root, env=env, text=True, capture_output=True, timeout=900
        )
        (root / (name + ".stdout")).write_text(p.stdout)
        (root / (name + ".stderr")).write_text(p.stderr)
        results.append({"check": name, "exit_code": p.returncode, "expected": list(expected)})
        (root / "verification.json").write_text(json.dumps(results, indent=2) + "\n")
        assert p.returncode in expected, f"{name} failed ({p.returncode}); see {root}/{name}.stderr"
        return json.loads(p.stdout) if json_output else p.stdout

    target = str(args.wheel.resolve()) if args.wheel else "reward-lens==" + args.version
    run(
        "install",
        [
            python,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-cache-dir",
            "--index-url",
            args.index_url,
            target,
        ],
    )
    run("pip-check", [python, "-m", "pip", "check"])
    code = """import importlib.metadata as m,importlib.resources as r,json,pathlib,sys
import reward_lens
import reward_lens.artifacts.claims
p=pathlib.Path(reward_lens.__file__).resolve()
assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()) and 'site-packages' in str(p), str(p)
assert 'torch' not in sys.modules
root=r.files('reward_lens')
resources=['schema/assay/1.0/assay.schema.json','spec/QUANTITIES.json',
'skills/reward-lens/SKILL.md','render/report/assets/report.js',
'product/serve/assets/index.html','_vendor/mcp2/MCP2_PROVENANCE.json',
'_vendor/verifiers_writer/save_utils.py','examples/e1/counts.golden.json']
for name in resources: assert root.joinpath(name).is_file(),name
print(json.dumps({'version':m.version('reward-lens'),'runtime':reward_lens.__version__,
'origin':str(p),'requires_python':m.metadata('reward-lens')['Requires-Python'],'resources':resources}))
"""
    info = run("installed-origin", [python, "-I", "-c", code], json_output=True)
    assert info["version"] == info["runtime"] == args.version
    claims_doc = root / "claims-smoke.md"
    claims_doc.write_text("This page contains no measurements.\n")
    default_claims = run("claims-default-store", [commons, "claims", claims_doc])
    assert "Claims checked: 0." in default_claims
    assert (root / "state/reward-lens/store").is_dir()
    store_code = """from pathlib import Path
from reward_lens.core.config import get_settings
from reward_lens.core.evidence import make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import ModelFP, SubjectRef
root = Path.cwd() / 'state/reward-lens'
assert get_settings().home == root
subject = SubjectRef(signals=(ModelFP('mfp:base-install'),),
                     dataset='ds:base-install', readout='reward')
evidence = make_evidence(observable='base.install', observable_version='1',
    subject=subject, value=1.0, provenance=Provenance(),
    created_at='2026-10-15T00:00:00+00:00')
store = EvidenceStore()
assert store.path == root / 'store'
assert store.append(evidence) == evidence.id
assert EvidenceStore().get(evidence.id).value == 1.0
print(evidence.id)
"""
    run("store-round-trip", [python, "-I", "-c", store_code])
    explicit_store = root / "explicit-evidence-store"
    explicit_claims = run(
        "claims-explicit-store", [commons, "claims", "--store", explicit_store, claims_doc]
    )
    assert "Claims checked: 0." in explicit_claims and explicit_store.is_dir()
    claims_doc.write_text("A result cites ev:deadbeef.\n")
    unresolved = run(
        "claims-unresolved-refusal",
        [commons, "claims", "--store", explicit_store, claims_doc],
        expected=(1,),
    )
    assert "referenced but not in the store" in unresolved
    version = run("cli-version", [cli, "--version"])
    assert re.search(r"(?<![0-9.])" + re.escape(args.version) + r"(?![0-9.])", version)
    doctor = run("doctor", [cli, "doctor", "--format", "json"], json_output=True)
    assert isinstance(doctor, dict)
    project = root / "demo"
    run("init", [cli, "init", "--example", "code-reward", project, "--format", "json"])
    audit = run(
        "audit",
        [cli, "audit", project, "--name", "release-smoke", "--offline", "--format", "json"],
        expected=(2,),
        json_output=True,
    )
    records = list((project / "assays").glob("*.assay.json"))
    assert len(records) == 1, records
    record = records[0]
    saved = json.loads(record.read_bytes())
    assert saved["decision"]["state"] == "unresolved"
    assert saved["holes"] and saved["producer"]["version"] == args.version
    report = record.with_suffix(".html")
    assert report.is_file(), report
    before = sha(record)
    reused = run(
        "reuse",
        [cli, "audit", project, "--name", "release-smoke", "--offline", "--format", "json"],
        expected=(2,),
        json_output=True,
    )
    assert sha(record) == before
    assert reused["execution"]["reused"] is not None
    assert reused["decision"] == audit["decision"]

    class AssetParser(HTMLParser):
        external: list[str] = []

        def handle_starttag(self, tag, attributes):
            for key, value in attributes:
                if tag in ("script", "img", "link", "iframe") and key in ("src", "href") and value:
                    if not value.startswith(("data:", "#")):
                        self.external.append(tag + ":" + value)

    assets = AssetParser()
    text = report.read_text()
    assets.feed(text)
    assert not assets.external, assets.external
    match = re.search(r'<script type="application/json" id="assay">(.*?)</script>', text, re.S)
    assert match and json.loads(match.group(1)) == saved
    bundle = root / "bundle"
    run(
        "export",
        [
            cli,
            "export",
            "--record",
            record,
            "--kind",
            "bundle",
            "--dest",
            bundle,
            "--backend",
            "ed25519",
            "--format",
            "json",
        ],
        expected=(0, 2),
    )
    imported = root / "imported"
    run(
        "import",
        [cli, "import", bundle, "--verify", "--save-to", imported, "--format", "json"],
        expected=(2,),
    )
    imported_records = list(imported.rglob("*.assay.json"))
    if not imported_records:
        imported_records = [
            p for p in imported.rglob("*.json") if p.read_bytes() == record.read_bytes()
        ]
    assert imported_records and any(p.read_bytes() == record.read_bytes() for p in imported_records)
    tampered = root / "tampered-bundle"
    shutil.copytree(bundle, tampered)
    candidates = [p for p in tampered.rglob("*.json") if p.read_bytes() == record.read_bytes()]
    assert len(candidates) == 1
    with candidates[0].open("ab") as out:
        out.write(b" ")
    run(
        "tamper-refusal",
        [cli, "import", tampered, "--verify", "--save-to", root / "rejected", "--format", "json"],
        expected=(4,),
    )
    assert "RL0632" in (root / "tamper-refusal.stderr").read_text()
    assert sha(record) == before
    summary = {
        "status": "PASS",
        "version": args.version,
        "source": target,
        "installed": info,
        "record_sha256": before,
        "checks": results,
    }
    (root / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "report": str(root / "verification.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
