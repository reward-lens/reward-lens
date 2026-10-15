#!/usr/bin/env python3
"""Check the exact release archives, including narrow retained fixture closure."""

from __future__ import annotations

import argparse
import pathlib
import tarfile
import zipfile

VERSION = "3.1.0"
WHEEL = f"reward_lens-{VERSION}-py3-none-any.whl"
SDIST = f"reward_lens-{VERSION}.tar.gz"
REQUIRED_WHEEL = (
    "reward_lens/schema/assay/1.0/assay.schema.json",
    "reward_lens/spec/QUANTITIES.json",
    "reward_lens/skills/reward-lens/SKILL.md",
    "reward_lens/render/report/assets/report.js",
    "reward_lens/product/serve/assets/index.html",
    "reward_lens/_vendor/mcp2/MCP2_PROVENANCE.json",
    "reward_lens/examples/e1/counts.golden.json",
)
# These source-only files serve retained regression tests. None belongs in the wheel.
ALLOWED_FLEET = {
    "fleet/ALLOWED_INCOMPLETE.md",
    "fleet/CLI_COMMANDS.json",
    "fleet/port/SLICES.json",
    "fleet/gates/fixtures/controls/g39/positive/sdk.assay.json",
    "fleet/golden/CURRENT_WAVE",
    "fleet/golden/DEMO_NUMBERS.md",
    "fleet/golden/VOLATILE.md",
    "fleet/golden/run_demo.py",
    "fleet/golden/candidates/current/doctor.txt",
    "fleet/golden/candidates/current/root-help.txt",
    "fleet/golden/current/trace-fixture.txt",
    "fleet/golden/target/init-detect.txt",
    "fleet/golden/target/root-help.txt",
    "fleet/golden/target/error-lines.txt",
    "fleet/golden/wave-1/audit-unfamiliar.txt",
    "fleet/golden/wave-1/doctor.txt",
    "fleet/golden/wave-1/first-hour.txt",
    "fleet/golden/wave-2/connect/init-detect.txt",
    "fleet/golden/wave-2/trace/root-help.txt",
    "fleet/golden/wave-3/e1/counts.json",
    "fleet/golden/wave-3/policy/absent-assay.json",
}


def check_name(name: str) -> None:
    parts = pathlib.PurePosixPath(name).parts
    assert parts and not name.startswith("/") and ".." not in parts, name
    assert "development-snapshots" not in parts and "reproducibility" not in parts, name
    assert "g07" not in parts and "planted_secret" not in name, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", nargs="?", type=pathlib.Path, default=pathlib.Path("dist"))
    args = parser.parse_args()
    assert {p.name for p in args.dist.iterdir()} == {WHEEL, SDIST}, "unexpected dist file set"
    with zipfile.ZipFile(args.dist / WHEEL) as archive:
        names = set(archive.namelist())
        for name in names:
            check_name(name)
            assert "/fleet/" not in "/" + name, name
            info = archive.getinfo(name)
            assert (info.external_attr >> 16) & 0o170000 != 0o120000, name
            if name.endswith((".py", ".json", ".md", ".html", ".js", ".css", ".txt")):
                assert b"/home/suhail-nadaf" not in archive.read(name), name
        assert set(REQUIRED_WHEEL) <= names, sorted(set(REQUIRED_WHEEL) - names)
        assert any(name.endswith("THIRD_PARTY_NOTICES.md") for name in names)
    with tarfile.open(args.dist / SDIST) as archive:
        names = set()
        fleet = set()
        for item in archive.getmembers():
            check_name(item.name)
            assert item.isfile() or item.isdir(), item.name
            relative = "/".join(pathlib.PurePosixPath(item.name).parts[1:])
            if item.isfile():
                names.add(relative)
                if relative.startswith("fleet/"):
                    fleet.add(relative)
        assert fleet <= ALLOWED_FLEET, sorted(fleet - ALLOWED_FLEET)
        assert "fleet/gates/fixtures/controls/g39/positive/sdk.assay.json" in fleet
        for required in (
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "hatch_build.py",
            "frontend/package.json",
            "tests/conftest.py",
            "release/validation-matrix.json",
            ".pre-commit-hooks.yaml",
            ".github/actions/reward-lens-static-validity/pre_commit.py",
        ):
            assert required in names, required
    print(f"PASS: {WHEEL}, {SDIST}; retained source-only fleet files: {len(fleet)}")


if __name__ == "__main__":
    main()
