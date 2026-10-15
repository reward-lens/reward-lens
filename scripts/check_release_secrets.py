#!/usr/bin/env python3
"""Fail on unaudited detect-secrets findings in every tracked release file."""

from __future__ import annotations

import json
import pathlib
import subprocess

BASELINE = pathlib.Path(".secrets.baseline")


def main() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["version"] == "1.5.0", "review scanner-version changes"
    reviewed = 0
    for path, findings in data["results"].items():
        assert pathlib.Path(path).is_file(), f"stale secret review: {path}"
        assert "planted_secret" not in path and "/g07/" not in path, path
        for finding in findings:
            assert finding.get("is_secret") is False, (
                f"unreviewed finding: {path}:{finding['line_number']}"
            )
            reviewed += 1
    paths = [
        p.decode() for p in subprocess.check_output(["git", "ls-files", "-z"]).split(b"\0") if p
    ]
    for offset in range(0, len(paths), 100):
        result = subprocess.run(
            ["detect-secrets-hook", "--baseline", str(BASELINE), *paths[offset : offset + 100]],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise SystemExit(
                f"detect-secrets found unreviewed tracked content in batch {offset // 100 + 1}; "
                "run detect-secrets-hook locally and review before updating the baseline"
            )
    print(
        f"PASS: detect-secrets scanned {len(paths)} tracked files; {reviewed} labelled false positives"
    )


if __name__ == "__main__":
    main()
