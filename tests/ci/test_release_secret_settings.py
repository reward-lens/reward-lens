"""Portable detect-secrets baseline settings are a release gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_release_secrets.py"
SPEC = importlib.util.spec_from_file_location("release_secret_settings_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
secret_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(secret_check)


def _data(filename: str = ".secrets.baseline") -> dict:
    return {
        "version": "1.5.0",
        "filters_used": [
            {"path": "detect_secrets.filters.allowlist.is_line_allowlisted"},
            {
                "path": "detect_secrets.filters.common.is_baseline_file",
                "filename": filename,
            },
        ],
        "results": {},
    }


def test_portable_baseline_filter_is_valid() -> None:
    secret_check.validate_baseline_settings(_data())


@pytest.mark.parametrize(
    "invalid_filters",
    [
        [
            {
                "path": "detect_secrets.filters.common.is_baseline_file",
                "filename": "/home/user/private/.secrets.baseline",
            }
        ],
        [
            {
                "path": "detect_secrets.filters.common.is_baseline_file",
                "filename": "elsewhere/.secrets.baseline",
            }
        ],
        [{"path": "detect_secrets.filters.allowlist.is_line_allowlisted"}],
        [
            {
                "path": "detect_secrets.filters.common.is_baseline_file",
                "filename": ".secrets.baseline",
            },
            {
                "path": "detect_secrets.filters.common.is_baseline_file",
                "filename": ".secrets.baseline",
            },
        ],
        "not-a-list",
    ],
)
def test_missing_ambiguous_or_private_baseline_filter_is_rejected(
    invalid_filters: object,
) -> None:
    data = _data()
    data["filters_used"] = invalid_filters
    with pytest.raises(ValueError, match="baseline filter"):
        secret_check.validate_baseline_settings(data)


def test_main_rejects_private_filename_before_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / ".secrets.baseline"
    baseline.write_text(json.dumps(_data("/home/user/private/.secrets.baseline")))
    monkeypatch.setattr(secret_check, "BASELINE", baseline)

    def unexpected_scan(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("baseline settings must fail before Git or detect-secrets runs")

    monkeypatch.setattr(secret_check.subprocess, "check_output", unexpected_scan)
    with pytest.raises(ValueError, match="baseline filter"):
        secret_check.main()
