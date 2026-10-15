"""Archive path identity controls for the 3.1.0 release gate."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import warnings
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/verify_release_archives.py"
SPEC = importlib.util.spec_from_file_location("verify_release_archives", SCRIPT)
assert SPEC and SPEC.loader
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)

SDIST_ROOT = f"reward_lens-{verify.VERSION}"
REQUIRED_SDIST = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "VERIFYING.md",
    "hatch_build.py",
    "frontend/package.json",
    "tests/conftest.py",
    "release/validation-matrix.json",
    ".pre-commit-hooks.yaml",
    ".github/actions/reward-lens-static-validity/pre_commit.py",
    "fleet/gates/fixtures/controls/g39/positive/sdk.assay.json",
)


def make_archives(
    root: Path,
    *,
    wheel_extra: tuple[str, ...] = (),
    sdist_extra: tuple[str, ...] = (),
    sdist_missing: tuple[str, ...] = (),
) -> None:
    root.mkdir()
    with zipfile.ZipFile(root / verify.WHEEL, "w") as archive:
        archive.writestr("reward_lens/assets/", b"")
        for name in (*verify.REQUIRED_WHEEL, "reward_lens/THIRD_PARTY_NOTICES.md"):
            archive.writestr(name, b"public test")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name in wheel_extra:
                archive.writestr(name, b"second payload")
    with tarfile.open(root / verify.SDIST, "w:gz") as archive:
        directory = tarfile.TarInfo(SDIST_ROOT + "/")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for relative in (*REQUIRED_SDIST, *sdist_extra):
            if relative in sdist_missing:
                continue
            name = f"{SDIST_ROOT}/{relative}"
            payload = b"public test"
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload))


def run_verifier(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["verify_release_archives.py", str(root)])
    verify.main()


def test_canonical_archive_paths_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dist"
    make_archives(root)
    run_verifier(root, monkeypatch)


def test_sdist_missing_verification_guide_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dist"
    make_archives(root, sdist_missing=("VERIFYING.md",))
    with pytest.raises(AssertionError, match="VERIFYING.md"):
        run_verifier(root, monkeypatch)


@pytest.mark.parametrize(
    "extra",
    [
        verify.REQUIRED_WHEEL[0],
        "reward_lens/assets/",  # directory spelling aliases the same member
        "reward_lens//extra.py",
        "reward_lens/./extra.py",
        r"reward_lens\extra.py",
        "C:/reward_lens/extra.py",
    ],
)
def test_wheel_rejects_duplicate_or_ambiguous_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: str
) -> None:
    root = tmp_path / "dist"
    make_archives(root, wheel_extra=(extra,))
    with pytest.raises(AssertionError):
        run_verifier(root, monkeypatch)


@pytest.mark.parametrize(
    "extra",
    [
        "LICENSE",
        "./LICENSE",
        "sub//extra.txt",
        r"sub\extra.txt",
        "../escape.txt",
    ],
)
def test_sdist_rejects_duplicate_or_ambiguous_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: str
) -> None:
    root = tmp_path / "dist"
    make_archives(root, sdist_extra=(extra,))
    with pytest.raises(AssertionError):
        run_verifier(root, monkeypatch)
