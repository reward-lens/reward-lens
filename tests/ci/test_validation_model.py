"""Bounded security controls for the pinned public validation-model cache."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from urllib.error import URLError

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "release/validation_model/provision_model.py"
SPEC = importlib.util.spec_from_file_location("validation_model_provisioner", SCRIPT)
assert SPEC and SPEC.loader
provision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provision)


def _entry(data: bytes) -> dict:
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha1": hashlib.sha1(blob).hexdigest(),  # noqa: S324 - Git identity
        "lfs_sha256": None,
    }


@pytest.fixture
def sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    files = {"config.json": b"{}", "model.safetensors": b"tiny synthetic weights"}
    manifest = {
        "schema": "reward-lens-validation-model/1",
        "model": provision.MODEL,
        "revision": provision.REVISION,
        "files": {name: _entry(data) for name, data in files.items()},
    }
    manifest_path = tmp_path / "model-files.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(provision, "MANIFEST", manifest_path)
    monkeypatch.setattr(provision, "EXPECTED_FILES", set(files))
    home = tmp_path / "hf-home"
    snapshot = (
        home
        / "hub/models--trl-internal-testing--tiny-Qwen3ForCausalLM/snapshots"
        / provision.REVISION
    )
    return home, snapshot, files


def _run(
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    *,
    offline: bool = False,
    fixture: str = "tiny-qwen",
) -> None:
    args = ["provision_model.py", "--hf-home", str(home)]
    if fixture != "tiny-qwen":
        args.extend(["--fixture", fixture])
    if offline:
        args.append("--offline")
    monkeypatch.setattr(sys, "argv", args)
    provision.main()


def test_synthetic_fetch_then_offline_verification(sample, monkeypatch):
    home, snapshot, files = sample

    class Response(io.BytesIO):
        status = 200

        def __init__(self, data: bytes):
            super().__init__(data)
            self.headers = {"Content-Length": str(len(data))}

    def fetch(request, timeout):
        assert timeout == 45
        name = request.full_url.split("/resolve/", 1)[1].split("/", 1)[1].split("?", 1)[0]
        return Response(files[name])

    monkeypatch.setattr(provision, "urlopen", fetch)
    _run(monkeypatch, home)
    assert {entry.name for entry in snapshot.iterdir()} == set(files)
    _run(monkeypatch, home, offline=True)


def test_offline_missing_cache_does_not_create_directories(sample, monkeypatch):
    home, _, _ = sample
    with pytest.raises(FileNotFoundError):
        _run(monkeypatch, home, offline=True)
    assert not home.exists()


@pytest.mark.parametrize("level", ["home", "hub"])
def test_symlink_directory_ancestor_is_rejected_without_writes(sample, monkeypatch, level):
    home, snapshot, _ = sample
    external = home.parent / "external"
    external.mkdir()
    if level == "home":
        home.symlink_to(external, target_is_directory=True)
    else:
        home.mkdir()
        (home / "hub").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe model cache"):
        _run(monkeypatch, home, offline=True)
    assert list(external.iterdir()) == []
    assert not snapshot.exists()


def test_unexpected_adapter_config_is_rejected_offline(sample, monkeypatch):
    home, snapshot, files = sample
    snapshot.mkdir(parents=True)
    for name, data in files.items():
        (snapshot / name).write_bytes(data)
    adapter = snapshot / "adapter_config.json"
    adapter.write_text('{"base_model_name_or_path":"other-model"}')
    with pytest.raises(ValueError, match="unexpected model snapshot entry"):
        _run(monkeypatch, home, offline=True)
    assert adapter.exists()


@pytest.mark.parametrize("offline", [False, True])
def test_symlink_partial_is_rejected_without_target_mutation(sample, monkeypatch, offline):
    home, snapshot, _ = sample
    snapshot.mkdir(parents=True)
    target = home.parent / "outside.txt"
    target.write_text("preserve")
    (snapshot / "config.json.part").symlink_to(target)
    with pytest.raises(ValueError, match="unsafe model snapshot entry"):
        _run(monkeypatch, home, offline=offline)
    assert target.read_text() == "preserve"
    assert (snapshot / "config.json.part").is_symlink()


def test_regular_partial_retried_online_but_offline_is_read_only(sample, monkeypatch):
    home, snapshot, files = sample
    snapshot.mkdir(parents=True)
    partial = snapshot / "config.json.part"
    partial.write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected model snapshot entry"):
        _run(monkeypatch, home, offline=True)
    assert partial.read_bytes() == b"x"

    class Response(io.BytesIO):
        status = 200

        def __init__(self, data):
            super().__init__(data)
            self.headers = {"Content-Length": str(len(data))}

    def fetch(request, timeout):
        name = request.full_url.split("/resolve/", 1)[1].split("/", 1)[1].split("?", 1)[0]
        return Response(files[name])

    monkeypatch.setattr(provision, "urlopen", fetch)
    _run(monkeypatch, home)
    assert not partial.exists()
    assert (snapshot / "config.json").read_bytes() == files["config.json"]


def test_transient_download_retries_from_clean_partial(sample, monkeypatch):
    home, snapshot, files = sample
    attempts = 0

    class Response(io.BytesIO):
        status = 200

        def __init__(self, data):
            super().__init__(data)
            self.headers = {"Content-Length": str(len(data))}

    def fetch(request, timeout):
        nonlocal attempts
        name = request.full_url.split("/resolve/", 1)[1].split("/", 1)[1].split("?", 1)[0]
        attempts += 1
        if attempts == 1:
            raise URLError("synthetic transient failure")
        return Response(files[name])

    monkeypatch.setattr(provision, "urlopen", fetch)
    monkeypatch.setattr(provision.time, "sleep", lambda _: None)
    _run(monkeypatch, home)
    assert attempts == 3  # one failure, then two pinned synthetic files
    assert {entry.name for entry in snapshot.iterdir()} == set(files)


@pytest.fixture
def tokenizer_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    files = {"config.json": b"{}", "tokenizer.json": b'{"version":"1.0"}'}
    manifest = {
        "schema": "reward-lens-validation-tokenizer/1",
        "model": provision.TOKENIZER_MODEL,
        "upstream_model": provision.TOKENIZER_UPSTREAM_MODEL,
        "revision": provision.TOKENIZER_REVISION,
        "files": {name: _entry(data) for name, data in files.items()},
    }
    manifest_path = tmp_path / "gpt2-tokenizer-files.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(provision, "TOKENIZER_MANIFEST", manifest_path)
    monkeypatch.setattr(provision, "TOKENIZER_EXPECTED_FILES", set(files))
    home = tmp_path / "hf-home"
    snapshot = home / "hub/models--gpt2/snapshots" / provision.TOKENIZER_REVISION
    return home, snapshot, files, manifest_path


def test_pinned_gpt2_tokenizer_fetches_official_revision_then_verifies_offline(
    tokenizer_sample, monkeypatch
):
    home, snapshot, files, _ = tokenizer_sample

    class Response(io.BytesIO):
        status = 200

        def __init__(self, data):
            super().__init__(data)
            self.headers = {"Content-Length": str(len(data))}

    def fetch(request, timeout):
        assert timeout == 45
        prefix = (
            f"https://huggingface.co/{provision.TOKENIZER_UPSTREAM_MODEL}/resolve/"
            f"{provision.TOKENIZER_REVISION}/"
        )
        assert request.full_url.startswith(prefix)
        name = request.full_url.removeprefix(prefix).split("?", 1)[0]
        return Response(files[name])

    monkeypatch.setattr(provision, "urlopen", fetch)
    _run(monkeypatch, home, fixture="gpt2-tokenizer")
    assert {entry.name for entry in snapshot.iterdir()} == set(files)
    _run(monkeypatch, home, fixture="gpt2-tokenizer", offline=True)
    (snapshot / "adapter_config.json").write_text('{"base_model_name_or_path":"other-model"}')
    with pytest.raises(ValueError, match="unexpected model snapshot entry"):
        _run(monkeypatch, home, fixture="gpt2-tokenizer", offline=True)


def test_gpt2_tokenizer_bad_manifest_and_missing_offline_cache_do_not_write(
    tokenizer_sample, monkeypatch
):
    home, _, _, manifest_path = tokenizer_sample
    with pytest.raises(FileNotFoundError):
        _run(monkeypatch, home, fixture="gpt2-tokenizer", offline=True)
    assert not home.exists()
    manifest = json.loads(manifest_path.read_text())
    manifest["upstream_model"] = "other-model"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest identity"):
        _run(monkeypatch, home, fixture="gpt2-tokenizer")
    assert not home.exists()


def test_gpt2_tokenizer_manifest_has_bounded_public_source_identity():
    manifest = json.loads(provision.TOKENIZER_MANIFEST.read_text())
    assert manifest["model"] == "gpt2"
    assert manifest["upstream_model"] == "openai-community/gpt2"
    assert manifest["revision"] == provision.TOKENIZER_REVISION
    assert manifest["license"] == "mit"
    assert set(manifest["files"]) == provision.TOKENIZER_EXPECTED_FILES
    assert sum(row["bytes"] for row in manifest["files"].values()) == 2_854_566
    assert provision._files(provision._fixture("gpt2-tokenizer")) == manifest["files"]
