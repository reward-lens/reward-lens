"""The reward-lens CLI runs its pure subcommands over the store and gates the model-touching ones (section 2.16, M13).

This exercises the operator surface with typer's CliRunner. The subcommands that are views over the
evidence store (``card``, ``scoreboard``, ``atlas export``, and ``claims`` on a sound document) run
here and exit 0; the ``claims`` subcommand exits nonzero on a document that cites a number the store
does not contain (the unbound-number acceptance); and a model-touching subcommand (``score``) exits
with the GPU-gated code rather than pretending to load a model. The CLI imports torch-free, which is
also asserted, so an operator can introspect the surface without a GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from typer.testing import CliRunner

from reward_lens.artifacts.atlas import Atlas
from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.gates import CalibrationRef
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import GaugeStatus, ModelFP, SubjectRef
from reward_lens.operate.cli import GPU_GATED_EXIT, app

runner = CliRunner()


def test_legacy_mcp_exports_are_lazy_typed_refusals() -> None:
    """The deleted v3 server keeps its public shape without loading any MCP runtime."""

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import inspect
import json
import sys

import reward_lens.operate as operate

def optional_runtime_modules():
    return sorted(
        name for name in sys.modules
        if name == 'torch' or name.startswith('torch.') or name == 'mcp' or name.startswith('mcp.')
        or name == 'reward_lens.integrations.mcp' or name.startswith('reward_lens._vendor.mcp2')
    )

before = optional_runtime_modules()
symbols = [operate.Tool, operate.MCPServer, operate.build_server]
after_symbols = optional_runtime_modules()
from reward_lens.contracts import CapabilityUnavailable
errors = []
for symbol, invoke in (
    ('Tool', lambda: operate.Tool('name', 'description', {}, lambda _: 'ok')),
    ('MCPServer', lambda: operate.MCPServer()),
    ('build_server', lambda: operate.build_server()),
):
    try:
        invoke()
    except CapabilityUnavailable as error:
        errors.append({
            'symbol': symbol,
            'code': error.code,
            'message': error.message,
            'remediation': error.remediation,
        })
    else:
        raise AssertionError(f'{symbol} did not refuse')

try:
    importlib.import_module('reward_lens.operate.mcp')
except ModuleNotFoundError:
    removed_module = True
else:
    removed_module = False

print(json.dumps({
    'public': operate.__all__,
    'tool_parameters': list(inspect.signature(operate.Tool).parameters),
    'tool_gpu_default': inspect.signature(operate.Tool).parameters['gpu_gated'].default,
    'server_parameters': list(inspect.signature(operate.MCPServer).parameters),
    'server_store_default': inspect.signature(operate.MCPServer).parameters['store'].default,
    'build_parameters': list(inspect.signature(operate.build_server).parameters),
    'build_store_default': inspect.signature(operate.build_server).parameters['store'].default,
    'before': before,
    'after_symbols': after_symbols,
    'errors': errors,
    'removed_module': removed_module,
}, sort_keys=True))
""",
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)

    assert observed["public"] == ["app", "main", "MCPServer", "Tool", "build_server"]
    assert observed["tool_parameters"] == [
        "name",
        "description",
        "input_schema",
        "handler",
        "gpu_gated",
    ]
    assert observed["tool_gpu_default"] is False
    assert observed["server_parameters"] == ["store"]
    assert observed["server_store_default"] is None
    assert observed["build_parameters"] == ["store"]
    assert observed["build_store_default"] is None
    assert observed["before"] == []
    assert observed["after_symbols"] == []
    assert observed["removed_module"] is True
    assert [error["symbol"] for error in observed["errors"]] == [
        "Tool",
        "MCPServer",
        "build_server",
    ]
    for error in observed["errors"]:
        assert error["code"] == "RL0701"
        assert "removed transport-less v3 MCP API" in error["message"]
        assert "reward-lens mcp" in error["remediation"]
        assert "reward_lens.integrations.mcp" in error["remediation"]


def _seed(tmp_path) -> tuple[str, str, str]:
    """Seed a store with a calibrated and an uncalibrated index on Skywork-v0.2. Returns the paths."""
    store = EvidenceStore(tmp_path)
    fp = Atlas.standard().by_name("Skywork-Reward-Llama-3.1-8B-v0.2").fingerprint
    subj = SubjectRef(signals=(ModelFP(fp),), dataset="ds:diag", readout="reward")
    calibrated = make_evidence(
        observable="BiasBattery",
        observable_version="1.0",
        subject=subj,
        value={"verbosity": -0.05},
        uncertainty=Uncertainty(ci_low=-0.1, ci_high=0.0, n=30, n_effective=6.0),
        gauge=GaugeStatus.INVARIANT,
        calibration=CalibrationRef("ev:scorecard", "planted-bias"),
    )
    store.append(calibrated)
    store.append(
        make_evidence(
            observable="DistortionV2",
            observable_version="1.0",
            subject=subj,
            value=0.42,
            gauge=GaugeStatus.INVARIANT,
        )
    )
    return str(tmp_path), fp, calibrated.id


def test_card_subcommand_runs(tmp_path):
    store_dir, fp, _ = _seed(tmp_path)
    result = runner.invoke(app, ["card", fp, "--store", store_dir])
    assert result.exit_code == 0
    assert fp in result.stdout
    assert "BiasBattery" in result.stdout


def test_scoreboard_subcommand_runs():
    result = runner.invoke(app, ["scoreboard"])
    assert result.exit_code == 0
    assert "T9" in result.stdout  # a candidate-law row from the default scoreboard


def test_atlas_export_subcommand_runs(tmp_path):
    store_dir, fp, _ = _seed(tmp_path)
    result = runner.invoke(
        app, ["atlas", "export", "--store", store_dir, "--observables", "BiasBattery,DistortionV2"]
    )
    assert result.exit_code == 0
    assert "uncalibrated_cells" in result.stdout


def test_claims_subcommand_passes_sound_document(tmp_path):
    store_dir, fp, eid = _seed(tmp_path)
    doc = tmp_path / "sound.md"
    doc.write_text(
        f"The verbosity slope is [[claim value=-0.05 ev={eid} field=verbosity tol=0.01]].",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["claims", str(doc), "--store", store_dir])
    assert result.exit_code == 0


def test_claims_subcommand_fails_unbound_number(tmp_path):
    store_dir, fp, _ = _seed(tmp_path)
    doc = tmp_path / "unbound.md"
    # A number tagged to an Evidence id the store does not contain: the unbound-number failure.
    doc.write_text(
        "We report [[claim value=0.99 ev=ev:deadbeefdeadbeef field=x tol=0.01]].",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["claims", str(doc), "--store", store_dir])
    assert result.exit_code != 0


def test_score_subcommand_is_gpu_gated():
    result = runner.invoke(app, ["score", "some/reward-model"])
    assert result.exit_code == GPU_GATED_EXIT


def test_cli_imports_torch_free():
    # Importing the operator surface must not pull torch; an operator introspects without a GPU.
    import builtins

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"operate.cli import pulled torch via {name}")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guard
    try:
        import importlib

        importlib.reload(importlib.import_module("reward_lens.operate.cli"))
    finally:
        builtins.__import__ = real_import
