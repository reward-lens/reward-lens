"""
IO helpers: resumable JSONL writers, atomic JSON saves, manifest creation.

Per-pair JSONL is the safety net: if aggregation logic changes (it will,
once BH-FDR lands), we don't want to redo the model passes.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional


def _convert(obj: Any) -> Any:
    """Convert numpy/torch/dataclass objects to JSON-friendly forms."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except ImportError:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
    except ImportError:
        pass
    if is_dataclass(obj) and not isinstance(obj, type):
        return _convert(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    if isinstance(obj, set):
        return [_convert(v) for v in obj]
    return obj


def save_json(data: Any, path: str | os.PathLike, indent: int = 2) -> None:
    """Atomically write data as JSON. Numpy/torch are converted recursively."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_convert(data), f, indent=indent, default=str)
    tmp.replace(p)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r") as f:
        return json.load(f)


class JsonlWriter:
    """Append-mode JSONL writer with auto-flush per record.

    Used as the resumable per-pair intermediate: each record corresponds to
    one (pair, model, dimension) triple, so re-running an experiment can
    skip records already in the file (matched by ``record_id``).
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._existing_ids: set[str] = set()
        if self.path.exists():
            with open(self.path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        rid = rec.get("record_id")
                        if rid is not None:
                            self._existing_ids.add(str(rid))
                    except json.JSONDecodeError:
                        # corrupt line — ignore; resume will overwrite
                        pass

    def has(self, record_id: str) -> bool:
        return record_id in self._existing_ids

    def write(self, record: dict) -> None:
        rid = record.get("record_id")
        with open(self.path, "a") as f:
            f.write(json.dumps(_convert(record), default=str) + "\n")
        if rid is not None:
            self._existing_ids.add(str(rid))

    def read_all(self) -> list[dict]:
        records: list[dict] = []
        if not self.path.exists():
            return records
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def hardware_string() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            return f"{name} ({mem} GB) cuda={torch.version.cuda}"
        return "cpu"
    except Exception:
        return "unknown"


def peak_gpu_memory_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated() / (1024 ** 3))
    except Exception:
        return 0.0


def write_manifest(out_dir: str | os.PathLike,
                   experiment: str,
                   config: dict,
                   *,
                   status: str = "running",
                   model: Optional[str] = None,
                   seed: int = 0,
                   runtime_seconds: Optional[float] = None,
                   notes: Optional[str] = None,
                   extra: Optional[dict] = None) -> Path:
    p = Path(out_dir) / "manifest.json"
    data = {
        "experiment": experiment,
        "status": status,
        "git_commit": git_commit(),
        "host": socket.gethostname(),
        "hardware": hardware_string(),
        "peak_gpu_gb": peak_gpu_memory_gb(),
        "timestamp": datetime.now().isoformat(),
        "config": _convert(config),
        "model": model,
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "notes": notes,
        **(extra or {}),
    }
    save_json(data, p)
    return p


@contextmanager
def manifest_run(out_dir: str | os.PathLike, experiment: str, config: dict,
                 model: Optional[str] = None, seed: int = 0,
                 swallow_exceptions: bool = False) -> Iterator[Path]:
    """Context manager that writes a manifest with status tracking.

    Args:
        out_dir: Where to write ``manifest.json``.
        experiment: Experiment registry name.
        config: Config dict to embed in the manifest.
        model: Optional per-model marker for the manifest.
        seed: Seed for reproducibility tracking.
        swallow_exceptions: When True, write a ``failed`` manifest and
            return *without* re-raising. Use this when iterating per-model
            so a single broken model doesn't abort the whole experiment.
            See deep_analysis_v2 §2.7 — the v2 run had ~10 partial-failure
            cells flagged orchestrator-FAILED because ONE model raised.
            Also frees GPU memory on failure so the next loop iteration
            has clean memory to load into.
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    write_manifest(p, experiment, config, status="running", model=model, seed=seed)
    t0 = time.time()
    try:
        yield p
        write_manifest(p, experiment, config, status="complete", model=model, seed=seed,
                       runtime_seconds=time.time() - t0)
    except Exception as exc:
        write_manifest(p, experiment, config, status="failed", model=model, seed=seed,
                       runtime_seconds=time.time() - t0,
                       notes=f"{type(exc).__name__}: {exc}")
        if swallow_exceptions:
            # Clean up GPU memory before the next iteration of the caller's
            # for-loop tries to load another model. Without this, a partial
            # forward pass that died mid-allocation leaves CUDA fragmented
            # and the next big load OOMs (deep_analysis_v2 §2.4).
            try:
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            return
        raise


def write_csv(rows: list[dict], path: str | os.PathLike, columns: Optional[list[str]] = None) -> None:
    """Write a list of dicts as CSV."""
    import csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return
    cols = columns or sorted({k for r in rows for k in r.keys()})
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _convert(r.get(k)) for k in cols})
