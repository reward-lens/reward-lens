"""
Configuration dataclasses + YAML loader for experiments.

Every experiment receives a single ``ExperimentConfig`` object. The YAML
loader is permissive: unknown keys land in ``cfg.extra`` so individual
experiments can carry their own knobs without changing this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelConfig:
    """Reward model load specification."""
    name: str  # HuggingFace ID or local path
    short: Optional[str] = None  # short label for filenames; defaults to last name component
    torch_dtype: str = "bfloat16"  # "bfloat16" / "float16" / "float32"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None  # "flash_attention_2" / "eager" / None
    device: Optional[str] = None  # auto-detect when None
    extra: dict[str, Any] = field(default_factory=dict)

    def short_name(self) -> str:
        return self.short or self.name.split("/")[-1]


@dataclass
class ExperimentConfig:
    """Per-experiment configuration."""
    name: str  # the experiment registry name (e.g. "e04_faithfulness")
    out_dir: str  # where to write artifacts
    models: list[ModelConfig] = field(default_factory=list)
    n_pairs_per_dim: int = 150
    dimensions: Optional[list[str]] = None  # None = use all from diagnostic_v2
    batch_size: int = 32
    max_length: int = 2048
    seed: int = 0
    n_resamples: int = 10_000
    ci: float = 0.95
    skip_models_on_error: bool = True
    resume: bool = True
    progress: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def out_path(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_yaml(path: str | os.PathLike) -> dict:
    """Load a YAML file. Falls back to a minimal hand-rolled parser if
    PyYAML isn't installed — covers flat key:value files (the format every
    config in this repo uses)."""
    p = Path(path)
    text = p.read_text()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        return _minimal_yaml(text)


def _minimal_yaml(text: str) -> dict:
    """Tiny indent-aware YAML subset parser. Supports nested dicts, lists of
    scalars, and lists of dicts — enough for our config files."""
    import re
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]

    def parse_value(v: str) -> Any:
        v = v.strip()
        if v.lower() in ("true", "yes"): return True
        if v.lower() in ("false", "no"): return False
        if v.lower() in ("null", "none", "~", ""): return None
        if re.fullmatch(r"-?\d+", v): return int(v)
        if re.fullmatch(r"-?\d*\.\d+(e-?\d+)?", v): return float(v)
        if v.startswith('"') and v.endswith('"'): return v[1:-1]
        if v.startswith("'") and v.endswith("'"): return v[1:-1]
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner: return []
            return [parse_value(x) for x in _smart_split(inner)]
        return v

    def _smart_split(s: str) -> list[str]:
        out, depth, cur = [], 0, []
        for ch in s:
            if ch in "[{": depth += 1
            elif ch in "]}": depth -= 1
            if ch == "," and depth == 0:
                out.append("".join(cur)); cur = []
            else:
                cur.append(ch)
        if cur: out.append("".join(cur))
        return [x.strip() for x in out]

    root: dict = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        indent = len(line) - len(line.lstrip())
        stripped = line.lstrip()

        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else root

        if stripped.startswith("- "):
            item_text = stripped[2:]
            if ":" in item_text and not item_text.strip().startswith("[") and not item_text.strip().startswith("{"):
                # list of dicts
                if not isinstance(parent, list):
                    raise ValueError(f"unexpected '-' at indent {indent}")
                d: dict = {}
                parent.append(d)
                stack.append((indent, d))
                # process the rest of the first key:value
                key, _, val = item_text.partition(":")
                if val.strip():
                    d[key.strip()] = parse_value(val)
                else:
                    d[key.strip()] = {}
                    stack.append((indent + 2, d[key.strip()]))
            else:
                if not isinstance(parent, list):
                    raise ValueError(f"unexpected '-' at indent {indent}")
                parent.append(parse_value(item_text))
        else:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                # could be dict or list — peek ahead
                next_idx = i + 1
                next_indent = None
                while next_idx < len(lines):
                    nl = lines[next_idx]
                    if nl.strip():
                        next_indent = len(nl) - len(nl.lstrip())
                        break
                    next_idx += 1
                if next_indent is not None and next_indent > indent and lines[next_idx].lstrip().startswith("- "):
                    parent[key] = []
                    stack.append((indent, parent[key]))
                else:
                    parent[key] = {}
                    stack.append((indent, parent[key]))
            else:
                parent[key] = parse_value(val)
        i += 1

    return root


def config_from_dict(d: dict) -> ExperimentConfig:
    """Construct an ExperimentConfig from a parsed YAML dict."""
    models_raw = d.pop("models", [])
    models = []
    for m in models_raw:
        if isinstance(m, str):
            models.append(ModelConfig(name=m))
        else:
            extra = {k: v for k, v in m.items()
                     if k not in {"name", "short", "torch_dtype", "trust_remote_code",
                                   "attn_implementation", "device"}}
            models.append(ModelConfig(
                name=m["name"],
                short=m.get("short"),
                torch_dtype=m.get("torch_dtype", "bfloat16"),
                trust_remote_code=m.get("trust_remote_code", True),
                attn_implementation=m.get("attn_implementation"),
                device=m.get("device"),
                extra=extra,
            ))

    known = {"name", "out_dir", "n_pairs_per_dim", "dimensions", "batch_size",
             "max_length", "seed", "n_resamples", "ci", "skip_models_on_error",
             "resume", "progress"}
    extra = {k: v for k, v in d.items() if k not in known}

    return ExperimentConfig(
        name=d["name"],
        out_dir=d.get("out_dir", "outputs/default"),
        models=models,
        n_pairs_per_dim=d.get("n_pairs_per_dim", 150),
        dimensions=d.get("dimensions"),
        batch_size=d.get("batch_size", 32),
        max_length=d.get("max_length", 2048),
        seed=d.get("seed", 0),
        n_resamples=d.get("n_resamples", 10_000),
        ci=d.get("ci", 0.95),
        skip_models_on_error=d.get("skip_models_on_error", True),
        resume=d.get("resume", True),
        progress=d.get("progress", True),
        extra=extra,
    )


def load_config(path: str | os.PathLike, name: Optional[str] = None,
                out_dir: Optional[str] = None) -> ExperimentConfig:
    """Load and parse an experiment YAML config."""
    d = load_yaml(path)
    if name is not None:
        d.setdefault("name", name)
    if out_dir is not None:
        d["out_dir"] = out_dir
    return config_from_dict(d)
