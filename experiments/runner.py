"""
CLI entrypoint: ``python -m experiments.runner run <name> --config <path>``.

Subcommands:
  list                       — list registered experiments
  run NAME --config PATH     — run an experiment with a YAML config
  run NAME --models M1 M2... — run with default config + explicit models
  report --runs DIR          — emit REPORT.md by aggregating manifest.json files
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import ExperimentConfig, ModelConfig, load_config
from . import registry


def _build_default_config(name: str, out_dir: str, models: list[str]) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        out_dir=out_dir,
        models=[ModelConfig(name=m) for m in models],
    )


def cmd_list(args):
    for n in registry.list_experiments():
        print(n)


def cmd_run(args):
    if args.config:
        cfg = load_config(args.config, name=args.name, out_dir=args.out_dir or None)
    else:
        if not args.models:
            print("error: --models is required when --config is omitted", file=sys.stderr)
            sys.exit(2)
        cfg = _build_default_config(args.name, args.out_dir or f"outputs/{args.name}", args.models)

    if args.n_pairs is not None:
        cfg.n_pairs_per_dim = args.n_pairs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.dimensions:
        cfg.dimensions = args.dimensions

    fn = registry.resolve(args.name)
    print(f"[runner] {args.name}: out_dir={cfg.out_dir}, n_pairs={cfg.n_pairs_per_dim}, "
          f"models={[m.short_name() for m in cfg.models]}")
    t0 = time.time()
    result = fn(cfg)
    print(f"[runner] {args.name}: done in {time.time() - t0:.1f}s")
    return result


def cmd_report(args):
    """Aggregate manifest.json + summary outputs across a runs/ directory."""
    from .utils.io import load_json
    runs_dir = Path(args.runs)
    manifests = sorted(runs_dir.rglob("manifest.json"))
    lines = ["# reward-lens v2 run report", ""]
    lines.append(f"Runs root: `{runs_dir}`  ·  {len(manifests)} manifests found")
    lines.append("")
    by_exp: dict[str, list] = {}
    for m_path in manifests:
        try:
            m = load_json(m_path)
            by_exp.setdefault(m.get("experiment", "unknown"), []).append((m_path, m))
        except Exception:
            continue
    for exp, entries in sorted(by_exp.items()):
        lines.append(f"## {exp}")
        lines.append("")
        lines.append("| status | model | runtime (s) | dir |")
        lines.append("|---|---|---|---|")
        for path, m in entries:
            lines.append(
                f"| {m.get('status','?')} | {m.get('model','-') or '-'} | "
                f"{m.get('runtime_seconds','-') or '-'} | "
                f"`{path.parent.relative_to(runs_dir)}` |"
            )
        lines.append("")
    out = runs_dir / "REPORT.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="experiments.runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="list registered experiments")
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("run", help="run an experiment")
    sp.add_argument("name", help="experiment name (e.g. e04_faithfulness_population)")
    sp.add_argument("--config", help="YAML config path")
    sp.add_argument("--out-dir", help="override output directory")
    sp.add_argument("--models", nargs="+", help="HF model IDs (when --config is omitted)")
    sp.add_argument("--n-pairs", type=int, help="override n_pairs_per_dim")
    sp.add_argument("--batch-size", type=int, help="override batch_size")
    sp.add_argument("--dimensions", nargs="+", help="override dimensions list")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("report", help="aggregate REPORT.md from runs dir")
    sp.add_argument("--runs", required=True)
    sp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    main()
