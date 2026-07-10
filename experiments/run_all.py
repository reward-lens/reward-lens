"""
Master orchestrator: run the full v2 plan in dependency order.

Phases (per the user's gate sequence):
  Phase 1: library infra (already complete, not re-run here)
  Phase 2: E04 spine — gates further work on signal
  Phase 3: E17 causal validation
  Phase 4: remaining experiments
  Phase 5: E20 only if E04 + E17 both land cleanly (NOT IMPLEMENTED — gated)

This module is invoked by ``python -m experiments.run_all --models <m1> <m2>``.
It writes a single run-id directory with per-experiment subfolders and
finally emits REPORT.md by aggregating manifest.json files.

Expected runtime on H200: many hours; resumable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import ExperimentConfig, ModelConfig
from . import registry
from .runner import cmd_report
from .utils.io import save_json, git_commit
from .utils.parallel import tprint


# Phase 4 experiments — order matters for those reading upstream artifacts.
PHASE_4 = [
    "e01_baseline_and_diagnostics",
    "e02_lens_population",
    "e03_attribution_population",
    "e05_circuit_overlap",        # reads e03 intermediates
    "e06_hacking_at_scale",
    "e07_cascade_at_scale",
    "e08_concept_population",
    "e09_conflict_population",
    "e10_distortion_index",
    "e11_divergence_patching",
    "e12_sae_feature_decomposition",
    "e13_scale_study",            # reads e02/e03/e04/e08
    "e14_cross_architecture",
    "e15_head_path_patching",
    "e16_prompt_robustness",      # new (deep_analysisv1 follow-up)
    "e18_armorm_multi_objective",
    "e19_finetune_delta",         # new — needs Llama-orig + Llama-v0.2
    "e20_arch_vs_finetune",       # new — needs all 3 Skywork-family models
]


def _make_cfg(name: str, root: Path, models: list[ModelConfig], **overrides) -> ExperimentConfig:
    base = dict(
        name=name,
        out_dir=str(root / name),
        models=models,
        n_pairs_per_dim=150,
        batch_size=32,
        max_length=2048,
        seed=0,
        n_resamples=10_000,
        ci=0.95,
    )
    base.update(overrides)
    cfg = ExperimentConfig(**{k: v for k, v in base.items() if k != "extra"})
    cfg.extra = overrides.get("extra", {})
    return cfg


def _run_one(name: str, cfg: ExperimentConfig, *, fail_fast: bool) -> bool:
    """Run a single experiment; return True on success, False on failure."""
    fn = registry.resolve(name)
    tprint(f"\n{'='*70}\n[run_all] {name}\n{'='*70}")
    t0 = time.time()
    # Pre-experiment GPU hygiene (deep_analysis_v2 §2.4): make sure the
    # previous experiment's leftover allocations are released before this
    # one starts loading models. Without this, a 27B model sometimes OOMs
    # because pytorch's allocator is fragmented from the prior 8B run.
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    try:
        fn(cfg)
        tprint(f"[run_all] {name} done in {time.time()-t0:.1f}s")
        return True
    except Exception as e:
        tprint(f"[run_all] {name} FAILED: {type(e).__name__}: {e}")
        if fail_fast:
            raise
        return False
    finally:
        # Post-experiment GPU hygiene — same reason, opposite end.
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="HF model IDs (or local paths)")
    p.add_argument("--root", default=None,
                   help="run root; defaults to outputs/<timestamp>-<git>")
    p.add_argument("--n-pairs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128,
                   help="forward batch size (default 128 — bumped from 32 in the "
                        "deep_analysisv1 follow-up; H200 sustains 256+ for 8B and "
                        "192 for the 27B Gemma)")
    p.add_argument("--auto-batch-size", action="store_true",
                   help="probe free CUDA memory at load time and pick a batch "
                        "size that fits. Overrides --batch-size when set. Falls "
                        "back to --batch-size on CPU.")
    p.add_argument("--length-bucket", action="store_true",
                   help="length-bucket pairs in forward_with_cache_batch to "
                        "reduce padding waste. Recommended when running mixed "
                        "RewardBench + diagnostic_v2 corpora.")
    p.add_argument("--shared-activation-cache", action="store_true",
                   help="precompute activations once per (model, pair-set) and "
                        "reuse across experiments e02/e03/e04/e07/e09. Big win "
                        "in wall-clock when running the full population suite.")
    p.add_argument("--skip", nargs="*", default=[],
                   help="experiment names to skip")
    p.add_argument("--only", nargs="*", default=[],
                   help="if set, only run these experiments")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--gate-on-e04", action="store_true",
                   help="abort phase 4 if E04 produces no rows")
    p.add_argument("--attn-impl", default="flash_attention_2",
                   help="attention implementation (default flash_attention_2). "
                        "Use 'eager' to debug numerics; 'sdpa' on machines "
                        "without flash-attn installed.")
    p.add_argument("--preflight", action="store_true",
                   help="run the preflight harness (CPU-only) before launching the "
                        "real campaign. Aborts if any check fails.")
    args = p.parse_args(argv)

    # CUDA allocator hygiene (deep_analysis_v2 §2.4): expandable_segments
    # avoids fragmentation when the run alternates between 27B and 8B
    # models. Also disable HF tokenizer parallelism to prevent fork-after-
    # threads warnings polluting the log.
    cur_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in cur_alloc:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            (cur_alloc + "," if cur_alloc else "") + "expandable_segments:True"
        )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.root:
        root = Path(args.root)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        commit = git_commit()[:8]
        root = Path("outputs") / f"v2_{ts}_{commit}"
    root.mkdir(parents=True, exist_ok=True)
    save_json({"models": args.models, "args": vars(args), "git_commit": git_commit()},
              root / "run_args.json")

    if args.preflight:
        tprint("[run_all] running CPU preflight before launching the real campaign...")
        from .preflight import main as preflight_main
        rc = preflight_main(["--out-root", str(root / "preflight")])
        if rc != 0:
            tprint(f"[run_all] preflight failed ({rc} experiments broken) — aborting")
            return
        tprint("[run_all] preflight passed; proceeding to model loads.")

    # Resolve effective batch size. ``--auto-batch-size`` probes free
    # CUDA memory at runtime and picks a tensor-core friendly batch.
    eff_batch = args.batch_size
    if args.auto_batch_size:
        try:
            from reward_lens.model import auto_batch_size
            # Use the largest model in args.models as the reference for
            # memory headroom: a 27B Gemma will set the batch size for
            # the whole run.
            big = max(args.models, key=lambda m: 1 if "gemma-2-27b" in m.lower() else 0)
            d_model = 4608 if "27b" in big.lower() else 4096
            n_layers = 42 if "27b" in big.lower() else 32
            # deep_analysis_v2 §2.4: the v2 run hit OOM at batch=176 on
            # SKG27 because the working-set headroom is too tight.
            # Bump weight_gb (54 → 60) and headroom (8 → 16) for the 27B
            # path so the auto-probe leaves real slack.
            weight_gb = 60.0 if "27b" in big.lower() else 16.0
            headroom_gb = 16.0 if "27b" in big.lower() else 8.0
            eff_batch = auto_batch_size(d_model=d_model, n_layers=n_layers,
                                        weight_gb=weight_gb,
                                        headroom_gb=headroom_gb,
                                        seq_len=2048)
            # Hard cap for the 27B Gemma (the empirical OOM ceiling sits
            # around batch=128 even with expandable_segments).
            if "27b" in big.lower():
                eff_batch = min(eff_batch, 96)
            tprint(f"[run_all] auto-batch-size selected {eff_batch} "
                   f"(reference model: {big})")
        except Exception as e:
            tprint(f"[run_all] auto-batch-size probe failed: {e}; falling back to "
                   f"--batch-size={args.batch_size}")
            eff_batch = args.batch_size

    models = [ModelConfig(name=m, attn_implementation=args.attn_impl) for m in args.models]

    def make(name, **extra) -> ExperimentConfig:
        if args.length_bucket:
            extra.setdefault("length_bucket", True)
        if args.shared_activation_cache:
            extra.setdefault("shared_cache_root", str(root / "_shared_cache"))
        return _make_cfg(name, root, models, n_pairs_per_dim=args.n_pairs,
                         batch_size=eff_batch, extra=extra)

    only = set(args.only) if args.only else None
    skip = set(args.skip)

    # Phase 2 — E04 spine
    if (only is None or "e04_faithfulness_population" in only) and \
       "e04_faithfulness_population" not in skip:
        ok = _run_one("e04_faithfulness_population",
                      make("e04_faithfulness_population", patching_pairs_per_dim=30),
                      fail_fast=args.fail_fast)
        if args.gate_on_e04 and not ok:
            tprint("[run_all] E04 failed — gate triggered, exiting")
            cmd_report(argparse.Namespace(runs=str(root)))
            return

    # Phase 3 — E17
    if (only is None or "e17_reward_editing" in only) and \
       "e17_reward_editing" not in skip:
        _run_one("e17_reward_editing", make("e17_reward_editing"),
                 fail_fast=args.fail_fast)

    # Phase 4 — remaining
    for name in PHASE_4:
        if only is not None and name not in only:
            continue
        if name in skip:
            continue
        extras = {}
        if name == "e05_circuit_overlap":
            extras["e03_root"] = str(root / "e03_attribution_population")
        if name == "e13_scale_study":
            extras["upstream_root"] = str(root)
        _run_one(name, make(name, **extras), fail_fast=args.fail_fast)

    # E20 deliberately not invoked — gated on user review of E04 + E17.

    cmd_report(argparse.Namespace(runs=str(root)))
    tprint(f"\n[run_all] complete. results under: {root}")


if __name__ == "__main__":
    main()
