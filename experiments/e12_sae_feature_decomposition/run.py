"""
E12 — SAE feature decomposition (the §5.7 experiment).

For one or more models:
  1. Collect ~1-5M tokens of activations from a configurable layer set.
  2. Train TopK SAEs (default: 8x dictionary, k=32).
  3. Headline plots:
     - histogram of w_r^T d_i across the dictionary
     - cumulative reward variance explained vs k
     - top-15 features by |w_r^T d_i| with top-5 activating contexts each

Defaults are tuned for tractability — full ~5M token training is gated
behind cfg.extra["full_train"] = True. Otherwise we run a small smoke
training sized so this completes in <1h on H200.

Bug history (preflight, deep_analysisv1 follow-up): the original runner
was authored against a hypothetical API and never executed (e12 was in
the "never ran" tail of the v2 campaign). It imported a non-existent
``SAEFeatureAnalyzer`` (real name: :class:`FeatureAnalyzer`), passed a
``layers=`` kwarg to :class:`ActivationCollector` (which only takes
``model``), called ``trainer.train(buf, n_steps=...)`` (real signature
takes ``activations``/``n_epochs``), and called the bogus ``analyze()``
method instead of ``analyze_features()``. The runner now uses the real
library API end-to-end.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig, PALETTE
from ..utils.parallel import tprint, clear_gpu
from ..utils.diagnostics import load_diagnostic_v2
from ..utils.models import load_reward_model


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)
    layers = cfg.extra.get("layers", [16, 24, 28, 31])
    expansion = int(cfg.extra.get("expansion", 8))
    k = int(cfg.extra.get("k", 32))
    n_epochs = int(cfg.extra.get("n_epochs", cfg.extra.get("n_steps", 5)))
    full_train = bool(cfg.extra.get("full_train", False))
    if full_train:
        n_epochs = max(n_epochs, 20)
    sae_batch_size = int(cfg.extra.get("sae_batch_size", 4096))
    collect_max_pairs = int(cfg.extra.get("collect_pairs", cfg.n_pairs_per_dim))
    top_k_examples = int(cfg.extra.get("top_k_examples", 5))

    master_rows: list[dict] = []
    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e12_sae_feature_decomposition", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e12] load failed: {e}")
                raise

            from reward_lens.sae import SAETrainer, ActivationCollector, FeatureAnalyzer

            # Clamp requested layers to the actual model depth — the
            # default ``[16, 24, 28, 31]`` overflows on tiny preflight
            # models (n_layers=2) and on smaller production checkpoints.
            n_layers_model = rm.n_layers
            valid_layers = [L for L in layers if 0 <= L < n_layers_model]
            if not valid_layers:
                # Fall back to the deepest available layer.
                valid_layers = [n_layers_model - 1]
            tprint(f"[e12] {short}: layers={valid_layers} of {n_layers_model}")

            # 1. Collect activations.
            corpus = list(load_diagnostic_v2(limit_per_dim=collect_max_pairs))
            prompts = [p.prompt for p in corpus]
            responses = [p.preferred for p in corpus]
            collector = ActivationCollector(rm)

            for L in valid_layers:
                t0 = time.time()
                try:
                    buf = collector.collect(
                        prompts, responses, layer=L,
                        max_length=cfg.max_length, show_progress=cfg.progress,
                    )
                except Exception as e:
                    tprint(f"[e12] L{L}: activation collection failed: {e}")
                    raise
                tprint(f"[e12] {short} L{L}: collected {tuple(buf.shape)} in {time.time()-t0:.1f}s")

                # 2. Train SAE.
                d_model = rm.d_model
                # The dictionary size needs to be at least k; clamp k so
                # tiny preflight runs (d_model=32, k_default=32) don't
                # blow up.
                target_features = expansion * d_model
                eff_k = min(k, max(1, target_features // 2))
                # Adapt batch_size to the actual buffer size — the default
                # 4096 with drop_last=True silently drops everything when
                # the buffer is smaller.
                eff_batch = min(sae_batch_size, max(2, buf.shape[0] // 2))
                trainer = SAETrainer(
                    d_model=d_model,
                    n_features=target_features,
                    k=eff_k,
                    batch_size=eff_batch,
                    device=str(rm.device),
                )
                t0 = time.time()
                try:
                    sae = trainer.train(
                        buf, n_epochs=n_epochs, show_progress=cfg.progress,
                    )
                except Exception as e:
                    tprint(f"[e12] {short} L{L}: SAE training failed: {e}")
                    raise
                tprint(f"[e12] {short} L{L}: trained in {time.time()-t0:.1f}s")

                # 3. Feature analysis.
                analyzer = FeatureAnalyzer(sae, rm)
                features_info = analyzer.analyze_features(
                    buf, top_k_examples=top_k_examples, show_progress=cfg.progress,
                )

                # Save numeric outputs.
                w_r = rm.reward_direction.detach().cpu().float().numpy()
                D = sae.W_dec.detach().cpu().float().numpy()  # (n_features, d_model)
                if D.shape[1] != w_r.shape[0]:
                    D = D.T
                alignments = D @ w_r  # (n_features,)
                save_json({
                    "alignments": alignments.tolist(),
                    "n_features": int(D.shape[0]),
                }, model_out / f"sae_alignments_L{L}.json")

                # Cumulative reward variance explained vs k_top.
                order = np.argsort(np.abs(alignments))[::-1]
                cum = np.cumsum(alignments[order] ** 2)
                cum_norm = cum / max(cum[-1], 1e-12)
                save_json({"k_top": list(range(1, len(cum) + 1)),
                           "cumulative_variance": cum_norm.tolist()},
                          model_out / f"sae_cumvar_L{L}.json")
                _plot_alignment_hist(alignments, out / "figures" / f"e12_aln_{short}_L{L}")
                _plot_cumvar(cum_norm, out / "figures" / f"e12_cumvar_{short}_L{L}")

                # Top-15 features.
                top = order[:15].tolist()
                # Pull the top-activating examples for each, from features_info.
                info_by_idx = {fi.feature_idx: fi for fi in features_info}
                top_examples = []
                for fidx in top:
                    fi = info_by_idx.get(int(fidx))
                    top_examples.append({
                        "feature": int(fidx),
                        "alignment": float(alignments[fidx]),
                        "freq": float(fi.activation_frequency) if fi else None,
                        "mean_act": float(fi.mean_activation) if fi else None,
                        "top_indices": fi.top_activating_indices[:top_k_examples] if fi else [],
                    })
                save_json(top_examples, model_out / f"sae_top_features_L{L}.json")
                master_rows.append({
                    "model": short, "layer": L,
                    "top_15_indices": top,
                    "top_15_alignments": [float(alignments[i]) for i in top],
                })

            del rm
            clear_gpu()

    write_csv(master_rows, out / "e12_sae_top_features.csv")
    return {"rows": master_rows}


def _plot_alignment_hist(alignments: np.ndarray, path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(alignments, bins=80, color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("w_r^T d_i (feature alignment with reward direction)")
    ax.set_ylabel("# features")
    ax.set_yscale("log")
    ax.set_title("E12 SAE feature alignment distribution")
    savefig(fig, path)


def _plot_cumvar(cum: np.ndarray, path: Path) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, color=PALETTE[1])
    ax.set_xscale("log")
    ax.set_xlabel("k (top features by |alignment|)")
    ax.set_ylabel("cumulative reward-variance fraction")
    ax.set_title("E12 cumulative reward variance vs k")
    savefig(fig, path)
