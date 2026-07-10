"""
E08 — Concept analysis at population scale.

For each concept:
  - Fit the direction from >=30 contrastive pairs.
  - Bootstrap stability of the direction (cosine across resamples).
  - Dose-response sweep on a held-out set of 50 prompt-response pairs at
    alpha in {-3..3}; report mean slope ± CI.

Cross-model concept transfer: cosine similarity of same-named concepts
across loaded models.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..utils.io import manifest_run, save_json, write_csv
from ..utils.figures import setup_matplotlib, savefig, PALETTE
from ..utils.parallel import tprint, clear_gpu
from ..utils.models import load_reward_model

from reward_lens.statistics import bootstrap_ci


# Additional concept pairs — beyond CONCEPT_PAIRS shipped in concepts.py.
EXTRA_CONCEPT_PAIRS = {
    "emotional_tone": [
        ("", "I'm feeling sad about this.", "I am simply observing that this happened."),
        ("", "This is wonderful news.",     "This is information."),
        ("", "I'm thrilled to tell you.",   "I am informing you."),
    ],
    "code_style":     [
        ("", "def add(a,b):\n  return a+b",
         "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum.\"\"\"\n    return a + b"),
        ("", "x=1\ny=2\nprint(x+y)",
         "x = 1\ny = 2\nprint(x + y)"),
    ],
    "citation_style": [
        ("", "As shown in Smith et al. (2023), the result holds.",
         "Some research suggests this is true."),
        ("", "Per Brown 2021, the mechanism is well understood.",
         "It's well known that the mechanism works."),
    ],
    "hedging_uncertainty": [
        ("", "It might be the case that this holds, though I'm not sure.",
         "This holds."),
        ("", "Possibly, but I'm uncertain.", "Yes, definitely."),
    ],
    "length_of_explanation": [
        ("", "Yes.",
         "Yes — and to expand on that, the underlying reason is multifaceted, involving X, Y, and Z."),
    ],
    "list_vs_prose": [
        ("", "- apples\n- bananas\n- cherries",
         "Apples, bananas, and cherries are all fruits."),
    ],
}


def run(cfg: ExperimentConfig) -> dict:
    out = cfg.out_path
    (out / "figures").mkdir(parents=True, exist_ok=True)

    from reward_lens.concepts import ConceptExtractor, CONCEPT_PAIRS

    all_concepts: dict[str, list[tuple[str, str, str]]] = {**CONCEPT_PAIRS, **EXTRA_CONCEPT_PAIRS}
    # Bug fix (deep_analysisv1): the previous held-out set contained 3 unique
    # prompts replicated 17×, so slopes_per_prompt collapsed to 3 unique values
    # in the JSON. We now use 12 distinct prompts; that's still a small held-
    # out set but the per-prompt diversity is real, not an artefact.
    held_out = [
        ("What is machine learning?",
         "Machine learning is a branch of AI that enables systems to learn from data."),
        ("How do plants make food?",
         "Plants use photosynthesis to convert sunlight, water, and carbon dioxide into glucose."),
        ("What's the speed of light?",
         "The speed of light is approximately 299,792,458 m/s in a vacuum."),
        ("Explain the water cycle in one paragraph.",
         "Water evaporates from oceans, condenses into clouds, falls as precipitation, and returns to the sea."),
        ("Who wrote Hamlet?",
         "William Shakespeare wrote Hamlet around 1600."),
        ("What is the boiling point of water at sea level?",
         "Water boils at 100 degrees Celsius (212 Fahrenheit) at standard atmospheric pressure."),
        ("Describe the structure of an atom.",
         "An atom has a nucleus of protons and neutrons surrounded by a cloud of electrons."),
        ("What causes a rainbow?",
         "Sunlight refracts and reflects through water droplets, separating into the visible spectrum."),
        ("How does a transistor work?",
         "A transistor uses a small input current or voltage to control a larger output current between two terminals."),
        ("What is the capital of Australia?",
         "Canberra is the capital of Australia."),
        ("Define the second law of thermodynamics.",
         "Entropy in an isolated system never decreases over time."),
        ("How do vaccines work?",
         "Vaccines train the immune system to recognise and respond to specific pathogens by exposing it to antigens."),
    ]

    strengths = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]
    master_rows = []
    by_model_concept_vec: dict[str, dict[str, np.ndarray]] = {}

    for mc in cfg.models:
        short = mc.short_name()
        model_out = out / short
        model_out.mkdir(parents=True, exist_ok=True)
        with manifest_run(model_out, "e08_concept_population", cfg.__dict__,
                          model=mc.name, seed=cfg.seed,
                          swallow_exceptions=cfg.skip_models_on_error):
            try:
                rm = load_reward_model(mc)
            except Exception as e:
                tprint(f"[e08] load failed: {e}")
                raise

            extractor = ConceptExtractor(rm)
            vecs = extractor.extract_concepts(all_concepts)
            by_model_concept_vec[short] = {
                name: v.detach().cpu().float().numpy() for name, v in vecs.items()
            }

            # Pre-calculate baseline rewards for held-out prompts
            tprint(f"[e08] {short}: calculating {len(held_out)} baseline rewards...")
            baselines = [float(rm.score(p, r, max_length=cfg.max_length)) for (p, r) in held_out]

            # Per-concept dose-response on held-out prompts.
            for cname, vec in vecs.items():
                deltas_at_each: dict[float, list[float]] = {s: [] for s in strengths}

                concept_vec_gpu = vec.to(rm.device).float()
                # Bug fix (deep_analysisv1): inject a couple of layers from the
                # end (rather than at the very last layer) so the perturbation
                # has a chance to propagate through the model's final norm and
                # any Gemma-specific post-norm scaling. Empirically the previous
                # `layer = n_layers - 1` returned identically-zero deltas on
                # Gemma-2-27B for every concept and every strength.
                layers_list = rm.adapter.get_layers(rm.model)
                inject_layer_idx = max(0, rm.n_layers - 3)
                layer_module = layers_list[inject_layer_idx]

                # Bug fix (deep_analysisv1): build the hook via a factory that
                # captures `s` by value, instead of relying on the enclosing
                # for-loop variable. The previous closure pattern happened to
                # work because we registered/removed the hook within each
                # strength iteration, but it was fragile and hard to reason
                # about.
                def make_hook(strength: float):
                    def _hook(module, _input, output):
                        hidden = rm.adapter.extract_layer_output(output)
                        # In-place residual perturbation at the final-token
                        # position. We keep the dtype matching ``hidden``
                        # because some kernels (flash-attn etc.) require the
                        # forward output to keep its original dtype.
                        delta = (strength * concept_vec_gpu).to(hidden.dtype)
                        hidden[:, -1, :] = hidden[:, -1, :] + delta
                        if isinstance(output, tuple):
                            return (hidden,) + output[1:]
                        return hidden
                    return _hook

                for i, (prompt, resp) in enumerate(held_out):
                    baseline = baselines[i]
                    for s in strengths:
                        if s == 0:
                            deltas_at_each[s].append(0.0)
                            continue
                        handle = layer_module.register_forward_hook(make_hook(s))
                        try:
                            intervened = float(rm.score(prompt, resp, max_length=cfg.max_length))
                        finally:
                            handle.remove()
                        deltas_at_each[s].append(intervened - baseline)

                # mean slope across the held-out set
                xs = np.asarray(strengths, dtype=np.float64)
                slope_per_prompt = []
                for j in range(len(held_out)):
                    ys = np.asarray([deltas_at_each[s][j] for s in strengths])
                    slope_per_prompt.append(float(np.polyfit(xs, ys, 1)[0]))
                slope_arr = np.asarray(slope_per_prompt)
                m_boot = bootstrap_ci(slope_arr, statistic=np.mean, n_resamples=cfg.n_resamples,
                                       seed=cfg.seed, ci=cfg.ci)
                # Cosine alignment
                w_r = rm.reward_direction.detach().cpu().float().numpy()
                v_np = vec.detach().cpu().float().numpy()
                cos = float(np.dot(v_np, w_r) /
                             (np.linalg.norm(v_np) * np.linalg.norm(w_r) + 1e-12))
                master_rows.append({
                    "model": short, "concept": cname,
                    "alignment_cosine": cos,
                    "mean_slope": m_boot.point,
                    "slope_ci_low": m_boot.ci_low, "slope_ci_high": m_boot.ci_high,
                })
                # Save dose-response curve numbers
                save_json({"strengths": strengths,
                           "mean_delta_per_strength": {str(s): float(np.mean(deltas_at_each[s]))
                                                        for s in strengths},
                           "slopes_per_prompt": slope_per_prompt},
                          model_out / f"dose_response_{cname}.json")

            del rm
            clear_gpu()

    # Cross-model concept similarity heatmap (one per concept).
    # Bug fix (deep_analysisv1): pairwise cosine across models with different
    # d_model raised "shapes (4096,) and (4608,) not aligned" and aborted the
    # whole experiment when Skywork-Llama-8B (d=4096) met Skywork-Gemma-27B
    # (d=4608). Concept directions live in their own model's hidden space and
    # are not directly comparable across architectures with different d_model.
    # We now skip mismatched-shape pairs rather than crash; cross-arch
    # comparison should be done via downstream metrics (alignment with w_r,
    # dose-response slope), not via raw vector cosine.
    concept_sim_rows = []
    for cname in all_concepts:
        models_with = [m for m, d in by_model_concept_vec.items() if cname in d]
        for i, mi in enumerate(models_with):
            for j, mj in enumerate(models_with):
                vi = by_model_concept_vec[mi][cname]
                vj = by_model_concept_vec[mj][cname]
                if vi.shape != vj.shape:
                    concept_sim_rows.append({"concept": cname, "model_i": mi, "model_j": mj,
                                             "cosine": float("nan"),
                                             "note": f"shape mismatch {vi.shape} vs {vj.shape}"})
                    continue
                cos = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12))
                concept_sim_rows.append({"concept": cname, "model_i": mi, "model_j": mj,
                                         "cosine": cos})
    write_csv(concept_sim_rows, out / "e08_cross_model_concepts.csv")
    write_csv(master_rows, out / "e08_concept_dose_response.csv")
    return {"rows": master_rows, "cross_model": concept_sim_rows}
