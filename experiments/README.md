# reward-lens v2 experiments

Population-scale, statistically-grounded successor to `run_weekend_experiment.py`.
Every aggregate stat ships with a bootstrap CI, multi-test corrections are
first-class, and every experiment writes per-pair JSONL intermediates so
re-aggregation is free.

## Quick start

List registered experiments:
```
python -m experiments.runner list
```

Run a single experiment from a config:
```
python -m experiments.runner run e04_faithfulness_population \
    --config configs/experiments/e04_faithfulness.yaml
```

Run the full v2 plan (gated phase order: E04 spine → E17 → rest):
```
python -m experiments.run_all --models \
    Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    RLHFlow/ArmoRM-Llama3-8B-v0.1 \
    Skywork/Skywork-Reward-Gemma-2-27B-v0.2
```

Generate REPORT.md:
```
python -m experiments.report --runs outputs/v2_<timestamp>_<commit>
```

## Per-experiment summary

| ID  | Module                          | What it answers |
|-----|---------------------------------|-----------------|
| E01 | e01_baseline_and_diagnostics    | Per-dim accuracy with bootstrap CI; adapter health-check. |
| E02 | e02_lens_population             | Distribution of crystallisation depth per dim; ridgelines. |
| E03 | e03_attribution_population      | Top-k component frequency with bootstrap CI; rank stability. |
| E04 | e04_faithfulness_population     | **Spine.** Distribution of per-pair Spearman ρ between |attr| and |patch| over n≥150 pairs/dim. |
| E05 | e05_circuit_overlap             | Per-pair Jaccard between dims; bootstrap CI; dendrogram. |
| E06 | e06_hacking_at_scale            | Bootstrap-CI'd Cohen's d per (dim × model); BH-FDR; forest plot. |
| E07 | e07_cascade_at_scale            | Cross-dim correlation matrix with CIs and BH-FDR. |
| E08 | e08_concept_population          | Concept dose-response with CIs; cross-model concept transfer. |
| E09 | e09_conflict_population         | Pairwise term-direction cosines with bootstrap CIs. |
| E10 | e10_distortion_index            | Distortion under (full / honesty-removed / safety-removed / only-helpfulness) strategies. |
| E11 | e11_divergence_patching         | % pernicious patches; reliability distribution. |
| E12 | e12_sae_feature_decomposition   | Reward-direction alignment of SAE features; cumulative variance. |
| E13 | e13_scale_study                 | 4-metric panel across the model ladder. |
| E14 | e14_cross_architecture          | Per-pair lens-trajectory correlation across models; family vs cross-family. |
| E15 | e15_head_path_patching          | Top heads per dim + 2-hop head→MLP path effects. |
| E17 | e17_reward_editing              | Bias-vs-accuracy frontier from interpretability-guided edits. |
| E18 | e18_armorm_multi_objective      | 19-objective deep dive on ArmoRM (per-objective lens + conflict). |

## What is intentionally NOT here

- **E16** — merged into E15 per gate amendment 2.
- **E19** — process reward models, deferred per gate amendment 3.
- **E20** — distortion-index validation, gated on E04 + E17 results per gate amendment 1. Run only after the user reviews phase-3 outputs.

## Statistical bars

Per the v2 plan §7:
- No `inf` Cohen's d (NaN with documented reason instead).
- No claims grounded in n < 30 without a `directional` tag.
- All correlations ≥ |0.3| ship with 95% bootstrap CI.
- All multi-test claims (across dims, across models) get BH-FDR.
- Effect sizes always with their CI.

## Resumability

Every experiment writes a `manifest.json` (status: running / complete / failed)
plus per-pair JSONL. Re-running an experiment skips records by `record_id`.

## Adding a new experiment

1. Create `experiments/eNN_<name>/run.py` exposing `run(cfg) -> dict`.
2. Register it in `experiments/registry.py`.
3. Add a YAML under `configs/experiments/`.
