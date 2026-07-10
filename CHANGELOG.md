# Changelog

## [2.0.0] - 2026-07-10

Major redesign. The library is reorganized around a single kernel with a lazy,
torch-free epistemics layer; the 1.0 public API is preserved under
`reward_lens.legacy`.

### Changed
- Reorganized around one kernel of subsystems: `core`, `stats`, `runtime`,
  `signals`, `data`, `concepts`, `interventions`, `geometry`, `measure`,
  `attribution`, `organisms`, `dynamics`, `loops`, `studies`, and `artifacts`.
- The top-level import is now lazy: `import reward_lens`, `reward_lens.core`, and
  `reward_lens.stats` pull only numpy, so the pure epistemics layer is usable
  without torch. Model-touching code is imported on first access.

### Added
- Sixteen reward-science studies over the kernel, plus three runtime gates
  (calibration, gauge, registration) in the stats/evidence layer.
- `reward-lens` command-line interface (console script) and an operate MCP surface.
- Artifact builders: atlas, cards, claims, safety case, and site.
- Training-loop integrations for TRL, veRL, and OpenRLHF, with tilt, anneal,
  and best-of-N.
- E-parity golden-fixture test suite.

### Compatibility
- The 1.0 public API is preserved under `reward_lens.legacy` and remains
  importable from the top level through the lazy accessor.

## [1.0.0] - 2026-04-12

### Added
- Initial release: RewardLens, ComponentAttribution, ActivationPatcher, HackingDetector
- DistortionAnalyzer: predictive reward hacking analysis
- MisalignmentCascadeDetector
- RewardConflictAnalyzer
- ConceptExtractor and quick_concept_analysis
- DivergenceAwarePatching

### Validated
- Ran experiments on RewardBench (~695 pairs) across Skywork-Reward-Llama-3.1-8B-v0.2 and ArmoRM
- Key finding: late-layer crystallization (90-97% depth for Skywork)
- Key limitation: attribution does not predict causal importance
