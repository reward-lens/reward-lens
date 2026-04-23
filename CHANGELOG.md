# Changelog

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
