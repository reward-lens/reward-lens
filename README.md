# reward-lens

[![PyPI version](https://badge.fury.io/py/reward-lens.svg)](https://pypi.org/project/reward-lens/)

**Mechanistic interpretability toolkit for reward models.**

The first comprehensive open-source library for understanding *what happens inside* the models that define the RLHF training signal. Reward-lens is to reward model interpretability what TransformerLens is to generative model interpretability — the foundation that makes the work possible.

---

## Known Limitations (Read Before Using)

### Attribution ≠ Causal Importance
The most important finding from our own validation experiments:
**component attribution does NOT reliably predict causal importance.**
Spearman ρ between attribution and patch effects was -0.256 (Skywork) and
-0.027 (ArmoRM) — negative to zero, never positive.

This means: use the Reward Lens and attribution for *exploration*,
then validate important claims with activation patching.
This actually strengthens your credibility in the mech interp community — researchers respect honesty about limitations far more than overselling.

---

## Why This Exists

Every RLHF-trained language model was shaped by a reward model. The reward model is the mathematical object that encodes "what we want." It is the most safety-critical component in the alignment pipeline — and as of early 2026, it has received approximately 0.5% of the interpretability community's attention.

This is not because reward models are hard to study. They may actually be *easier* than generative models:

- **Scalar output** — attribution targets a single number, not a 50K-token distribution
- **Built-in contrastive structure** — preference pairs give natural controlled comparisons
- **Known "answer direction"** — the reward head weight vector defines exactly what the model is optimizing

Reward-lens provides the tools to exploit these structural advantages.

---

## Architectural Decisions

### Why not TransformerLens?

TransformerLens was built for generative models. Its core abstractions — the logit lens, direct logit attribution, the unembedding matrix — all assume the model outputs a distribution over vocabulary tokens. Reward models replace the unembedding with a scalar head, which breaks every one of these tools.

Rather than fighting TransformerLens's abstractions, reward-lens builds purpose-built primitives directly on HuggingFace `transformers` models using lightweight PyTorch hooks. This means:

- **Any HuggingFace reward model works out of the box** — `AutoModelForSequenceClassification`, custom reward heads, multi-objective models
- **No model zoo dependency** — if HuggingFace can load it, reward-lens can analyze it
- **The hook system is minimal and auditable** — ~200 lines, not thousands

### Why not nnsight?

nnsight is a powerful general-purpose intervention library. But reward model interpretability needs domain-specific primitives — reward lens plots, differential reward attribution, preference circuit identification — that would be clumsy to build on top of a generic framework. We build these as first-class citizens.

### The Core Insight

The reward head is a linear projection: `r(x,y) = w_r^T @ h_final + b`. The weight vector `w_r` defines the **reward direction** in activation space. Every tool in this library is, at its core, a projection onto or decomposition along this direction:

- **Reward Lens**: project each layer's residual stream onto `w_r` to see when preference forms
- **Component Attribution**: decompose `h_final` into per-head, per-MLP contributions and project each onto `w_r`
- **Feature Attribution**: decompose through SAE features and measure each feature's alignment with `w_r`
- **Activation Patching**: swap components between preferred/dispreferred and measure reward change

---

## Installation

Install from PyPI (recommended):
```bash
pip install reward-lens
```

### Advanced Installation (from source)

Clone the repository and install:
```bash
git clone https://github.com/suhailnadaf509/reward-lens.git
cd reward-lens
pip install -e .
```

For SAE training support:
```bash
pip install -e ".[sae]"
```

For development:
```bash
pip install -e ".[all]"
```

---

## Quick Start

### 5-Line Reward Lens

```python
from reward_lens import RewardModel, reward_lens_plot

model = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

prompt = "Explain quantum computing."
good = "Quantum computing uses qubits that can exist in superposition..."
bad = "Quantum computing is when computers are really fast..."

reward_lens_plot(model, prompt, good, bad, save_path="reward_lens.png")
```

### Full Analysis Pipeline

```python
from reward_lens import RewardModel
from reward_lens.lens import RewardLens
from reward_lens.attribution import ComponentAttribution
from reward_lens.patching import ActivationPatcher

# Load model
rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

# Define preference pair
prompt = "What is 2+2?"
preferred = "2+2 equals 4."
dispreferred = "2+2 equals 5."

# 1. Reward Lens — when does preference form?
lens = RewardLens(rm)
result = lens.trace(prompt, preferred, dispreferred)
result.plot()  # Layer-by-layer preference formation
print(f"Preference crystallizes at layer {result.crystallization_layer}")

# 2. Component Attribution — which heads/MLPs drive the preference?
attrib = ComponentAttribution(rm)
components = attrib.attribute(prompt, preferred, dispreferred)
components.plot_top_k(k=15)  # Top 15 components by reward contribution

# 3. Activation Patching — which components are causally necessary?
patcher = ActivationPatcher(rm)
effects = patcher.patch_all_components(prompt, preferred, dispreferred)
effects.plot()  # Heatmap of patch effects
```

### Reward Hacking Detection

```python
from reward_lens import RewardModel
from reward_lens.hacking import HackingDetector

rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
detector = HackingDetector(rm)

# Test for known failure modes
report = detector.scan(
    prompt="Explain relativity.",
    response="Einstein's theory of relativity...",
    tests=["length", "confidence", "formatting", "sycophancy"],
)
report.print_summary()
# Length bias: +0.34 reward per 100 tokens (SIGNIFICANT)
# Confidence bias: +0.12 for authoritative vs hedged (moderate)
# Formatting bias: +0.08 for markdown vs plain (low)
```

### Predictive Hacking Analysis (v0.2.0)

```python
from reward_lens import RewardModel, DistortionAnalyzer
from reward_lens.diagnostic_data import get_diagnostic_pairs

rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

# Predict which quality dimensions are under-covered (will be hacked)
analyzer = DistortionAnalyzer(rm)
report = analyzer.compute_distortion_index(
    quality_dimensions=["helpfulness", "safety", "honesty"],
    evaluation_probes={
        "helpfulness": get_diagnostic_pairs(["helpfulness"]),
        "safety": get_diagnostic_pairs(["safety"]),
        "honesty": [],  # No probes - will be flagged as under-covered!
    },
)
report.print_summary()
# Shows "honesty" has high distortion index (likely to be hacked)
```

### Misalignment Cascade Detection (v0.2.0)

```python
from reward_lens import MisalignmentCascadeDetector

detector = MisalignmentCascadeDetector(rm)
report = detector.detect_cascade()  # Tests multiple misalignment dimensions
report.print_summary()
# Shows if failures are correlated (systemic vulnerability)
```

### Concept Vector Analysis (v0.2.0)

```python
from reward_lens import quick_concept_analysis

report = quick_concept_analysis(rm)
report.print_summary()
# Shows which concepts (confidence, verbosity, sycophancy)
# align with reward and may be hackable
```

---

## Core Modules

### `reward_lens.model` — Reward Model Wrapper

Wraps any HuggingFace reward model with hooks for activation caching and intervention. Handles the architectural differences between single-scalar models (Skywork, Starling) and multi-objective models (ArmoRM, Nemotron).

### `reward_lens.lens` — Reward Lens

The core primitive. Projects intermediate residual stream states onto the reward direction to trace preference formation across layers. The reward model analogue of the logit lens.

### `reward_lens.attribution` — Component Attribution

Decomposes the reward score into signed per-component contributions (each attention head and MLP layer). Answers: "why did the model assign this score?"

### `reward_lens.patching` — Activation Patching

Causal intervention tool. Swaps component activations between preferred and dispreferred completions to identify causally necessary components for each preference dimension.

### `reward_lens.hacking` — Reward Hacking Detection

Automated detection of hackable features in reward models. Tests for length bias, confidence bias, formatting bias, sycophancy, and more. Produces vulnerability reports.

### `reward_lens.sae` — Sparse Autoencoder Integration

Train and apply SAEs to reward model activations. Decompose reward into interpretable feature-level contributions. Identify features aligned with the reward direction.

### `reward_lens.diagnostic_data` — Diagnostic Datasets

Curated preference pairs for controlled experiments across preference dimensions: helpfulness, safety, verbosity, sycophancy, formatting, confidence.

---

## New Modules (v0.2.0)

Based on cutting-edge interpretability research (2025-2026):

### `reward_lens.distortion` — Distortion Index

Predicts which quality dimensions are under-covered by evaluation and thus likely to be hacked. Based on "Reward Hacking as Equilibrium under Finite Evaluation" — moves from detecting hacking to **predicting** it.

### `reward_lens.divergence_patching` — Divergence-Aware Patching

Extends activation patching with out-of-distribution detection. Flags when interventions create divergent representations that may make causal claims unreliable. Based on "Addressing Divergent Representations from Causal Interventions."

### `reward_lens.cascade` — Misalignment Cascade Detection

Tests for correlations between different misalignment dimensions. Based on "Natural Emergent Misalignment from Reward Hacking" — reward hacking onset correlates with broad emergent misalignment.

### `reward_lens.conflict` — Reward Conflict Analysis

Classifies relationships between reward terms as aligned/orthogonal/in-conflict. In-conflict terms may cause models to hide reasoning. Based on "When Can We Safely Optimize CoT?"

### `reward_lens.concepts` — Concept Vector Extraction

Extracts linear concept vectors from activations and analyzes their reward alignment. Identifies concepts that may enable hacking (e.g., confidence, verbosity, sycophancy). Based on "Emotion Concepts and their Function in an LLM."

---

## Supported Models

| Model | Architecture | Type | Status |
|-------|-------------|------|--------|
| Skywork-Reward-Llama-3.1-8B-v0.2 | Llama 3.1 + classification head | Single scalar | ✅ Full support |
| ArmoRM-Llama3-8B-v0.1 | Llama 3 + multi-objective head + MoE gating | Multi-objective | ✅ Full support |
| Nemotron-4-340B-Reward | Nemotron + 5-dim linear head | Multi-dimensional | ⚠️ Requires multi-GPU |
| FsfairX-LLaMA3-RM-v0.1 | Llama 3 + classification head | Single scalar | ✅ Full support |
| Any `AutoModelForSequenceClassification` | Varies | Single scalar | ✅ Auto-detected |

**Adding new models:** Any model loadable via `AutoModelForSequenceClassification` with a linear reward head works automatically. Models with custom architectures (like ArmoRM's MoE gating) need a thin adapter — see `reward_lens/model_adapters/`.

---

## What This Toolkit Can and Cannot Do

### Can Do
- Trace preference formation across layers for any HuggingFace reward model
- Decompose reward scores into per-component (head/MLP) signed contributions
- Identify causally necessary components via activation patching
- Detect reward hacking vulnerabilities (length, confidence, formatting, sycophancy)
- Train SAEs on reward model activations and decompose reward through features
- Compare preference circuits across different reward models

### Cannot Do (Honestly)
- **Process reward models (PRMs)** are partially supported — per-step analysis works, but step-boundary detection and accumulated quality tracking are not yet implemented
- **Proprietary models** — this toolkit requires access to model weights. API-only models cannot be analyzed
- **Causal claims from correlational tools** — the reward lens and component attribution are observational. Only activation patching provides causal evidence. We are explicit about this distinction in the API
- **Guaranteed completeness** — mechanistic interpretability never guarantees you've found everything. The toolkit helps you find what's there, but absence of evidence is not evidence of absence

---

## Compute Requirements

All analyses run in inference mode. No training of the reward model is required.

| Analysis | 8B model | Hardware | Time |
|----------|----------|----------|------|
| Reward Lens (single pair) | ~2 forward passes | 1× GPU (16GB+) | ~5 seconds |
| Component Attribution (single pair) | ~2 forward passes | 1× GPU (16GB+) | ~10 seconds |
| Activation Patching (all components) | ~n_components × 2 forward passes | 1× GPU (24GB+) | ~30 minutes |
| SAE Training (single layer) | Activation collection + training | 1× GPU (24GB+) | ~8-24 hours |
| Full Hacking Scan | ~50 paired forward passes | 1× GPU (16GB+) | ~5 minutes |

---

## Citation

```bibtex
@software{nadaf2026rewardlens,
    title = {reward-lens: Mechanistic Interpretability Toolkit for Reward Models},
    author = {Nadaf, Mohammed Suhail B},
    year = {2026},
    url = {https://github.com/suhailnadaf509/reward-lens},
}
```

---

## License

MIT
