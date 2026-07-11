<div class="rl-chips">
  <span class="rl-chip rl-chip--obs">Observational</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">gauge</span> raw only</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">works on</span> a multi-objective readout</span>
</div>

# Multi-objective geometry

**When a reward head scores nineteen objectives at once, do they point the same way or nineteen different ways?**

Most of the battery leans on a comforting fact: the reward is one direction, \(w_r\), and everything projects onto it. Some reward models break that assumption on purpose. ArmoRM reads out nineteen objectives, each with its own direction, and combines them through a gating network. For a head like that, "the reward direction" is an approximation, a weighted average standing in for nineteen genuine directions. Multi-objective geometry declines the approximation. It reads every objective's readout vector and reports the cosine between them, the full geometry, never a single collapsed number.

The chip says it works on a *multi-objective readout*, and it means it. Hand it a scalar reward model and it refuses rather than pretend a one-objective head has a geometry.

## The intuition

A multi-objective head is a matrix, one row per objective, each row a direction in the residual stream. Two objectives whose rows point the same way rise and fall together: an answer that scores well on one scores well on the other. Two rows at right angles are independent objectives. Two rows pointing apart are objectives in tension, where improving one tends to cost the other. The cosine matrix of the rows is that whole picture in one object.

The reason to look is that collapsing nineteen objectives into one average hides exactly the structure you care about. Two heads with the same average reward direction can have completely different internal geometry, one a tight bundle of aligned objectives, the other a loose spray of competing ones, and they will behave differently under optimization. The average cannot tell them apart; the geometry can.

## The math

Let the head have objective directions \(w_1, \dots, w_K\), the rows of the readout matrix. The geometry is the matrix of pairwise cosines,

\[ G_{kj} = \cos\bigl(w_k, w_j\bigr) = \frac{w_k^\top w_j}{\lVert w_k\rVert\,\lVert w_j\rVert}, \]

with ones on the diagonal by construction and symmetric off it. The instrument summarizes the off-diagonal into a mean, the most aligned and most opposed pairs, and a count of conflicting pairs. The gauge is *raw only*: these cosines are computed in the model's raw weight basis, so an objective index and its cosines mean nothing outside this specific head. Comparing the objective geometry of two different multi-objective models needs both expressed in a shared [frame](../discipline/gauge-and-frames.md) first.

## A worked run

The tiny thread here needs a head with more than one readout, so build a small nineteen-label classifier and wrap it. On an untrained head the nineteen directions are essentially random, which is itself the informative null.

```python
import torch
from transformers import LlamaConfig, LlamaForSequenceClassification
from reward_lens.signals import wrap_hf_model
from reward_lens.signals.loaders import _build_tokenizer
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import MultiObjectiveGeometry

tok = _build_tokenizer("gpt2")
torch.manual_seed(2)
cfg = LlamaConfig(vocab_size=getattr(tok, "vocab_size", 1000), hidden_size=32,
                  intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=4, max_position_embeddings=256,
                  pad_token_id=getattr(tok, "pad_token_id", 0) or 0,
                  num_labels=19, attn_implementation="eager")
signal = wrap_hf_model(LlamaForSequenceClassification(cfg).eval(), tok, device="cpu",
                       architecture="LlamaForSequenceClassification", conformance_quickcheck=False)

view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:5])
ev = mb.run(MultiObjectiveGeometry(), mb.Context(signal=signal, view=view))

import numpy as np
cos = np.array(ev.value["cosine_matrix"])
off = cos[np.triu_indices(19, k=1)]
print(ev.trust, ev.gauge, "| objectives:", ev.value["n_objectives"])
print("cosine matrix:", cos.shape, "| symmetric:", np.allclose(cos, cos.T, atol=1e-6))
print("off-diagonal cosine  min:", round(float(off.min()), 4),
      "max:", round(float(off.max()), 4), "mean:", round(float(off.mean()), 4))
```

```text
EXPLORATORY raw_only | objectives: 19
cosine matrix: (19, 19) | symmetric: True
off-diagonal cosine  min: -0.4213 max: 0.4963 mean: 0.0048
```

A mean off-diagonal cosine of \(0.005\) is the signature of random directions: nineteen untrained readout rows sit near-orthogonal, neither aligned nor opposed, because there is no trained structure to align them. That null is the baseline a real head is measured against. Ask the same instrument for a geometry it cannot compute, a scalar head, and it refuses:

```python
from reward_lens.signals import from_tiny
scalar = from_tiny(seed=0)                    # num_labels=1, one readout
mb.run(MultiObjectiveGeometry(), mb.Context(signal=scalar, view=view))
# CapabilityError: MultiObjectiveGeometry needs a multi-objective head
#                  (>=2 objective readouts); signal mfp:... exposes 0
```

### On ArmoRM the objectives are aligned, but not one direction

The real head is ArmoRM's nineteen objectives, a measured result from committed artifacts. Its geometry is the opposite of the random null: mostly positive, because every objective encodes some notion of quality, but well short of a single direction, because the objectives are genuinely distinct.

![The 19-by-19 cosine matrix of ArmoRM's objective readouts: mostly warm and positive, with cooler off-diagonal cells where objectives diverge.](../assets/figures/armo-objective-cosine-light.svg#only-light){ .rl-fig }
![The 19-by-19 cosine matrix of ArmoRM's objective readouts: mostly warm and positive, with cooler off-diagonal cells where objectives diverge.](../assets/figures/armo-objective-cosine-dark.svg#only-dark){ .rl-fig }

/// caption
**Nineteen objectives, mostly aligned, not identical.** Each cell is the cosine between two of ArmoRM's objective readouts. The matrix runs warm, so the objectives broadly agree, but the off-diagonal is far from a solid block of ones, and the cooler cells mark objectives that pull apart. Collapsing this to one average reward direction would erase every distinction the matrix shows.
///

This is why the single-direction instruments carry a footnote on ArmoRM. When you run [attribution](attribution.md) or the [reward lens](lens-crystallization.md) against a multi-objective head, they read a weighted average of these nineteen directions, and the average is a real approximation with real error. Multi-objective geometry is the instrument that shows you what the approximation smooths over.

To read ArmoRM's own head you need to load it.

!!! warning "Needs a GPU"
    ArmoRM in fp32 does not fit a laptop GPU. `load_signal` refuses a hub id unless you pass `allow_download=True`, and the forward passes are the gated part.

    ```python
    from reward_lens.signals import load_signal
    signal = load_signal("RLHFlow/ArmoRM-Llama3-8B-v0.1", allow_download=True)
    ev = mb.run(MultiObjectiveGeometry(), mb.Context(signal=signal, view=view))
    ev.value["n_objectives"]        # 19
    ```

## How to read the output

- **`n_objectives`** confirms how many readouts the head exposes, nineteen for ArmoRM.
- **`cosine_matrix`** is the full geometry; read it as a heatmap. A warm block means aligned objectives, a cool or negative cell means objectives in tension.
- **`min_cosine`** and **`max_offdiagonal_cosine`** name the most opposed and most aligned pairs, and **`n_conflicting_pairs`** counts the negative entries, a quick read on how much the objectives disagree.
- **`mean_offdiagonal_cosine`** is the overall tone, and its distance from zero tells you whether there is trained structure at all.

## When not to reach for it

It applies only to multi-objective heads and refuses scalar ones, which is a feature: a scalar reward has no objective geometry, and row-meaning one into existence would be a fabrication. Its cosines are *raw only*, so use them to understand one head's internal structure, not to compare two models' objective geometries directly; that comparison needs a shared frame. And the objective *labels* are model-specific, so "objective 7" on ArmoRM has no counterpart on another head.

It is observational. The cosine between two objective directions is a statement about the head's weights, not a proof that optimizing one objective will move another in your training run; a negative cosine is a lead to test causally, not a settled trade-off.

## How much to trust this

The number arrives at trust level **EXPLORATORY**. The cosine geometry is exact arithmetic on the head's weights, but no calibration provider is wired in this release, so no scorecard yet relates an objective geometry to behavior under multi-objective optimization. Exploratory means unaudited, not wrong. Use the matrix to see how a multi-objective head's goals relate and to know when the single-direction instruments are approximating; do not carry a raw cosine across models, and do not yet read a negative cell as a guaranteed training conflict. The trust level and what would raise it are covered in [the trust ladder](../discipline/trust-ladder.md).

Full signatures and return fields: [`MultiObjectiveGeometry`](../reference/measure.md#reward_lens.measure.battery.geometry.MultiObjectiveGeometry). For where these heads come from, see [rubric heads](../models-and-signals/rubric-heads.md).
