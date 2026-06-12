# Contributing

The most useful thing you can add is reach: a reward model `reward-lens` cannot open yet. The library only earns its keep when it runs natively on the model you actually care about, so a new adapter is worth more than almost any other change. Everything below is in service of keeping that easy and keeping the core honest.

## What the project values

Three beliefs shape every decision, and a good contribution respects them.

- **Minimal abstractions.** The interventions are lightweight PyTorch hooks placed directly on a HuggingFace model. No custom model wrapper, no hooks buried in a dictionary you have to reverse-engineer. If you can read `transformers`, you can read the whole thing.
- **An auditable core.** The mechanism that reads a reward out of a hidden state is about two hundred lines, not thousands. That ceiling is deliberate. A change that doubles the core to save a few lines at a call site is usually the wrong trade.
- **Honesty over polish.** The library's own headline result is a limitation: component attribution anti-correlates with causal patching at Spearman \(\rho = -0.256\) on Skywork, and the docs lead with that rather than bury it. Hold new tools to the same standard. If a technique is exploratory, say so in the docstring. If a claim only holds after patching validation, write that down.

## Setting up a dev checkout

Fork the repo, then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/reward-lens.git
cd reward-lens
```

Make a virtual environment and install the package with its dev extras:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Confirm the environment works before you change anything by running the suite:

```bash
pytest tests/ -v
```

If that passes, you have a working checkout of `reward-lens` 1.0.0.

## Working on a change

- **Branch off with a name that says what it does:** `git checkout -b feature/model-adapter-armorm`, or `git checkout -b fix/tensor-mismatch-divergence`. Prefix with `feature/` or `fix/`.
- **Write a test for every new tool or hook,** and put it in `tests/`. Keep it light. Tests should run on a small stand-in model or on mock activations, not by downloading 8B weights on every CI run. Reserve the real 8B loads for cases explicitly marked as integration tests.
- **Format with `black` and sort imports with `isort`** before you commit. The code follows PEP 8; the formatters settle the rest, so review can stay on the substance.
- **Keep a pull request to one idea.** Do not fold an unrelated refactor into a new interpretability feature. If the change is large, open an issue first and agree on its shape with the maintainer before writing it.

## Adding a model adapter

This is the contribution that decides whether the library reaches your model or stops at the families it already ships. `reward-lens` reads a scalar reward out of a hidden state as \( r = w_r^{\top} h + b \), so for any new model it needs exactly one thing: where the reward head lives, and what its weight vector and bias are. An adapter is the object that answers that.

The adapters sit in `src/reward_lens/model_adapters/`. Each subclasses the `ModelAdapter` abstract base class and implements its eleven methods. The one that matters most, the one everything else projects onto, is `get_reward_head_params(model) -> (w_r, bias)`:

```python
class MyRewardModelAdapter(ModelAdapter):

    def get_reward_head_params(self, model):
        """Return (w_r, bias): the reward-head weight vector and its scalar bias."""
        ...
```

Get that right and the observational tools work immediately, because the reward direction \(w_r\) it returns is the direction the whole library measures against. The remaining methods point the library at the residual stream, the attention and MLP outputs, and the model's layer and head counts, so attribution and patching know where to hook. Two more methods are optional: `get_attn_o_proj`, which exposes per-head outputs for head-level patching, and `extract_reward_batch`, for batched scoring.

Once the class exists, wire it into dispatch. `get_adapter(model, model_name)` is a plain if-chain, not a registry, so adding a family means adding a branch that recognizes your model and returns your adapter. If your reward model is already an `AutoModelForSequenceClassification` with a linear head, try `GenericAdapter` first: it auto-detects the head and may cover you with no new code at all.

The [write-an-adapter how-to](../how-to/write-an-adapter.md) walks through a full adapter end to end, method by method, with a worked example.

## How to cite

If `reward-lens` helped your work, please cite it:

```bibtex
@software{nadaf2026rewardlens,
    title = {reward-lens: Mechanistic Interpretability Toolkit for Reward Models},
    author = {Nadaf, Mohammed Suhail B},
    year = {2026},
    url = {https://github.com/suhailnadaf509/reward-lens},
}
```

`reward-lens` 1.0.0 is released under the MIT license. Author: Mohammed Suhail B Nadaf.
