# Contributing to Reward Lens

Thank you for your interest in contributing to Reward Lens, the mechanistic interpretability toolkit for reward models! We're building the foundation that makes understanding the RLHF optimization target possible.

## Philosophy

Reward Lens is built on a few core beliefs:
- **Minimal Abstractions:** Prefer lightweight PyTorch hooks directly on HuggingFace models over complex, opaque abstractions.
- **Auditable Core Code:** No hooks hidden in mysterious dictionaries. ~200 lines vs thousands.
- **Truth over Hype:** As shown in our validation with component attribution vs causal importance, honesty about limitations (like Spearman $\rho$ values) builds credibility. If a technique is exploratory, say so. If it requires rigorous patching validation, document that.

## Getting Started

1. **Fork and Clone:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/reward-lens.git
   cd reward-lens
   ```
2. **Install Dev Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```
3. **Run Checks First:** Validate your env works by running the test suite:
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

1. **Branch Naming:** Create a new branch pointing directly to the feature or fix: `git checkout -b feature/model-adapter-armorm` or `fix/tensor-mismatch-divergence`.
2. **Write Tests:** Every new mechanistic tool or hook requires a corresponding test case in `tests/`. Tests should ideally run on lightweight placeholder models or mock inputs rather than downloading full 8B HF models (unless marked as an integration test).
3. **Format and Lint:** Use `black` and `isort` for formatting. We enforce PEP 8 standards.
4. **Docs:** Update inline docstrings (Sphinx/Google style) and add an example script if you are introducing a major new interpretability technique.

## Adding a New Model Adapter
Because reward models use different head architectures on top of standard LLMs:
- Check `src/reward_lens/model_adapters/` to see if the structure already exists.
- If it's a completely bespoke reward head (e.g. multi-objective output), you may need to implement a new `RewardHeadAdapter`.
- Ensure it properly returns a single scalar or exposes the contrastive diff structure natively.

## Publishing Pull Requests

- Before throwing an enormous PR, open an Issue to discuss the design with the core maintainers.
- Keep PRs focused. Do not mix unrelated refactoring with a new interpretability feature.
- Ensure all CI tests pass.
