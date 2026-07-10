"""
reward-lens v2 experiments package.

Each experiment is a self-contained module under e<NN>_*/. Modules expose
``run(cfg)`` returning a JSON-serialisable result dict, and ``save(result,
out_dir)`` writing artifacts (per-pair JSONL intermediates, aggregated
JSON, figures, CSVs).

The runner discovers modules via the registry and dispatches by name:

    python -m experiments.runner run e04_faithfulness --config configs/...

Resumability: every long-running experiment writes per-shard JSONL so that
a partial run is recoverable. See experiments.utils.io.
"""

from . import registry  # noqa: F401  — registers built-in experiments on import

__all__ = ["registry"]
