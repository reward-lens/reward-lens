"""The campaign data plane: converters from public sources to locked, lineage-honest slices.

Every slice named in ``campaign.config.SLICES`` that is a text dataset has exactly one
converter here (``campaign.data.loaders``). A converter produces a ``reward_lens`` DataView
whose items are built through the schema builders, so every item carries a lineage whose
``seed_id`` traces to a source row, whose ``builder_id`` is ``campaign.data``, and whose ops
record how the item was derived. Subsetting is deterministic everywhere: a seeded shuffle
(``config.task_seed("slice", name)``) over the source's stable row ordering, then the first n
rows that parse.

Layout:

- ``loaders``: the per-slice converters, the cache, and ``load_slice``.
- ``serde``: the JSONL codec for schema items (cache files under ``campaign/data/cache/``).
- ``tournaments``: rank and score edge extraction plus the TOPO-HODGE edge-matrix format.
- ``ingest``: ``python -m campaign.data.ingest`` builds slices and writes
  ``campaign/data/slices.lock.json`` (content hash, n, effective n, duplicate fraction,
  source, pinned revision per slice).
- ``tiny``: small synthetic stand-ins per slice for the CPU rehearsal.

Two ``SLICES`` names have no converter on purpose: ``caltransfer-organisms`` and
``adjavp-organism`` are trained-organism artifacts produced by the organism arcs, not text
datasets, so there is nothing to ingest for them here.
"""

from campaign.data.loaders import CONVERTERS, SliceResult, cache_path, load_slice

__all__ = ["CONVERTERS", "SliceResult", "cache_path", "load_slice"]
