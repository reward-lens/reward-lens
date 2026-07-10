"""
Preference-pair dataset loaders.

Each loader returns a list of ``PreferencePair``. Loaders cache to
``~/.cache/reward_lens/datasets/``. Network-dependent calls are guarded —
if a dataset can't be reached the loader returns an empty list and logs a
warning, never raises.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreferencePair:
    prompt: str
    preferred: str
    dispreferred: str
    dimension: str = "unknown"
    source: str = "unknown"
    pair_id: Optional[str] = None
    metadata: Optional[dict] = None


CACHE_DIR = Path(os.path.expanduser("~/.cache/reward_lens/datasets"))


def _try_hf_load(*args, **kwargs):
    """Best-effort wrapper around ``datasets.load_dataset``.

    Robust to a couple of v2-observed failure modes:
      - The requested ``split`` doesn't exist (e.g. RewardBench-2 has only
        ``test``; SafetyBench has ``claude``/``gpt``). Pre-v2 the loader
        warned and returned ``None``, leaving downstream calls to score on
        whatever empty fallback the consumer used. Now we ask
        ``get_dataset_split_names`` first and pick a present split when the
        requested one is missing.
      - Network is unreachable (returns ``None`` after a single warn, no
        retry — the surrounding call sites tolerate empty lists).
    """
    try:
        from datasets import load_dataset, get_dataset_split_names  # type: ignore
    except Exception as e:
        warnings.warn(f"datasets.load_dataset failed: {e}")
        return None

    try:
        return load_dataset(*args, **kwargs)
    except Exception as e:
        msg = str(e)
        # Check if the failure is "Unknown split" — try to recover by
        # picking a real split.
        if "split" in msg.lower() and ("unknown" in msg.lower() or "should be one of" in msg.lower()):
            try:
                # args[0] is the dataset id; the requested split may be in
                # kwargs["split"] or args[1].
                ds_name = args[0] if args else kwargs.get("path")
                cache_dir = kwargs.get("cache_dir")
                available = get_dataset_split_names(ds_name)
                if available:
                    new_kwargs = dict(kwargs)
                    new_kwargs["split"] = available[0]
                    warnings.warn(
                        f"datasets.load_dataset: requested split not found for "
                        f"'{ds_name}'; falling back to '{available[0]}' "
                        f"(available: {available})"
                    )
                    if "split" in new_kwargs and len(args) > 1:
                        return load_dataset(*args, **{k: v for k, v in new_kwargs.items() if k != "split"})
                    return load_dataset(*args, **new_kwargs)
            except Exception as e2:
                warnings.warn(f"datasets.load_dataset split-recovery failed: {e2}")
                return None
        warnings.warn(f"datasets.load_dataset failed: {e}")
        return None


def load_rewardbench(split: str = "filtered", subset: Optional[str] = None,
                     limit: Optional[int] = None) -> list[PreferencePair]:
    """RewardBench (allenai/reward-bench)."""
    ds = _try_hf_load("allenai/reward-bench", split=split, cache_dir=str(CACHE_DIR))
    if ds is None:
        return []
    pairs: list[PreferencePair] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        sub = row.get("subset", "unknown")
        if subset is not None and sub != subset:
            continue
        pairs.append(PreferencePair(
            prompt=row.get("prompt", ""),
            preferred=row.get("chosen", ""),
            dispreferred=row.get("rejected", ""),
            dimension=sub,
            source="rewardbench",
            pair_id=f"rb-{row.get('id', i)}",
        ))
    return pairs


def load_rewardbench2(limit: Optional[int] = None) -> list[PreferencePair]:
    ds = _try_hf_load("allenai/reward-bench-2", split="train", cache_dir=str(CACHE_DIR))
    if ds is None:
        return []
    pairs = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        # RB-2 has a chosen + multiple rejected; take the first rejected.
        rejected = row.get("rejected") or row.get("rejected_responses") or []
        if isinstance(rejected, list) and rejected:
            rej = rejected[0]
        elif isinstance(rejected, str):
            rej = rejected
        else:
            continue
        pairs.append(PreferencePair(
            prompt=row.get("prompt", ""),
            preferred=row.get("chosen", ""),
            dispreferred=rej,
            dimension=row.get("subset", "unknown"),
            source="rewardbench2",
            pair_id=f"rb2-{row.get('id', i)}",
        ))
    return pairs


def load_rmbench(limit: Optional[int] = None) -> list[PreferencePair]:
    ds = _try_hf_load("THU-KEG/RM-Bench", split="train", cache_dir=str(CACHE_DIR))
    if ds is None:
        return []
    pairs = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        chosen = row.get("chosen") or row.get("response_chosen")
        rejected = row.get("rejected") or row.get("response_rejected")
        if not chosen or not rejected:
            continue
        pairs.append(PreferencePair(
            prompt=row.get("prompt", row.get("instruction", "")),
            preferred=chosen if isinstance(chosen, str) else chosen[0] if chosen else "",
            dispreferred=rejected if isinstance(rejected, str) else rejected[0] if rejected else "",
            dimension=row.get("domain", row.get("category", "unknown")),
            source="rm-bench",
            pair_id=f"rmb-{i}",
        ))
    return pairs


def load_judgebench(limit: Optional[int] = None) -> list[PreferencePair]:
    ds = _try_hf_load("ScalerLab/JudgeBench", split="train", cache_dir=str(CACHE_DIR))
    if ds is None:
        return []
    pairs = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        chosen = row.get("chosen") or row.get("response_a")
        rejected = row.get("rejected") or row.get("response_b")
        if not chosen or not rejected:
            continue
        pairs.append(PreferencePair(
            prompt=row.get("question", row.get("prompt", "")),
            preferred=chosen, dispreferred=rejected,
            dimension=row.get("category", "unknown"),
            source="judgebench",
            pair_id=f"jb-{i}",
        ))
    return pairs


def load_helpsteer2(limit: Optional[int] = None) -> list[PreferencePair]:
    """Construct synthetic pairs from HelpSteer2 by pairing high vs low
    helpfulness responses to the same prompt."""
    ds = _try_hf_load("nvidia/HelpSteer2", split="train", cache_dir=str(CACHE_DIR))
    if ds is None:
        return []
    by_prompt: dict[str, list[dict]] = {}
    for row in ds:
        p = row.get("prompt", "")
        by_prompt.setdefault(p, []).append(row)
    pairs = []
    for p, rows in by_prompt.items():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: r.get("helpfulness", 0))
        lo, hi = rows[0], rows[-1]
        if hi.get("helpfulness", 0) <= lo.get("helpfulness", 0):
            continue
        pairs.append(PreferencePair(
            prompt=p, preferred=hi["response"], dispreferred=lo["response"],
            dimension="helpfulness", source="helpsteer2",
            pair_id=f"hs2-{len(pairs)}",
        ))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def load_pku_safe(limit: Optional[int] = None) -> list[PreferencePair]:
    ds = _try_hf_load("PKU-Alignment/PKU-SafeRLHF", split="train", cache_dir=str(CACHE_DIR))
    if ds is None:
        return []
    pairs = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        better = row.get("better_response_id", 0)
        r0, r1 = row.get("response_0", ""), row.get("response_1", "")
        if not r0 or not r1:
            continue
        if better == 0:
            chosen, rejected = r0, r1
        else:
            chosen, rejected = r1, r0
        pairs.append(PreferencePair(
            prompt=row.get("prompt", ""),
            preferred=chosen, dispreferred=rejected,
            dimension="safety", source="pku-safe-rlhf",
            pair_id=f"pku-{i}",
        ))
    return pairs


# Convenience: a single function that returns a sensible default mix.
def load_default_population(per_dim_target: int = 150) -> list[PreferencePair]:
    """Return a mixed population suitable for E04 faithfulness.

    Includes RewardBench helpfulness/safety/reasoning + diagnostic_v2 as fallback.
    """
    pairs: list[PreferencePair] = []
    for sub in ("chat", "chat-hard", "safety", "reasoning"):
        pairs.extend(load_rewardbench(subset=sub, limit=per_dim_target))
    if not pairs:
        from reward_lens.diagnostic_data_v2 import get_pairs_v2
        pairs = [
            PreferencePair(
                prompt=p.prompt, preferred=p.preferred, dispreferred=p.dispreferred,
                dimension=p.dimension, source="diagnostic_v2",
                pair_id=f"dv2-{i}",
            )
            for i, p in enumerate(get_pairs_v2())
        ]
    return pairs
