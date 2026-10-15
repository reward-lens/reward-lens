"""Data-converter configuration derived from the pinned campaign source.

Only imported slice definitions, source-lock location, and seed derivation are
retained. Research rosters, infrastructure paths, and spend planning are omitted.
See vendor-provenance.json for the original and derived source hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SLICES_LOCK_PATH = REPO_ROOT / "campaign" / "data" / "slices.lock.json"

@dataclass(frozen=True)
class SliceDef:
    """One named, deterministic slice of a source dataset."""

    name: str
    source: str  # HF dataset id, or "in-repo"
    split: str = "train"
    n: int | None = None  # None = full
    seed: int = 20260716  # the deterministic shuffle seed for subsampled slices
    config: str | None = None
    notes: str = ""

SLICES: dict[str, SliceDef] = {
    s.name: s
    for s in (
        SliceDef("rb2-full", "allenai/reward-bench-2", n=None,
                 notes="1,865 items, best-of-4; the capture backbone; Ties handled separately"),
        SliceDef("rb2-helpfulness-512", "allenai/reward-bench-2", n=512,
                 notes="the fp32 frame reference corpus; helpfulness-flavored pairs"),
        SliceDef("rb2-calibration-200", "allenai/reward-bench-2", n=200,
                 notes="the FORECAST-CAT calibration slice; disjoint from held-out categories "
                       "by construction of the leave-one-out"),
        SliceDef("rb2-patch-40", "allenai/reward-bench-2", n=40,
                 notes="the ADJ-AVP real component patch grid"),
        SliceDef("rmbench-full", "THU-KEG/RM-Bench", n=None, notes="1,327 items, Hard tier"),
        SliceDef("ppe-best-of-k", "lmarena-ai/PPE", n=None,
                 notes="five Best-of-K sets: MMLU-Pro, MATH, GPQA, MBPP-Plus, IFEval"),
        SliceDef("ppe-human", "lmarena-ai/PPE-Human-Preference-V1", n=None,
                 notes="16,038 arena battles; the human anchor and the off-pipeline "
                       "independence control"),
        SliceDef("helpsteer3-pref-2000", "nvidia/HelpSteer3", n=2000, config="preference",
                 notes="pairs with per-annotator scores; disagreement = per-pair spread"),
        SliceDef("ultrafeedback-bank", "openbmb/UltraFeedback", n=2000,
                 notes="the CHI-DRIFT bank: 2,000 prompts x 4 completions = 8,000 responses"),
        SliceDef("ultrafeedback-train-20k", "openbmb/UltraFeedback", n=20000,
                 notes="EMB-LORA training pairs, binarized"),
        SliceDef("ultrafeedback-tournaments", "openbmb/UltraFeedback", n=5000,
                 notes="k=4 tournaments for TOPO-HODGE"),
        SliceDef("nectar-tournaments", "berkeley-nest/Nectar", n=5000,
                 notes="k=7 tournaments for TOPO-HODGE"),
        SliceDef("processbench-full", "Qwen/ProcessBench", n=None,
                 notes="3,400 step-split solutions, earliest-error-step label"),
        SliceDef("receipt-triples-1600", "PKU-Alignment/PKU-SafeRLHF", n=1600,
                 notes="receipt-corrupted triples built from PKU and hh-rlhf responses"),
        SliceDef("hh-ood", "Anthropic/hh-rlhf", n=1000, notes="OOD captures"),
        SliceDef("wildchat-organic", "allenai/WildChat-4.8M", n=1000,
                 notes="the freshness/comparability arm, NOT the independence control"),
        SliceDef("arena-organic", "lmarena-ai/PPE-Human-Preference-V1", n=1000,
                 notes="the genuinely off-pipeline organic slice, via PPE-Human prompts"),
        SliceDef("evalaware-paired", "mixed", n=1000,
                 notes="paired RB2-shaped and organic prompts for the eval-awareness probe"),
        SliceDef("judge-pairs-1000", "allenai/reward-bench-2", n=1000,
                 notes="pairwise judgments for JUDGE-VBC, batched"),
        SliceDef("hump-prompts", "openbmb/UltraFeedback", n=500,
                 notes="held-out prompt set for the HUMP generation arm; disjoint from "
                       "ultrafeedback-bank by construction"),
        SliceDef("hack-probes", "in-repo", n=None,
                 notes="hand-built matched probe sets per hack family"),
        SliceDef("diagnostic-v3", "in-repo", n=None,
                 notes="65 seeds x 14 dims, matched-prompt probes with honest lineage"),
        SliceDef("caltransfer-organisms", "in-repo", n=None,
                 notes="three planted doses on the real 0.6B trunk"),
        SliceDef("adjavp-organism", "in-repo", n=None,
                 notes="hack-direction organism with chi > 0, Cov(feature, gold) <= 0"),
    )
}

CAMPAIGN_SEED = 20260716  # the single base seed; per-task seeds derive from it deterministically

def task_seed(*parts: str) -> int:
    """A deterministic per-task seed derived from the campaign seed and a task identity."""
    import hashlib

    h = hashlib.blake2b("/".join(parts).encode(), digest_size=4).hexdigest()
    return (CAMPAIGN_SEED + int(h, 16)) % (2**31 - 1)
