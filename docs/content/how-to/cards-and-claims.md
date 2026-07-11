# Build a card, check a manuscript

**How do you stop a model card, or a paper, from claiming a number nothing measured?**

Make the evidence store the only source of numbers, and let the card and the claim checker be views onto it. A card assembles every stored measurement about one signal and shows each one's trust and calibration, gaps included. The claim checker reads a document, finds every number tagged with the Evidence it came from, and fails if the store cannot back it. Neither computes anything; both can only report what was measured.

## A card is a view over the store

`build_card` queries the store for the latest Evidence about one signal fingerprint and returns a `Card`. Assume a store holding a couple of measurements on the tiny model (the earlier how-tos show how they get there):

```python
from reward_lens.artifacts import build_card

card = build_card(signal.meta.fingerprint, store)
for e in card.entries:
    print(e.observable, "|", e.trust, "|", e.gauge, "| validated", e.validated)
# DirectLinearAttribution | EXPLORATORY | invariant | validated False
# LensCrystallization | EXPLORATORY | invariant | validated False
print("unvalidated:", [e.observable for e in card.unvalidated])
# unvalidated: ['DirectLinearAttribution', 'LensCrystallization']
```

Both entries are EXPLORATORY, so both land in `unvalidated` and the card renders them as gaps rather than hiding them. A card that dropped its uncalibrated numbers would be a marketing document; this one carries every measurement with its trust attached, which is the only version worth signing.

## A claim must cite the evidence that backs it

The checker parses a claim tag of the form `[[claim value=… ev=… field=… tol=…]]`: the number as written, the Evidence id it came from, an optional dotted field into the Evidence value, and an optional tolerance. `check_text` (and `check_files` over a document set) loads that Evidence, extracts the comparable value, and passes only if the claim matches within tolerance:

```python
from reward_lens.artifacts import check_text

ev = store.get(card.entries[1].evidence_id)          # the LensCrystallization evidence
frac = ev.value["mean_crystal_frac"]
good = f"Crystallization sits at [[claim value={frac:.4f} ev={ev.id} field=mean_crystal_frac tol=1e-3]] of depth."
bad  = f"Crystallization sits at [[claim value=0.90 ev={ev.id} field=mean_crystal_frac tol=1e-3]] of depth."

print(check_text(good, store).ok, check_text(good, store).n_failures)   # True 0
print(check_text(bad,  store).ok, check_text(bad,  store).n_failures)    # False 1
```

The backed claim resolves: the stored `mean_crystal_frac` is \(0.1667\), the prose says \(0.1667\), they agree. The overclaim fails: the prose says \(0.90\), the store says \(0.1667\), and \(0.73\) is well past the tolerance. A tag pointing at an Evidence id the store does not hold fails the same way, so a citation to a measurement that was never taken cannot slip through.

## Run it from the command line

Both surfaces are CPU-pure and shipped on the `reward-lens` command. The card prints JSON (or HTML with `--format html`); the claim checker prints a report and exits nonzero when any number is unbound, which is the shape a CI step wants:

```console
$ reward-lens card mfp:586b55dd932158ef9d21a4e4e71e1276 --store store/
{
  "signal": "mfp:586b55dd932158ef9d21a4e4e71e1276",
  "total_gpu_seconds": 0.0,
  "entries": [ ... ]
}

$ reward-lens claims paper_ok.md --store store/
Claims checked: 1. Failures: 0.
  [ok] ev:a70c1fbc… mean_crystal_frac: claimed 0.1667, stored 0.166667, |diff|=3.3e-05 <= tol 0.001
$ echo $?
0

$ reward-lens claims paper_bad.md --store store/
Claims checked: 1. Failures: 1.
  [FAIL] ev:a70c1fbc… mean_crystal_frac: claimed 0.9 but stored 0.166667 (|diff|=0.73 > tol 0.001)
$ echo $?
1
```

The nonzero exit is the whole point. Wire `reward-lens claims` over your manuscript and its evidence store in CI, and a number that drifts from what was measured stops the build instead of shipping.

![A card and a claim checker are read-only views over the evidence store.](../assets/figures/artifacts-as-views-light.svg#only-light){ .rl-fig .rl-fig--hero }
![A card and a claim checker are read-only views over the evidence store.](../assets/figures/artifacts-as-views-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**A card can only say what the store can back.** The store is the single source of measured numbers; the card and the claim checker are views that read from it and add nothing, so neither can assert a value that was not measured.
///

The figure puts the store at the center with the card and the checker hanging off it as views. Nothing flows the other way: a card cannot write a number into the store, and the checker cannot soften a claim to make it pass. The arrows point outward from measurement to document, never from wish to record, which is exactly the property that made the old habit of numbers drifting between the paper and the CSVs impossible to sustain here.

See also: [the evidence store](../discipline/evidence-store.md), [the command line](../cli.md), [`build_card`](../reference/artifacts-operate.md#reward_lens.artifacts.card.build_card), [`check_files`](../reference/artifacts-operate.md#reward_lens.artifacts.claims.check_files).
