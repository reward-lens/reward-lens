# The anatomy of evidence

Why was a floating-point number the bug? It sounds like an odd thing to blame. But nearly every failure the first version of this library produced traces back to the same root: a measurement returned a bare `float`, and a bare float cannot tell you whether it is a result or a rumor.

A float carries no uncertainty, so a number from five cloned stimuli looks identical to a number from five hundred independent ones. It carries no record of whether it depends on an arbitrary basis, so a cross-model cosine that means nothing looks exactly like one that means something. It carries no provenance, so you cannot trace it back to the inputs that produced it. And it carries no notion of whether anyone ever checked it, so the anti-correlation that was never calibrated sat on the page next to a bit-exact reproduction with equal authority. Every one of those was a float pretending to be more than it was.

So in 2.0 a measurement does not return a float. It returns an `Evidence`.

## The object

`Evidence` is a frozen record. Once made, it cannot be edited, and its identity is a content hash of what it says, so two runs that computed the same thing collapse to the same evidence and a tampered value is a different object. Here is one, produced on CPU.

```python
from reward_lens.signals import from_tiny
from reward_lens.measure import base as mb
from reward_lens.measure.battery import BiasBattery
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["sycophancy"].items)[:12])
ev = mb.run(BiasBattery(), mb.Context(signal=signal, view=view))
```

![One measurement as a specimen card, the value at the center and its credentials around it.](../assets/figures/anatomy-of-evidence-light.svg#only-light){ .rl-fig .rl-fig--hero }
![One measurement as a specimen card, the value at the center and its credentials around it.](../assets/figures/anatomy-of-evidence-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**Every field earns its place.** The value is one part of the record. The rest is what you would need to decide whether to believe it: how much independent data it rests on, whether it depends on a basis, where it came from, what it cost, and the trust level computed from all of it.
///

## The fields

- `value`. The payload. For a bias measurement it is a per-axis effect size; for the lens it is a set of per-layer arrays. It is typed, not a loose float.
- `observable` and `observable_version`. Which tool produced it, and which version of that tool, so a number outlives the code that made it.
- `subject`. What the measurement is *about*: the signals, the dataset, the readout, and, when the measurement is a comparison, the frame it was taken in.
- `uncertainty`. Not one number but a small record: the confidence interval and its level, the nominal row count `n`, and the effective sample size `n_effective`. Those last two are the clone problem made visible. Thirty rows cloned from six seeds report `n = 30` and `n_effective = 6`, and the interval is computed from the six, because that is how much the data actually constrains the answer. The `method` field names how the interval was built, and a clone-inflated interval is labelled as such rather than passed off as honest.
- `gauge`. Whether the value is `invariant` (it survives any change of basis), `covariant` (it means something only once a frame is fixed), or `raw_only` (it lives in raw coordinates and must not be compared across models). This is the field the [gauge gate](gauge-and-frames.md) reads.
- `calibration`. A reference to the scorecard that graded this tool against a known answer, or `None`. When it is `None`, the tool is uncalibrated and cannot climb past exploratory.
- `provenance`. The lineage: the git commit, the config hash, the seeds, the cost in GPU-seconds and tokens and wall time, any oracle calls, the parent evidence it was derived from, and the study it belongs to if any. Provenance is what makes the store a graph rather than a pile.
- `trust`. The one summary, computed from calibration, registration, and adjudication. You do not pass it in. The [trust ladder](trust-ladder.md) is the whole story of how it is set.
- `created_at`. A timestamp, kept out of the content hash so identity depends on what was measured, not when.

## Why frozen, and why hashed

Two properties fall out of making evidence immutable and content-addressed, and both matter more than they look.

Immutability means trust cannot be edited after the fact. There is no setter, so no code path, yours or the library's, can promote a number by writing a higher rung onto it. The rung is a function of the facts, recomputed from them, and the frozen dataclass makes that the only way.

Content addressing means the record is a node in a graph. Because each evidence names its parents, a card or a paper that cites an evidence id is not quoting a loose number, it is pointing at a specific node whose whole derivation is recoverable. That is what lets the [evidence store](evidence-store.md) refuse to accept a derived result whose parents it cannot find, and what lets the claims checker fail a manuscript that cites a number the store cannot back.

The value on that card was a bias effect size. It is real, it is reproducible, and it is `EXPLORATORY`, because nothing has yet scored the bias battery against a case where the true bias was known. That last fact is not a caveat bolted on. It is a field. Next: [what computes it](trust-ladder.md).
