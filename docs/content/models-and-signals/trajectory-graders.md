<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">grader</span> trajectory RM</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">score gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">bring</span> a classifier and typed episodes</span>
</div>

# Trajectory graders

**An agent's reward is over a whole episode. Can you point an experiment at the receipt it lied about?**

Only if the receipt survives tokenization as a typed span, and that is what this adapter guarantees. An agent trajectory has structure the receipt and narrative sciences read: a *receipt* is the evidence a step produced, a tool result or a computed value; a *narrative* is the agent's own account of it; an *action* is what it did. The trajectory grader renders an episode to text while carrying those typed spans into token coordinates, then scores at the end of the episode. Without the span carry-through, a receipt-falsification or narrative-patching experiment addresses the wrong tokens and misaligns in silence.

## Load or wrap one

The scorer is a sequence classifier, so `TrajectoryRM.from_sequence_classifier(model, tokenizer)` wraps a reward head. The tiny CPU vehicle:

```python
from reward_lens.signals import TrajectoryRM

traj_rm = TrajectoryRM.from_tiny(seed=0)
traj_rm.caps    # SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT|SPAN_TYPES
```

The distinctive capability is `SPAN_TYPES`. The reward direction and the whole linear-readout battery come along, because underneath it is a classifier.

## Typed spans survive the render

Build a two-step episode where one step produces a receipt and the next narrates a conclusion, then tokenize it and watch the span types come through:

```python
from reward_lens.data.schema import Trajectory, TrajStep
from reward_lens.data.lineage import make_lineage
from reward_lens.core.types import Span

step1 = "Ran the query; the table has 512 rows."
step2 = "So the dataset is large enough to proceed."
r0 = step1.find("512 rows")
steps = (
    TrajStep(action="query_db", text=step1, receipts=(Span(r0, r0 + len("512 rows"), "receipt"),)),
    TrajStep(action="decide",   text=step2, narrative=(Span(0, len(step2), "narrative"),)),
)
traj = Trajectory(steps=steps, outcome={"success": True},
                  lineage=make_lineage("demo", "docs", (), ["demo"]),
                  prompt="Check the table size.")

tk = traj_rm.tokenize(traj)
sorted({s.kind for s in tk.spans})    # ['action', 'narrative', 'receipt']
len(tk.spans)                         # 4
traj_rm.score([traj]).value.values    # array([-0.0847], dtype=float32)
```

Four spans survive: an action and a receipt from the first step, an action and a narrative from the second, each mapped from its character range in the step text into exact token coordinates over the rendered whole. That mapping is the unglamorous, load-bearing part of the adapter. Once a receipt is a typed token span you can patch exactly the tokens the reward is supposed to be about, which is what the [receipt-reliance and skepticism indices](../instruments/index-library.md) do.

A trajectory grader is still an ordinary signal for ordinary pairs, so the generic instruments and the conformance suite exercise it unchanged:

```python
traj_rm.score([("hi", "there")]).value.values   # array([0.0194], dtype=float32)  -- classifier fallback
```

## Honest caveats

The span types are only as good as the spans you feed in. If a trajectory arrives without receipt and narrative spans, the grader still scores it, but the receipt sciences have nothing to point at, and no adapter can recover a structure that was never recorded. Separating receipt from narrative is the whole point, because it is what makes "the agent rewarded its own story over the tool's result" a measurable claim rather than a suspicion. As with every raw score, a trajectory reward is `INVARIANT` and `EXPLORATORY` until calibrated. To grade an episode shape reward-lens does not yet render, see [write an adapter](../how-to/write-an-adapter.md).

## Reference

[`TrajectoryRM`](../reference/signals.md#reward_lens.signals.trajectory.TrajectoryRM), and the trajectory schema in [the data reference](../reference/data.md).
