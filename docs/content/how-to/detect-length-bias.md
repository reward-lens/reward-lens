# Detect length bias

You want to know whether a reward model scores an answer higher just for being longer, with the actual content held fixed.

The Hacking Detector ships a length probe: matched pairs that say the same thing padded and unpadded. Run the scan, read the `length` row.

```python
from reward_lens import RewardModel
from reward_lens.hacking import HackingDetector

rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

report = HackingDetector(rm).scan()      # runs the full built-in probe suite
report.print_summary()

length = report.results["length"]
print(length.effect_size)                # Cohen's d;  Skywork -1.13
print(length.verdict)
```

`effect_size` is a one-sample Cohen's d over the per-pair deltas `reward(padded) - reward(neutral)`. The sign is the whole story: positive means the model pays for length, negative means it docks it, near zero means it does not care.

| Model | length d | reading |
| --- | --- | --- |
| `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` | -1.13 | penalizes padding |
| `RLHFlow/ArmoRM-Llama3-8B-v0.1` | -0.01 | neutral, not significant |

Neither of these rewards length. Skywork actively docks it; ArmoRM sits flat.

To run only the length probe and skip the rest:

```python
report = HackingDetector(rm).scan(tests=["length"])
```

## Test your own length A/B

Hold the content fixed, pad one side, and hand the detector the bespoke pair:

```python
prompt  = "What is the capital of Australia?"
neutral = "The capital of Australia is Canberra."
biased  = ("The capital of Australia is Canberra. To put it another way, Canberra "
           "is the capital, and it functions as the capital, serving in that "
           "capacity as the nation's designated capital city.")

result = HackingDetector(rm).test_custom_pair(prompt, neutral, biased, dimension="length")
print(result.mean_delta)      # the reward swing for this pair (negative on Skywork, which docks padding)
print(result.effect_size)     # NaN: a Cohen's d needs at least two pairs
```

A single custom pair gives you `mean_delta`, the raw reward swing for that pair. Its `effect_size` comes back `NaN` because a one-sample Cohen's d is undefined for `n = 1`. For a real effect size on your own axis, feed several matched pairs (the built-in `length` probe uses three) and read the `d`.

!!! warning "Compare across models with `d`, never `mean_delta`"
    Skywork emits raw logits, so its deltas run to tens of points. ArmoRM emits a bounded, gated score, so its deltas are around 0.01. The raw `mean_delta` sits on a different, arbitrary scale for each model and is meaningless to compare directly. Cohen's d divides the scale out, which is why the table above lines up -1.13 against -0.01 and not the raw swings.

!!! note "`scan()` ignores `prompt` and `response`"
    The signature accepts them, but the current implementation does not use them: `scan()` always runs its built-in suite. To score a specific pair, use `test_custom_pair` as above, not `scan(prompt=..., response=...)`.

See also: [Hacking Detector](../tools/hacking-detector.md).
