# Calibration and organisms

**How do you calibrate an instrument when there is no ground truth to calibrate against?** For a real reward model there is no answer key. Nobody can tell you which components truly cause the length bias, so nobody can tell you whether your attribution tool got it right. The honest move is to stop pretending and manufacture the ground truth: build a model whose answer you already know, then see which methods recover it.

That built object is a **model organism**. You plant a rule you fully control, train a reward head that obeys it, and now the rule *is* the answer key. Every method you care about gets scored against it, and the scores go on a published scorecard. No answer key, no calibration, and without calibration a measurement stays [exploratory](trust-ladder.md).

![The calibration loop: an answer key is planted in an organism, trained into the reward head, recovered by an instrument, scored against a scorecard, and gated into a calibrated trust level.](../assets/figures/calibration-loop-light.svg#only-light){ .rl-fig .rl-fig--hero }
![The calibration loop: an answer key is planted in an organism, trained into the reward head, recovered by an instrument, scored against a scorecard, and gated into a calibrated trust level.](../assets/figures/calibration-loop-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**No answer key, no promotion.** The organism is a rule planted by construction. Training pushes that rule into \(w_r\). An instrument then tries to recover it; a scorecard scores the recovery (here AUC \(1.0\) on the out-of-distribution split, cosine \(0.96\) with the trained head); and only a passing scorecard clears the gate that promotes the measurement from exploratory to calibrated.
///

## Answer keys by construction

The loop is five steps, and each one is a place a dishonest tool would get caught.

1. **Plant a rule.** Pick a preference rule and instantiate it exactly: a spurious feature confounded with the label at a chosen agreement rate, a hidden second objective, a planted 3-cycle that makes preferences intransitive, a known annotator mixture. The [foundry](../reference/organisms.md) has a family of generators for these, each returning a data view and the answer key that governs it.
2. **Train it in.** Fine-tune a reward head on data drawn from the rule until the rule lives in \(w_r\), not just in the labels.
3. **Verify it governs behavior out of distribution.** In-distribution accuracy proves nothing; a lookup table would pass. The organism only counts if the planted rule still drives preferences on a held-out split it was never trained on. That out-of-distribution check is what separates a rule the model *learned* from one it *memorized*.
4. **Score every method against the key.** Run attribution, patching, a probe, a detector, whatever you want to certify, and measure how well each recovers the thing you planted.
5. **Publish the scorecard.** The numbers go in the store, and a difficulty dial lets you turn the rule from easy to subtle and watch which methods hold up.

Run at the full go/no-go, the recovered rule matches by construction: the detector hits AUC \(1.0\) on the out-of-distribution split, its recovered direction has cosine \(0.96\) with the trained reward head, and the scorecard AUC climbs monotonically with the planted confound strength, from \(0.475\) at the hardest setting to \(0.919\) at the easiest. The difficulty dial cleanly separates strong methods from weak ones, which is the entire point: a scorecard that ranked every method the same would certify nothing.

## The micro-organism that runs in CI

The full go/no-go needs real training runs, but a shrunken version of the loop runs on every continuous-integration pass, on a two-layer model, on CPU, in about fifteen seconds. It plants a spurious rule (a `cites` feature confounded with a `factual` label at agreement rate \(0.85\)), trains a tiny head, and checks that a mean-difference detector recovers it:

```python
from reward_lens.organisms import micro_organism_calibration

result = micro_organism_calibration()          # ~15s on CPU, no download

print(result.recovered)
# -> True
print(result.detector.ood_auc, round(result.detector.cosine_with_reward, 3))
# -> 1.0 0.963
print(result.verification.reason)
# -> rule governs OOD: 100.0% of 160 held-out pairs preferred correctly (>= 90%)
```

`result.recovered` is the go/no-go bit. If a change to the library ever broke the calibration machinery so that a planted rule could no longer be recovered, that boolean flips and the build fails. The discipline is not a document; it is a test. The same numbers the full loop reports at scale, out-of-distribution AUC \(1.0\) and cosine near \(0.96\), show up here on the toy, because the rule is known and the detector either finds it or it does not.

## The auditing game

Scoring a method you designed against a rule you planted is friendly. The adversarial version is the **auditing game**: hide the objective, hand an auditor only the model and a budget, and ask them to name what it rewards. The planted key is the referee. A method that can only reproduce a bias it was told to look for scores badly here, because it was never told. The game is exposed through `run_auditing_game` and the CLI `audit` subcommand, which is gated behind the hardware a population-scale audit needs and refuses rather than fabricate a result.

This is also where the library's own most uncomfortable result gets adjudicated instead of argued. On Skywork, attribution and patching rank components in nearly opposite order, Spearman \(\rho = -0.171\) averaged over dimensions. Which one is *right* is not answerable on a real model, because there is no key. On an organism it is: plant the rule, ask each method to recover it, read the scorecard. Calibration is the resolution path for [the observational-versus-causal disagreement](../concepts/observational-vs-causal.md), and it is honest about its status. The loop is designed and runs green on organisms; the population-scale audit across real models is pending the hardware.

## What a scorecard does and does not say

A scorecard is a receipt for a regime, not a certificate of universal correctness. Calibrating a detector on a spurious-correlation organism at agreement rate \(0.85\) certifies that the detector catches *that kind* of confound at *that difficulty*. It says nothing about a confound you never planted, and little about a model far from the organism's regime. The `regime_match` field on a calibration reference exists precisely so a measurement can record whether the subject actually resembles the organism it was calibrated on.

So read the trust level literally. <span class="rl-chip rl-chip--fill rl-chip--calibrated">calibrated</span> means "audited against a known answer, in a stated regime." It does not mean "true." A measurement that stays <span class="rl-chip rl-chip--fill rl-chip--exploratory">exploratory</span> is unaudited, which is not the same as wrong. And a clean scorecard on the confounds you thought to plant is no evidence about the one you did not. The gate caps overclaiming; it cannot manufacture insight. The full anatomy of what a calibrated measurement carries is in [the anatomy of evidence](anatomy-of-evidence.md), and the honest limits sit in [interpreting results honestly](../caveats.md).
