# Organisms

**How do you know an instrument works? Give it a question whose answer you planted.** `reward_lens.organisms` generates reward models with a known defect built in: a spurious feature, a hidden objective, an intransitivity, a reward-hacking direction. Each generator returns a data view and an answer key, so a scorecard can grade what the instrument recovered against the truth. This is the machinery behind [calibration and organisms](../discipline/calibration-and-organisms.md).

## The generators

Eleven generators, each returning a `(DataView, AnswerKey)` pair. The first four plant a structural defect: a rule composed from parts, a spurious correlate, a hidden second objective, and a safety-then-quality gate the model must learn.

::: reward_lens.organisms.foundry.compositional_rule_organism
    options:
      heading_level: 3

::: reward_lens.organisms.foundry.spurious_correlation_organism
    options:
      heading_level: 3

::: reward_lens.organisms.foundry.hidden_objective_organism
    options:
      heading_level: 3

::: reward_lens.organisms.foundry.gate_organism
    options:
      heading_level: 3

The next three plant a defect in the preference structure itself: a three-cycle \(A > B > C > A\) no scalar can honor, a mixture of annotators with known entropy \(H(V)\), and a set of rubric directions with exact pairwise cosines.

::: reward_lens.organisms.foundry.intransitivity_organism
    options:
      heading_level: 3

::: reward_lens.organisms.foundry.annotator_mixture_organism
    options:
      heading_level: 3

::: reward_lens.organisms.foundry.rubric_organism
    options:
      heading_level: 3

The last four are the pathologies the discipline cares most about: a direction with positive susceptibility that anti-correlates with the gold reward, fabricated receipts, inverted preferences, and a curl-plus-harmonic field with no consistent potential.

::: reward_lens.organisms.foundry.hack_direction_organism
    options:
      heading_level: 3

::: reward_lens.organisms.foundry.epistemic_error_organism
    options:
      heading_level: 3

::: reward_lens.organisms.foundry.value_error_organism
    options:
      heading_level: 3

::: reward_lens.organisms.foundry.curl_harmonic_organism
    options:
      heading_level: 3

A twelfth generator, `kinship_organism`, needs a GPU-trained population and raises until that hardware is present, so it is not documented as working here.

## The answer key

`AnswerKey` is what makes an organism an organism: the planted truth an instrument is scored against, including which channels are meant to govern behavior out of distribution.

::: reward_lens.organisms.spec.AnswerKey
    options:
      heading_level: 3

## The scorecard

`MethodScorecard` grades an instrument against an organism's answer key. It turns "did the tool recover the planted defect" into an AUC, the number a calibration rests on, and across the organism suite that AUC rises monotonically with the size of the planted defect.

::: reward_lens.organisms.scorecard.MethodScorecard
    options:
      heading_level: 3

## Calibration in CI

`micro_organism_calibration` runs the whole loop small enough to fit in a unit test: a spurious organism, a tiny CPU trunk, and a detector, returning the go/no-go bit that says calibration still recovers the planted rule.

::: reward_lens.organisms.micro.micro_organism_calibration
    options:
      heading_level: 3
