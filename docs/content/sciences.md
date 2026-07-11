# The sixteen sciences

**What is a reward model, taken seriously as an object of study?** Not a black box to benchmark and move past. A trained artifact with a geometry, a development history, a failure surface, and a set of claims you can make about it and then try to break. The sciences are sixteen families of those claims. Each one is a question put to a reward model, a prediction registered before the run, and a kill criterion that says what result would sink it.

The point of the kernel, the signals and measures and geometry and gates the rest of these docs describe, is that the research on top can be thin. A science is not a new subsystem. It is a frozen [study](discipline/studies-and-preregistration.md) spec, a short analysis function, and kill criteria, run against the same measurement discipline as everything else. Adding one adds a hypothesis, not infrastructure. That is what makes sixteen of them tractable for one author, and what makes any of them yours to run or refute.

![A map of one kernel feeding sixteen thin studies](assets/figures/kernel-map-light.svg#only-light){ .rl-fig .rl-fig--wide }
![A map of one kernel feeding sixteen thin studies](assets/figures/kernel-map-dark.svg#only-dark){ .rl-fig .rl-fig--wide }

/// caption
**One kernel underneath, so every study is thin.** The subsystems in the middle do the measuring; each science on the rim is a spec plus an analysis function that consumes them. Nothing on the rim reimplements what is in the middle.
///

Read that map as an argument about cost. If each science had to build its own way to load a model, read activations, fix a gauge, and count effective sample size, sixteen of them would be sixteen libraries. Instead they share one, and a new science is a weekend, not a quarter.

## What a result counts as here

Every science reports through the same gates, so a headline number is not a claim until it has earned one. A confirmatory result needs a preregistration frozen before the run ([registration gate](discipline/studies-and-preregistration.md)). A basis-dependent quantity needs a shared frame before it can be compared across models ([gauge gate](discipline/gauge-and-frames.md)). And no instrument claims more than exploratory trust until it has been graded on an organism where the truth is planted by construction ([calibration gate](discipline/calibration-and-organisms.md)). The standing claims collect on a [scoreboard](reference/studies.md), and refutations post as plainly as confirmations.

```python
from reward_lens.studies import Scoreboard, DEFAULT_ROWS

board = Scoreboard()
for r in DEFAULT_ROWS:
    print(f"{r.id:>3}  {r.kind:16}  {r.science:8}  {r.title}")
#  T1  standing_theorem  S4        Constructive unhackable-subspace finder
#  T2  standing_theorem  S8/S12    Distortion equilibrium
#  T3  standing_theorem  S12/S3    RLHF speed proportional to teacher variance
#  T4  standing_theorem  S2/S12    Proxy-true reward angle
#  T5  standing_theorem  S3/S4     Heavy tail defeats KL control
#  T6  standing_theorem  S2        Identifiability up to shift and scale
#  T7  standing_theorem  S11       No single scalar for a population
#  T8  standing_theorem  S2        Scalar head cannot express intransitivity
#  T9  candidate_law     S3        Fluctuation-dissipation for reward hacking
# T10  candidate_law     S8/S2     Belief factorization and gauge=channel-kernel
# T11  candidate_law     S13/AT    Evaluator-model divergence precedes hacking
# T12  candidate_law     S5/S6     Coherence/Welch law and Hodge obstruction
# T13  candidate_law     AT        Value convergence excess
# T14  candidate_law     S15       Honesty unraveling law
```

T1 through T8 are standing theorems the program instantiates inside real reward models. T9 through T14 are candidate laws this work proposes and then tries to kill. The sciences below are grouped not by topic but by how far each bet has actually been carried, because that is the honest thing a reader deciding what to run or break needs to know.

## Run and confirmed during the build

These five were frozen and then run, on organisms where the recovered quantity could be checked against a planted ground truth. The numbers are real and registered. What is still gated is the population-scale run on real 8B and flagship reward models, which needs hardware this was not built on.

**Thermodynamics** asks which features optimization will exploit, read off base-policy statistics before any RL. The bet is a fluctuation-dissipation identity: the initial per-feature drift under optimization is a base-policy covariance, \(\chi_i = \mathrm{Cov}_0(f_i, r)\), so the susceptibility spectrum previews where a best-of-\(n\) policy is headed with no gradient step taken. On the calibration draw the predicted susceptibility ranked the realized best-of-\(n\) drift at Spearman \(0.958\). The kill criterion is sharp: if the two did not rank-correlate, the tilted-exponential family would be the wrong idealization of RLHF, which is itself a publishable negative. This is candidate law T9.

**Values** asks whether a scalar reward model encodes "this pair is contested," annotators would disagree, in a direction separate from what it rewards. If it does, ensemble-grade disagreement can be read from one model with one probe. The contested-direction probe decoded disagreement at \(0.824\) and landed orthogonal to the reward direction \(w_r\), exactly as a signal the reward throws away should. A disagreement probe that decoded at chance would mean Bradley-Terry training destroys the hidden context, the kill criterion. This is theorem T7, no single scalar for a population.

**Coupling** watches the policy and the grader as one optimizing loop and asks whether representational divergence shows up before the behaviour does. A recorder built in reward-feature space named the planted hack direction three optimization steps before the gold reward diverged. Under evasion pressure that silences a free-floating probe on a stylistic tell, the monitor anchored in \(w_r\) kept \(0.98\) of its signal, because a policy cannot hide from a monitor pointed at the direction it is paid to excite. This is candidate law T11.

**Phase** asks whether the reward-hacking transition is reversible. If it were a gradual drift, a policy pushed past onset could be annealed back by lowering the pressure. On a bistable reward system where the answer is analytic, the order parameter followed a different branch coming down than going up, and the two branches enclosed an area of \(3.06\). A nonzero loop area means the transition is first-order and a hacked policy cannot simply be annealed home. This deepens T3 and T9.

**Forensics** asks how a grader weighs evidence: does it rank a caught fabrication below saying nothing? Two statistics summarize it, how much a valid receipt is worth and how much better silence is than a receipt that fails on checking. Across a population of graders with planted values, the reliance score recovered its planted value at \(0.993\), which is what licenses reading it off a grader you have not planted. This is candidate law T14, the honesty unraveling law.

## Frozen, calibrated, and waiting on a real model

For these nine the spec is frozen and the method is calibrated, meaning it recovers a planted answer on a constructed organism, but the confirmatory arm on a production reward model is GPU-gated. The instrument works. What it has not yet done is report on a real model at population scale. Each is an open bet you could take further.

**Gauge** asks whether two reward directions can be compared at all, when a head is fixed only up to shift and scale. The identifiability claim (T6) and the intransitivity claim (T8) both live here; the [identifiability](theory/identifiability.md) page works through the first and [preference rank](theory/preference-rank.md) the second.

**Capacity** asks how much bias a scalar head forces just by routing many criteria through few dimensions. Once the number of criteria exceeds the effective dimension, the Welch bound makes some cross-criterion interference obligatory before any data is seen, and the surplus variance leaks into a dark channel a per-criterion audit cannot see. This is half of T12.

**Topology** asks what share of reward error is topologically obligatory, beyond any scalar reward's reach. Hodge-decompose a corpus of pairwise preferences and the curl and harmonic parts are precisely what a Bradley-Terry reward provably cannot represent, a coordinate-free lower bound on its error. If that intransitive mass were uniformly tiny, it would be a clean defense of scalar reward modeling; the registered prediction is the opposite. This is the other half of T12.

**Embryology** asks whether the reward direction forms gradually or in jumps, and which features enter first. The sharp version is an ordering claim: cheap surface biases such as length and format enter \(w_r\) before quality features such as correctness and harmlessness, because they are cheaper to fit from preference data.

**Factorization** asks how much the reward knows but fails to use, and whether its error is epistemic (it believes something false) or axiological (it values the wrong thing). A property the model can decode but does not price is the mechanistic precondition of a hack, and the sycophancy case is the crown test: does the premium for agreement route through belief, or bypass it? This feeds T2 and T10.

**Verification** asks whether the reward checks the work or just reads the style around it. Patch the activations at the exact span where a corruption lives and measure how much of the clean-versus-corrupted reward gap comes back: near one means the verifier is anchored at the error, near zero means it is reacting to surface style everywhere but the error.

**Decompiling** asks how much of the decision function can be put into words, and what stays tacit. Fit a natural-language surrogate from a budget of predicates and trace where fidelity stops rising; the residual no short rubric captures is the tacit part, and the hypothesis is that it is where reward hacks are financed.

**Hackability** asks whether a number read off the weights can name the dimension that gets hacked, before training starts, and whether editing that one direction removes the hack. If both land, a weights-derived forecast closes an exploit before RL rather than after, which is the strongest single breakthrough this program bets on. This feeds T2.

**Robustness** asks whether the model knows it is being tested, and whether that recognition inflates the score. A reward model that scores benchmark-shaped responses higher for looking like a benchmark is contaminated at the grader, detectable as a decodable benchmark-versus-organic direction with a positive causal loading on the reward.

## Laws that need a whole population

The last two are not about one reward model but about how a population of them behaves. They are calibrated on synthetic pairs where the sign of the effect is known; the real run needs many models and a live optimization loop.

**Universality** asks whether two reward models converge on values beyond what shared world-modeling already forces. Two heads on the same base inherit the same capability subspace for free, so the question is the excess: is their reward alignment greater than that shared structure explains? The value convergence excess measures it as a difference of alignments read against a random-utility null. This is candidate law T13.

**Performative** asks how fast a metric decays once developers start optimizing against it. A metric that is optimized changes the population it measures, so its correlation with the truth it tracked has a half-life. The registered prediction is that a causally grounded metric outlives an observational one, because the observational metric can be raised by a truth-independent proxy while the causal one can only be raised by moving the truth.

## Taking one further

None of this is finished, and that is the invitation. A science is a spec you can read, freeze, and run against your own model; the [studies and preregistration](discipline/studies-and-preregistration.md) page shows the mechanics, and the how-to on [freezing and running a study](how-to/freeze-and-run-a-study.md) walks one end to end. The honest caps apply here as everywhere: a registered result means the prediction predated the data, nothing more, and a calibrated instrument speaks to the regime it was graded in, not to every model. What the scoreboard buys you is that a refutation is as legible as a confirmation, so an idea that does not survive contact shows it plainly. Where even the surviving instruments stop being trustworthy is the subject of [interpreting results honestly](caveats.md).
