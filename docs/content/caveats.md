# Interpreting results honestly

Mechanistic interpretability has an overclaiming problem. A clean figure gets read as a mechanism, an observational correlation gets written up as a cause, and a result on one example gets stated as a general law. Reward models make all three temptations worse, because the scalar target produces unusually crisp pictures, and a crisp picture is easy to trust. This page is the counterweight. It is not an appendix of disclaimers. It is the part of the docs that tells you which of your conclusions you have actually earned.

## Attribution is not causation, and here is the proof

The headline limitation is the one the library discovered about itself. Component attribution, which reads how much of the reward each component carries, does not predict which components causally matter.

![Attribution against patch effect: the cloud tilts the wrong way.](assets/figures/attribution-vs-patching-light.svg#only-light){ .rl-fig }
![Attribution against patch effect: the cloud tilts the wrong way.](assets/figures/attribution-vs-patching-dark.svg#only-dark){ .rl-fig }

/// caption
**A tool that read cause would put the cloud on the diagonal.** Each dot is a component, attribution rank against patch-effect rank. The late MLPs that attribution credits most carry little causal weight; the early components patching finds most necessary are credited almost nothing.
///

Ranked and correlated, the two agree at Spearman \(\rho = -0.171\) on Skywork-v0.2, averaged across dimensions, and down to \(-0.44\) on code correctness. On ArmoRM the same correlation is \(+0.05\), near zero. Negative to zero, never reliably positive. The reason is [crystallization](concepts/crystallization.md): the reward becomes *visible* in the last layers, so attribution credits them, but it is *computed* earlier, so patching needs the early ones. Attribution can only see where the reward ended up, not where it came from.

!!! danger "The rule this forces"
    Treat attribution, the reward lens, feature alignment, and concept alignment as **hypothesis generators**, never as evidence of cause. The instant a claim becomes causal, "this component is responsible," "this head implements the bias," run [patching](instruments/patch-grid.md) and let the intervention decide. If you have not patched, you do not have a causal result, no matter how clean the attribution bar looks. And the library agrees with you in the type system: an attribution number and a patch number both come back `EXPLORATORY` until scored, so neither outranks the other by default.

## Patching has its own failure mode

So patch everything and trust the causal numbers? Not quite. Activation patching swaps an activation from one run into another, and that swapped activation can land somewhere the model never actually goes. You have created a representation off the model's natural distribution, and the reward you read off it may reflect a circuit that never co-occurs in real inputs. The causal claim is then about a counterfactual the model never faces.

This is a genuine and much-discussed concern in the patching literature. The library's answer is to keep the distribution in view rather than pretend the problem away: fit the activation distribution from clean data and flag any patched activation that lands too far off it, so a large effect from a badly off-distribution intervention arrives with a warning attached rather than as a clean causal number. Use that check when a patch effect is surprising or load-bearing.

## Small samples make loud effect sizes, and clones make them louder

Effect sizes computed from a handful of pairs are noisy, and cloned pairs are worse than noisy, they are dishonest, because they inflate your apparent sample size while adding no information. A confidence interval computed over thirty stimuli that were really six seeds mutated five ways is describing the cloning, not the model.

This was a real failure in the first version, and it is now something the type system refuses to let happen quietly. Every measurement's uncertainty carries both the nominal row count and an effective sample size that counts unique content. Thirty clones of six seeds report `n = 30` and `n_effective = 6`, and the interval is computed from the six.

![Thirty rows from six seeds collapse to an effective sample size of six.](assets/figures/clone-collapse-light.svg#only-light){ .rl-fig }
![Thirty rows from six seeds collapse to an effective sample size of six.](assets/figures/clone-collapse-dark.svg#only-dark){ .rl-fig }

/// caption
**Thirty clones are not thirty data points.** Six seeds mutated into thirty rows constrain the answer as much as six independent ones do. The bootstrap resamples at the seed level, not the row level, so the interval reflects what the data actually pins down.
///

Read the effective sample size, not the row count, and read the interval, not just the point estimate. The [effective-sample-size how-to](how-to/effective-sample-size.md) runs the check on your own eval set.

## A single reward direction is an approximation for multi-objective models

The whole [reward-direction picture](concepts/reward-direction.md) rests on there being one direction. For a single-head model like Skywork that is exact. For a multi-objective model it is a summary.

![ArmoRM's 19 objective directions: mostly positively aligned, some near-orthogonal.](assets/figures/armo-objective-cosine-light.svg#only-light){ .rl-fig }
![ArmoRM's 19 objective directions: mostly positively aligned, some near-orthogonal.](assets/figures/armo-objective-cosine-dark.svg#only-dark){ .rl-fig }

/// caption
**One direction is a gated average of nineteen that disagree.** Cosine similarity between ArmoRM's nineteen objective directions. Most pairs are weakly to moderately aligned, but several are near-orthogonal or slightly negative. The single \(w_r\) the library uses for ArmoRM genuinely represents no one of them.
///

ArmoRM has nineteen objective heads and a learned gate that reweights them per input. reward-lens collapses that to one aggregate direction so the standard tools run, but read every ArmoRM result knowing the direction is an average over objectives that disagree. When the disagreement itself is the question, use [multi-objective geometry](instruments/multi-objective-geometry.md), which measures the geometry between the objective directions rather than pretending they are one. This is also why ArmoRM's crystallization is earlier and noisier than Skywork's: you are watching a mixture form, not a single judgment.

## What the discipline does not do

The trust machinery is the best thing the library has, and it will still mislead you if you oversell it. Being exact about its limits is part of using it honestly.

**Calibration does not transfer for free.** A scorecard earned on a family of planted-rule organisms tells you how a tool behaves on organisms like those. It is evidence about the instrument in a regime, not a warranty on your reward model. When the regime does not match, the scorecard does not carry over, and the calibrated rung is a claim about the test bed, not a promise about your model.

**Registered does not mean true.** It means the prediction predated the data. A frozen prediction can be frozen and wrong, and the most useful outcome of a preregistered study is often a clean refutation, which the library records as prominently as a confirmation.

**Exploratory does not mean wrong.** It means unaudited. Most of what this site shows you is exploratory, the anti-correlation included, and it is still the most useful thing here. The trust ladder is a ceiling on overclaiming, not a source of insight. It cannot make a measurement true. It can only stop you from calling it more settled than it is.

## Absence of evidence is not evidence of absence

Interpretability never proves it has found everything. If attribution and patching both come back quiet on a component, that means those two tools did not find it important, not that it is unimportant. A circuit can be distributed in a way a per-component sweep misses, matter only in interaction, or hide in a direction none of these tools looked along. The library is built to help you find what is there. It cannot certify that nothing else is.

## This is alpha, and honest about it

The parts of the library that run without a large GPU, the evidence layer, the statistics, the data plane, the index math, are tested and usable now. The parts that need an 8B model, a flagship GPU, a real dataset download, or an external judge are gated, and gated means the library names the exact call it would make and refuses rather than fabricate a result. A number you read in these docs was either produced on hardware that could compute it, or drawn from a committed artifact and labelled as measured. None were invented to fill a page. The science layer is young and its interfaces may still move. When something is a bet rather than a result, the docs say what would falsify it, because this library, unusually, ships with the machinery to say so.

## The short version

- Observational tools locate the reward. Causal tools explain it. Do not swap them.
- A patch effect is only as trustworthy as the intervention is on-distribution.
- Read the effective sample size, not the row count, and the interval, not the point estimate.
- For multi-objective models, the single reward direction is a summary, not the truth.
- Calibration speaks to a regime, registered means predicted-in-advance, exploratory means unaudited. None of them mean "true."
- Finding nothing is not proving nothing is there.

Every instrument page folds its own caveat into its teaching, so you meet the limits where you meet the tool. This page is where they live together, for the reader who wants to know, before trusting any of it, exactly where it breaks.
