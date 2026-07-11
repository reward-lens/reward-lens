# Observational versus causal

Two tools, same model, same pair. One says the last MLP layers wrote the reward. The other says the early layers cause it. Which do you believe?

The honest answer is that they measure different things, and the moment you forget that is the moment you publish a plausible, backwards result. This is the single most important idea in the library, so it gets its own page and a measured result to make it concrete.

## The two questions

An **observational** tool reads. It takes an activation, projects it onto the reward direction, and reports how far the model has leaned toward its verdict by that point. The [reward lens](../instruments/lens-crystallization.md) and [component attribution](../instruments/attribution.md) are observational. They are cheap, one or two forward passes, and they answer "where does the reward *appear*?"

A **causal** tool intervenes. It changes an activation, runs the model forward from there, and measures how the reward moves. [Patching](../instruments/patch-grid.md) is causal. It is expensive, one forward pass per component, and it answers a different question: "which components, if you change them, *change the reward*?"

Appear and cause sound like the same thing. On real reward models they are not. Every instrument page in the site wears a chip that says which of the two claims it licenses, and the runner will not let an observational tool hand back a causal one.

## The result

Rank Skywork's components by attribution. Rank them again by patch effect. Correlate the two rankings and you get Spearman \(\rho = -0.171\). Negative. That is the mean across quality dimensions, and on the strongest ones it is sharper: about \(-0.31\) on helpfulness, \(-0.44\) on code correctness. The components that carry the most attribution are, if anything, mildly *anti*-predictive of the components that carry the most causal weight.

![Attribution rank against patch-effect rank on Skywork; the cloud tilts the wrong way.](../assets/figures/attribution-vs-patching-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Attribution rank against patch-effect rank on Skywork; the cloud tilts the wrong way.](../assets/figures/attribution-vs-patching-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**Attribution and causation point different ways.** Each dot is a component: its attribution rank against its patch-effect rank. A tool that read cause would put the cloud on the rising diagonal. On Skywork it tilts the other way. The late MLPs dominate attribution; the top causal head sits at layer 0.
///

Concretely: attribution on the sky-is-blue pair is led by the final MLPs, `mlp_L31` at \(+3.99\), then `mlp_L30`, then `mlp_L29`. Patching tells a different story. The single component whose change moves the reward most is an early head, `head_L0_H29` for helpfulness, and across dimensions the largest patch effects cluster early.

The mechanism behind the number is legible once you see it. Break an early layer and the whole chain downstream computes on garbage, so the reward swings. Break the last MLP and the model has mostly already decided, so it recovers. The reward becomes *visible* late, which is exactly what [crystallization](crystallization.md) measures, and it is *built* early, which is what patching finds. Attribution can only see where it is visible.

## What it is not

This is not a claim that attribution is broken. Attribution answers its own question correctly: the late MLPs really do carry most of the reward's final expression, and if you want to know where the score is written, that is a true and useful answer. The error is only in the substitution. "The late MLPs explain the reward" quietly becomes "the late MLPs cause the reward," and the second sentence is false on this model.

The result is also model-dependent, which is itself the point. On ArmoRM the same correlation is about \(+0.05\), near zero, no reliable relationship in either direction. So you cannot even memorize "attribution anti-predicts patching" as a law. What a tool sees and what is true line up differently on different models, which is precisely why you cannot take any single tool's word for it.

## The rule, and where it leads

Read with the observational tools, confirm with the causal ones, and never quote one as if it were the other.

That rule is easy to state and easy to forget under deadline, so the library stopped relying on you to remember it. In the first version both tools returned a bare number, and nothing stopped you from ranking components by the cheap one and calling them causal. Now every measurement comes back marked with how far it has earned your belief, and a tool earns a stronger claim only by being checked against a case where the answer is known. This anti-correlation is the exact case that motivated that machinery. An instrument you cannot check against a known answer is a rumor, and until you can plant that answer and score the tool against it, "attribution predicts causation" is a rumor you should not repeat.

That is the door into the second half of the site. [A measurement you can trust](measurement-you-can-trust.md) says what the receipt on every number means, and [calibration and organisms](../discipline/calibration-and-organisms.md) shows how a tool earns a stronger one. The full accounting of this and the other traps lives on the [honesty page](../caveats.md).
