# API reference

**Which import surface owns a name, and will reaching for it pull in torch?** Those two questions cause most import errors, and this section answers them one page per kernel subsystem.

The pages are generated. mkdocstrings reads the source in `src/` through griffe, which parses the code statically instead of importing it, so the reference builds with no torch, no model, and no GPU, and it cannot drift from the signatures it documents. What you read here is what the source says.

The layout follows the kernel. `import reward_lens` and the two pure-Python subsystems under it, `reward_lens.core` and `reward_lens.stats`, pull only numpy and scipy. Everything that touches a model lives one level down in its own subsystem and is imported from there: `from reward_lens.signals import load_signal`, `from reward_lens.measure.battery import DirectLinearAttribution`. The top level stays deliberately thin. The 1.0 names it still exposes are lazy, resolved on first access, so importing the package costs nothing you did not ask for.

<div class="grid cards" markdown>

-   __[Core and evidence](core.md)__

    The `Evidence` object, the trust and gauge enums, the three gates, and the append-only store. Torch-free.

-   __[Stats](stats.md)__

    Effective sample size, clone detection, cluster bootstrap, effect sizes, ROC, and mutual information. Pure numpy and scipy.

-   __[Signals](signals.md)__

    The `RewardSignal` protocol, the loaders, and the eight grader adapters behind one interface.

-   __[Data](data.md)__

    The preference schema, the built-in diagnostic set, lineage for honest sample sizes, and span maps.

-   __[Measure and indices](measure.md)__

    The runner, the eleven battery observables, and the eighteen scalar indices.

-   __[Interventions](interventions.md)__

    Patch, steer, ablate, edit, erase, and the erasure certificate that grades its own success.

-   __[Geometry](geometry.md)__

    Frames, canonicalization, the cross-model angle that arrives with a receipt, and the Hessian spectrum.

-   __[Concepts](concepts.md)__

    Concept directions, reward alignment, dose-response slopes, and calibrated probes.

-   __[Organisms](organisms.md)__

    The twelve planted-rule generators and the scorecard that grades an instrument against a known answer.

-   __[Dynamics and loops](dynamics-loops.md)__

    Checkpoint chains, best-of-N accounting, susceptibility, and the rollout recorder.

-   __[Studies](studies.md)__

    The spec, freeze, run, and the theorem scoreboard.

-   __[Artifacts and operate](artifacts-operate.md)__

    Cards, the population Atlas, the manuscript claims checker, the CLI, and the in-process MCP surface.

-   __[Legacy (1.0 API)](legacy.md)__

    The preserved 1.0 compatibility layer and the submodule-only tools it did not fold in.

</div>
