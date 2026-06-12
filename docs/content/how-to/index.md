# How-to guides

Short recipes for one job each. If you know what you want to do and just need the calls, start here. If you want to understand why the calls are shaped the way they are, the [tool pages](../tools/index.md) and [concepts](../concepts/index.md) are the longer read.

<div class="grid cards" markdown>

-   __[Detect length bias](detect-length-bias.md)__

    Find out whether a reward model pays for longer answers, and read it as an effect size.

-   __[Attribute a reward score](attribute-a-score.md)__

    Break one preference into per-component contributions, and know why that is only the first look.

-   __[Compare two reward models](compare-two-models.md)__

    See where and how two models form the same preference, on one GPU.

-   __[Patch without running out of memory](patching-memory.md)__

    Triage with the cheap tools, then patch only the components that matter.

-   __[Train an SAE on reward activations](train-an-sae.md)__

    Collect activations, fit a sparse dictionary, read the reward through its features.

-   __[Write an adapter for your model](write-an-adapter.md)__

    Make `reward-lens` work with a model family it does not ship support for.

</div>
