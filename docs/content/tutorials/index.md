# Tutorials

**Two ways in, and they answer different questions.** One takes a single reward model apart and shows you where its opinion lives, which parts of it wrote the score, and the moment two of our own tools contradict each other. The other never touches a big model: it teaches the thing that makes 2.0 different, that every number here arrives with a receipt saying how far to trust it, and it runs start to finish on a CPU in about a minute.

Read them in either order. If you came for interpretability, start with the first. If you came because a clean-looking metric has burned you before, start with the second. If you would rather not set anything up at all, the [Colab tour](https://colab.research.google.com/drive/1x5zG07HdsWlNsJmkl2ddJ1yalmwwujfY?usp=sharing) runs both stories, and a good deal more of the library, in a browser tab.

<div class="grid cards" markdown>

-   __Inside one reward model__

    Trace the sky-is-blue pair and watch the preference form late. Attribute the score to components. Patch to find what actually causes it. See attribution and patching disagree, with the real numbers. Then scan the model for bias. The scientific results come from a real 8B grader; the same instruments run on a toy model on your CPU.

    [:octicons-arrow-right-24: Inside one reward model](inside-one-reward-model.md)

-   __Measurements you can trust__

    Six steps, all on the CPU, all runnable now. Build a tiny signal, run an instrument, and read why the number comes back untrusted. Plant a rule you know the answer to, calibrate a detector against it, and watch trust climb. Freeze a question, run it, and let the frozen prediction adjudicate itself.

    [:octicons-arrow-right-24: Measurements you can trust](measurements-you-can-trust.md)

-   __The Colab tour__

    Both arcs and the rest of the library, executing, in a browser tab. The reward direction, evidence and the trust ladder, the eight graders behind one protocol, interventions, gauge, training loops, studies. Nothing to install.

    [:octicons-arrow-right-24: Open the tour in Colab](https://colab.research.google.com/drive/1x5zG07HdsWlNsJmkl2ddJ1yalmwwujfY?usp=sharing)

</div>

The first arc is the classic white-box story, refreshed for the 2.0 API. The second is the one to internalize before you publish a number, because it is the difference between a plot and a claim. The notebook is both of them with the code running, and it is the one to send someone who has fifteen minutes and no environment set up.
