"""The science layer: the sixteen sciences as preregistered studies over the kernel (DESIGN Part III).

Each science is a directory here holding a frozen study spec and a thin analysis function (R9); none
adds a kernel subsystem. The analysis functions consume the kernel (signals, data, geometry,
measure, organisms, loops) and emit REGISTERED Evidence through the studies engine
(`reward_lens.studies`). This top-level package is deliberately separate from the engine package of
the same name inside `reward_lens`: the engine is machinery, this is the corpus of studies that run
on it.
"""
