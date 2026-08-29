"""Framework adapters: what binds ``tap/`` to somebody else's training loop (section 8.2).

Nothing in ``tap/`` knows what a framework is. That is deliberate, and it leaves three jobs that
only an adapter can do.

It has to find *every* call site. TRL calls its reward functions from two places,
``grpo_trainer.py:1659-1661`` synchronously and ``:1671-1673`` under ``await``, and the branch
between them is ``inspect.iscoroutinefunction(reward_func)`` at ``:1655``. A wrapper that is
synchronous around an asynchronous grader does not merely miss half the calls, it changes which
branch TRL takes and hands the host a coroutine where it expects a list.

It has to wrap the individual reward function rather than the aggregator, because an aggregator is
usually where an exception gets turned into a zero and the whole point of ``CallOutcome`` is to be
upstream of that.

And it owns the step boundary. ``tap/`` measures and buffers; it does not know where a step ends,
so it never emits. The adapter is what calls ``effect()`` at a boundary and turns a pile of
``GraderCall`` records into the five-level hierarchy in ``record/``.

Every module in here imports its framework lazily, inside functions. ``tap/`` is part of the
torch-free core and importing this package must not change that, so nothing at module scope here
imports ``trl``, ``transformers`` or ``torch``.
"""

from __future__ import annotations

from reward_lens.tap.adapters.trl_contract import ContractTRLTap, MappingSources
from reward_lens.tap.adapters.trl_signature import verify

#: The four adapter modules, named so that ``reward_lens.tap.adapters.trl_contract`` and
#: ``reward_lens.tap.adapters.trl_signature`` are public module paths (BUG_LEDGER P-LIB1-1), and the
#: three symbols a caller was previously obliged to reach into a submodule for.
#:
#: This list was empty on purpose, and the guarantee written against it was that a star import of
#: this package could never be the thing that drags a framework into a process that did not ask for
#: one. The guarantee that actually holds is the one the module docstring above already states: no
#: adapter imports its framework at module scope. Naming the four modules binds four module objects;
#: naming the three symbols additionally imports ``trl_contract`` and ``trl_signature`` here, and the
#: module-scope closure of both is the standard library plus ``record.contract`` and ``tap.trl``,
#: with no framework anywhere in it. That is measured rather than argued, in
#: ``test_tap_trl.py::test_the_adapters_package_star_import_pulls_no_framework``, which runs the
#: star import in a subprocess and asserts no framework reached ``sys.modules``.
#:
#: What the empty list did buy and this does not: ``from reward_lens.tap.adapters import *`` now
#: binds a name ``trl`` in the caller's namespace, and it is this package's adapter module rather
#: than the ``trl`` distribution, and a bare ``verify``, which is generic enough to collide. An
#: explicit ``from reward_lens.tap.adapters.trl import TRLTap`` is still the import to write.
__all__: list[str] = [
    "ContractTRLTap",
    "MappingSources",
    "trl",
    "trl_contract",
    "trl_signature",
    "verifiers",
    "verify",
]
