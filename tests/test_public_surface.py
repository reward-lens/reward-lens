"""What this library exports, and whether an outside caller can reach it without a private path.

The C1 chain experiment is a separate repository that imports `reward_lens` the way any external
user would, and its seam gate fails the build if any import reaches a private module path rather
than a package-level `__init__`. The strongest thing that run can say is that its intervention
worked through the shipped public API rather than through a hook somebody added for the demo, and
that sentence is only true if the API is wide enough to have been used.

**What "private" means here**, and it is the gate's definition rather than a new one: a dotted path
is private if any component of it starts with an underscore, or if any component after the root is
not named in its parent package's `__all__`. A symbol is publicly reachable if some public prefix
of the module it lives in names it in `__all__` *and* binds the same object. The identity check is
not decoration. Without it a symbol reached at `a.b.c` would match an unrelated same-named symbol
exported from `a`, and the checker would report a route that does not lead where the caller went.

**Why there is a negative case in here.** A checker that only ever confirms presence has not been
tested; it has been demonstrated. `reward_lens.geometry.hessian.gradient_ascent_probe` is the
white-box attack search, it is dual-use, and this library deliberately does not re-export it from
any package. If the checker below reports it public, the checker is wrong, and every green answer
above it is worth nothing. Same for `reward_lens.monitor._base`, whose leading underscore is the
other half of the definition.

That the negatives can fail was checked by breaking things on purpose rather than by reading them.
Two mutants of the checker, one making `is_public_module` return `True` unconditionally and one
dropping the object-identity comparison from `public_route`, turn eight tests red, including the
four `xfail(strict=True)` entries this file carried at the time, which flipped to failures because
they started passing. A third mutant, a copy of `src/` in which `geometry` re-exports
`gradient_ascent_probe`, turns the dual-use test red on its own with
`assert 'reward_lens.geometry' is None`.

**The xfails are gone, and that is the point of having had them.** Three of the sixteen sites the
frozen experiment tree imports live under `reward_lens.tap.adapters`, whose `__all__` was empty on
purpose. Those three sites and the matching module-path check were marked `xfail(strict=True)` so
that the day the decision went the other way this file would go red and force the ledger row closed
rather than quietly agreeing. P-LIB3 made that decision, on a measurement rather than on the fear
the empty list was written against: naming the four adapter modules drags no framework into a
process that did not ask for one, because none of them imports its framework at module scope. The
marks came off in the same change. `BUG_LEDGER.md` P-LIB1-1 carries the measurement and what was
traded for what.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from types import ModuleType

import pytest

from reward_lens.core.extras import ExtraRequiredError

ROOT = "reward_lens"


# ---------------------------------------------------------------------------
# the checker
# ---------------------------------------------------------------------------


def _import(dotted: str) -> ModuleType:
    return importlib.import_module(dotted)


def needs_an_extra(dotted: str) -> str:
    """The message a subsystem raised because an optional extra is not installed, or "".

    The `base-install` CI job builds the base wheel, installs it into an interpreter with nothing
    compiled in it, and runs `pytest tests/` with no marker deselection. Ten of the twenty-four
    subpackages named in the top-level `__all__` call `require_extra` at module scope and refuse to
    import there. Refusing is correct, and it is the behaviour
    `tests/acceptance/test_w0_3_dependencies.py` exists to hold, so this file has to tell a refusal
    apart from a broken export rather than counting both as failure.
    """
    try:
        _import(dotted)
    except ExtraRequiredError as exc:
        return str(exc)
    except ImportError:
        return ""
    return ""


def skip_without_extras(*dotted: str) -> None:
    """Skip the calling test if any of these subsystems is gated behind an absent extra.

    The refusal's own first sentence is the skip reason, because it names the extra and the command
    that installs it, and a skip line reading "needs an extra" would send the reader back to the
    source to find out which.
    """
    for name in dotted:
        message = needs_an_extra(name)
        if message:
            pytest.skip(message.split(". ")[0].strip() + ".")


def is_public_module(dotted: str) -> bool:
    """Is every component of `dotted` reachable through its parent package's `__all__`?

    The root is public by definition; it is the thing being imported. Every component after it has
    to be named in the `__all__` of the package above it, and no component may start with an
    underscore.

    **The name has to refer to the module.** `reward_lens.monitor` exports a function called
    `eprocess` and also contains a module called `eprocess`, and a check that stopped at "the
    string is in `__all__`" would call `reward_lens.monitor.eprocess` a public module path on the
    strength of the function. It is not one: nothing that walks that dotted path arrives at what
    `__all__` promised. So where the parent has the attribute bound, it has to be the submodule
    itself. Where the parent declares the name without binding it, which is how `measure` and
    `record` name a submodule they do not import, the declaration can only mean the submodule and
    a spec for it is enough.
    """
    parts = dotted.split(".")
    if parts[0] != ROOT:
        return False
    if any(p.startswith("_") for p in parts[1:]):
        return False
    for depth in range(1, len(parts)):
        parent = _import(".".join(parts[:depth]))
        name = parts[depth]
        if name not in getattr(parent, "__all__", ()):
            return False
        child = ".".join(parts[: depth + 1])
        bound = getattr(parent, name, None)
        if bound is None:
            if importlib.util.find_spec(child) is None:
                return False
        elif not (isinstance(bound, ModuleType) and bound.__name__ == child):
            return False
    return True


def public_route(module_path: str, symbol: str) -> str | None:
    """The shortest public package that exports `symbol` as the same object, or None.

    Returning the package rather than a bool is what makes a failure message useful: the caller
    finds out where to import from, not merely that it imported from the wrong place.
    """
    target = getattr(_import(module_path), symbol)
    parts = module_path.split(".")
    for depth in range(1, len(parts) + 1):
        candidate = ".".join(parts[:depth])
        if not is_public_module(candidate):
            continue
        module = _import(candidate)
        if symbol in getattr(module, "__all__", ()) and getattr(module, symbol, None) is target:
            return candidate
    return None


# ---------------------------------------------------------------------------
# the checker distinguishes its two answers
# ---------------------------------------------------------------------------


def test_the_checker_reports_an_exported_symbol_as_public():
    assert public_route("reward_lens.record.contract", "ContractSchema") == "reward_lens.record"


def test_the_checker_reports_the_dual_use_attack_search_as_unreachable():
    """`gradient_ascent_probe` is deliberately not re-exported from anywhere. It must not be found.

    This is the test that lets the ones above mean something. It fails if the checker is a rubber
    stamp, and it fails if somebody widens an `__all__` far enough to publish a dual-use symbol
    that `test_interventions_erase.py::test_attack_generator_is_not_reexported` says stays private.
    """
    skip_without_extras("reward_lens.geometry")
    assert public_route("reward_lens.geometry.hessian", "gradient_ascent_probe") is None


def test_the_checker_reports_an_underscore_component_as_private():
    """`monitor._base` holds `MonitorInstrument`, which `monitor` re-exports.

    So the symbol is publicly reachable and the module it lives in is not, and the checker has to
    give the two different answers. A checker that reasoned from the symbol alone would report the
    underscore path public, which is the one mistake this definition exists to prevent.
    """
    assert is_public_module("reward_lens.monitor._base") is False
    assert public_route("reward_lens.monitor._base", "MonitorInstrument") == "reward_lens.monitor"
    assert public_route("reward_lens.monitor._base", "_step_value") is None


def test_the_checker_reports_an_unnamed_module_as_private():
    """`readjudicate` the module cannot be named: the package already exports a function of that
    name. So the module path stays private while every symbol in it is publicly reachable."""
    assert is_public_module("reward_lens.record.convert.readjudicate") is False
    assert (
        public_route("reward_lens.record.convert.readjudicate", "readjudicate")
        == "reward_lens.record.convert"
    )


def test_the_checker_refuses_a_same_named_impostor():
    """`core.quantity.Registry` and `core.registry.Registry` are different classes.

    `core.__all__` exports the second. A checker matching on name alone would call the first one
    public and point the caller at a class that is not the one they were using.
    """
    quantity_registry = _import("reward_lens.core.quantity").Registry
    assert _import("reward_lens.core").Registry is not quantity_registry
    assert public_route("reward_lens.core.quantity", "Registry") == "reward_lens.core.quantity"


# ---------------------------------------------------------------------------
# every name in every widened __all__ resolves
# ---------------------------------------------------------------------------


def _resolves(package: ModuleType, name: str) -> bool:
    """A name in `__all__` is either an attribute or an importable submodule.

    Both are legitimate. A package may name a submodule it does not import, which is how `measure`
    and `record.convert` stay cheap, and `from package import name` resolves either.

    A subsystem that refuses because an optional extra is missing counts as resolved. The name
    points at something real and the import reached it; what stopped it is a typed refusal naming
    an extra that `pip` can install, which is the library's documented behaviour rather than a hole
    in the export list. `ExtraRequiredError` subclasses `ImportError`, so it has to be caught first
    or the two answers collapse into one.
    """
    if hasattr(package, name):
        return True
    try:
        _import(f"{package.__name__}.{name}")
    except ExtraRequiredError:
        return True
    except ImportError:
        return False
    return True


def test_the_top_level_names_more_than_three_things():
    """The starting point was `["__version__", "core", "stats"]` against 28 subpackages on disk."""
    top = _import(ROOT)
    assert len(top.__all__) > 3
    assert "__version__" in top.__all__


def test_every_top_level_name_is_importable_from_the_package_level():
    top = _import(ROOT)
    unresolved = [n for n in top.__all__ if not _resolves(top, n)]
    assert unresolved == [], f"named in reward_lens.__all__ but not importable: {unresolved}"


@pytest.mark.parametrize(
    "dotted",
    [
        "reward_lens.artifacts",
        "reward_lens.core",
        "reward_lens.interventions",
        "reward_lens.measure",
        "reward_lens.monitor",
        "reward_lens.record",
        "reward_lens.record.convert",
        "reward_lens.tap",
        "reward_lens.verifier",
    ],
)
def test_every_name_a_widened_package_declares_resolves(dotted: str):
    skip_without_extras(dotted)
    package = _import(dotted)
    names = list(package.__all__)
    assert len(names) == len(set(names)), f"{dotted}.__all__ repeats a name"
    unresolved = [n for n in names if not _resolves(package, n)]
    assert unresolved == [], f"named in {dotted}.__all__ but not importable: {unresolved}"


def test_every_subpackage_the_top_level_names_is_a_real_package():
    top = _import(ROOT)
    on_disk = {m.name for m in pkgutil.iter_modules(top.__path__)}
    named = [n for n in top.__all__ if n != "__version__"]
    assert set(named) <= on_disk, f"named but not on disk: {sorted(set(named) - on_disk)}"


# ---------------------------------------------------------------------------
# the G8 pair, by the exact line the design asks for
# ---------------------------------------------------------------------------


def test_the_two_subspace_control_families_import_from_the_package_level():
    """The line the design asks for, written out rather than paraphrased."""
    skip_without_extras("reward_lens.interventions")

    from reward_lens.interventions import subspace_matched_random, target_orthogonal_random

    assert callable(subspace_matched_random)
    assert callable(target_orthogonal_random)
    assert public_route("reward_lens.interventions.rescue", "subspace_matched_random") == (
        "reward_lens.interventions"
    )
    assert public_route("reward_lens.interventions.rescue", "target_orthogonal_random") == (
        "reward_lens.interventions"
    )


def test_the_ambient_draw_is_exported_beside_them_and_is_a_different_object():
    """`norm_matched_random` draws over the full residual space and is not a subspace control.

    Part 9.2 rules the ambient draw out of the subspace control family in as many words. Exporting
    all three is right; letting anything treat them as interchangeable is not, so this pins that
    they are three distinct callables rather than aliases.
    """
    skip_without_extras("reward_lens.interventions")

    from reward_lens.interventions import (
        norm_matched_random,
        subspace_matched_random,
        target_orthogonal_random,
    )

    assert len({norm_matched_random, subspace_matched_random, target_orthogonal_random}) == 3


# ---------------------------------------------------------------------------
# what the frozen experiment tree actually imports
# ---------------------------------------------------------------------------

#: Every `reward_lens` symbol the frozen tree at `experiments/chain_c1/` imports, found by an AST
#: walk of all 136 of its modules rather than by reading the ones somebody remembered. This is the
#: set the seam gate tests. `import reward_lens` itself, which `library_pin.py` and
#: `recorder/rehearsal.py` do, is a bare package import and needs no entry.
FROZEN_TREE_IMPORTS = [
    ("reward_lens.record.contract", "ContractField"),
    ("reward_lens.record.contract", "ContractRecord"),
    ("reward_lens.record.contract", "ContractRefusal"),
    ("reward_lens.record.contract", "ContractSchema"),
    ("reward_lens.record.hardware", "DecodingParameters"),
    ("reward_lens.record.hardware", "EngineIdentity"),
    ("reward_lens.record.hardware", "HardwareIdentity"),
    ("reward_lens.record.hardware", "ReadContext"),
    ("reward_lens.record.hardware", "ReadContextMismatch"),
    ("reward_lens.record.labels", "adjudicate"),
    ("reward_lens.tap.contract", "DEFAULT"),
    ("reward_lens.tap.adapters.trl_contract", "ContractTRLTap"),
    ("reward_lens.tap.adapters.trl_contract", "MappingSources"),
    ("reward_lens.tap.adapters.trl_signature", "verify"),
]


@pytest.mark.parametrize("module_path,symbol", FROZEN_TREE_IMPORTS)
def test_the_frozen_tree_reaches_no_private_path(module_path: str, symbol: str):
    route = public_route(module_path, symbol)
    assert route is not None, (
        f"{module_path}.{symbol} is reachable only through a private path. "
        f"The frozen experiment tree imports it, and the seam gate fails the build on it."
    )


def test_the_frozen_trees_own_import_lines_are_public_paths():
    """The gate reads the import line, and the frozen tree cannot be edited to move an import.

    So it is not enough that the symbol is reachable somewhere public; the module the tree names
    has to be a public module in its own right. `reward_lens.record.contract` is, now that
    `record.__all__` names it.
    """
    for module_path in (
        "reward_lens.record.contract",
        "reward_lens.record.hardware",
        "reward_lens.record.labels",
        "reward_lens.tap.contract",
    ):
        assert is_public_module(module_path), f"{module_path} is not a public module path"


def test_the_frozen_trees_adapter_import_lines_are_public_paths():
    for module_path in (
        "reward_lens.tap.adapters.trl_contract",
        "reward_lens.tap.adapters.trl_signature",
    ):
        assert is_public_module(module_path), f"{module_path} is not a public module path"


# ---------------------------------------------------------------------------
# the three subsystems the four feature rungs call into
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "attack",
        "coverage",
        "fuzz",
        "growth",
        "metamorphic",
        "mutate",
        "replay",
        "sensitivity",
        "static",
    ],
)
def test_every_verifier_instrument_module_is_a_public_path(module: str):
    """Nine instruments, D1 to D6 and D8 to D10. There is no D7."""
    skip_without_extras("reward_lens.verifier")
    assert is_public_module(f"reward_lens.verifier.{module}")


def test_there_is_no_d7_module_to_find():
    """Stated so that a later reader does not go looking for the one that is not missing."""
    skip_without_extras("reward_lens.verifier")
    verifier = _import("reward_lens.verifier")
    modules = {m.name for m in pkgutil.iter_modules(verifier.__path__)}
    assert len([m for m in modules if not m.startswith("_")]) == 9


@pytest.mark.parametrize(
    "module",
    ["ablate", "base", "certify", "edit", "erase", "patch", "rescue", "steer"],
)
def test_every_intervention_module_is_a_public_path(module: str):
    skip_without_extras("reward_lens.interventions")
    assert is_public_module(f"reward_lens.interventions.{module}")


@pytest.mark.parametrize(
    "module",
    ["arl", "check_standard", "conjunction", "cusum", "ewma", "operating_point"],
)
def test_every_monitor_instrument_module_is_a_public_path(module: str):
    assert is_public_module(f"reward_lens.monitor.{module}")


def test_the_monitor_eprocess_module_is_the_one_that_cannot_be_named():
    """`monitor` exports a function called `eprocess`, so the module of that name cannot join
    `__all__` without shadowing it. Recorded here so the gap is deliberate rather than an oversight.
    """
    monitor = _import("reward_lens.monitor")
    assert "eprocess" in monitor.__all__
    assert callable(monitor.eprocess)
    assert is_public_module("reward_lens.monitor.eprocess") is False


# ---------------------------------------------------------------------------
# the widening did not cost the guarantee that pays for it
# ---------------------------------------------------------------------------


def test_the_named_modules_the_design_calls_out_are_all_reachable():
    """`interventions/rescue`, `artifacts/sealed` and `record/convert/inspect_eval`."""
    skip_without_extras("reward_lens.interventions")

    from reward_lens.artifacts import scan
    from reward_lens.interventions import knockout_and_rescue
    from reward_lens.record.convert import read_eval

    assert callable(scan)
    assert callable(knockout_and_rescue)
    assert callable(read_eval)


def test_core_exports_the_five_families_it_used_to_hide():
    from reward_lens.core import (  # noqa: F401
        QUANTITIES,
        EnvelopeSpec,
        LimitOfDetection,
        ReferenceMaterial,
        Refusal,
    )


# ---------------------------------------------------------------------------
# the seven the C1 seam reaches through a submodule rather than a package
# ---------------------------------------------------------------------------

#: `(package, symbol, module the symbol is defined in)` for every symbol the C1 chain experiment
#: reaches by naming a submodule. `public_route` already calls three of these public, because
#: `tap.adapters.__all__` names `trl_contract` and `trl_signature` and a public module path is a
#: public route by that checker's definition. It is not the route the caller wants to write.
#: `from reward_lens.tap.adapters import ContractTRLTap` is, and until this list is green that line
#: raises `ImportError` while the module-path check reports green beside it.
#:
#: Commission section 14 is the reason the widening happens in the library rather than in the
#: experiment, and D-09 is the reason it is a widening rather than a shim: an experiment that
#: reaches past the public surface has not demonstrated the public surface.
PACKAGE_INIT_SEAM = [
    ("reward_lens.measure.ledger", "surface_features", "reward_lens.measure.ledger.features"),
    ("reward_lens.measure.ledger", "transition_window", "reward_lens.measure.ledger.prediction"),
    ("reward_lens.measure.ledger", "DEFAULT_SHIFTS", "reward_lens.measure.ledger.reconstruct"),
    (
        "reward_lens.measure.ledger",
        "REGISTERED_ANALYSIS_LAGS",
        "reward_lens.measure.ledger.reconstruct",
    ),
    ("reward_lens.tap.adapters", "MappingSources", "reward_lens.tap.adapters.trl_contract"),
    ("reward_lens.tap.adapters", "ContractTRLTap", "reward_lens.tap.adapters.trl_contract"),
    ("reward_lens.tap.adapters", "verify", "reward_lens.tap.adapters.trl_signature"),
]


@pytest.mark.parametrize("package,symbol,defined_in", PACKAGE_INIT_SEAM)
def test_the_seam_symbol_is_reachable_from_its_package_init(
    package: str, symbol: str, defined_in: str
):
    """Three assertions, and the third is the one that stops this being a rename.

    The name has to be declared in the package's `__all__`, it has to be bound on the package so
    that `from <package> import <symbol>` resolves, and the object it binds has to be the object
    the private module defines. Without the identity check a same-named unrelated export would
    satisfy the first two and send the caller somewhere else.
    """
    pkg = _import(package)
    private = _import(defined_in)
    target = getattr(private, symbol)

    assert symbol in getattr(pkg, "__all__", ()), (
        f"{package}.__all__ does not name {symbol}; the only route to it is {defined_in}, "
        f"which is a submodule reach and what the C1 seam gate fails the build on."
    )
    bound = getattr(pkg, symbol, None)
    assert bound is not None, f"{package} declares {symbol} in __all__ but does not bind it"
    assert bound is target, (
        f"{package}.{symbol} is not {defined_in}.{symbol}; the public name resolves to a "
        f"different object, so the public route does not lead where the caller went"
    )
