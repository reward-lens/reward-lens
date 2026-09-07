"""W0.3 acceptance: the core installs with no compiled dependency, and says so honestly.

The clause this file discharges: *a clean venv installs the base wheel with no compiled
dependency; the torch-free test subset passes; importing a white-box module without the extra
raises a typed error naming an extra that exists; a test asserts `import reward_lens` does not
import torch.*

The first two clauses are verified by actually building the wheel and installing it into a fresh
interpreter, which is slower than trusting the metadata and is the only way to catch a transitive
compiled dependency that nobody declared. They are marked so the fast run skips them; CI does not.

The third clause is the one with history. `loops/integrations/base.py` used to raise
"Install reward-lens[trl]" for three frameworks, none of which was declared in `pyproject.toml`.
The error was clear, actionable, and impossible to act on. So the test here is not that an error
is raised; it is that every extra any error can name is an extra `pip` can install.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

# `tomllib` is 3.11+. The CI matrix runs 3.10, and on 3.10 this import took the whole module down at
# collection: ModuleNotFoundError, collection interrupted, zero tests run on the oldest interpreter
# the package claims to support. It is only used to read this repository's own `pyproject.toml`,
# which is a development-time concern rather than anything a user of the wheel touches, so on 3.10
# the module skips rather than pulling in a new dependency to parse a file the other two versions
# already cover.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 matrix entry
    tomllib = pytest.importorskip(
        "tomli",
        reason=(
            "reading pyproject.toml needs tomllib (3.11+) or tomli. Neither is present on 3.10, and "
            "these assertions are about the source tree rather than the installed package, so the "
            "3.11 and 3.12 matrix entries cover them."
        ),
    )

from reward_lens.core.extras import EXTRA_PROBE, EXTRA_PURPOSE, ExtraRequiredError, require_extra

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PYPROJECT = _ROOT / "pyproject.toml"

#: Distributions that ship compiled artifacts. The base wheel must pull none of them.
_COMPILED = {
    "torch",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "triton",
    "safetensors",
    "tokenizers",
    "transformers",  # not compiled itself, but it drags tokenizers, which is
}


def _declared() -> dict:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]


def test_import_reward_lens_does_not_import_torch():
    """Checked in a fresh interpreter, because this process has already imported torch."""
    r = subprocess.run(
        [sys.executable, "-c", "import reward_lens, sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert r.stdout.strip() == "False", r.stdout


def test_the_base_dependency_list_declares_nothing_compiled():
    deps = _declared()["dependencies"]
    names = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in deps}
    assert not (names & _COMPILED), f"base install pulls a compiled dependency: {names & _COMPILED}"


def test_every_extra_the_code_can_name_is_an_extra_pip_can_install():
    """The dead-end error, made impossible.

    `require_extra` raises with an extra name; this asserts every name it can produce is declared.
    A typo is a failing test rather than a user in a loop.
    """
    declared = set(_declared()["optional-dependencies"])
    assert set(EXTRA_PROBE) <= declared, f"named but not declared: {set(EXTRA_PROBE) - declared}"
    assert set(EXTRA_PURPOSE) == set(EXTRA_PROBE), "every extra needs a stated purpose"


#: The 4.0 extras, and `all`. The v3 file asserted `{"trl", "verl"} <= declared`, which was the
#: shape of the specific defect it was written for: three extras named in an error message and
#: declared nowhere. D-58 replaced that list wholesale, so the assertion below is the same claim
#: against the set 4.0 actually declares, and `tests/packaging/test_wheel_metadata.py` makes it
#: again against the built wheel's `Provides-Extra`, which is the copy that binds a user.
EXPECTED_EXTRAS_4_0 = frozenset(
    {
        "judge",
        "organisms",
        "record",
        "sigstore",
        "trace",
        "train",
        "verifier",
        "viz",
        "white-box",
    }
)


def test_the_declared_extras_are_the_four_zero_set():
    declared = set(_declared()["optional-dependencies"])
    assert declared - {"all"} == set(EXPECTED_EXTRAS_4_0), sorted(declared)


def test_all_is_a_superset_of_every_other_extra():
    """`reward-lens[all]` has to mean all of them, or the name is a lie a user finds out late.

    `all` is declared by self-reference, `reward-lens[trace]` and so on, so the check is over the
    extra names it recurses into and not over distributions. Written against the distribution
    lists it would report every package in every extra as missing, which is how this assertion
    fails in the wrong direction.
    """
    optional = _declared()["optional-dependencies"]
    recursed = {
        r.split("[", 1)[1].split("]", 1)[0].strip()
        for r in optional["all"]
        if r.split("[", 1)[0].strip().lower().replace("_", "-") == "reward-lens" and "[" in r
    }
    missing = sorted(set(optional) - {"all"} - recursed)
    assert missing == [], f"`all` does not pull in {missing}"


def test_requiring_a_missing_extra_raises_a_typed_error_naming_it():
    # `sampling` was the v3 example and is not a 4.0 extra, so the call under test used to be a
    # `KeyError` from `EXTRA_PROBE` before it could raise the error this test is about.
    # `white-box` is a real 4.0 extra whose probe is torch.
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed, so `white-box` is present and cannot be required")
    with pytest.raises(ExtraRequiredError) as exc:
        require_extra("white-box", subsystem="reward_lens.sae")
    msg = str(exc.value)
    assert "reward-lens[white-box]" in msg
    assert "reward_lens.sae" in msg
    # It subclasses ImportError, so the `except ImportError` most callers already write still works.
    assert isinstance(exc.value, ImportError)


def test_requiring_an_extra_that_does_not_exist_is_a_keyerror_not_a_message():
    """Better to fail at import than to hand a user an install command that cannot work."""
    with pytest.raises(KeyError):
        require_extra("gpu", subsystem="somewhere")


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="the guard is only observable with torch"
)
def test_a_white_box_module_raises_the_typed_error_when_the_extra_is_absent():
    """Importing a white-box module in an interpreter where torch is blocked.

    The guard sits on the subsystem's `__init__`, so importing anything beneath it fails at the
    boundary with the actionable message rather than somewhere in the middle with a bare
    ModuleNotFoundError.
    """
    # find_module/load_module were removed in 3.12; the finder protocol is find_spec.
    prog = (
        "import sys\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        "try:\n"
        "    import reward_lens.runtime.backend\n"
        "except Exception as e:\n"
        "    print(type(e).__name__)\n"
        "    print(str(e))\n"
    )
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
    assert "ExtraRequiredError" in r.stdout, r.stdout + r.stderr
    assert "reward-lens[white-box]" in r.stdout


#: Base dependencies that nothing under `src/` imports yet, each with the packet that will. A name
#: here is a promise, not an exemption: when its packet lands the import appears and the entry comes
#: out, and until then the list says out loud what the closure carries on spec.
DECLARED_BEFORE_THEIR_IMPORTER = {
    "click": "P-CLI. D-30 makes the CLI click; `reward_lens.cli` does not exist yet and the "
    "console scripts in `pyproject.toml` point at `reward_lens.cli.main:main`.",
    "jsonschema-rs": "the packet that lands schema validation over the record. Declared on D-58's "
    "reasoning that the Rust validator is the runtime one and the pure-Python `jsonschema` stays "
    "a dev cross-check; nothing under `src/` calls either yet.",
    "rfc8785": "the packet that lands the canonical byte form. D-10 makes RFC 8785 JCS the form "
    "every digest is taken over; `core/types.content_hash` does not use it yet.",
}


def test_no_declared_dependency_is_unused():
    """Declaring what you do not import is the same defect as importing what you do not declare.

    The old list carried `jaxtyping`, `einops`, `seaborn` and a `[sae]` extra whose only entry was
    `wandb`, and not one of those four is imported anywhere in the package.

    The distribution-to-module map is explicit because two of the seven base names differ from
    their import name, and guessing wrong there reads as an unused dependency: `pyyaml` imports as
    `yaml` and `jsonschema-rs` as `jsonschema_rs`. The v3 version mapped neither, so it would have
    reported both as unused had it ever reached them.
    """
    module_of = {
        "pyyaml": "yaml",
        "jsonschema-rs": "jsonschema_rs",
        "scikit-learn": "sklearn",
        "pydantic-settings": "pydantic_settings",
    }
    src = _ROOT / "src" / "reward_lens"
    blob = "\n".join(p.read_text(encoding="utf-8") for p in src.rglob("*.py"))
    unused = []
    for dep in _declared()["dependencies"]:
        name = dep.split(">")[0].split("=")[0].split("[")[0].strip()
        module = module_of.get(name, name.replace("-", "_"))
        if f"import {module}" not in blob and f"from {module}" not in blob:
            unused.append(name)
    unexpected = [n for n in unused if n not in DECLARED_BEFORE_THEIR_IMPORTER]
    assert unexpected == [], f"declared and never imported: {unexpected}"
    landed = [n for n in DECLARED_BEFORE_THEIR_IMPORTER if n not in unused]
    assert landed == [], f"{landed} is imported now; take it out of DECLARED_BEFORE_THEIR_IMPORTER"
