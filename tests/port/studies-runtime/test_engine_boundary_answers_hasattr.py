"""Defect 6 of the release diff, carried into 4.0.

`EngineBoundary` is raised out of `ServingPolicy.__getattr__`. Python decides that an attribute is
absent by catching `AttributeError` out of the lookup, so under a plain `RuntimeError` every
`hasattr` on a serving policy propagated the exception instead of answering False. `ServingPolicy`
documents `isinstance(serving, PolicySubject)` to be False, and before 3.12 that answer is decided
by walking `typing._get_protocol_attrs`, which is a set: whether the walk reached `capture` before
some other absent name was hash order, per-process randomised, so the same commit passed and
failed on the same interpreter.

The policy package reaches a subsystem that refuses without the `white-box` extra, so this reads
the class statement out of the source with `ast` rather than importing it, and pairs that with the
mechanism itself run in-process on a class carrying the same bases. Between them the two bind the
source to the behaviour without a torch install. This is recorded in the handoff.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SOURCE = (
    pathlib.Path(__file__).resolve().parents[2].parent
    / "src"
    / "reward_lens"
    / "policy"
    / "vllm.py"
)
PLANE_B = ("capture", "grad_h", "token_gradients", "hvp", "with_interventions")


def _class_def(name: str) -> ast.ClassDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {SOURCE}")


def test_engine_boundary_subclasses_attribute_error_as_well_as_runtime_error() -> None:
    """The language's own contract for an absent attribute, and the whole of the fix."""
    bases = [b.id for b in _class_def("EngineBoundary").bases if isinstance(b, ast.Name)]
    assert "AttributeError" in bases, (
        "EngineBoundary is raised out of __getattr__, so hasattr propagates it instead of "
        "answering False unless it is an AttributeError"
    )
    assert "RuntimeError" in bases, "the existing contract is not dropped to add the new one"


def test_the_boundary_is_what_getattr_raises_for_the_plane_b_names() -> None:
    """The fix is only worth anything at the site that raises it."""
    getattr_def = next(
        n
        for n in ast.walk(_class_def("ServingPolicy"))
        if isinstance(n, ast.FunctionDef) and n.name == "__getattr__"
    )
    raised = {
        n.exc.func.id
        for n in ast.walk(getattr_def)
        if isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and isinstance(n.exc.func, ast.Name)
    }
    assert "EngineBoundary" in raised
    names = {
        c.value
        for n in ast.walk(getattr_def)
        for c in ast.walk(n)
        if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value in PLANE_B
    }
    assert names == set(PLANE_B), f"the intercepted names moved: {sorted(names)}"


@pytest.mark.parametrize("bases", [(RuntimeError, AttributeError)])
def test_hasattr_answers_false_when_getattr_raises_that_exception(bases: tuple) -> None:
    """The mechanism itself, in process: an AttributeError subclass is what hasattr catches."""
    boundary = type("Boundary", bases, {})

    class Subject:
        def __getattr__(self, name: str):
            if name in PLANE_B:
                raise boundary(f"`{name}` is not available behind a serving engine")
            raise AttributeError(name)

    subject = Subject()
    for name in PLANE_B:
        assert hasattr(subject, name) is False


def test_a_plain_runtime_error_is_the_failure_this_carries() -> None:
    """The counterfactual, so the test above is not passing for an unrelated reason."""
    boundary = type("Boundary", (RuntimeError,), {})

    class Subject:
        def __getattr__(self, name: str):
            raise boundary(name)

    with pytest.raises(RuntimeError):
        hasattr(Subject(), "capture")


def test_the_boundary_still_names_the_backend_that_can_take_the_reading() -> None:
    """Nothing about the message changes: the sentence is what the exception is for."""
    text = SOURCE.read_text(encoding="utf-8")
    assert "is not available behind a serving engine" in text
    assert "reward_lens.policy.hf.from_pretrained" in text
