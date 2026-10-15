"""Acceptance tests for the optional local neural reward-model adapter.

The model and tokenizer are deliberately tiny local fixtures.  They are constructed in memory;
no hub identifier, provider, GPU, or network request participates in these tests.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

from reward_lens.contracts import CapabilityUnavailable

REPO = Path(__file__).resolve().parents[2]


def _load():
    return importlib.import_module("reward_lens.graders.reward_model")


def test_missing_train_extra_refuses_with_the_allocated_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    original_make = module.errors.make
    calls: list[tuple[str, dict[str, str]]] = []

    def recorded_make(code: str, **context: str):
        calls.append((code, context))
        return original_make(code, **context)

    monkeypatch.setattr(module.errors, "make", recorded_make)
    monkeypatch.setattr(
        module.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name))
    )
    with pytest.raises(CapabilityUnavailable) as caught:
        module.RewardModel.from_pretrained(object(), object())
    error = caught.value
    assert error.code == "RL0704"
    assert error.exit_code == 5
    assert error.context == {"extra": "train", "capability": "neural_reward_model"}
    assert "train" in error.message
    assert calls == [("RL0704", {"extra": "train", "capability": "neural_reward_model"})]


def test_cli_import_graph_never_reaches_torch() -> None:
    """The walk begins at the CLI package, rather than checking an import list by hand."""
    root = REPO / "src" / "reward_lens"
    pending = ["reward_lens.cli", "reward_lens.cli.main"]
    resolved: set[str] = set()
    seen: set[Path] = set()
    imported: set[str] = set()

    def source_for(module: str) -> Path | None:
        relative = Path(*module.split("."))
        for candidate in (root.parent / f"{relative}.py", root.parent / relative / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    while pending:
        module = pending.pop()
        if module in resolved:
            continue
        resolved.add(module)
        path = source_for(module)
        assert path is not None, f"could not resolve the package-local CLI import {module}"
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    names = [node.module or ""]
                else:
                    prefix = package
                    for _ in range(node.level - 1):
                        prefix = prefix.rpartition(".")[0]
                    names = [
                        (prefix + "." if prefix and node.module else prefix) + (node.module or "")
                    ]
            else:
                continue
            imported.update(name.split(".")[0] for name in names if name)
            pending.extend(name for name in names if name.startswith("reward_lens"))
    assert seen, "the graph walk must reach the CLI source"
    assert "torch" not in imported
    before = set(sys.modules)
    importlib.import_module("reward_lens.cli")
    importlib.import_module("reward_lens.cli.main")
    assert not ({"torch", "transformers"} & (set(sys.modules) - before))


def test_identity_and_scores_keep_every_transform_input() -> None:
    module = _load()
    model, tokenizer = module.tiny_local_model()
    adapter = module.RewardModel.from_pretrained(
        model,
        tokenizer,
        template="{prompt}\\nAssistant: {response}",
        preprocessing=module.Preprocessing(
            max_length=12, truncation_side="left", padding_side="right", batch_size=2
        ),
        transform=module.sigmoid,
        direction=module.ScoreDirection.HIGHER_IS_BETTER,
    )
    identity = adapter.identity
    assert all(
        value.startswith("sha256:")
        for value in (
            identity.weights_digest,
            identity.tokenizer_digest,
            identity.template_digest,
            identity.preprocessing_digest,
        )
    )
    result = adapter.score_item("say hi", "hello")
    assert result.raw_score != result.transformed_score
    assert result.direction is module.ScoreDirection.HIGHER_IS_BETTER
    assert result.transformed_score == pytest.approx(module.sigmoid(result.raw_score))
    envelope = adapter.score({"id": "t1", "prompt": "say hi"}, "hello", sandbox=None, limits=None)
    assert envelope.score == pytest.approx(result.transformed_score)
    assert envelope.observations["raw_score"] == pytest.approx(result.raw_score)
    assert envelope.observations["transformed_score"] == pytest.approx(result.transformed_score)


def test_tokenizer_digest_binds_configuration_and_observed_tokenisation() -> None:
    module = _load()
    first_model, first_tokenizer = module.tiny_local_model(token_scores={"low": -4.0, "high": 4.0})
    second_model, second_tokenizer = module.tiny_local_model(
        token_scores={"low": -4.0, "high": 4.0}
    )
    second_tokenizer.normalizer = "reverse"
    policy = module.Preprocessing(max_length=1, truncation_side="right")
    first = module.RewardModel.from_pretrained(
        first_model, first_tokenizer, template="{prompt} {response}", preprocessing=policy
    )
    second = module.RewardModel.from_pretrained(
        second_model, second_tokenizer, template="{prompt} {response}", preprocessing=policy
    )
    assert first_tokenizer.get_vocab() == second_tokenizer.get_vocab()
    assert first.identity.tokenizer_digest != second.identity.tokenizer_digest
    assert first.score_item("", "low high").raw_score == pytest.approx(-4.0)
    assert second.score_item("", "low high").raw_score == pytest.approx(4.0)


def test_declared_direction_orders_a_known_preference_pair() -> None:
    module = _load()
    model, tokenizer = module.tiny_local_model(token_scores={"good": 3.0, "bad": -3.0})
    adapter = module.RewardModel.from_pretrained(model, tokenizer, direction="higher_is_better")
    pair = module.PreferencePair(
        id="pair-1", prompt="p", chosen="good", rejected="bad", preferred="chosen"
    )
    outcome = adapter.score_pairs([pair])[0]
    assert outcome.passed is True
    assert outcome.chosen.transformed_score > outcome.rejected.transformed_score
    assert outcome.direction is module.ScoreDirection.HIGHER_IS_BETTER


def test_batching_is_an_observed_invariance_probe() -> None:
    module = _load()
    model, tokenizer = module.tiny_local_model()
    adapter = module.RewardModel.from_pretrained(
        model, tokenizer, preprocessing=module.Preprocessing(batch_size=2)
    )
    probe = adapter.probe([("p", "alpha"), ("p", "beta"), ("p", "gamma")])
    assert probe.batching.passed is True
    assert adapter.describe().capabilities.probed["batching_invariant"] is True


def test_padding_is_an_observed_invariance_probe() -> None:
    module = _load()
    model, tokenizer = module.tiny_local_model()
    adapter = module.RewardModel.from_pretrained(
        model, tokenizer, preprocessing=module.Preprocessing(batch_size=2)
    )
    probe = adapter.probe([("p", "short"), ("p", "a much longer response")])
    assert probe.padding.passed is True
    assert adapter.describe().capabilities.probed["padding_invariant"] is True


def test_truncation_probe_retains_the_configured_side_and_its_raw_score() -> None:
    module = _load()
    left_model, left_tokenizer = module.tiny_local_model(
        token_scores={"discard": -3.0, "tail": 2.0}
    )
    right_model, right_tokenizer = module.tiny_local_model(
        token_scores={"discard": -3.0, "tail": 2.0}
    )
    left = module.RewardModel.from_pretrained(
        left_model,
        left_tokenizer,
        template="{prompt} {response}",
        preprocessing=module.Preprocessing(max_length=1, truncation_side="left"),
    )
    right = module.RewardModel.from_pretrained(
        right_model,
        right_tokenizer,
        template="{prompt} {response}",
        preprocessing=module.Preprocessing(max_length=1, truncation_side="right"),
    )
    left_probe = left.probe([("", "discard tail")])
    right_probe = right.probe([("", "discard tail")])
    assert left_probe.truncation.evidence["configured_side"] == "left"
    assert left_probe.truncation.evidence["raw_score"] == pytest.approx(2.0)
    assert right_probe.truncation.evidence["configured_side"] == "right"
    assert right_probe.truncation.evidence["raw_score"] == pytest.approx(-3.0)
    assert left_probe.truncation.evidence["configured_side_matches_full_control"] is True
    assert right_probe.truncation.evidence["configured_side_matches_full_control"] is True
    assert left.describe().capabilities.probed["truncation_observed"] is True


def test_lying_tokenizer_cannot_claim_left_truncation_from_its_side_label() -> None:
    module = _load()
    model, honest = module.tiny_local_model(token_scores={"discard": -3.0, "tail": 2.0})

    class LyingTokenizer:
        """Advertises left truncation but always delegates a right-truncated sequence."""

        def __init__(self, inner) -> None:
            object.__setattr__(self, "inner", inner)
            object.__setattr__(self, "padding_side", "right")
            object.__setattr__(self, "truncation_side", "right")

        def __setattr__(self, name, value) -> None:
            if name == "truncation_side":
                object.__setattr__(self, name, "right")
            else:
                object.__setattr__(self, name, value)

        def get_vocab(self):
            return self.inner.get_vocab()

        def __call__(self, texts, **kwargs):
            self.inner.padding_side = self.padding_side
            self.inner.truncation_side = "right"
            return self.inner(texts, **kwargs)

    adapter = module.RewardModel.from_pretrained(
        model,
        LyingTokenizer(honest),
        template="{prompt} {response}",
        preprocessing=module.Preprocessing(max_length=1, truncation_side="left"),
    )
    probe = adapter.probe([("", "discard tail")])
    assert probe.truncation.evidence["configured_side"] == "left"
    assert probe.truncation.evidence["raw_score"] == pytest.approx(-3.0)
    assert probe.truncation.evidence["expected_raw_score"] == pytest.approx(2.0)
    assert probe.truncation.passed is False
    assert adapter.describe().capabilities.probed["truncation_observed"] is False


def test_labelled_pairs_are_scored_and_manifest_refuses_unproven_claims() -> None:
    module = _load()
    model, tokenizer = module.tiny_local_model(token_scores={"chosen": 2.0, "rejected": -2.0})
    adapter = module.RewardModel.from_pretrained(model, tokenizer)
    outcomes = adapter.score_pairs(
        [
            module.PreferencePair(
                id="a", prompt="p", chosen="chosen", rejected="rejected", preferred="chosen"
            ),
            module.PreferencePair(
                id="b", prompt="p", chosen="rejected", rejected="chosen", preferred="rejected"
            ),
        ]
    )
    assert [item.passed for item in outcomes] == [True, True]
    assert adapter.describe().capabilities.probed["labelled_pairs_scored"] is True
    assert adapter.labelled_pair_evidence == tuple(outcomes)
    assert adapter.with_capabilities({"labelled_pairs_scored": True}) is adapter.describe()
    with pytest.raises(module.UnprovenRewardModelCapability):
        adapter.with_capabilities({"batching_invariant": True})
