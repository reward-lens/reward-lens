"""Optional, local neural reward-model adapter.

Importing this module is cheap: neither torch nor transformers is imported until a caller asks to
construct an adapter.  That boundary is intentional.  A reward model is optional machinery and
must not become part of the CLI's base import graph.

The adapter accepts an already loaded scoring-head model and tokenizer.  It never resolves a hub
name, never contacts a provider, and always moves the supplied model to CPU.  ``tiny_local_model``
exists solely as an offline conformance fixture: it is an actual ``torch.nn.Module`` with a learned
token-score table, paired with a wholly local whitespace tokenizer.
"""

from __future__ import annotations

import datetime as _datetime
import importlib
import math
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Callable, Mapping, Sequence

from reward_lens import errors
from reward_lens.contracts import CapabilityUnavailable, Counters, digest
from reward_lens.graders.base import CapabilityVector, EvidenceEnvelope, InvalidInput, Manifest

__all__ = [
    "PreferencePair",
    "PreferenceResult",
    "Preprocessing",
    "ProbeCheck",
    "RewardModel",
    "RewardModelIdentity",
    "RewardModelProbe",
    "RewardScore",
    "ScoreDirection",
    "UnprovenRewardModelCapability",
    "sigmoid",
    "tiny_local_model",
]


class ScoreDirection(StrEnum):
    """The interpretation of a transformed reward.  It is declared, never inferred."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class Preprocessing:
    """The tokenisation policy that can silently change a score."""

    max_length: int = 512
    truncation_side: str = "right"
    padding_side: str = "right"
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.truncation_side not in {"left", "right"}:
            raise ValueError("truncation_side is left or right")
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side is left or right")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "truncation_side": self.truncation_side,
            "padding_side": self.padding_side,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True)
class RewardModelIdentity:
    """Content-addressed provenance for every object that affects an RM score."""

    weights_digest: str
    tokenizer_digest: str
    template_digest: str
    preprocessing_digest: str

    def to_dict(self) -> dict[str, str]:
        return {
            "weights_digest": self.weights_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "template_digest": self.template_digest,
            "preprocessing_digest": self.preprocessing_digest,
        }


@dataclass(frozen=True)
class RewardScore:
    raw_score: float
    transformed_score: float
    direction: ScoreDirection


@dataclass(frozen=True)
class PreferencePair:
    id: str
    prompt: str
    chosen: str
    rejected: str
    preferred: str

    def __post_init__(self) -> None:
        if self.preferred not in {"chosen", "rejected"}:
            raise ValueError("preferred is chosen or rejected")


@dataclass(frozen=True)
class PreferenceResult:
    pair_id: str
    chosen: RewardScore
    rejected: RewardScore
    preferred: str
    passed: bool
    direction: ScoreDirection


@dataclass(frozen=True)
class ProbeCheck:
    name: str
    passed: bool
    evidence: dict[str, Any]


@dataclass(frozen=True)
class RewardModelProbe:
    batching: ProbeCheck
    padding: ProbeCheck
    truncation: ProbeCheck


class UnprovenRewardModelCapability(CapabilityUnavailable):
    """An RM-specific guard for a claim whose probe was not green."""

    def __init__(self, capability: str) -> None:
        super().__init__(
            code="RL0702",
            message=f"the reward-model manifest cannot claim {capability}: its probe did not pass",
            remediation="run the adapter probe and retain its evidence before claiming this capability",
            context={"capability": capability},
        )


def sigmoid(value: float) -> float:
    """A numerically stable, named calibration transform used by the local fixture."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _train_stack() -> Any:
    """Import the scoring runtime only on the adapter construction path.

    The declared ``train`` extra also carries transformers for callers who load Hugging Face model
    classes.  This adapter receives an already loaded model and tokenizer, so importing
    transformers here would add a false runtime requirement and would not establish anything
    about the supplied scoring head.
    """
    try:
        return importlib.import_module("torch")
    except ImportError as caught:
        raise errors.make("RL0704", extra="train", capability="neural_reward_model") from caught


def _utc_now() -> str:
    return _datetime.datetime.now(tz=_datetime.timezone.utc).isoformat(timespec="microseconds")


def _plain(value: Any) -> Any:
    """Turn supported model/tokenizer state into canonical JSON data without importing torch."""
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _weights_digest(model: Any) -> str:
    state = model.state_dict() if callable(getattr(model, "state_dict", None)) else model
    return digest(
        {
            "state_dict": _plain(state),
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
        }
    )


def _tokenizer_material(tokenizer: Any, preprocessing: Preprocessing) -> dict[str, Any]:
    """The state and executed behaviour that determine token IDs, not merely vocabulary names."""
    vocab = tokenizer.get_vocab() if callable(getattr(tokenizer, "get_vocab", None)) else None
    material: dict[str, Any] = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab": _plain(vocab),
        "configured": preprocessing.to_dict(),
    }
    for name in (
        "init_kwargs",
        "special_tokens_map",
        "added_tokens_encoder",
        "chat_template",
        "model_max_length",
        "padding_side",
        "truncation_side",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
    ):
        value = getattr(tokenizer, name, None)
        if value is not None:
            material[name] = _component_state(value)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        material["backend"] = {
            name: _component_state(getattr(backend, name, None))
            for name in ("normalizer", "pre_tokenizer", "post_processor")
            if getattr(backend, name, None) is not None
        }
    # A compact execution witness detects an implementation whose public configuration lies about
    # how it tokenises. It is deliberately token IDs, not decoded text: those are what the model sees.
    try:
        prior_padding, prior_truncation = (
            getattr(tokenizer, "padding_side", None),
            getattr(tokenizer, "truncation_side", None),
        )
        if prior_padding is not None:
            tokenizer.padding_side = preprocessing.padding_side
        if prior_truncation is not None:
            tokenizer.truncation_side = preprocessing.truncation_side
        encoded = tokenizer(
            ["rm-identity alpha beta", "rm-identity beta alpha"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=preprocessing.max_length,
        )
        material["behaviour_witness"] = _plain(dict(encoded))
    finally:
        if "prior_padding" in locals() and prior_padding is not None:
            tokenizer.padding_side = prior_padding
        if "prior_truncation" in locals() and prior_truncation is not None:
            tokenizer.truncation_side = prior_truncation
    return material


def _component_state(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, Mapping, list, tuple)):
        return _plain(value)
    for name in ("to_str", "__getstate__"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                return _plain(method())
            except (TypeError, ValueError):
                pass
    return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "text": str(value)}


def _tokenizer_digest(tokenizer: Any, preprocessing: Preprocessing) -> str:
    return digest(_tokenizer_material(tokenizer, preprocessing))


class RewardModel:
    """A CPU-local scoring-head adapter with identity and observed conformance probes."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        template: str,
        preprocessing: Preprocessing,
        transform: Callable[[float], float],
        direction: ScoreDirection,
        torch: Any,
    ) -> None:
        self.model = model.to("cpu").eval() if callable(getattr(model, "to", None)) else model
        self.tokenizer = tokenizer
        self.template = template
        self.preprocessing = preprocessing
        self.transform = transform
        self.direction = direction
        self._torch = torch
        self.identity = RewardModelIdentity(
            weights_digest=_weights_digest(model),
            tokenizer_digest=_tokenizer_digest(tokenizer, preprocessing),
            template_digest=digest({"template": template}),
            preprocessing_digest=digest(preprocessing.to_dict()),
        )
        self._manifest = Manifest(
            family="NEURAL_REWARD_MODEL",
            implementation_revision="reward-lens/graders/reward_model/1",
            input_schema_digest=digest({"task": {"prompt": "str"}, "response": "str"}),
            output_schema_digest=digest({"raw_score": "float", "transformed_score": "float"}),
            shape="pretrained_model",
            score_domain="transformed: explicit calibration transform; not comparable across revisions",
            score_direction=direction.value,
            determinism_class="declared_deterministic",
            state_model="stateless",
            access_level="source_visible",
            declared_inputs=("prompt", "response"),
            declared_outputs=("raw_score", "transformed_score"),
            resource_needs={"device": "cpu", "threads": 1, "network": False},
            replay_mode="local_replay_unproven",
            capabilities=CapabilityVector(
                probed={
                    "batching_invariant": False,
                    "padding_invariant": False,
                    "truncation_observed": False,
                    "labelled_pairs_scored": False,
                }
            ),
        )
        self._labelled_pair_evidence: tuple[PreferenceResult, ...] = ()

    @classmethod
    def from_pretrained(
        cls,
        model: Any,
        tokenizer: Any,
        *,
        template: str = "{prompt}\\n{response}",
        preprocessing: Preprocessing | None = None,
        transform: Callable[[float], float] | None = None,
        direction: ScoreDirection | str = ScoreDirection.HIGHER_IS_BETTER,
    ) -> "RewardModel":
        torch = _train_stack()
        if "{prompt}" not in template or "{response}" not in template:
            raise ValueError("template must name both {prompt} and {response}")
        parsed_direction = ScoreDirection(direction)
        return cls(
            model,
            tokenizer,
            template=template,
            preprocessing=preprocessing or Preprocessing(),
            transform=transform or (lambda value: value),
            direction=parsed_direction,
            torch=torch,
        )

    def describe(self) -> Manifest:
        return self._manifest

    def source(self) -> None:
        return None

    def reset(self) -> None:
        return None

    def validate_input(self, task: dict, response: str | dict) -> None | InvalidInput:
        if not isinstance(task, Mapping) or not isinstance(task.get("prompt"), str):
            return InvalidInput(
                field="prompt", reason="a reward-model task requires a string prompt"
            )
        if not isinstance(response, str):
            return InvalidInput(field="response", reason="a reward-model response is text")
        return None

    def _render(self, prompt: str, response: str) -> str:
        return self.template.format(prompt=prompt, response=response)

    def _tokenize(
        self,
        texts: Sequence[str],
        *,
        padding: bool,
        truncation: bool = True,
        max_length: int | None = None,
    ) -> Mapping[str, Any]:
        prior_padding = getattr(self.tokenizer, "padding_side", None)
        prior_truncation = getattr(self.tokenizer, "truncation_side", None)
        if prior_padding is not None:
            self.tokenizer.padding_side = self.preprocessing.padding_side
        if prior_truncation is not None:
            self.tokenizer.truncation_side = self.preprocessing.truncation_side
        try:
            arguments = {"return_tensors": "pt", "padding": padding, "truncation": truncation}
            if truncation:
                arguments["max_length"] = (
                    self.preprocessing.max_length if max_length is None else max_length
                )
            encoded = self.tokenizer(list(texts), **arguments)
        finally:
            if prior_padding is not None:
                self.tokenizer.padding_side = prior_padding
            if prior_truncation is not None:
                self.tokenizer.truncation_side = prior_truncation
        return {
            key: value.to("cpu") if callable(getattr(value, "to", None)) else value
            for key, value in dict(encoded).items()
        }

    def _raw_from_encoded(self, encoded: Mapping[str, Any]) -> list[float]:
        with self._torch.inference_mode():
            output = self.model(**encoded)
        logits = getattr(output, "logits", output)
        if getattr(logits, "ndim", 0) == 2:
            logits = logits[:, 0]
        return [float(item) for item in logits.detach().cpu().reshape(-1).tolist()]

    def _raw_scores(
        self, rows: Sequence[tuple[str, str]], *, batch_size: int, padding: bool
    ) -> list[float]:
        values: list[float] = []
        for start in range(0, len(rows), batch_size):
            texts = [
                self._render(prompt, response)
                for prompt, response in rows[start : start + batch_size]
            ]
            encoded = self._tokenize(texts, padding=padding)
            values.extend(self._raw_from_encoded(encoded))
        return values

    def score_item(self, prompt: str, response: str) -> RewardScore:
        raw = self._raw_scores([(prompt, response)], batch_size=1, padding=False)[0]
        return RewardScore(
            raw_score=raw, transformed_score=float(self.transform(raw)), direction=self.direction
        )

    def score_batch(self, rows: Sequence[tuple[str, str]]) -> list[RewardScore]:
        raw = self._raw_scores(rows, batch_size=self.preprocessing.batch_size, padding=True)
        return [RewardScore(value, float(self.transform(value)), self.direction) for value in raw]

    def score(
        self, task: dict, response: str | dict, *, sandbox: Any = None, limits: Any = None
    ) -> EvidenceEnvelope:
        started = _utc_now()
        invalid = self.validate_input(task, response)
        if invalid is not None:
            return self._envelope(
                task,
                response,
                started,
                "invalid_input",
                None,
                errors=(f"{invalid.field}: {invalid.reason}",),
            )
        assert isinstance(response, str)
        value = self.score_item(task["prompt"], response)
        return self._envelope(task, response, started, "scored", value)

    def _envelope(
        self,
        task: Any,
        response: Any,
        started: str,
        verdict: str,
        value: RewardScore | None,
        errors: tuple[str, ...] = (),
    ) -> EvidenceEnvelope:
        observations: dict[str, Any] = {
            "identity": self.identity.to_dict(),
            "direction": self.direction.value,
            "preprocessing": self.preprocessing.to_dict(),
        }
        if value is not None:
            observations.update(
                raw_score=value.raw_score, transformed_score=value.transformed_score
            )
        return EvidenceEnvelope(
            run_id=str(uuid.uuid4()),
            subject_digest=self.identity.weights_digest,
            manifest_digest=self._manifest.digest(),
            request_digest=digest({"task": task, "response": response}),
            start=started,
            end=_utc_now(),
            seed=None,
            verdict=verdict,
            score=None if value is None else value.transformed_score,
            observations=observations,
            counters=Counters(),
            provenance={
                "task": dict(task) if isinstance(task, Mapping) else task,
                "response": response,
            },
            errors=errors,
            limitations=("local CPU model; no provider or model download was used",),
        )

    def score_pairs(self, pairs: Sequence[PreferencePair]) -> list[PreferenceResult]:
        rows: list[tuple[str, str]] = []
        for pair in pairs:
            rows.extend(((pair.prompt, pair.chosen), (pair.prompt, pair.rejected)))
        scores = self.score_batch(rows)
        outcomes: list[PreferenceResult] = []
        for index, pair in enumerate(pairs):
            chosen, rejected = scores[index * 2 : index * 2 + 2]
            chosen_wins = chosen.transformed_score > rejected.transformed_score
            if self.direction is ScoreDirection.LOWER_IS_BETTER:
                chosen_wins = chosen.transformed_score < rejected.transformed_score
            outcomes.append(
                PreferenceResult(
                    pair.id,
                    chosen,
                    rejected,
                    pair.preferred,
                    chosen_wins == (pair.preferred == "chosen"),
                    self.direction,
                )
            )
        self._labelled_pair_evidence = tuple(outcomes)
        self._manifest = replace(
            self._manifest,
            capabilities=CapabilityVector(
                probed={
                    **self._manifest.capabilities.probed,
                    "labelled_pairs_scored": bool(outcomes),
                }
            ),
        )
        return outcomes

    @property
    def labelled_pair_evidence(self) -> tuple[PreferenceResult, ...]:
        """The actual labelled-pair outcomes behind the manifest's calibration capability."""
        return self._labelled_pair_evidence

    def probe(self, rows: Sequence[tuple[str, str]]) -> RewardModelProbe:
        if not rows:
            raise ValueError("the reward-model probe needs at least one prompt/response row")
        single = self._raw_scores(rows, batch_size=1, padding=False)
        batched = self._raw_scores(rows, batch_size=self.preprocessing.batch_size, padding=True)
        batching = ProbeCheck(
            "batching", _same(single, batched), {"single": single, "batched": batched}
        )
        padded = self._raw_scores(rows, batch_size=len(rows), padding=True)
        padding = ProbeCheck(
            "padding", _same(single, padded), {"unpadded": single, "padded": padded}
        )
        text = self._render(*rows[0])
        encoded = self._tokenize([text], padding=False)
        token_count = int(encoded["attention_mask"].sum().item())
        retained_ids = (
            encoded["input_ids"][0][encoded["attention_mask"][0].bool()].detach().cpu().tolist()
        )
        truncation_raw = self._raw_from_encoded(encoded)[0]
        full = self._tokenize([text], padding=False, truncation=False)
        full_mask = full["attention_mask"][0].bool()
        full_ids = full["input_ids"][0][full_mask].detach().cpu().tolist()
        expected_ids = (
            full_ids[-self.preprocessing.max_length :]
            if self.preprocessing.truncation_side == "left"
            else full_ids[: self.preprocessing.max_length]
        )
        full_width = int(full["input_ids"].shape[-1])
        expected_encoded: dict[str, Any] = {}
        for key, value in full.items():
            if getattr(value, "ndim", 0) >= 2 and int(value.shape[-1]) == full_width:
                expected_encoded[key] = (
                    value[:, -len(expected_ids) :]
                    if self.preprocessing.truncation_side == "left"
                    else value[:, : len(expected_ids)]
                )
            else:
                expected_encoded[key] = value
        expected_raw = self._raw_from_encoded(expected_encoded)[0]
        side_matches = retained_ids == expected_ids and math.isclose(
            truncation_raw, expected_raw, rel_tol=0.0, abs_tol=1e-6
        )
        truncation = ProbeCheck(
            "truncation",
            token_count <= self.preprocessing.max_length and bool(retained_ids) and side_matches,
            {
                "token_count": token_count,
                "max_length": self.preprocessing.max_length,
                "configured_side": self.preprocessing.truncation_side,
                "retained_token_ids": retained_ids,
                "raw_score": truncation_raw,
                "full_token_ids": full_ids,
                "expected_retained_token_ids": expected_ids,
                "expected_raw_score": expected_raw,
                "configured_side_matches_full_control": side_matches,
            },
        )
        self._manifest = replace(
            self._manifest,
            capabilities=CapabilityVector(
                probed={
                    **self._manifest.capabilities.probed,
                    "batching_invariant": batching.passed,
                    "padding_invariant": padding.passed,
                    "truncation_observed": truncation.passed,
                }
            ),
        )
        return RewardModelProbe(batching=batching, padding=padding, truncation=truncation)

    def with_capabilities(self, claimed: Mapping[str, bool]) -> Manifest:
        for capability, value in claimed.items():
            if value and self._manifest.capabilities.probed.get(capability) is not True:
                raise UnprovenRewardModelCapability(capability)
        return self._manifest


def _same(left: Sequence[float], right: Sequence[float], tolerance: float = 1e-6) -> bool:
    return len(left) == len(right) and all(
        abs(a - b) <= tolerance for a, b in zip(left, right, strict=True)
    )


def tiny_local_model(token_scores: Mapping[str, float] | None = None) -> tuple[Any, Any]:
    """Build an in-memory CPU reward model and tokenizer for offline conformance tests.

    This helper does not call ``from_pretrained``, download weights, consult a provider, or create
    a GPU model.  Its score is the masked sum of a trainable embedding table, hence is an actual
    neural scoring head rather than a numeric test double.
    """
    torch = _train_stack()
    torch.set_num_threads(1)

    class LocalTokenizer:
        pad_token_id = 0
        padding_side = "right"
        truncation_side = "right"
        normalizer = "identity"

        def __init__(self, vocabulary: Mapping[str, int]) -> None:
            self.vocabulary = dict(vocabulary)

        def get_vocab(self) -> dict[str, int]:
            return dict(self.vocabulary)

        def __call__(
            self,
            texts: Sequence[str],
            *,
            return_tensors: str,
            padding: bool,
            truncation: bool,
            max_length: int | None = None,
        ):
            # A template may contain a literal ``\\n`` from a config file as well as a newline.
            # Treat both as a separator, so the local fixture tests template handling rather than
            # accidentally hiding the response token behind an escape sequence.
            pieces = [text.replace(chr(92) + "n", " ").split() for text in texts]
            if self.normalizer == "reverse":
                pieces = [list(reversed(row)) for row in pieces]
            tokens = [[self.vocabulary.get(piece, 1) for piece in row] or [1] for row in pieces]
            if truncation:
                assert max_length is not None
                tokens = [
                    row[-max_length:] if self.truncation_side == "left" else row[:max_length]
                    for row in tokens
                ]
            width = max(len(row) for row in tokens) if padding else None
            if width is not None:
                tokens = [
                    (
                        [0] * (width - len(row)) + row
                        if self.padding_side == "left"
                        else row + [0] * (width - len(row))
                    )
                    for row in tokens
                ]
            input_ids = torch.tensor(tokens, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": (input_ids != 0).long()}

    scores = dict(token_scores or {})
    vocabulary = {"<unk>": 1, **{token: index + 2 for index, token in enumerate(sorted(scores))}}

    class TinyRewardHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.table = torch.nn.Embedding(len(vocabulary) + 1, 1, padding_idx=0)
            with torch.no_grad():
                self.table.weight.zero_()
                for token, value in scores.items():
                    self.table.weight[vocabulary[token], 0] = float(value)

        def forward(self, input_ids, attention_mask=None):
            values = self.table(input_ids).squeeze(-1)
            if attention_mask is not None:
                values = values * attention_mask
            return type("RewardOutput", (), {"logits": values.sum(dim=1, keepdim=True)})()

    return TinyRewardHead(), LocalTokenizer(vocabulary)
