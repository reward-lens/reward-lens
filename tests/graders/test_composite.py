"""Contract tests for the record-preserving composite grader."""

from __future__ import annotations

import json

import pytest

from reward_lens.contracts import Counters, digest
from reward_lens.examples.connect_fixture import shapes
from reward_lens.graders.base import (
    AdapterCapabilityUnproven,
    EvidenceEnvelope,
    Limits,
    Manifest,
)
from reward_lens.graders.composite import Component, CompositeGrader, reconstruct
from reward_lens.graders.connect import bind, detect


class Child:
    """A deterministic child grader whose envelope makes forwarding observable."""

    def __init__(self, name: str, score: float | None, *, verdict: str = "scored") -> None:
        self.name = name
        self.value = score
        self.verdict = verdict
        self.calls: list[tuple[object, object]] = []
        self._manifest = Manifest(
            family="fixture",
            implementation_revision="test/1",
            input_schema_digest=digest({"task": "fixture"}),
            output_schema_digest=digest({"score": "float"}),
            source=f"fixture:{name}",
        )

    def describe(self) -> Manifest:
        return self._manifest

    def validate_input(self, task: dict, response: str | dict):  # noqa: ANN201
        return None

    def source(self) -> str | None:
        return None

    def reset(self) -> None:
        return None

    def score(self, task: dict, response: str | dict, *, sandbox, limits) -> EvidenceEnvelope:  # noqa: ANN001
        self.calls.append((sandbox, limits))
        return EvidenceEnvelope(
            run_id=f"run-{self.name}",
            subject_digest=digest({"child": self.name}),
            manifest_digest=self._manifest.digest(),
            request_digest=digest({"task": task, "response": response}),
            start="2026-09-26T00:00:00Z",
            end="2026-09-26T00:00:01Z",
            seed=None,
            verdict=self.verdict,
            score=self.value,
            counters=Counters(),
        )


def composite(
    *components: Component, final_clip: tuple[float, float] | None = None
) -> CompositeGrader:
    return CompositeGrader(
        components=components,
        final_clip=final_clip,
        probed={"component_dag": True, "returns_components": True},
    )


def _mutable_serialized(envelope: EvidenceEnvelope) -> dict:
    """A serialized external record, deliberately separate from issued public evidence."""
    return json.loads(json.dumps(envelope.to_dict()))


def _attest_serialized(data: dict) -> None:
    composite_record = data["observations"]["composite"]
    composite_record["integrity_digest"] = digest(
        {
            "declared": composite_record["declared"],
            "actual": composite_record["actual"],
            "envelope_score": data["score"],
            "verdict": data["verdict"],
            "raw_components": data["components"],
        }
    )


def test_ordered_dag_records_raw_values_gates_normalisation_weights_and_clip() -> None:
    quality = Child("quality", 2.0)
    penalty = Child("penalty", 3.0)
    skipped = Child("skipped", 99.0)
    grader = composite(
        Component("quality", quality, weight=0.5, normalisation="clamp_0_1", clip=(0.0, 1.0)),
        Component("penalty", penalty, weight=-0.5, depends_on=("quality",)),
        Component(
            "skipped", skipped, weight=1.0, gate=lambda values: values["quality"].raw_score == 2.0
        ),
    )
    sandbox, limits = object(), Limits(cpu_s=1)

    envelope = grader.score({"prompt": "p"}, "r", sandbox=sandbox, limits=limits)

    record = envelope.observations["composite"]
    assert [row["name"] for row in record["actual"]["components"]] == [
        "quality",
        "penalty",
        "skipped",
    ]
    assert envelope.components == {"quality": 2.0, "penalty": 3.0, "skipped": 99.0}
    assert record["actual"]["components"][0]["normalised_score"] == 1.0
    assert record["actual"]["components"][0]["weighted_score"] == 0.5
    assert record["actual"]["components"][1]["weighted_score"] == -1.5
    assert record["actual"]["components"][2]["gate_passed"] is True
    assert record["actual"]["pre_clip"] == 98.0
    assert envelope.score == 98.0
    assert quality.calls == [(sandbox, limits)]
    assert penalty.calls == [(sandbox, limits)]
    assert skipped.calls == [(sandbox, limits)]


def test_false_gate_is_recorded_without_calling_the_component() -> None:
    child = Child("blocked", 5.0)
    envelope = composite(Component("blocked", child, gate=lambda _values: False)).score(
        {"prompt": "p"}, "r", sandbox=object(), limits=Limits()
    )

    row = envelope.observations["composite"]["actual"]["components"][0]
    assert child.calls == []
    assert row["gate_passed"] is False
    assert row["raw_score"] is None
    assert row["missing_action"] == "excluded_by_gate"
    assert envelope.score == 0.0


def test_declared_missing_zero_is_visible_and_undeclared_missing_refuses() -> None:
    missing = Child("missing", None, verdict="unscored")
    zeroed = composite(Component("missing", missing, missing="zero"))
    accepted = zeroed.score({"prompt": "p"}, "r", sandbox=object(), limits=Limits())
    row = accepted.observations["composite"]["actual"]["components"][0]
    assert accepted.verdict == "scored"
    assert accepted.score == 0.0
    assert row["raw_score"] is None
    assert row["missing_action"] == "zero"
    assert row["weighted_score"] == 0.0

    with pytest.raises(ValueError, match="missing"):
        Component("missing", missing, missing=None)

    refused = composite(Component("missing", missing, missing="refuse")).score(
        {"prompt": "p"}, "r", sandbox=object(), limits=Limits()
    )
    assert refused.verdict == "unscored"
    assert refused.score is None
    assert refused.observations["composite"]["actual"]["aggregate"] is None
    assert "missing" in refused.errors[0]

    excluded = composite(Component("missing", missing, missing="exclude")).score(
        {"prompt": "p"}, "r", sandbox=object(), limits=Limits()
    )
    assert excluded.verdict == "unscored"
    assert excluded.score is None
    assert excluded.observations["composite"]["actual"]["pre_clip"] is None
    assert excluded.observations["composite"]["actual"]["aggregate"] is None
    assert excluded.observations["composite"]["actual"]["components"][0]["raw_score"] is None


def test_accepted_connector_identity_capabilities_and_evidence_are_component_records() -> None:
    connected = bind(detect(shapes.single_fn))
    grader = composite(Component("connected", connected, weight=2.0))
    envelope = grader.score({"prompt": "p"}, "r", sandbox=object(), limits=Limits(cpu_s=1))

    row = envelope.observations["composite"]["actual"]["components"][0]
    declared = envelope.observations["composite"]["declared"]["components"][0]
    assert row["identity"]["manifest_digest"] == connected.describe().digest()
    assert row["evidence"]["manifest_digest"] == connected.describe().digest()
    assert declared["child_capabilities"] == connected.describe().capabilities.to_dict()
    assert envelope.components["connected"] == row["raw_score"]
    assert grader.describe().family == "COMPOSITE"
    assert grader.source() is None


def test_reconstructs_agreeing_record_and_reports_mismatch_without_adjustment() -> None:
    envelope = composite(Component("a", Child("a", 2.0), weight=3.0), final_clip=(0.0, 5.0)).score(
        {"prompt": "p"}, "r", sandbox=object(), limits=Limits()
    )
    agreeing = reconstruct(envelope)
    assert agreeing.matches is True
    assert agreeing.recorded_aggregate == agreeing.reconstructed_aggregate == 5.0
    assert agreeing.finding is None

    serialized = _mutable_serialized(envelope)
    serialized["score"] = 4.0
    serialized["observations"]["composite"]["actual"]["aggregate"] = 4.0
    _attest_serialized(serialized)
    forged = EvidenceEnvelope.from_dict(serialized)
    disagreeing = reconstruct(forged)
    assert disagreeing.matches is False
    assert disagreeing.recorded_aggregate == 4.0
    assert disagreeing.reconstructed_aggregate == 5.0
    assert disagreeing.finding["kind"] == "composite_reconstruction_mismatch"
    assert disagreeing.finding["recorded_aggregate"] == 4.0
    assert disagreeing.finding["reconstructed_aggregate"] == 5.0
    assert forged.score == 4.0


def test_issued_actual_aggregate_refuses_public_mutation() -> None:
    envelope = composite(Component("a", Child("a", 2.0), weight=3.0)).score(
        {"prompt": "p"}, "r", sandbox=object(), limits=Limits()
    )
    with pytest.raises(TypeError):
        envelope.observations["composite"]["actual"]["aggregate"] = 999.0

    check = reconstruct(envelope)

    assert check.matches is True
    assert check.recorded_aggregate == 6.0


def test_issued_declared_refuse_hole_refuses_public_zero_mutation() -> None:
    envelope = composite(Component("missing", Child("missing", None, verdict="unscored"))).score(
        {"prompt": "p"}, "r", sandbox=object(), limits=Limits()
    )
    with pytest.raises(TypeError):
        envelope.observations["composite"]["actual"]["components"][0]["missing_action"] = "zero"

    check = reconstruct(envelope)

    assert check.matches is True
    assert check.recorded_aggregate is None


def test_reconstruct_replays_declared_refuse_policy_not_a_recorded_zero_action() -> None:
    envelope = composite(Component("missing", Child("missing", None, verdict="unscored"))).score(
        {"prompt": "p"}, "r", sandbox=object(), limits=Limits()
    )
    serialized = _mutable_serialized(envelope)
    serialized["observations"]["composite"]["actual"]["components"][0]["missing_action"] = "zero"
    serialized["observations"]["composite"]["actual"]["aggregate"] = 0.0
    serialized["score"] = 0.0
    serialized["verdict"] = "scored"
    _attest_serialized(serialized)
    forged = EvidenceEnvelope.from_dict(serialized)

    check = reconstruct(forged)

    assert check.matches is False
    assert check.recorded_aggregate == 0.0
    assert check.reconstructed_aggregate is None
    assert check.finding["kind"] == "composite_reconstruction_mismatch"


def test_issued_evidence_refuses_every_consistent_forgery_write() -> None:
    envelope = composite(Component("a", Child("a", 2.0), weight=3.0)).score(
        {"prompt": "p"}, "r", sandbox=object(), limits=Limits()
    )
    actual = envelope.observations["composite"]["actual"]

    with pytest.raises(TypeError):
        actual["components"][0]["raw_score"] = 99.0
    with pytest.raises(TypeError):
        envelope.components["a"] = 99.0
    with pytest.raises(TypeError):
        actual["pre_clip"] = 297.0
    with pytest.raises(TypeError):
        actual["aggregate"] = 297.0
    with pytest.raises(TypeError):
        envelope.observations["composite"]["integrity_digest"] = "re-attested"
    with pytest.raises(TypeError):
        dict.__setitem__(actual, "aggregate", 297.0)
    with pytest.raises(TypeError):
        dict.__ior__(envelope.components, {"a": 99.0})
    with pytest.raises(TypeError):
        dict.__setitem__(envelope.observations["composite"], "integrity_digest", "re-attested")

    assert reconstruct(envelope).matches is True


def test_manifest_cannot_claim_component_dag_without_an_observed_probe() -> None:
    with pytest.raises(AdapterCapabilityUnproven) as caught:
        CompositeGrader(components=(Component("a", Child("a", 1.0)),), probed={})
    assert caught.value.code == "RL0702"
    assert caught.value.exit_code == 5
    assert caught.value.context["capability"] == "component_dag"


def test_rejects_non_topological_or_cyclic_component_dag() -> None:
    child = Child("a", 1.0)
    with pytest.raises(ValueError, match="after"):
        composite(Component("second", child, depends_on=("first",)), Component("first", child))
    with pytest.raises(ValueError, match="depend"):
        composite(Component("a", child, depends_on=("missing",)))
