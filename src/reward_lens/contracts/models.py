"""The assay record as pydantic v2 models (D-66).

`schema/assay/1.0/assay.schema.json` is the specification. These models are an implementation of
it, one class per schema object, strict mode and `extra="forbid"` throughout, and a parity gate
holds the two together. Nothing here is regenerated from the schema and the schema is never
regenerated from this.

Two shapes need explaining. `Entry` stays one class, because that is the name every other packet
imports, so the schema's five per-kind conditionals live in an after-validator and are re-emitted
into the model's own JSON Schema, which is what lets a generator working from the model produce
only entries the frozen schema would accept. `Decision` does the same for its three.

Optional fields that the schema does not allow to be null are declared with a `None` default and a
non-optional annotation: pydantic does not validate defaults, so the field is absent-or-typed and
never null, and the serialiser on `Base` drops it when it holds None.

Serialisation is a function of the record's values and never of how it was built. `to_dict()` is
`model_dump(mode="json", by_alias=True)` through a wrap serialiser that omits exactly the fields
the schema leaves optional when they hold None; a required field is always written, null included,
because the schema requires the key. `exclude_unset` is never used on a digest path: it makes the
bytes depend on the construction path, so an SDK-built record and the same record loaded from disk
would hash differently.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    model_serializer,
    model_validator,
)

from .quantise import JsonInt, quantised

__all__ = [
    "Assay",
    "Subject",
    "Intent",
    "Entry",
    "Witness",
    "Check",
    "Detection",
    "Absence",
    "Uncertainty",
    "NO_INTERVAL",
    "Power",
    "Method",
    "EntryProvenance",
    "Counters",
    "Finding",
    "Rule",
    "Hole",
    "JoinRow",
    "Decision",
    "Experiment",
    "CalibrationLink",
    "Cost",
    "TableRef",
    "Embedding",
    "Attestation",
    "Provenance",
    "Measurement",
    "Producer",
    "Section",
    "Kind",
    "Scope",
    "HoleState",
    "DecisionState",
    "Verdict",
    "SandboxTier",
    "SCHEMA_URL",
    "KIND_EVIDENCE",
    "absence",
    "estimate",
    "model_for_def",
]

SCHEMA_URL = "https://reward-lens.github.io/schema/assay/1.0/assay.schema.json"
ZERO_DIGEST = "sha256:" + "0" * 64

# --- the validated strings ----------------------------------------------------------------------

Section = Literal[
    "validity",
    "soundness",
    "reach",
    "exploits",
    "framing",
    "reward_statistics",
    "signal",
    "trace",
    "forecast",
    "calibration",
]
Kind = Literal["witness", "check", "estimate", "detection", "absence"]
Scope = Literal[
    "evaluator_comparison",
    "selection_stress",
    "reconstructed_training_pressure",
    "controlled_update_response",
    "short_fork_behavior",
    "held_out_transfer",
]
HoleState = Literal["NOT_MEASURED", "COULD_NOT_CHECK", "REFUSED"]
DecisionState = Literal["qualified", "rejected", "unresolved"]
Verdict = Literal[
    "scored",
    "invalid_input",
    "unscored",
    "grader_error",
    "timeout",
    "resource_exhausted",
    "provider_unavailable",
    "parse_error",
    "cancelled",
]
EntryState = Literal["complete", "partial", "absent"]

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Timestamp = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]
Money = Annotated[str, Field(pattern=r"^-?[0-9]+\.[0-9]{2}$")]
Ident = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$")]
EntryId = Annotated[
    str,
    Field(
        pattern=r"^(validity|soundness|reach|exploits|framing|reward_statistics|signal|trace"
        r"|forecast|calibration)\.[a-z0-9_.]+$"
    ),
]
RuleId = Annotated[str, Field(pattern=r"^[a-z_]+\.[a-z0-9_.]+$")]
FindingId = Annotated[str, Field(pattern=r"^(RGX-[0-9]{4}-[0-9]{4}|RGX-local-[0-9]{4})$")]
SandboxTier = Annotated[str, Field(pattern=r"^(T0|L[0-3]|M[01]|W[0-2])$")]
ExtensionKey = Annotated[str, Field(pattern=r"^https?://[^\s]+$")]
EXTENSION_KEY_PATTERN = r"^https?://[^\s]+$"


def bounded_int(**bounds: int) -> Any:
    """An integer whose bound reaches the emitted schema.

    `JsonInt` wraps `int` in a before-validator, and pydantic cannot write `minimum` through one:
    `bounded_int(ge=0)` emits the constraint under its own name and a reader of the
    model's schema never sees the bound. Constraining the `int` first and coercing around it emits
    `minimum` and `maximum`, which is what the frozen schema writes (D-66).
    """
    return Annotated[int, Field(**bounds), *JsonInt.__metadata__]
Text = Annotated[str, Field(min_length=1)]
NonNegativeInt = bounded_int(ge=0)
Extensions = Annotated[
    dict[ExtensionKey, dict[str, Any]],
    # pydantic writes the key pattern as `patternProperties` and leaves every other key admitted,
    # so the emitted schema was weaker than the validation. This is the frozen schema's form.
    WithJsonSchema(
        {
            "type": "object",
            "propertyNames": {"pattern": EXTENSION_KEY_PATTERN},
            "additionalProperties": {"type": "object"},
        }
    ),
]


def schema_rule() -> Any:
    """The correspondence between the frozen assay schema and these models.

    Imported here rather than at module scope because `validate` imports this module: by the time
    a record is serialised, both are loaded. The seeds are the schema's root object and each
    `$def` that a model implements; `bind` walks outward from those through `properties`, `items`
    and `$ref` and reaches the nested objects (`Version`, `Context`, `Digests`, `Decision` and the
    rest) that are neither the root nor a `$def`.
    """
    from .serialise import bind
    from .validate import assay_schema

    schema = assay_schema()
    seeds: dict[type[BaseModel], Any] = {Assay: schema}
    for name, model in MODEL_FOR_DEF.items():
        seeds[model] = schema["$defs"][name]
    return bind(schema, seeds)


def omitted_when_none(cls: type[BaseModel]) -> frozenset[str]:
    """The keys this model omits when they hold None: the ones the schema leaves optional.

    A property the frozen schema does not list under `required` is one where absent and null are
    the same record, so the canonical bytes must not be able to tell them apart. A property it
    does require is written even when it is null. The set is read out of the schema, never out of
    the model's own defaults, and read at call time, so a `required` list changed in memory is a
    serialiser changed in memory.
    """
    return schema_rule().omitted(cls)


class Base(BaseModel):
    """Strict, closed, and serialised by alias so `$schema` and `schema` keep their names."""

    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True)

    @model_serializer(mode="wrap")
    def _drop_absent_optionals(self, handler: Any) -> dict[str, Any]:
        """Serialise by value: optional fields holding None are absent, required ones stay."""
        dumped = handler(self)
        if not isinstance(dumped, dict):
            return dumped
        omit = omitted_when_none(type(self))
        return {k: v for k, v in dumped.items() if v is not None or k not in omit}

    def to_dict(self) -> dict[str, Any]:
        """The record as JSON-ready data: aliases restored, optional-and-None fields absent.

        A pure function of the values. `Assay.model_validate(d).to_dict() == d` for every valid
        instance, so `canonical_bytes()` gives one answer for one record however it was built.
        """
        return self.model_dump(mode="json", by_alias=True)


# --- producer and subject -----------------------------------------------------------------------


class Producer(Base):
    tool: Literal["reward-lens"]
    version: Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+([-.+][0-9A-Za-z.]+)?$")]
    method_set: Digest


class RewardSystemRef(Base):
    id: Ident
    name: Annotated[str, Field(min_length=1, max_length=200)]


DeclaredChange = Literal[
    "scorer",
    "judge",
    "weights",
    "task_mixture",
    "trainer_configuration",
    "outcome_protocol",
    "other",
]


class VersionRef(Base):
    id: Ident
    digest: Digest
    parents: list[Ident]
    declared_change: Annotated[list[DeclaredChange], Field(min_length=1)] | None


class ContextRef(Base):
    policy: Digest | None
    task_set: Digest | None
    configuration: Annotated[str, Field(max_length=200)] | None


class Digests(Base):
    source: Digest | None
    environment: Digest | None
    scorer_config: Digest | None
    task_distribution: Digest | None
    samples: Digest | None
    outcome_protocol: Digest | None
    policy: Digest | None
    training_semantics: Digest | None
    instrument_method: Digest


class Subject(Base):
    reward_system: RewardSystemRef
    version: VersionRef
    context: ContextRef
    digests: Digests


# --- intent ---------------------------------------------------------------------------------------


class NonEquivalentPair(Base):
    a: str
    b: str
    why: str


class OutcomeCheck(Base):
    state: Literal["qualified", "unqualified", "ambiguous"]
    kind: (
        Literal["protected_test_suite", "reference_verifier", "labelled_set", "panel", "other"]
        | None
    )
    provenance: Digest | None
    measured_disagreement: quantised(4, ge=0, le=1) | None


class AttackBudget(Base):
    seeker_calls: NonNegativeInt
    usd: Money


class Partition(Base):
    id: Ident
    kind: Literal["development", "attack_development", "acceptance", "calibration", "transfer"]
    digest: Digest
    access_log: list[str]


class Intent(Base):
    success: Annotated[str, Field(min_length=1, max_length=2000)] | None
    constraints: list[Annotated[str, Field(max_length=500)]]
    non_equivalent_pairs: list[NonEquivalentPair]
    outcome_check: OutcomeCheck
    attack_budget: AttackBudget
    partitions: list[Partition]


# --- the metrology blocks ---------------------------------------------------------------------


class Counters(Base):
    calls: NonNegativeInt = None  # type: ignore[assignment]
    cpu_s: quantised(3, ge=0) = None  # type: ignore[assignment]
    wall_s: quantised(3, ge=0) = None  # type: ignore[assignment]
    peak_rss_bytes: NonNegativeInt = None  # type: ignore[assignment]
    processes: NonNegativeInt = None  # type: ignore[assignment]
    disk_bytes: NonNegativeInt = None  # type: ignore[assignment]
    output_bytes: NonNegativeInt = None  # type: ignore[assignment]
    api_calls: NonNegativeInt = None  # type: ignore[assignment]
    input_tokens: NonNegativeInt = None  # type: ignore[assignment]
    output_tokens: NonNegativeInt = None  # type: ignore[assignment]
    usd: Money = None  # type: ignore[assignment]


Interval = Annotated[list[quantised(6)], Field(min_length=2, max_length=2)]


class UncertaintyComponent(Base):
    source: str
    type: Literal["A", "B"]
    u: quantised(6, ge=0)


#: One method is the stated absence of an interval rather than an interval (D-75, rule 9).
NO_INTERVAL = "no_interval_below_15_clusters"

_UNCERTAINTY_CONDITIONALS = [
    {
        "if": {"required": ["method"], "properties": {"method": {"const": NO_INTERVAL}}},
        "then": {"not": {"required": ["interval"]}, "required": ["reason", "clusters"]},
        "else": {"required": ["interval"]},
    }
]


class Uncertainty(Base):
    """An interval is never unnamed, and never a Wald interval (D-17).

    Below about fifteen clusters rule 9 says to report no interval and say why, so one method is
    the stated absence of an interval: under `no_interval_below_15_clusters` the block carries no
    `interval` and must carry `reason` and `clusters`, and under every other method `interval` is
    required (D-75). Inventing endpoints to fill the field is the failure this refuses. The
    measured value on the estimate is untouched either way; `bounded` is what a policy reads.
    """

    interval: Interval = None  # type: ignore[assignment]
    method: Literal[
        "wilson",
        "clopper_pearson",
        "exact_zero_event",
        "cluster_bootstrap",
        "wild_cluster_bootstrap",
        "paired_difference",
        "task_level_bootstrap",
        "normal_approximation_named",
        "no_interval_below_15_clusters",
    ]
    level: quantised(3, ge=0, le=1)
    reason: Text = None  # type: ignore[assignment]
    standard: quantised(6, ge=0) = None  # type: ignore[assignment]
    expanded: quantised(6, ge=0) = None  # type: ignore[assignment]
    coverage_factor: quantised(3, ge=0) = None  # type: ignore[assignment]
    resamples: NonNegativeInt = None  # type: ignore[assignment]
    cluster_unit: str = None  # type: ignore[assignment]
    clusters: NonNegativeInt = None  # type: ignore[assignment]
    design_effect: quantised(4, ge=0) = None  # type: ignore[assignment]
    icc: quantised(4) = None  # type: ignore[assignment]
    icc_interval: Interval = None  # type: ignore[assignment]
    n_eff: quantised(2, ge=0) = None  # type: ignore[assignment]
    components: list[UncertaintyComponent] = None  # type: ignore[assignment]

    @property
    def bounded(self) -> bool:
        """False when the record states that no interval is available (D-75)."""
        return self.method != NO_INTERVAL

    @model_validator(mode="after")
    def _an_interval_or_the_stated_reason_there_is_none(self) -> Uncertainty:
        if self.method == NO_INTERVAL:
            if self.interval is not None:
                raise ValueError(
                    f"{NO_INTERVAL} states that no interval is available, so the block carries "
                    "none; endpoints invented to fill the field are what this refuses"
                )
            missing = [f for f in ("reason", "clusters") if getattr(self, f) is None]
            if missing:
                raise ValueError(
                    "an uncertainty with no interval says why and states the cluster count that "
                    f"forced it; missing {', '.join(missing)}"
                )
        elif self.interval is None:
            raise ValueError(f"method {self.method} carries the interval it produced")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):  # type: ignore[no-untyped-def]
        schema = handler(core_schema)
        schema = handler.resolve_ref_schema(schema)
        schema["allOf"] = _UNCERTAINTY_CONDITIONALS
        return schema


class Power(Base):
    """What a null result could have detected."""

    alpha: quantised(3)
    power: quantised(3)
    n: JsonInt
    mde: quantised(6)
    design_effect: quantised(4) = None  # type: ignore[assignment]
    equivalence_margin: quantised(6) = None  # type: ignore[assignment]


class Method(Base):
    id: RuleId
    version: str
    params_digest: Digest
    procedure: Text
    credited_to: str = None  # type: ignore[assignment]


class EntryProvenance(Base):
    started: Timestamp
    duration_s: quantised(3, ge=0)
    sandbox_tier: SandboxTier
    offline: bool
    counters: Counters = None  # type: ignore[assignment]
    actor: str = None  # type: ignore[assignment]
    arm: Literal["white_box", "black_box"] = None  # type: ignore[assignment]
    budget_usd: Money = None  # type: ignore[assignment]


# --- the five kinds of evidence -----------------------------------------------------------------


class Witness(Base):
    inputs: dict[str, Any]
    procedure: Text
    observed: Text
    inputs_ref: str = None  # type: ignore[assignment]
    artifact_identity: Digest = None  # type: ignore[assignment]
    path: str = None  # type: ignore[assignment]


class Check(Base):
    predicate: Text
    passed: bool
    scope_tested: Text


class Detection(Base):
    """Limit of blank, limit of detection and limit of quantification are three levels."""

    target: str
    material: Ident | None
    threshold: quantised(6) = None  # type: ignore[assignment]
    LoB: quantised(6) = None  # type: ignore[assignment]
    LoD: quantised(6) = None  # type: ignore[assignment]
    LoQ: quantised(6) = None  # type: ignore[assignment]
    alpha: quantised(3) = None  # type: ignore[assignment]
    beta: quantised(3) = None  # type: ignore[assignment]
    sensitivity: quantised(4) = None  # type: ignore[assignment]
    false_alarm_rate: quantised(4) = None  # type: ignore[assignment]
    arl_in_control: quantised(1) = None  # type: ignore[assignment]
    arl_out_of_control: quantised(1) = None  # type: ignore[assignment]
    unqualified: bool = None  # type: ignore[assignment]
    unqualified_reason: str = None  # type: ignore[assignment]


class Absence(Base):
    state: HoleState
    missing_access: Text
    affected_claims: list[str]
    remedy: Text


DependsOn = Annotated[
    str,
    Field(
        pattern=r"^digest:(source|environment|scorer_config|task_distribution|samples"
        r"|outcome_protocol|policy|training_semantics|instrument_method)$"
    ),
]

#: What each kind must carry, from D-11 and the schema's `allOf`.
KIND_EVIDENCE: dict[str, tuple[str, ...]] = {
    "witness": ("witness", "result"),
    "check": ("check", "result"),
    "estimate": ("value", "unit", "n", "sampling_unit", "uncertainty", "result"),
    "detection": ("detection", "result"),
    "absence": ("absence",),
}

#: A detection that does not declare itself unqualified carries its limits and its error rates.
DETECTION_LIMITS = ("LoB", "LoD", "alpha", "beta", "sensitivity", "false_alarm_rate")

_ENTRY_CONDITIONALS = [
    {
        "if": {"properties": {"kind": {"const": kind}}},
        "then": {
            "required": list(fields),
            "properties": {"state": {"enum": ["complete", "partial"]}},
        },
    }
    for kind, fields in KIND_EVIDENCE.items()
    if kind != "absence"
] + [
    {
        "if": {"properties": {"kind": {"const": "absence"}}},
        "then": {
            "required": ["absence"],
            "not": {"required": ["result"]},
            "properties": {"state": {"const": "absent"}},
        },
    },
    {
        "if": {
            "properties": {
                "kind": {"const": "detection"},
                "detection": {
                    "not": {"properties": {"unqualified": {"const": True}}, "required": ["unqualified"]}
                },
            }
        },
        "then": {"properties": {"detection": {"required": list(DETECTION_LIMITS)}}},
    },
]


class Entry(Base):
    """One measurement, carrying its method, scope, provenance and the evidence its kind calls for."""

    entry_id: EntryId
    section: Section
    kind: Kind
    measurand: Text
    method: Method
    scope: Scope
    subject_ref: Digest
    depends_on: list[DependsOn]
    state: EntryState
    provenance: EntryProvenance
    limitations: list[str]

    result: dict[str, Any] = None  # type: ignore[assignment]
    value: quantised(6) = None  # type: ignore[assignment]
    unit: Text = None  # type: ignore[assignment]
    n: NonNegativeInt = None  # type: ignore[assignment]
    denominator: str = None  # type: ignore[assignment]
    sampling_unit: Text = None  # type: ignore[assignment]
    uncertainty: Uncertainty = None  # type: ignore[assignment]
    power: Power = None  # type: ignore[assignment]
    exclusions: list[str] = None  # type: ignore[assignment]
    assumptions: list[str] = None  # type: ignore[assignment]
    witness: Witness = None  # type: ignore[assignment]
    check: Check = None  # type: ignore[assignment]
    detection: Detection = None  # type: ignore[assignment]
    absence: Absence = None  # type: ignore[assignment]
    verdicts: dict[Verdict, NonNegativeInt] = None  # type: ignore[assignment]
    rate_definition: Literal[
        "false_accept", "false_reject", "false_discovery", "contrast_fraction", "other"
    ] = None  # type: ignore[assignment]
    extensions: Extensions = None  # type: ignore[assignment]

    @model_validator(mode="after")
    def _the_evidence_the_kind_calls_for(self) -> Entry:
        missing = [f for f in KIND_EVIDENCE[self.kind] if getattr(self, f, None) is None]
        if missing:
            raise ValueError(
                f"a {self.kind} entry carries {', '.join(KIND_EVIDENCE[self.kind])}; "
                f"{self.entry_id} is missing {', '.join(missing)}"
            )
        if self.kind == "absence":
            if self.result is not None:
                raise ValueError("an absence is the only kind with no result, and this one has one")
            if self.state != "absent":
                raise ValueError(f"an absence entry has state absent, not {self.state}")
        elif self.state == "absent":
            raise ValueError(f"state absent belongs to an absence entry, not a {self.kind}")
        if self.kind == "detection" and self.detection.unqualified is not True:
            unmet = [f for f in DETECTION_LIMITS if getattr(self.detection, f) is None]
            if unmet:
                raise ValueError(
                    "a qualified detection carries its limits and its error rates; "
                    f"{self.entry_id} is missing {', '.join(unmet)}"
                )
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):  # type: ignore[no-untyped-def]
        schema = handler(core_schema)
        schema = handler.resolve_ref_schema(schema)
        schema["allOf"] = _ENTRY_CONDITIONALS
        return schema


class Measurement(Base):
    validity: list[Entry]
    soundness: list[Entry]
    reach: list[Entry]
    exploits: list[Entry]
    framing: list[Entry]
    reward_statistics: list[Entry]
    signal: list[Entry]
    trace: list[Entry]
    forecast: list[Entry]
    calibration: list[Entry]


# --- findings, holes, the join ------------------------------------------------------------------


class Hole(Base):
    section: Section
    state: HoleState
    missing_access: Text
    affected_claims: list[str]
    remedy: Text
    entry_id: EntryId = None  # type: ignore[assignment]


class Rule(Base):
    id: RuleId
    name: str
    short_description: str
    help: str = None  # type: ignore[assignment]
    help_uri: str = None  # type: ignore[assignment]


class FindingLocation(Base):
    file: str = None  # type: ignore[assignment]
    line: bounded_int(ge=1) = None  # type: ignore[assignment]


class Finding(Base):
    id: FindingId
    rule: RuleId
    level: Literal["none", "note", "warning", "error"]
    kind: Literal["pass", "fail", "open", "review", "informational", "notApplicable"]
    severity_rationale: Text
    scope: Scope
    entries: Annotated[list[EntryId], Field(min_length=1)]
    partial_fingerprint: Annotated[str, Field(min_length=8)]
    message: str = None  # type: ignore[assignment]
    witness_path: str = None  # type: ignore[assignment]
    content_id: Digest = None  # type: ignore[assignment]
    arm: Literal["white_box", "black_box", "both", "static"] = None  # type: ignore[assignment]
    code: Annotated[str, Field(pattern=r"^RL[0-9]{4}$")] = None  # type: ignore[assignment]
    location: FindingLocation = None  # type: ignore[assignment]


class JoinRow(Base):
    family: Text
    selected: Literal["yes", "no", "not_measured"]
    applied_update_weight: quantised(4, ge=0, le=1) | None
    fix_changes_pressure: quantised(4) | None
    fix_cost: Annotated[str, Field(max_length=200)] | None
    missing: str = None  # type: ignore[assignment]


# --- the decision --------------------------------------------------------------------------------


class PolicyRef(Base):
    id: Ident
    digest: Digest


class TradeoffOption(Base):
    option: str
    consequences: list[str]


class Tradeoff(Base):
    competing: Annotated[list[TradeoffOption], Field(min_length=2)]


class Signature(Base):
    backend: Literal["ed25519", "sigstore", "github"]
    statement_digest: Digest
    signer: str = None  # type: ignore[assignment]


DecisionReason = Annotated[
    str,
    Field(
        pattern=r"^(blocking_finding|required_missing|required_stale|required_invalid"
        r"|required_inconclusive|constraint_violated|tradeoff_unsettled|policy_absent"
        r"|outcome_unqualified):[A-Za-z0-9_.:/-]+$"
    ),
]

_DECISION_CONDITIONALS = [
    {
        "if": {"properties": {"state": {"const": "qualified"}}},
        "then": {"properties": {"policy": {"type": "object"}, "action": {"type": "string"}}},
    },
    {
        "if": {"properties": {"state": {"const": "rejected"}}},
        "then": {
            "properties": {
                "policy": {"type": "object"},
                "action": {"type": "string"},
                "reasons": {"minItems": 1},
            }
        },
    },
    {
        "if": {"properties": {"tradeoff": {"type": "object"}}},
        "then": {"properties": {"state": {"const": "unresolved"}}},
    },
]


class Decision(Base):
    """Qualification needs a named action under a named policy; a tradeoff is never a verdict."""

    state: DecisionState
    action: (
        Literal[
            "train_on_this_version",
            "merge_this_grader",
            "use_for_selection",
            "publish_this_environment",
            "other",
        ]
        | None
    )
    policy: PolicyRef | None
    reasons: list[DecisionReason]
    tradeoff: Tradeoff | None
    signature: Signature | None
    regression_cases: list[Ident]

    @model_validator(mode="after")
    def _a_verdict_names_its_action(self) -> Decision:
        if self.state in ("qualified", "rejected"):
            if self.policy is None or self.action is None:
                raise ValueError(
                    f"a {self.state} decision names a policy and an action (D-13)"
                )
        if self.state == "rejected" and not self.reasons:
            raise ValueError("a rejected decision names at least one typed reason (D-13)")
        if self.tradeoff is not None and self.state != "unresolved":
            raise ValueError(
                "a tradeoff is a subtype of unresolved and is never collapsed into a score (D-13)"
            )
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):  # type: ignore[no-untyped-def]
        schema = handler(core_schema)
        schema = handler.resolve_ref_schema(schema)
        schema["allOf"] = _DECISION_CONDITIONALS
        return schema


# --- the tail ------------------------------------------------------------------------------------


class RejectedRung(Base):
    rung: JsonInt
    reason: str
    cost: str = None  # type: ignore[assignment]


class Experiment(Base):
    id: Ident
    rung: bounded_int(ge=1, le=6)
    scope: Scope
    registered_expectation: Text
    result: str | None
    informative: bool | None
    registered_at: Timestamp = None  # type: ignore[assignment]
    candidates: list[Ident] = None  # type: ignore[assignment]
    rejected_rungs: list[RejectedRung] = None  # type: ignore[assignment]


class CalibrationLink(Base):
    material: Ident
    grade: Literal["informational", "reported", "certified", "adopted"]
    transfer_gap: quantised(6, ge=0) | None
    transfer_tolerance: quantised(6, ge=0) = None  # type: ignore[assignment]
    evidence_id: str = None  # type: ignore[assignment]


class Cost(Base):
    wall_s: quantised(3, ge=0)
    cpu_s: quantised(3, ge=0)
    usd: Money
    api_calls: NonNegativeInt
    input_tokens: NonNegativeInt = None  # type: ignore[assignment]
    output_tokens: NonNegativeInt = None  # type: ignore[assignment]
    price_table_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None  # type: ignore[assignment]


class TableRef(Base):
    id: Ident
    digest: Digest
    rows: NonNegativeInt
    table_schema: Annotated[str, Field(pattern=r"^schemas/[A-Za-z0-9_.-]+\.json$")] = Field(
        alias="schema"
    )
    inline_rows: Annotated[
        list[dict[str, Any]], Field(max_length=50, json_schema_extra={"x-max-inline-rows": 50})
    ] = None  # type: ignore[assignment]


class Embedding(Base):
    tier: Literal["A", "B", "C"]
    omitted_tables: list[Ident]
    html_bytes: NonNegativeInt = None  # type: ignore[assignment]
    record_bytes: NonNegativeInt = None  # type: ignore[assignment]


class Attestation(Base):
    statement_digest: Digest | None
    backend: Literal["ed25519", "sigstore", "github"] | None
    manifest_digest: Digest | None = None


class Provenance(Base):
    offline: bool
    sandbox_tier: SandboxTier
    os: Literal["linux", "macos", "windows"]
    seed: JsonInt
    reproduce: list[str]
    run_id: Annotated[str, Field(pattern=r"^run-[0-9a-f]{8,32}$")] = None  # type: ignore[assignment]
    spend_usd: Money = None  # type: ignore[assignment]
    access_level: Literal[
        "source", "callable", "response_bank", "run_record", "trainer_log", "black_box"
    ] = None  # type: ignore[assignment]
    input_access_note: str = None  # type: ignore[assignment]


class Assay(Base):
    """One measurement of one reward-system version in one context (section 6, D-09 to D-18)."""

    schema_url: Literal[SCHEMA_URL] = Field(alias="$schema")
    assay_id: Digest
    created: Timestamp
    producer: Producer
    subject: Subject
    intent: Intent
    measurement: Measurement
    holes: list[Hole]
    findings: list[Finding]
    decision: Decision
    cost: Cost
    embedding: Embedding
    attestation: Attestation = None  # type: ignore[assignment]
    provenance: Provenance
    environment_excluded_from_digest: dict[str, Any]
    rules: list[Rule] = None  # type: ignore[assignment]
    join: list[JoinRow] = None  # type: ignore[assignment]
    experiments: list[Experiment] = None  # type: ignore[assignment]
    calibration_links: list[CalibrationLink] = None  # type: ignore[assignment]
    tables: list[TableRef] = None  # type: ignore[assignment]
    extensions: Extensions = None  # type: ignore[assignment]

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> "Assay":
        """A malformed record is RL0604, not a pydantic traceback.

        `validate_record` already answers in that code at the schema layer; a caller who builds
        the record through the models rather than through the validator is owed the same answer.
        pydantic's `ValidationError` is kept as the cause, so every location is still reachable
        through `__cause__.errors()` where this carries only the first.
        """
        from pydantic import ValidationError

        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as invalid:
            raise _record_invalid(invalid) from invalid

    def entries(self) -> list[Entry]:
        """Every entry, in section order."""
        out: list[Entry] = []
        for section in Measurement.model_fields:
            out.extend(getattr(self.measurement, section))
        return out

    def holes_from_entries(self) -> list[Hole]:
        """Rebuild the `holes` index from every absence entry, and install it (section 6.2)."""
        rebuilt = [
            Hole(
                section=entry.section,
                entry_id=entry.entry_id,
                state=entry.absence.state,
                missing_access=entry.absence.missing_access,
                affected_claims=list(entry.absence.affected_claims),
                remedy=entry.absence.remedy,
            )
            for entry in self.entries()
            if entry.kind == "absence"
        ]
        self.holes = rebuilt
        return rebuilt


#: The version of the absence procedure itself. It moves when the procedure below changes, never
#: with the record's schema.
ABSENCE_METHOD_VERSION = "1.0.0"


def absence_method(
    section: str,
    missing_access: str,
    remedy: str,
    affected_claims: tuple[str, ...] | list[str] = (),
) -> Method:
    """The method of an absence entry: the absence procedure, not a stand-in for a missing one.

    An entry's method says what was done. For an absence the thing that was done is the absence
    procedure: the attempt was made, the access it needed was not there, and the entry records
    which claims that leaves unsupported. So `params_digest` is a digest of those parameters,
    taken through `canonical` like every other digest in this project (D-10), and two absences
    that differ in any of them differ in their digest. A constant there would say that every
    absence was reached the same way and would make the digest evidence of nothing.
    """
    from .canonical import digest

    return Method(
        id=f"{section}.absence",
        version=ABSENCE_METHOD_VERSION,
        params_digest=digest(
            {
                "missing_access": missing_access,
                "remedy": remedy,
                "affected_claims": list(affected_claims),
            }
        ),
        procedure=(
            "the measurement was attempted, the access it needs was not available, and the "
            "entry records the missing access, the remedy that would supply it and the claims "
            "left unsupported"
        ),
    )


def absence(
    section: str,
    entry_id: str,
    measurand: str,
    missing_access: str,
    remedy: str,
    affected_claims: tuple[str, ...] | list[str] = (),
    *,
    subject_ref: str,
    provenance: EntryProvenance,
    depends_on: tuple[str, ...] | list[str] = (),
    state: str = "NOT_MEASURED",
    method: Method | None = None,
    scope: str = "evaluator_comparison",
    limitations: tuple[str, ...] | list[str] = (),
) -> Entry:
    """The honest-absence entry: a hole is a first-class state, never a zero and never an error.

    An absence entry records that a measurement was not made. It still has to say which subject
    the attempt was against and when the attempt happened, and those are facts only the caller
    holds, so `subject_ref` and `provenance` are required keyword arguments with no defaults
    (A-004). The `EntryProvenance` the caller builds carries the real start time, the real
    duration, the sandbox tier that actually held and whether the attempt ran offline. Filling
    them in here with a zero digest, the epoch and T0 would write a start time and a tier that
    never happened, which is the one thing an absence entry exists to avoid.
    """
    return Entry(
        entry_id=entry_id,
        section=section,  # type: ignore[arg-type]
        kind="absence",
        measurand=measurand,
        method=method or absence_method(section, missing_access, remedy, affected_claims),
        scope=scope,  # type: ignore[arg-type]
        subject_ref=subject_ref,
        depends_on=list(depends_on),
        state="absent",
        provenance=provenance,
        limitations=list(limitations),
        absence=Absence(
            state=state,  # type: ignore[arg-type]
            missing_access=missing_access,
            affected_claims=list(affected_claims),
            remedy=remedy,
        ),
    )


def estimate(
    section: str,
    entry_id: str,
    measurand: str,
    value: float,
    unit: str,
    n: int,
    sampling_unit: str,
    uncertainty: Uncertainty,
    *,
    subject_ref: str,
    provenance: EntryProvenance,
    method: Method,
    result: dict[str, Any],
    depends_on: tuple[str, ...] | list[str] = (),
    scope: str = "evaluator_comparison",
    state: str = "complete",
    limitations: tuple[str, ...] | list[str] = (),
    denominator: str | None = None,
    power: Power | None = None,
    exclusions: tuple[str, ...] | list[str] | None = None,
    assumptions: tuple[str, ...] | list[str] | None = None,
    verdicts: dict[str, int] | None = None,
    rate_definition: str | None = None,
    extensions: dict[str, dict[str, Any]] | None = None,
) -> Entry:
    """A measured number, carried with the unit, the count, the sampling unit and the uncertainty.

    Every field a `kind: estimate` entry is required to carry is a parameter here with no default,
    because each one is a fact only the caller holds: the value and its unit, the `n` and the
    sampling unit the `n` counts, the uncertainty, the raw `result` the value was computed from,
    the subject the measurement was against, the provenance of the attempt, and the method.

    The method is the difference from `absence()`. An absence derives its own, because the thing
    that was done is the absence procedure itself. An estimate's method is the instrument's: what
    the instrument computed, at what version, over which parameters. Deriving one here would put
    a procedure string and a `params_digest` into the record that no instrument ever ran, so
    `method` is required and is stored exactly as it arrives. `params_digest` is never recomputed;
    two records that disagree about how a number was reached have to disagree at that field.

    `uncertainty` arrives built, and the `Uncertainty` model is what holds D-75: under
    `no_interval_below_15_clusters` the block carries `reason` and the `clusters` that forced it
    and no interval at all, and `uncertainty.bounded` is False. Endpoints are never invented to
    fill the field, so there is no default uncertainty and no fallback interval here either.

    The optional fields are keywords defaulting to `None`, and a `None` is not passed on to the
    model: an optional field the schema will not admit as null is declared on `Entry` with a
    `None` default and a non-optional annotation, so the field is absent-or-typed, and the one
    serialisation rule then writes it as absent. An estimate with no `power` analysis says nothing
    about power rather than claiming an empty one.
    """
    optional: dict[str, Any] = {
        "denominator": denominator,
        "power": power,
        "exclusions": None if exclusions is None else list(exclusions),
        "assumptions": None if assumptions is None else list(assumptions),
        "verdicts": verdicts,
        "rate_definition": rate_definition,
        "extensions": extensions,
    }
    return Entry(
        entry_id=entry_id,
        section=section,  # type: ignore[arg-type]
        kind="estimate",
        measurand=measurand,
        method=method,
        scope=scope,  # type: ignore[arg-type]
        subject_ref=subject_ref,
        depends_on=list(depends_on),
        state=state,  # type: ignore[arg-type]
        provenance=provenance,
        limitations=list(limitations),
        result=result,
        value=value,  # type: ignore[arg-type]
        unit=unit,
        n=n,
        sampling_unit=sampling_unit,
        uncertainty=uncertainty,
        **{name: supplied for name, supplied in optional.items() if supplied is not None},
    )


def _record_invalid(invalid: Any) -> Exception:
    """pydantic's answer to a malformed record, in the code the record layer reserves for it."""
    from .errors import RecordInvalid

    first = invalid.errors()[0] if invalid.errors() else {"loc": (), "msg": str(invalid)}
    path = "/" + "/".join(str(part) for part in first["loc"])
    return RecordInvalid(
        message=f"the record does not validate against the frozen schema at {path}: {first['msg']}",
        remediation=(
            "the schema at schema/assay/1.0/assay.schema.json is the specification; the invalid "
            "fixtures beside it name each rule and the layer that catches it"
        ),
        context={"instance_path": path, "error_count": len(invalid.errors())},
    )


#: The `$defs` name in the frozen schema -> the model that implements it, for the parity gate.
MODEL_FOR_DEF: dict[str, type[Base]] = {
    "entry": Entry,
    "finding": Finding,
    "uncertainty": Uncertainty,
    "power": Power,
    "method": Method,
    "entry_provenance": EntryProvenance,
    "counters": Counters,
}


def model_for_def(name: str) -> type[Base]:
    return MODEL_FOR_DEF[name]
