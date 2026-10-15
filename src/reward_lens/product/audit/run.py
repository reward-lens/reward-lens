"""The audit spine: ten panels, and every one of them either runs or writes its absence.

Section 7.3 and D-41. The default path is a local grader, no key, no network, no paid call. The
runner discovers its built-in and extension panels once, carries their executed entries, and writes
an explicit absence for every protocol it cannot qualify. `holes` is rebuilt from those entries so
the index cannot drift from them.

The one forbidden thing is a panel that prints a plausible number it did not compute. Every number
in the record comes back from an instrument; every instrument that refused is a hole carrying the
refusal's own words; and every exception from loading or calling a user's grader is a
`COULD_NOT_CHECK` hole naming it, never a zero and never `inconclusive`.
"""

from __future__ import annotations

import hashlib
import http.client
import inspect
import json
import os
import platform
import re
import sys
import time
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from reward_lens import contracts
from reward_lens.instruments.base import (
    DetectionReferenceBinding,
    RunContext,
    TraceSourceBinding,
    utc_now,
)
from reward_lens.instruments.seeker.budget import BudgetExhausted
from reward_lens.store import RunDir, runs_root
from reward_lens.store.errors import RunNotFound
from reward_lens.verifier import VerifierUnderTest

from . import absences, instruments, registry, static_checks
from .plan import SECTIONS as _SECTIONS
from .plan import estimate_s as _estimate_s
from .plan import panels as _panels

__all__ = ["Reused", "active_run", "last_reuse", "persist_assay_bundle", "run"]

#: The finding id each code is recorded under. `RGX-local-0001` is the writable-then-read finding.
FINDING_IDS: dict[str, str] = {
    "RL0201": "RGX-local-0001",
    "RL0202": "RGX-local-0002",
    "RL0203": "RGX-local-0003",
    "RL0210": "RGX-local-0010",
    "RL0211": "RGX-local-0011",
    "RL0212": "RGX-local-0012",
    "RL0301": "RGX-local-0301",
}

#: The finding codes wave-2 interfaces section 6 allocates to a panel. A discovered panel raises its
#: own findings and the runner carries them; what the runner will not do is carry a code the
#: catalogue does not own under a panel's name, because `explain <code>` would then have nothing to
#: say about a code a reader just read off the record. A finding that declares no code is not
#: checked here: an id in the local band (P-REACH's `RGX-local-12xx`) is a name, not a catalogue
#: claim, and section 6 allocates codes, not ids.
PANEL_FINDING_CODES: frozenset[str] = frozenset(
    ["RL0213", "RL0214", "RL0215"]
    + [f"RL02{number}" for number in range(21, 30)]
    # The framing package is the sole source that names this code. The runner only includes its
    # allocated numeric slot when checking a panel's reported findings.
    + [f"RL0{23}{1}"]
    + [f"RL02{number}" for number in range(41, 46)]
    + ["RL0251", "RL0261"]
)


#: The reward system's name when nothing survives the cleaning. Stated once, not invented at the
#: call site, so every path that has to produce a name lands on the same string.
DEFAULT_SYSTEM_NAME = "reward-system"

#: The sealed record extension for the configured local OpenEnv route.  It carries binding identity
#: and the one executed step's redacted provenance; neither the raw task nor response is retained.
OPENENV_EXTENSION = "https://reward-lens.dev/extensions/openenv"

#: The bundle manifest's file name. It is written beside the record it names, and every path in it
#: is relative to it, so the digest the page carries survives the bundle being copied or moved.
MANIFEST_NAME = "manifest.json"

#: How many times `_write` may seal the record while the manifest digest settles. Three is what
#: the settling takes: no digest, then the digest that produced, then the pass that repeats it.
#: The rest is slack, and running out of it is a failure rather than a fallback.
MANIFEST_PASSES = 6

#: How much of the subject version's digest goes into a record's name. Eight hex is what tells two
#: versions of one subject apart on a day; it is a label on the file, not the identity, which stays
#: the whole digest inside the record.
NAME_DIGEST_CHARS = 8


def _source_digest(path: Path) -> str:
    """Digest source bytes, including a run directory's relative filenames.

    A path is not an identity: copied or edited training evidence may retain its old name.  The
    deterministic directory form deliberately includes filenames as well as bytes, because a
    reader can select meaning from a particular name such as ``results.jsonl``.
    """
    if path.is_file():
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _trace_declarations(request: Any) -> dict[str, Any]:
    """The complete caller declaration passed only to a TRL completions reader."""
    return {
        "reward_functions": tuple(getattr(request, "reward_functions", ()) or ()),
        "extra_columns": tuple(getattr(request, "extra_columns", ()) or ()),
        "weights": dict(getattr(request, "weight_map", {}) or {}),
        "group_column": getattr(request, "group_column", None),
        "group_size": getattr(request, "group_size", None),
    }


def _read_trace(path: Path, *, declarations: dict[str, Any]) -> Any:
    """Open one trace by the public reader contract, once and without panel-specific loaders."""
    from reward_lens.product.trace import RecordIncomplete, reader_for

    if not path.exists():
        raise RecordIncomplete(
            message=f"the declared training trace {path} does not exist",
            remediation="pass --run-dir for a readable supported training artifact",
            context={"source": str(path)},
        )
    reader = reader_for(path)
    if reader.name != "trl_completions":
        return reader.read(path, policy="strict")
    if not declarations["reward_functions"]:
        raise RecordIncomplete(
            message=(
                "the TRL completions parquet has no declared reward-function column set, "
                "so a flat table cannot establish which values are native reward components"
            ),
            remediation=(
                "pass --reward-function once for every reward-function column and "
                "--extra-column for every remaining non-reserved column"
            ),
            context={"series": ["reward_recorded"], "source": str(path)},
        )
    return reader.read(path, policy="strict", **declarations)


def _reference_trace(item: Any, *, source: Path, partition: str) -> tuple[Any, TraceSourceBinding]:
    """Read one named reference trace with declarations carried by that reference item."""
    from reward_lens.product.trace import RecordIncomplete

    if not isinstance(item, dict) or not isinstance(item.get("run_dir"), str):
        raise RecordIncomplete(
            message="each detector reference partition item needs a run_dir",
            remediation="name a readable run_dir for every calibration and assessment item",
            context={"source": str(source)},
        )
    candidate = Path(item["run_dir"])
    if not candidate.is_absolute():
        candidate = source.parent / candidate
    declarations = {
        "reward_functions": tuple(item.get("reward_functions", ()) or ()),
        "extra_columns": tuple(item.get("extra_columns", ()) or ()),
        "weights": dict(item.get("weights", {}) or {}),
        "group_column": item.get("group_column"),
        "group_size": item.get("group_size"),
    }
    trace = _read_trace(candidate, declarations=declarations)
    return trace, TraceSourceBinding(
        partition=partition,
        source=str(candidate),
        reader=trace.reader,
        run_id=trace.run_id,
        digest=_source_digest(candidate),
        declarations=declarations,
    )


def _open_detection_reference(path: Path) -> DetectionReferenceBinding:
    """Load the separately declared detector reference bundle before panel dispatch."""
    from reward_lens.instruments.detection import DetectorSpec, EffectReference, ReferenceMaterial
    from reward_lens.product.trace import RecordIncomplete

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as failure:
        raise RecordIncomplete(
            message=f"the detector reference {path} is not readable JSON",
            remediation="supply a registered detector reference JSON bundle",
            context={"source": str(path)},
        ) from failure
    if not isinstance(document, dict):
        raise RecordIncomplete(
            message="the detector reference must be a JSON object",
            remediation="supply detector settings and all four named reference partitions",
            context={"source": str(path)},
        )

    identities: list[TraceSourceBinding] = []

    def traces(name: str) -> tuple[Any, ...]:
        values = document.get(name)
        if not isinstance(values, list) or not values:
            raise RecordIncomplete(
                message=f"the detector reference has no {name} partition",
                remediation="supply non-empty calibration_controls, calibration_effects, assessment_controls, and assessment_effects",
                context={"source": str(path), "partition": name},
            )
        opened = tuple(_reference_trace(item, source=path, partition=name) for item in values)
        identities.extend(identity for _, identity in opened)
        return tuple(trace for trace, _ in opened)

    def effects(name: str) -> tuple[Any, ...]:
        values = document.get(name)
        if not isinstance(values, list) or not values:
            raise RecordIncomplete(
                message=f"the detector reference has no {name} partition",
                remediation="supply non-empty calibration_controls, calibration_effects, assessment_controls, and assessment_effects",
                context={"source": str(path), "partition": name},
            )
        result = []
        for item in values:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("onset_index"), int)
                or not isinstance(item.get("level"), (int, float))
            ):
                raise RecordIncomplete(
                    message=f"each {name} item needs an integer onset_index and finite numeric level",
                    remediation="record the effect onset and level for every effect reference trace",
                    context={"source": str(path), "partition": name},
                )
            trace, identity = _reference_trace(item, source=path, partition=name)
            identities.append(identity)
            result.append(EffectReference(trace, item["onset_index"], float(item["level"])))
        return tuple(result)

    try:
        material = ReferenceMaterial(
            str(document["identifier"]),
            calibration_controls=traces("calibration_controls"),
            calibration_effects=effects("calibration_effects"),
            assessment_controls=traces("assessment_controls"),
            assessment_effects=effects("assessment_effects"),
        )
        spec = DetectorSpec(
            series=str(document["series"]),
            registered_arl=float(document["registered_arl"]),
            effect_size=float(document["effect_size"]),
            alpha=float(document["alpha"]),
            beta=float(document["beta"]),
            reference_material=material,
        )
    except (KeyError, TypeError, ValueError) as failure:
        raise RecordIncomplete(
            message=f"the detector reference has invalid registered settings: {failure}",
            remediation="supply identifier, series, registered_arl, effect_size, alpha, beta, and four disjoint partitions",
            context={"source": str(path)},
        ) from failure
    return DetectionReferenceBinding(
        spec=spec, source=str(path), digest=_source_digest(path), trace_sources=tuple(identities)
    )


@dataclass(frozen=True)
class Reused:
    """The run that measured nothing: which stored record answered, and what made it the answer.

    This is the run's provenance rather than the record's. Nothing inside the sealed record says a
    word about reuse, because the record returned is the one already on disk, byte for byte, and a
    measurement that says which of the runs that handed it over is talking is not the same
    measurement. So the sentence lives beside the record, on the run, and `last_reuse()` is where a
    caller reads it. Section 5.7's `ResultEnvelope.execution` is where it belongs on the way out,
    and `contracts`, `api` and `cli` are frozen for this packet; the handoff proposes the field.
    """

    #: The stored record's name, which is also the stem of the report and the record file.
    name: str
    #: The record that answered. Unchanged: reuse never writes a new id.
    assay_id: str
    #: The subject version both the store and the project now agree on.
    subject_version: str
    #: The record file on disk, as a string so the report is JSON without a converter.
    record_path: str
    #: True when the bundle had lost its report and this run composed it again from the record.
    rendered_report: bool
    #: True when the bundle had lost its manifest and this run wrote it again from the record.
    wrote_manifest: bool
    #: The one sentence a surface can print without knowing any of the above.
    says: str


#: Set by every `run`: a `Reused` when the audit returned a stored record, `None` when it measured.
#: A context variable rather than a module global so a second audit on another thread or task
#: cannot answer for this one; a thread starts on an empty context, so its `set` stays its own.
_REUSE: ContextVar[Reused | None] = ContextVar("reward_lens_audit_reuse", default=None)
_ACTIVE_RUN: ContextVar[RunDir | None] = ContextVar("reward_lens_audit_active_run", default=None)


def last_reuse() -> Reused | None:
    """What the last `run` in this context did instead of measuring, or `None` if it measured."""
    return _REUSE.get()


def active_run() -> RunDir | None:
    """The durable audit run currently executing in this context, if one reached creation."""
    return _ACTIVE_RUN.get()


def method_identity(method: Any) -> str:
    """What tells one measurement method from another: its rule id and its parameter digest.

    The pair, not either half. The id alone says which rule was applied and not how: two audits
    that ran `reach.oversight` with different probe budgets both write `id="reach.oversight"`. The
    parameter digest alone is not a name, and `contracts.digest` of an empty parameter set is the
    same eight-hundred-times-over digest for every method that takes no parameters. `version` is
    deliberately not in here: it is the method's own release string, and an instrument that fixes a
    typo in its procedure text has not measured anything differently.
    """
    return f"{getattr(method, 'id', '')}@{getattr(method, 'params_digest', '')}"


def _openenv_params(project: Any) -> dict[str, Any] | None:
    """The typed binding identity a configured OpenEnv step actually uses."""
    if project is None or getattr(project.config.reward, "kind", None) != "openenv":
        return None
    binding = getattr(project.config.reward, "openenv", None)
    if binding is None:
        return None
    return {"entry": project.config.reward.entry, "binding": binding.model_dump(mode="json")}


_DYNAMIC_SOUNDNESS_IDS = frozenset(
    {
        "soundness.batteries",
        "soundness.mutation",
        "soundness.relations",
        "soundness.exploit_premium",
    }
)


def _dynamic_soundness(instrument: Any) -> bool:
    return (
        type(instrument).__module__ == "reward_lens.instruments.soundness"
        and getattr(instrument, "id", None) in _DYNAMIC_SOUNDNESS_IDS
    )


def _soundness_versions_answer(previous: contracts.Assay, discovered: Iterable[Any]) -> bool:
    """A case-level entry cannot answer a newer built-in procedure just by sharing its prefix."""
    for panel in discovered:
        for instrument in getattr(panel, "instruments", ()):
            if not _dynamic_soundness(instrument):
                continue
            prefix = str(instrument.id) + "."
            actual = [
                entry
                for entry in previous.entries()
                if (entry.entry_id == instrument.id or entry.entry_id.startswith(prefix))
                and not _unbuilt_placeholder(entry)
            ]
            if not actual or any(
                entry.method.version != instrument.method.version for entry in actual
            ):
                return False
    return True


def _requested(
    discovered: Iterable[Any],
    project: Any = None,
    *,
    trace_record: Any | None = None,
    include_builtin_checks: bool = False,
) -> dict[str, str | None]:
    """Entry id -> the method identity the plan would write under it, over the discovered panels.

    This is the request's half of section 6.1's question, and it has to be answerable before any
    instrument runs, so it is read off what the instruments declare rather than off what they
    return. The entry id is the one the runner itself uses at the call site: `instrument.id`, which
    is what `guarded_many` labels the instrument's entries and its absence with. The identity is
    `method_identity` of the `method` an instrument declares up front, or `None` where it declares
    none, meaning the plan can say which instrument will run but not with what parameters, and the
    check falls back to coverage for that one.

    Full audit callers set ``include_builtin_checks`` to also require the built-in
    validity controls. A panel-only request does not implicitly request those controls.

    Panel names are deliberately not the key. A panel is a section of the record and a section can
    hold many methods, so keying on the section answers "has a panel landed" and cannot answer "has
    a method landed inside a panel that was already there" -- the case the principal's reviewer
    reproduced, where a second instrument registered in an already-populated `reach` was handed the
    old assay and nothing ran.
    """
    wanted: dict[str, str | None] = {}
    for panel in discovered:
        section = str(getattr(panel, "section", ""))
        for instrument in getattr(panel, "instruments", ()):
            if _dynamic_soundness(instrument):
                # These four built-ins derive their method parameters and, for Relations, their
                # entry ids from the protected cases. Their static declaration is a dispatch
                # identity, not the identity of the retained case-level evidence. Project
                # dependencies and the stored version bind those cases; reuse asks whether the
                # built-in actually left non-placeholder evidence under its instrument prefix.
                wanted[str(instrument.id)] = None
                continue
            response_method = getattr(instrument, "response_bank_method", None)
            declared = (
                response_method
                if response_method is not None and trace_record is None
                else getattr(instrument, "method", None)
            )
            entry_id = str(
                getattr(declared, "id", None) or getattr(instrument, "id", f"{section}.instrument")
            )
            wanted[entry_id] = None if declared is None else method_identity(declared)
    # These built-in controls now execute. An older reserved "not in this build" absence under
    # the same id does not answer the current request, even though no plugin declares the method.
    if include_builtin_checks:
        for suffix, *_ in (*absences.DYNAMIC_CHECKS, *absences.TASK_SET_CHECKS):
            wanted[f"validity.{suffix}"] = None
    openenv_params = _openenv_params(project)
    if openenv_params is not None:
        wanted["validity.openenv_step"] = "validity.openenv_step@" + contracts.digest(
            openenv_params
        )
    return wanted


# The public Audit route writes these deterministic method identities even when a panel's answer
# is an absence. They are the method-set version of the record that a declared supervisor must
# bind before it runs. Deriving them from an assembled record is too late to prevent substitution.
_PUBLIC_AUDIT_METHODS: tuple[str, ...] = (
    "calibration.absence@1.0.0",
    "exploits.seeker@1.0.0",
    "forecast.absence@1.0.0",
    "framing.belief_assay@1.0.0",
    "framing.context_inference@1.0.0",
    "outcome.protected_suite@1.0.0",
    "reach.surfaces@2.1.0",
    "reward_statistics.latency@1.1.0",
    "reward_statistics.price@1.2.0",
    "reward_statistics.semantics@1.1.0",
    "soundness.batteries@1.0.0",
    "soundness.capability_floor@1.0.0",
    "soundness.exploit_premium@1.0.0",
    "soundness.mutation.score@1.0.0",
    "soundness.relations.injected_defect@1.0.0",
    "trace.absence@1.0.0",
    "trace.composition@1.0.0",
    "validity.absence@1.0.0",
    "validity.fresh_state_sterility@1.0.0",
    "validity.episode_isolation@1.0.0",
    "validity.timeout_behaviour@1.0.0",
    "validity.resource_exhaustion@1.0.0",
    "validity.task_validity@1.0.0",
    "validity.pass_rate_floor@1.0.0",
    "validity.flakiness@1.0.0",
    "validity.input_handling@1.0.0",
    "validity.input_leakage@1.0.0",
    "validity.replay_determinism@1.0.0",
    "validity.unreachable_score_branch@1.0.0",
    "validity.writable_then_read@1.0.0",
)


def _planned_project_method_ids(
    discovered: Iterable[Any],
    *,
    trace_record: Any | None = None,
    response_bank: bool = False,
    project: Any = None,
    corpus: Any = None,
) -> tuple[str, ...]:
    """Return the pre-execution method set for a public Audit record.

    A supplied canonical TraceRecord is already closed input.  Its entry identities can therefore
    be derived before any Audit instrument or qualified outcome execution runs.  Reading those
    identities here binds the supervisor to the exact serialized Trace evidence the public route
    will retain, including per-row evidence or its explicit absence, instead of expanding a
    generic Audit literal after the record has been assembled.
    """
    planned = set(_PUBLIC_AUDIT_METHODS)
    killing_planned = False
    if project is not None and corpus is not None:
        if response_bank:
            planned.discard("validity.absence@1.0.0")
            planned.add("validity.decision_coverage@1.0.0")
        from reward_lens.instruments import soundness
        from reward_lens.instruments.base import RunContext

        try:
            cases, _ = soundness._cases(None, corpus, RunContext(project=project))
        except (contracts.UsageError, OSError, ValueError, KeyError):
            # A malformed or empty protected battery is an instrument refusal, whose method is
            # the panel's absence. The independent outcome execution remains a separate route.
            cases = ()
        if cases:
            planned.add("soundness.absence@1.0.0")
            planned.add("soundness.gold_sanity@1.0.0")
            planned.add("soundness.mutation.survival@1.0.0")
            for case in cases:
                for name in (
                    *soundness.relations.relations_for(case.response),
                    soundness.relations.KILLING,
                ):
                    if soundness.relations.pair(name, case.response) is not None:
                        planned.add(f"soundness.relations.{name}@1.0.0")
                        if name == soundness.relations.KILLING:
                            killing_planned = True
    for panel in discovered:
        for instrument in getattr(panel, "instruments", ()):
            # Without a supplied Trace, Signal writes its response-bank method,
            # including its precise sampled-response absence.  A closed Trace is
            # the other actual signal input and uses the clustered method.
            response_method = getattr(instrument, "response_bank_method", None)
            method = (
                response_method
                if response_method is not None and trace_record is None
                else getattr(instrument, "method", None)
            )
            if method is not None:
                planned.add(f"{method.id}@{method.version}")
    # With no usable cases, Relations still emits its own absence under the
    # instrument's injected-defect method. Only a nonempty case set whose
    # responses cannot form that relation omits this method altogether.
    if project is not None and corpus is not None and cases and not killing_planned:
        planned.discard("soundness.relations.injected_defect@1.0.0")
    if trace_record is not None:
        planned.update(
            f"{entry.method.id}@{entry.method.version}" for entry in trace_record.entries()
        )
        # `answer_entry` is derived from the same closed TraceRecord and is retained only when
        # its input evidence resolves.  Its procedure identity is fixed by that public producer,
        # so it belongs to this Trace-input declaration even though the evidence references are
        # assembled with the rest of the Audit record later.
        from reward_lens.product.trace.answer import _ANSWER_METHOD_ID, _ANSWER_METHOD_VERSION

        planned.add(f"{_ANSWER_METHOD_ID}@{_ANSWER_METHOD_VERSION}")
    return tuple(sorted(planned))


def _bind_seeker_config(entries: Iterable[contracts.Entry], config: Any) -> None:
    """Make configured actor inputs part of the measured-method identity.

    The actor's recorded proposals live in the entry result.  Its configuration changes the
    procedure itself, so it also belongs in the method identity used by reuse and the store's
    no-rewrite check.  This is intentionally applied only to the discovered seeker entries.
    """
    for entry in entries:
        if not str(entry.entry_id).startswith("exploits.seeker."):
            continue
        entry.method = entry.method.model_copy(
            update={
                "params_digest": contracts.digest(
                    {"method": entry.method.params_digest, "seeker_config": config.digest}
                )
            }
        )


def _supervised_outcome_checker(execution: Any) -> Any:
    """Resolve only declared target tasks through the qualified project outcome execution.

    Audit never imports or calls a project outcome function in this process.  The loader has already
    qualified declared controls under the accepted sandbox; this callback only selects the exact
    target item and retains the resulting supervisor receipt identities with the proposal score.
    """
    from reward_lens.instruments.seeker.protocol import OutcomeCheckResult
    from reward_lens.outcome.resolution import OutcomeState

    def unavailable(reason: str) -> OutcomeCheckResult:
        return OutcomeCheckResult(
            None,
            {"state": "UNQUALIFIED", "reason": reason, "qualification": "unavailable"},
        )

    if execution is None:
        return lambda _task, _response: unavailable("missing_outcome_execution_config")

    qualification = execution.supervised.qualification

    def outcome(task: dict, response: str) -> OutcomeCheckResult:
        if not execution.qualified:
            return unavailable("outcome_controls_unqualified")
        try:
            item_id = execution.target_items_by_task_digest.get(contracts.digest(dict(task)))
            if item_id is None:
                return unavailable("task_is_not_a_declared_outcome_target")
            resolved = execution.resolve_for_actual(
                task=dict(task), response=response, item_id=item_id
            )
        except Exception as error:
            return unavailable(f"supervised_outcome_unavailable:{type(error).__name__}")
        result = resolved.outcome
        passed = (
            True
            if result.state is OutcomeState.PASS
            else False
            if result.state is OutcomeState.FAIL
            else None
        )
        return OutcomeCheckResult(
            passed,
            {
                "state": str(result.state),
                "reason": result.reason,
                "protocol_digest": result.protocol_digest,
                "qualification_entry_id": qualification.entry.entry_id,
                "qualification_method_digest": qualification.entry.method.params_digest,
                "control_partition_id": execution.control_partition_id,
                "control_partition_digest": execution.control_partition_digest,
                "target_partition_id": resolved.target_partition_id,
                "target_partition_digest": resolved.target_partition_digest,
                "target_item_digest": resolved.target_item_digest,
                "task_digest": resolved.task_digest,
                "response_digest": resolved.response_digest,
                "receipt": {
                    "receipt_digest": resolved.receipt.receipt_digest,
                    "run_id": resolved.receipt.run_id,
                    "subject_digest": resolved.receipt.subject_digest,
                    "manifest_digest": resolved.receipt.manifest_digest,
                    "request_digest": resolved.receipt.request_digest,
                    "payload": resolved.receipt.to_record(),
                },
            },
        )

    return outcome


def _outcome_identity(execution: Any, error: str | None) -> dict[str, Any]:
    """The qualification identity saved with every configured actor record."""
    if execution is None:
        return {
            "state": "UNQUALIFIED",
            "reason": error or "missing_outcome_execution_config",
        }
    qualification = execution.supervised.qualification
    return {
        "state": "QUALIFIED" if execution.qualified else "UNQUALIFIED",
        "qualification_entry_id": qualification.entry.entry_id,
        "qualification_method_digest": qualification.entry.method.params_digest,
        "control_partition_id": execution.control_partition_id,
        "control_partition_digest": execution.control_partition_digest,
        "target_partition_id": execution.target_partition_id,
        "target_partition_digest": execution.target_partition_digest,
        "evaluator_source_digest": execution.evaluator_source_digest,
        "evaluator_manifest_digest": execution.evaluator_manifest_digest,
        "config_source_digest": execution.config_source_digest,
    }


class _AuditHttpTransport:
    """The runner-owned OpenAI-compatible protocol transport for configured seeker actors."""

    def __init__(self, endpoint: str, *, credential_env: str | None = None) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise contracts.UsageError(
                code="RL0003",
                message="the seeker endpoint must be an absolute http or https URL",
                remediation="set endpoint in --seeker-config to the endpoint base URL",
                context={"field": "seeker_config.endpoint"},
            )
        self._parsed = parsed
        self._credential_env = credential_env

    def complete(self, payload: dict) -> dict:
        parsed = self._parsed
        target = parsed.path.rstrip("/") + "/chat/completions"
        if parsed.query:
            target += "?" + parsed.query
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._credential_env is not None:
            secret = os.environ.get(self._credential_env)
            if not secret:
                raise contracts.UsageError(
                    code="RL0003",
                    message=f"the seeker credential environment variable {self._credential_env} is unset",
                    remediation="set the named environment variable before running the api seeker",
                    context={"field": "seeker_config.credential_env"},
                )
            headers["Authorization"] = f"Bearer {secret}"
        connection = (
            http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=30)
            if parsed.scheme == "https"
            else http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
        )
        try:
            connection.request(
                "POST", target, body=json.dumps(payload).encode("utf-8"), headers=headers
            )
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()
        if response.status != 200:
            raise RuntimeError(f"seeker endpoint returned HTTP {response.status}")
        loaded = json.loads(body.decode("utf-8"))
        if not isinstance(loaded, dict):
            raise RuntimeError("seeker endpoint returned a non-object completion")
        return loaded


def _identities_under(record: contracts.Assay, entry_id: str) -> frozenset[str]:
    """Every method identity the record holds under `entry_id`; empty when it holds none.

    An instrument's entry ids are its own id or a name below it: `soundness.battery` returning
    three entries writes `soundness.battery.0` and its siblings, and returning nothing leaves the
    runner's absence under `soundness.battery` exactly. So the record answers for an instrument
    when it holds either, and the prefix carries the separating dot so that `reach.oversight` is
    not read as covering `reach.oversight_added`, which is a different instrument.
    """
    prefix = f"{entry_id}."
    return frozenset(
        method_identity(entry.method)
        for entry in record.entries()
        if (entry.entry_id == entry_id or entry.entry_id.startswith(prefix))
        and not _unbuilt_placeholder(entry)
    )


def _unbuilt_placeholder(entry: Any) -> bool:
    """An old build's reserved method id is not evidence that the method ran."""
    if entry.kind != "absence":
        return False
    # `missing_code` has a deliberately coarse fallback that also labels a present instrument's
    # refusal. The long missing-access field is what identifies the reserved old-build slot.
    return getattr(getattr(entry, "absence", None), "missing_access", None) in {
        absences.NOT_IN_BUILD,
        "instrument not in this build",  # older reserved-slot spelling
    }


def _answers(previous: contracts.Assay, wanted: dict[str, str | None]) -> bool:
    """Whether `previous` measured what the request would measure (section 6.1).

    Containment one way only: every requested entry id has to be in the record, measured by the
    identity the request names where it names one. What the record holds and the request does not
    ask for does not disqualify it, because a record of a wider build answers a narrower request
    on the part that was asked -- and a method that changed inside a panel nobody asked about
    changed nothing this request can see.
    """
    all_identities = {
        method_identity(entry.method)
        for entry in previous.entries()
        if not _unbuilt_placeholder(entry)
    }
    for entry_id, identity in wanted.items():
        if identity is not None:
            # Instruments may materialize several evidence rows under domain entry
            # ids rather than their declared method id (Signal, Mutation and
            # Relations do).  The stored method identity is the persistent
            # measurement key used by the no-rewrite store; requiring its planned
            # label as an entry prefix made an unchanged record miss reuse.
            if identity not in all_identities:
                return False
            continue
        held = _identities_under(previous, entry_id)
        if not held:
            return False
    return True


def _standing_record(
    project: Any, *, discovered: Iterable[Any] = ()
) -> tuple[str, contracts.Assay] | None:
    """The store's record of the subject as it stands now, if it already holds one.

    The index is the list of candidates and the newest is asked first, because a project that has
    moved and moved back wants the measurement of where it is, not the oldest one that matches.
    A candidate is the answer only when three things hold, and they are not one thing. The project,
    asked for its version through that record's own method set, has to produce exactly the version
    digest the record claims. And `changed_since` has to be empty: that is the store's own
    dependency check over the nine digests, and it is the question the brief asks, "has anything
    the record depends on changed". And the record has to have measured what this request would
    measure, which `_answers` asks as the pairs (entry id, method identity) the plan would produce
    -- the same key the store's `_check_not_a_rewrite` refuses a rewrite on, so a record that
    answers here is exactly a record a second write would have been refused as a rewrite of.

    A file the store cannot open is not an answer and not an error either. `_check_not_a_rewrite`
    already steps over unreadable and non-record files in the same directory, and a reuse check
    that raised where the write path shrugs would make an audit fail on a stray file. Nothing is
    suppressed by stepping over it: a project whose config or grader is the reason a candidate
    could not be versioned falls through to the measuring path, which asks the same questions and
    raises there.
    """
    if project is None:
        return None
    wanted = _requested(discovered, project, include_builtin_checks=True)
    try:
        names = project.names()
    except OSError:
        return None
    for name in reversed(names):
        try:
            previous = project.open_record(name)
            current = project.version_of(previous).digest()
        except (OSError, ValueError, contracts.RewardLensError):
            # `ValueError` covers a file that is not JSON; `RecordInvalid` (RL0604) covers a file
            # that is JSON and is not a record, which is what a decoy in `assays/` looks like.
            continue
        if current != previous.subject.version.digest:
            continue
        if project.changed_since(previous):
            continue
        if not _answers(previous, wanted):
            # Section 6.1: an unchanged dependency set is not enough. A record that answers has to
            # have measured what this request would measure. Three ways it has not: a panel that
            # was dark when the record was written and is filled now, a method added beside one
            # that was already in a populated panel, and a method whose parameters have moved since.
            # In all three, handing the record back would answer a wider or a different question
            # with the old measurement while reporting that no instrument ran.
            continue
        if not _soundness_versions_answer(previous, discovered):
            continue
        return name, previous
    return None


def _reuse(name: str, previous: contracts.Assay, *, project: Any) -> contracts.Assay:
    """Hand back the stored record and complete its bundle, without rewriting anything on disk.

    The record is the file: it is not re-serialised, so the bytes a caller gets back are the bytes
    a reader has. The manifest and the report are functions of those bytes and of nothing else, so
    a bundle that lost either can have it back exactly as it was first written, which is why the
    two are composed here rather than copied. A bundle that still has them is left alone, including
    the case where `manifest.json` has since been replaced by a later audit of a changed version:
    rewriting it would only move the superseded report from this bundle to that one.
    """
    from reward_lens.render.report import render_to

    path = project.path_for(name)
    payload = path.read_bytes()
    manifest = _manifest(record_name=path.name, payload=payload, assay_id=previous.assay_id)
    digest = contracts.digest(manifest)

    manifest_path = path.with_name(MANIFEST_NAME)
    wrote_manifest = not manifest_path.is_file()
    if wrote_manifest:
        manifest_path.write_bytes(manifest_bytes(manifest))

    report_path = path.with_name(f"{name}.assay.html")
    rendered_report = not report_path.is_file()
    if rendered_report:
        render_to(previous, report_path, bundle_manifest_digest=digest)

    _REUSE.set(
        Reused(
            name=name,
            assay_id=previous.assay_id,
            subject_version=previous.subject.version.digest,
            record_path=str(path),
            rendered_report=rendered_report,
            wrote_manifest=wrote_manifest,
            says=(
                f"no instrument ran: {name} already measures subject version "
                f"{previous.subject.version.digest}, and nothing it depends on has changed"
            ),
        )
    )
    return previous


def _clean(text: str) -> str:
    """The cleaning `_ident` does, without the identifier fallback `_ident` ends on."""
    return re.sub(r"[^A-Za-z0-9_.:/-]", "-", text).strip("-")[:200]


def _ident(text: str) -> str:
    cleaned = _clean(text)
    return cleaned if cleaned and cleaned[0].isalnum() else "subject"


def _resolved_name(root: Any) -> str:
    """The project directory's own name, resolved first.

    `Project.open` keeps the root exactly as it was handed in, so a project opened as `.` has
    `Path(root).name == ""`. Resolving is what gives that directory a name to be known by.
    """
    try:
        return Path(root).resolve().name
    except OSError:
        return Path(root).name


def _system_name(declared: Any, root: Any) -> str:
    """The reward system's name, total over every root a project can be opened from.

    A config that declares a name uses it. One that does not is named for its project directory,
    resolved and then cleaned the way `_ident` cleans. When nothing survives that,
    `DEFAULT_SYSTEM_NAME` is the name. The empty string is never returned, which is what
    `RewardSystemRef.name` refused with `string_too_short`.
    """
    if declared:
        cleaned = _clean(str(declared))
        if cleaned:
            return cleaned
    return _clean(_resolved_name(root)) or DEFAULT_SYSTEM_NAME


def _os_name() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    return "macos" if sys.platform == "darwin" else "windows"


@dataclass
class AuditSubject:
    """What the audit is pointed at, once the request and the project have been reconciled."""

    grader_path: Path
    entrypoint: str
    project_dir: Path | None
    tasks_path: Path | None
    responses_path: Path | None
    trainer: str | None
    name: str
    source: str


def resolve(req: Any, project: Any) -> AuditSubject:
    """The grader, its entrypoint and its inputs. A grader that is not there is RL0001."""
    path = Path(req.path)
    entrypoint = "score"
    tasks_path = Path(req.tasks) if getattr(req, "tasks", None) else None
    responses_path = Path(req.responses) if getattr(req, "responses", None) else None
    trainer = None
    if project is not None:
        root = Path(project.root)
        config = project.config
        entry = str(config.reward.entry)
        if ":" in entry:
            module, _, entrypoint = entry.partition(":")
            grader_path = (root / module.replace(".", "/")).with_suffix(".py")
        else:
            grader_path = root / entry
        trainer = getattr(config.reward, "trainer", None)
        if tasks_path is None and config.tasks is not None:
            tasks_path = root / config.tasks.path
        if responses_path is None and getattr(config, "responses", None) is not None:
            responses_path = root / config.responses.path
        project_dir: Path | None = root
        name = _ident(_resolved_name(root))
    elif path.is_dir():
        raise contracts.UsageError(
            code="RL0001",
            message=f"{path} is not a reward-lens project: it holds no rewardlens.yaml",
            remediation=(
                "run `reward-lens init --example code-reward <dir>` to make one, or point the "
                "audit straight at a grader file"
            ),
            context={"path": str(path)},
        )
    else:
        grader_path = path
        project_dir = None
        name = _ident(path.name)
    if not grader_path.is_file():
        raise contracts.UsageError(
            code="RL0001",
            message=f"no grader at {grader_path}",
            remediation=(
                "point the audit at a Python file that defines the scoring function, or at a "
                "project whose rewardlens.yaml names one under `reward.entry`"
            ),
            context={"path": str(grader_path)},
        )
    if project is None:
        project_dir = grader_path.parent if grader_path.parent != Path("") else None
    return AuditSubject(
        grader_path=grader_path,
        entrypoint=entrypoint,
        project_dir=project_dir,
        tasks_path=tasks_path,
        responses_path=responses_path,
        trainer=trainer,
        name=name,
        source=grader_path.read_text(encoding="utf-8"),
    )


def _openenv_step(
    *,
    project: Any,
    corpus: instruments.Corpus,
    joined_rollouts: tuple[Any, ...],
    joined_origin: str,
    ctx: RunContext,
    subject_ref: str,
) -> contracts.Entry | None:
    """Run one configured typed OpenEnv step, or make its non-execution explicit.

    The regular Audit carrier keeps analysing the declared bridge source.  This is the separate
    black-box operation that proves that bridge can construct the frozen OpenEnv binding.  It
    deliberately scores one joined response: an Audit must not turn one configured environment
    into an unbounded episode runner.
    """
    params = _openenv_params(project)
    if params is None:
        return None
    binding = project.config.reward.openenv
    assert binding is not None  # ProjectConfig rejects kind=openenv without this object.
    dependency = ("digest:source", "digest:scorer_config", "digest:samples")
    started = ctx.clock()
    identity: dict[str, Any] = {
        "config_digest": contracts.digest(binding.model_dump(mode="json")),
        "client_factory": project.config.reward.entry,
        "reset": binding.reset,
        "action": binding.action,
        "server": {
            "base_url": binding.base_url,
            "identity_digest": contracts.digest(
                {
                    "package": binding.package,
                    "version": binding.version,
                    "base_url": binding.base_url,
                }
            ),
        },
    }

    def absent(detail: str, remedy: str, *, state: str, code: str | None = None) -> contracts.Entry:
        identity["state"] = "absent"
        identity["missing"] = {"state": state, "detail": detail, "code": code}
        ctx.openenv_route = identity
        return ctx.absence(
            "validity",
            "validity.openenv_step",
            "the one configured OpenEnv score over the first joined task and response",
            detail,
            remedy,
            ("no claim that the configured OpenEnv environment scored a response",),
            subject_ref=subject_ref,
            state=state,
            depends_on=dependency,
            duration_s=ctx.clock() - started,
        )

    if not joined_rollouts:
        return absent(
            f"no joined task/response rollout is available: {joined_origin}",
            "supply both a task set and response bank with one matching response id",
            state="NOT_MEASURED",
        )
    rollout = joined_rollouts[0]
    task = rollout.inputs.get("task")
    response = rollout.inputs.get("response")
    if not isinstance(task, dict) or not isinstance(response, (str, dict)):
        return absent(
            "the first joined rollout does not carry the typed task and response OpenEnv requires",
            "supply a joined task mapping and string or mapping response",
            state="REFUSED",
        )
    selected_rollout = {
        "rollout_id": rollout.id,
        "task_digest": contracts.digest({"task": task}),
        "response_digest": contracts.digest({"response": response}),
        "request_digest": contracts.digest({"task": task, "response": response}),
    }
    identity["selected_rollout"] = selected_rollout
    try:
        from reward_lens.graders.openenv import OpenEnvGrader

        adapter = OpenEnvGrader.from_config(
            binding, root=Path(project.root), entry=project.config.reward.entry
        )
        identity["adapter_manifest_digest"] = adapter.describe().digest()
        identity["adapter_source_digest"] = contracts.digest({"source": adapter.source()})
        envelope = adapter.score(dict(task), response)
    except contracts.RewardLensError as failure:
        return absent(
            str(failure),
            "install reward-lens[openenv] and supply resolvable explicit client, reset, and action bindings",
            state="REFUSED",
            code=failure.code,
        )
    adapter_attempt = {
        "request_digest": envelope.request_digest,
        "reset_digest": envelope.provenance.get("reset_digest"),
        "action_class": envelope.provenance.get("action_class"),
        "observation_class": envelope.provenance.get("returned_observation_class"),
        "result_digest": contracts.digest(envelope.to_dict()),
    }
    identity["attempt"] = adapter_attempt
    if envelope.verdict != "scored" or envelope.score is None:
        return absent(
            "; ".join(envelope.errors) or f"OpenEnv returned verdict {envelope.verdict}",
            "make the configured client return one finite StepResult.reward",
            state="COULD_NOT_CHECK",
        )

    result = {
        "score": float(envelope.score),
        "done": bool(envelope.provenance.get("done")),
        "step_count": int(envelope.provenance.get("step_count", 0)),
        "observation_class": envelope.provenance.get("returned_observation_class"),
        "selected_rollout": selected_rollout,
        "adapter": adapter_attempt,
    }
    identity["state"] = "scored"
    identity["result"] = {
        "done": result["done"],
        "step_count": result["step_count"],
        "result_digest": adapter_attempt["result_digest"],
    }
    ctx.openenv_route = identity
    return contracts.estimate(
        "validity",
        "validity.openenv_step",
        "the one configured OpenEnv score over the first joined task and response",
        float(envelope.score),
        "OpenEnv reward",
        1,
        "joined task/response rollout",
        contracts.Uncertainty(
            method=contracts.NO_INTERVAL,
            level=0.95,
            reason="one configured environment step is a witnessed score, not a sample for an interval",
            clusters=max(1, int(getattr(corpus, "n_tasks", 0) or 0)),
            cluster_unit="task",
        ),
        subject_ref=subject_ref,
        provenance=ctx.provenance(duration_s=ctx.clock() - started, arm="black_box"),
        method=contracts.Method(
            id="validity.openenv_step",
            version="0.6.0",
            params_digest=contracts.digest(params),
            procedure=(
                "construct the configured explicit OpenEnv client, reset it once from the first "
                "joined task, build one typed Action from that task and response, and retain the "
                "finite StepResult reward and done flag"
            ),
            credited_to="reward_lens.graders.openenv.OpenEnvGrader",
        ),
        result=result,
        depends_on=dependency,
        limitations=tuple(envelope.limitations),
    )


def _static_entry(
    result: static_checks.StaticResult,
    ctx: RunContext,
    *,
    subject_ref: str,
    source_name: str,
    source_digest: str,
) -> contracts.Entry:
    """A static check as a `check`, or, when it failed, as a `witness` by artifact identity."""
    method = contracts.Method(
        id=result.entry_id,
        version="1.0.0",
        params_digest=contracts.digest({"check": result.entry_id, "predicate": result.predicate}),
        procedure=(
            "the grader's own source is parsed with the standard library's ast and the check's "
            "predicate is evaluated over the call sites and branches it names; nothing is executed"
        ),
    )
    common: dict[str, Any] = {
        "entry_id": result.entry_id,
        "section": "validity",
        "kind": "check",
        "measurand": result.measurand,
        "method": method,
        "scope": "evaluator_comparison",
        "subject_ref": subject_ref,
        "depends_on": ["digest:source"],
        "state": "complete",
        "provenance": ctx.provenance(arm="white_box"),
        "limitations": list(result.limitations),
        "result": dict(result.result) | {"observed": result.observed},
    }
    if result.passed:
        return contracts.Entry(
            **common,
            check=contracts.Check(
                predicate=result.predicate, passed=True, scope_tested=result.scope_tested
            ),
        )
    common["kind"] = "witness"
    return contracts.Entry(
        **common,
        witness=contracts.Witness(
            inputs={
                "source": source_name,
                "line": result.line,
                "path": result.target_path,
                "predicate": result.predicate,
            },
            procedure=method.procedure,
            observed=result.observed,
            artifact_identity=source_digest,
            path=source_name,
        ),
    )


def _explain_writable_then_read(
    entry: contracts.Entry, result: static_checks.StaticResult, *, n_responses: int
) -> contracts.Entry:
    """The headline and the paragraph a reader gets for RL0201, from the numbers measured (A-015).

    The two sentences are not the finding's message repeated. The message says what was read; the
    headline says what it means, and the paragraph says what is still not established, because a
    static read of the source cannot say whether any sampled response actually took the path. The
    response count is the corpus's, and when there is no corpus the sentence does not invent one.
    """
    directory = str(result.target_path or "").rsplit("/", 1)[0]
    where = f"{directory}/" if directory else str(result.target_path or "the path it reads")
    used_it = (
        f"Whether any of the {n_responses} responses used that path"
        if n_responses
        else "Whether any response used that path"
    )
    entry.result = dict(entry.result or {}) | {
        "headline": "This reward reads a file the graded process can write.",
        "explanation": (
            f"{result.source_name} line {result.line} runs the tests in {where}, and {where} sits "
            f"inside the directory the graded response runs in. {used_it} is not established "
            f"here: the executed witness is not in this build."
        ),
    }
    return entry


def _finding(
    code: str,
    *,
    entry_id: str,
    message: str,
    level: str,
    kind: str,
    rationale: str,
    arm: str,
    rule: str | None = None,
    file: str | None = None,
    line: int | None = None,
) -> contracts.Finding:
    """One finding. `rule` is the name it is raised under and defaults to the entry id.

    The two are not the same name. The entry id is where the hole or the witness sits in the
    record, and one measurand can be asked by more than one arm: the rule names the arm, the entry
    id does not move, and `entries` still points at the entry the finding came from (A-025 point 2).
    """
    finding = contracts.Finding(
        id=FINDING_IDS[code],
        rule=rule or entry_id,
        level=level,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        severity_rationale=rationale,
        scope="evaluator_comparison",
        entries=[entry_id],
        partial_fingerprint=contracts.digest(
            {"code": code, "entry": entry_id, "message": message}
        ).removeprefix("sha256:")[:16],
        message=message,
        code=code,
        arm=arm,  # type: ignore[arg-type]
    )
    if file is not None and line is not None:
        finding.location = contracts.models.FindingLocation(file=file, line=int(line))
    return finding


def _panel_package(instrument: Any) -> Any:
    """The instrument package a module-level `findings_for` would be declared on.

    Discovery imports `reward_lens.instruments.<name>` and reads `PANEL` off it, so the package is
    what section 8.1 means by "the package": the module the class is defined in may be a private one
    below it (P-REACH's instrument lives in `reach.probes`), and both packages re-export the
    function, so the harvest asks the package and not the defining module.
    """
    defined_in = str(getattr(type(instrument), "__module__", ""))
    prefix = f"{registry.PACKAGE}."
    if defined_in.startswith(prefix):
        defined_in = prefix + defined_in[len(prefix) :].split(".", 1)[0]
    return sys.modules.get(defined_in)


def _wants_every_entry(harvest: Any) -> bool:
    """Whether a package's `findings_for` takes all the entries at once rather than one of them.

    Section 8.1 names one shape for this form, `findings_for(entry)`, and P-FRAME's is that.
    P-REACH's is `findings_for(entries: Sequence[Entry])`, which is neither of the two forms the
    section allocates; called per entry it would iterate a single `Entry` and the panel's findings
    would be lost behind the exception. What the function declares is read rather than guessed at,
    so the harvest is decided by the signature and not by the package's name, and the disagreement
    is booked against the package rather than silently normalised.
    """
    try:
        parameters = list(inspect.signature(harvest).parameters.values())
    except (TypeError, ValueError):  # a callable that will not introspect: assume the named form
        return False
    positional = [
        one for one in parameters if one.kind in (one.POSITIONAL_ONLY, one.POSITIONAL_OR_KEYWORD)
    ]
    if not positional:
        return False
    first = positional[0]
    annotation = str(first.annotation).lower()
    return first.name.endswith("entries") or any(
        hint in annotation for hint in ("sequence", "iterable", "list[", "tuple[")
    )


def _panel_findings(instrument: Any, produced: list[contracts.Entry]) -> list[contracts.Finding]:
    """Every finding a discovered instrument raises over the entries it has just produced (8.1).

    A panel that measures something and cannot say what it means is half a panel: the entry carries
    the number and the finding carries the judgement, and without this the judgement never reached
    `Assay.findings`, the verdict's reasons or the text surface. The two forms are the section's:
    `findings(entries)` on the instrument object, or `findings_for` on the package. An instrument
    that exposes neither raises nothing, which is a panel whose entries speak for themselves and not
    an error.
    """
    if not produced:
        return []
    own = getattr(instrument, "findings", None)
    if callable(own):
        return list(own(list(produced)))
    package = _panel_package(instrument)
    harvest = getattr(package, "findings_for", None)
    if not callable(harvest):
        return []
    if _wants_every_entry(harvest):
        return list(harvest(list(produced)))
    raised: list[contracts.Finding] = []
    for entry in produced:
        raised.extend(harvest(entry))
    return raised


def _unallocated_codes(raised: Iterable[contracts.Finding]) -> list[str]:
    """The codes among `raised` that section 6 allocates to no panel, in the order they appear."""
    loose: list[str] = []
    for finding in raised:
        code = getattr(finding, "code", None)
        if code and code not in PANEL_FINDING_CODES and code not in loose:
            loose.append(str(code))
    return loose


def _code_allocation_check(
    ctx: RunContext, *, entry_id: str, codes: list[str], subject_ref: str
) -> contracts.Entry:
    """A failed `check` naming the codes a panel raised that section 6 allocates to no panel."""
    named = ", ".join(codes)
    predicate = (
        "every finding a discovered panel raises declares a code wave 2 allocates to a panel"
    )
    return contracts.Entry(
        entry_id="validity.panel_finding_codes",
        section="validity",
        kind="check",
        measurand="the finding codes a discovered panel declares",
        method=contracts.Method(
            id="validity.panel_finding_codes",
            version="1.0.0",
            params_digest=contracts.digest({"allocated": sorted(PANEL_FINDING_CODES)}),
            procedure=(
                "each finding a discovered instrument raised is read for its code and the code is "
                "looked up in the allocation wave-2 interfaces section 6 writes down; nothing is "
                "executed and the finding is carried either way"
            ),
        ),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=("digest:source",),
        state="complete",
        provenance=ctx.provenance(arm="static"),
        check=contracts.Check(
            predicate=predicate,
            passed=False,
            scope_tested=f"the findings {entry_id} raised in this run",
        ),
        result={
            "instrument": entry_id,
            "unallocated_codes": codes,
            "observed": (
                f"{entry_id} raised a finding under {named}, which wave-2 interfaces section 6 "
                "allocates to no panel; the finding is in the record and `explain` has nothing to "
                "say about the code"
            ),
        },
    )


def run(
    req: Any,
    *,
    project: Any = None,
    sandbox: Any = None,
    seeker_actor: Any = None,
    seeker_tasks: Sequence[Mapping[str, Any]] | None = None,
    seeker_intent: str | None = None,
) -> contracts.Assay:
    """Measure one reward system and return the record (section 7.3, D-41).

    ``seeker_actor`` is the narrow internal seam used by the MCP single-proposal route.  The
    caller supplies one already-constructed :class:`AgentActor` and one exact declared target
    task; this function still owns the normal instrument execution, record assembly, and sole
    persistence path.  It is deliberately not a generic actor or outcome injection API.
    """
    injected_seeker = (
        seeker_actor is not None or seeker_tasks is not None or seeker_intent is not None
    )
    injected_tasks: tuple[dict[str, Any], ...] = ()
    if injected_seeker:
        from reward_lens.instruments.seeker.actors.agent import AgentActor

        if project is None:
            raise contracts.UsageError(
                code="RL0003",
                message="an injected seeker proposal needs an accepted reward-lens project",
                remediation="open the declared project before proposing a target response",
                context={"field": "project_path"},
            )
        if type(seeker_actor) is not AgentActor:
            raise TypeError("the injected seeker actor must be the supported AgentActor")
        if seeker_tasks is None or len(seeker_tasks) != 1:
            raise contracts.UsageError(
                code="RL0003",
                message="an injected seeker proposal needs exactly one declared target task",
                remediation="select one task from the project outcome target declaration",
                context={"field": "task_id"},
            )
        if not isinstance(seeker_tasks[0], Mapping):
            raise TypeError("the injected seeker task must be a mapping")
        if seeker_intent is not None and not isinstance(seeker_intent, str):
            raise TypeError("the injected seeker intent must be a string")
        injected_tasks = (dict(seeker_tasks[0]),)
    started_monotonic = time.monotonic()
    started = utc_now()
    _REUSE.set(None)
    subject = resolve(req, project)
    trace_path = getattr(req, "run_dir", None)
    trace_declarations = _trace_declarations(req)
    trace = None
    trace_identity = None
    if trace_path is not None:
        trace_source = Path(trace_path)
        trace = _read_trace(trace_source, declarations=trace_declarations)
        trace_identity = {
            "reader": trace.reader,
            "run_id": trace.run_id,
            "source": str(trace_source),
            "source_digest": _source_digest(trace_source),
        }
    reference_path = getattr(req, "detection_reference", None)
    detection_reference = (
        _open_detection_reference(Path(reference_path)) if reference_path is not None else None
    )
    # Before any instrument, and after `resolve`, which is what turns a path into a subject: a
    # grader that is not there is RL0001 whether or not the store holds an old record of it.
    # D-28 and the store's "one subject version, one measurement" say a second measurement of an
    # unchanged subject is not a second measurement; asking first is what makes that true instead
    # of making the second audit an RL0620.
    # Discovery first, because the reuse check is scope-aware: what a stored record has to cover
    # is the plan this request would run, and the plan is not known until the panels are. One
    # discovery pass serves the reuse check, the absences and the panels themselves.
    discovered = registry.discover()
    filled = _panels(discovered)
    standing = (
        None
        if trace is not None
        or detection_reference is not None
        or injected_seeker
        or str(getattr(req, "seeker", "off") or "off") != "off"
        or getattr(req, "seeker_config", None) is not None
        else _standing_record(project, discovered=discovered)
    )
    if standing is not None and not (
        getattr(req, "resume", None) or getattr(req, "continue_run", False)
    ):
        return _reuse(standing[0], standing[1], project=project)
    ctx = RunContext(
        sandbox=sandbox,
        project=project,
        budget="0.00",
        clock=time.monotonic,
        offline=bool(getattr(req, "offline", True)),
        started=started,
        detection_reference=detection_reference,
    )
    # RunContext is intentionally open to instrument-specific context.  The seeker reads these
    # two values without making the audit runner import its implementation.
    ctx.seeker = "agent" if injected_seeker else str(getattr(req, "seeker", "off") or "off")
    ctx.budget = str(getattr(req, "max_budget_usd", None) or "0.00")
    seeker_config = None
    if ctx.seeker != "off" and not injected_seeker:
        requested_config = getattr(req, "seeker_config", None)
        if requested_config is None:
            raise contracts.UsageError(
                code="RL0003",
                message="an enabled seeker needs --seeker-config to construct its actor",
                remediation="pass a closed actor configuration with --seeker-config",
                context={"field": "seeker_config"},
            )
        from reward_lens.instruments.seeker.config import load as load_seeker_config

        seeker_config = load_seeker_config(requested_config, requested_mode=ctx.seeker)
        ctx.seeker_config = seeker_config
        ctx.seeker_max_proposals = seeker_config.max_proposals
    source_name = subject.grader_path.name
    source_digest = contracts.digest({"source": subject.source})
    response_bank_requested = bool(
        subject.tasks_path is not None
        and subject.responses_path is not None
        and subject.tasks_path.is_file()
        and subject.responses_path.is_file()
    )
    planning_corpus = instruments.build_corpus(
        tasks_path=subject.tasks_path, responses_path=subject.responses_path, trace=trace
    )
    has_declared_execution = (
        project is not None
        and getattr(getattr(project.config, "outcome", None), "execution", None) is not None
    )
    planned_project_methods = (
        _planned_project_method_ids(
            discovered,
            trace_record=trace,
            response_bank=response_bank_requested,
            project=project,
            corpus=planning_corpus,
        )
        if has_declared_execution
        else ()
    )
    precommitted_project_subject_ref = (
        project.version(methods=planned_project_methods).digest()
        if has_declared_execution
        else None
    )

    # The optional project declaration is the sole outcome route for public seeker proposals.
    # Loading it executes and qualifies the declared controls once under the Audit sandbox.  An
    # absent, unavailable, or malformed declaration leaves the arm explicitly unqualified.
    from reward_lens.execution import Limits
    from reward_lens.outcome.project_execution import (
        ProjectOutcomeExecution,
        load_project_outcome_execution,
    )

    outcome_limits = Limits(network=False)
    outcome_execution = None
    outcome_loader_error = None
    from reward_lens.instruments.detection import OnsetDetection

    unbound_instrument = next(
        (
            str(getattr(instrument, "id", "unnamed instrument"))
            for panel in discovered
            for instrument in getattr(panel, "instruments", ())
            if (
                getattr(
                    instrument,
                    "response_bank_method" if trace is None else "method",
                    None,
                )
                is None
                and getattr(instrument, "method", None) is None
                and type(instrument) is not OnsetDetection
            )
        ),
        None,
    )
    if project is not None and sandbox is not None and unbound_instrument is None:
        try:
            loaded_outcome = load_project_outcome_execution(
                project,
                expected_subject_ref=precommitted_project_subject_ref,
                sandbox=sandbox,
                limits=outcome_limits,
            )
            if isinstance(loaded_outcome, ProjectOutcomeExecution):
                outcome_execution = loaded_outcome
        except Exception as error:
            outcome_loader_error = f"{type(error).__name__}: {error}"
    elif unbound_instrument is not None and has_declared_execution:
        outcome_loader_error = (
            f"discovered instrument {unbound_instrument} has no predeclared method identity"
        )
    if precommitted_project_subject_ref is not None and (
        outcome_execution is None or not outcome_execution.qualified
    ):
        # A refused declaration bound no independent outcome evidence. Its retained entry is an
        # absence, and the assembled method set is authoritative even when plugins add methods
        # we could not know before running. Keep the exact precommit guard only for a supervisor
        # that was actually accepted and qualified above.
        precommitted_project_subject_ref = None
    subject_ref = precommitted_project_subject_ref or source_digest
    outcome_check = _supervised_outcome_checker(outcome_execution)
    vut = VerifierUnderTest(source_path=subject.grader_path, entrypoint=subject.entrypoint)

    corpus = planning_corpus
    joined_rollouts = corpus.rollouts
    joined_origin = corpus.origin
    n_responses = len(corpus.rollouts)
    tasks_for_probes = corpus.tasks
    if not corpus.rollouts:
        tasks_for_probes = ()
        if subject.tasks_path is not None and subject.tasks_path.is_file():
            tasks_for_probes = tuple(instruments.read_jsonl(subject.tasks_path, cap=1))
        corpus = instruments.probe_corpus(tasks_for_probes, trace=trace)
        has_bank = False
    else:
        has_bank = True

    # What this run was actually given, which is what an absence names when a reader can close it.
    # A task set is a task set whether or not a response bank was built from it, so these are read
    # off the request rather than off the corpus, which conflates the two.
    has_tasks = subject.tasks_path is not None and subject.tasks_path.is_file()
    has_responses = subject.responses_path is not None and subject.responses_path.is_file()
    has_outcome = project is not None and getattr(project.config, "outcome", None) is not None
    supplied = tuple(
        name
        for name, present in (
            (absences.TASK_SET, has_tasks),
            (absences.SAMPLED_RESPONSES, has_responses),
            (absences.PROTECTED_CHECK, has_outcome),
            # `run_dir` is not an Audit run directory here.  It is the public Trace input
            # route, opened above into the immutable canonical TraceRecord that is retained in
            # this Assay.  Treating it as absent made a supplied Trace look indistinguishable
            # from a bare grader to the absence catalogue.
            (absences.RUN_RECORD, trace is not None),
        )
        if present
    )

    entries: dict[str, list[contracts.Entry]] = {section: [] for section in _SECTIONS}
    findings: list[contracts.Finding] = []
    rules: dict[str, contracts.Rule] = {}

    # D-27 starts after resolving the real subject and before the first instrument.  A resume
    # therefore binds to the same source digest before anything is permitted to run again.
    project_root = Path(
        getattr(project, "root", None) or subject.project_dir or subject.grader_path.parent
    )
    from .plan import plan as audit_plan

    planned = audit_plan(req, project=project, sandbox=sandbox).model_dump(mode="json")
    resume_token = getattr(req, "resume", None)
    if resume_token:
        run_dir = RunDir.find(str(resume_token)).resume()
    elif getattr(req, "continue_run", False):
        candidate = next(
            (
                known
                for known in RunDir.list()
                if known.project == str(project_root)
                and known.manifest.get("command") == "audit"
                and not known.finished
            ),
            None,
        )
        if candidate is None:
            raise RunNotFound(wanted="unfinished audit", root=runs_root())
        run_dir = candidate.resume()
    else:
        run_dir = RunDir.create(
            project=project_root,
            name=getattr(req, "name", None),
            command="audit",
            plan=planned,
        )
    run_dir.bind_subject(subject_ref)
    run_dir.bind_execution(
        {
            "subject_ref": subject_ref,
            "plan": planned,
            "methods": _requested(
                discovered, project, trace_record=trace, include_builtin_checks=True
            ),
            "seeker": ctx.seeker,
            "budget_usd": ctx.budget,
            "seeker_config": None if seeker_config is None else seeker_config.public(),
            "injected_seeker_task_digest": (
                contracts.digest(injected_tasks[0]) if injected_seeker else None
            ),
            "only": sorted(str(item) for item in getattr(req, "only", ()) or ()),
            "trace": trace_identity,
            "trace_declarations": trace_declarations if trace is not None else None,
            "detection_reference": (
                {
                    "source": detection_reference.source,
                    "source_digest": detection_reference.digest,
                    "spec_digest": detection_reference.spec.digest(),
                    "trace_sources": [
                        {
                            "partition": item.partition,
                            "source": item.source,
                            "reader": item.reader,
                            "run_id": item.run_id,
                            "source_digest": item.digest,
                            "declarations": dict(item.declarations),
                        }
                        for item in detection_reference.trace_sources
                    ],
                }
                if detection_reference is not None
                else None
            ),
        }
    )
    _ACTIVE_RUN.set(run_dir)

    # A finished run forks rather than mutating its source.  Its record is already sealed and the
    # fork carries the same partials, so returning that record is the only replay that preserves
    # the old bytes and its named path.  An interrupted run has no record_path and falls through
    # to normal partial replay below.
    carried_record = run_dir.manifest.get("record_path")
    if getattr(req, "resume", None) and run_dir.manifest.get("forked_from") and carried_record:
        path = Path(str(carried_record))
        if path.is_file():
            replayed_record = contracts.Assay.model_validate(json.loads(path.read_text("utf-8")))
            run_dir.finish(record_path=path)
            _ACTIVE_RUN.set(None)
            return replayed_record

    def partial_payload(
        produced: list[contracts.Entry], raised: list[contracts.Finding] | None = None
    ) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in produced],
            "findings": [finding.to_dict() for finding in (raised or [])],
        }

    def replayed(
        instrument_id: str, section: str, *, include_findings: bool = True
    ) -> list[contracts.Entry] | None:
        payload = run_dir.partial().get(instrument_id)
        if payload is None:
            return None
        loaded = [contracts.Entry.model_validate(item) for item in payload.get("entries", [])]
        entries[section].extend(loaded)
        if include_findings:
            findings.extend(
                contracts.Finding.model_validate(item) for item in payload.get("findings", [])
            )
        run_dir.log(f"{utc_now()} instrument {instrument_id} replayed {len(loaded)} entries")
        return loaded

    def seal(
        instrument_id: str,
        produced: list[contracts.Entry],
        raised: list[contracts.Finding] | None = None,
    ) -> None:
        run_dir.write_partial(instrument_id, partial_payload(produced, raised))
        run_dir.log(f"{utc_now()} instrument {instrument_id} ran {len(produced)} entries")

    # The grader is loaded once, after the source-only static boundary, and the failure is held.
    # Letting each instrument discover the failure for itself wrote a different hole per entry: the
    # first named the real `ModuleNotFoundError`, and the rest named whatever they hit downstream of
    # it (an adapter holding `None`), which named neither the module nor the import that raised it.
    # One load means one true statement, repeated on every entry that depended on it, while the
    # static checks, which never touch the loaded module, still run and still seal first.
    grader_failure: BaseException | None = None

    def guarded(
        entry_id: str,
        section: str,
        measurand: str,
        work,
        remedy: str,
        *,
        needs_grader: bool = True,
        defer_seal: bool = False,
    ) -> list[contracts.Entry]:
        prior = replayed(entry_id, section)
        if prior is not None:
            return prior
        before = len(entries[section])
        if needs_grader and grader_failure is not None:
            entries[section].append(
                absences.could_not_check(
                    ctx,
                    section=section,
                    entry_id=entry_id,
                    measurand=measurand,
                    failure=grader_failure,
                    remedy=remedy,
                    subject_ref=subject_ref,
                    affected_claims=(f"no {section} claim from {entry_id}",),
                )
            )
            produced = entries[section][before:]
            if not defer_seal:
                seal(entry_id, produced)
            return produced
        try:
            entries[section].append(work())
        except BudgetExhausted:
            raise
        except Exception as failure:  # the audit catches it; the instrument does not swallow it
            entries[section].append(
                absences.could_not_check(
                    ctx,
                    section=section,
                    entry_id=entry_id,
                    measurand=measurand,
                    failure=failure,
                    remedy=remedy,
                    subject_ref=subject_ref,
                    affected_claims=(f"no {section} claim from {entry_id}",),
                )
            )
        produced = entries[section][before:]
        if not defer_seal:
            seal(entry_id, produced)
        return produced

    def guarded_many(
        entry_id: str,
        section: str,
        measurand: str,
        work,
        remedy: str,
        *,
        declared_method: Any = None,
    ) -> tuple[list[contracts.Entry], list[contracts.Finding], bool]:
        """Every entry an instrument returned, not the first of them.

        Returns what the instrument itself produced, which is empty when it failed or returned
        nothing: the absence written in either case is the runner's own entry. When the instrument
        declared its method, that absence retains the declaration, so a later parameter change
        cannot reuse a refusal made by another method. The caller harvests findings from the
        instrument's entries, never from the runner's hole.

        An instrument that returns nothing measured nothing, and that is what goes in the record: an
        absence naming the instrument. Keeping `[0]` here dropped every entry after the first, and
        turned an empty return into an `IndexError` the guard relabelled as the grader failing to
        load, which is a different and untrue statement about the subject.
        """
        payload = run_dir.partial().get(entry_id)
        if payload is not None:
            prior = [contracts.Entry.model_validate(item) for item in payload.get("entries", [])]
            stored_findings = [
                contracts.Finding.model_validate(item) for item in payload.get("findings", [])
            ]
            entries[section].extend(prior)
            findings.extend(stored_findings)
            run_dir.log(f"{utc_now()} instrument {entry_id} replayed {len(prior)} entries")
            return prior, stored_findings, True
        before = len(entries[section])
        try:
            produced = list(work())
        except BudgetExhausted as stopped:
            # A Seeker budget stop is interrupted execution, but its completed proposals are
            # already evidence. Seal them under the instrument id before the CLI emits RL0501.
            produced = list(getattr(stopped, "entries", ()))
            if produced:
                entries[section].extend(produced)
                seal(entry_id, produced)
            raise
        except Exception as failure:  # the audit catches it; the instrument does not swallow it
            refusal = absences.could_not_check(
                ctx,
                section=section,
                entry_id=entry_id,
                measurand=measurand,
                failure=failure,
                remedy=remedy,
                subject_ref=subject_ref,
                affected_claims=(f"no {section} claim from {entry_id}",),
            )
            if declared_method is not None:
                refusal.method = declared_method
            entries[section].append(refusal)
            produced = entries[section][before:]
            return produced, [], False
        if not produced:
            refusal = ctx.absence(
                section,
                entry_id,
                measurand,
                f"{entry_id} ran and returned no entry",
                remedy,
                (f"no {section} claim from {entry_id}",),
                subject_ref=subject_ref,
            )
            if declared_method is not None:
                refusal.method = declared_method
            entries[section].append(refusal)
            produced = entries[section][before:]
            return produced, [], False
        if declared_method is not None:
            for entry in produced:
                if (
                    entry.entry_id == entry_id
                    and entry.kind == "absence"
                    and entry.method.id == f"{section}.absence"
                ):
                    entry.method = declared_method
        entries[section].extend(produced)
        return produced, [], False

    openenv_entry = _openenv_step(
        project=project,
        corpus=corpus,
        joined_rollouts=joined_rollouts,
        joined_origin=joined_origin,
        ctx=ctx,
        subject_ref=subject_ref,
    )
    if openenv_entry is not None:
        entries["validity"].append(openenv_entry)
        seal("validity.openenv_step", [openenv_entry])

    # --- validity: the three source checks ------------------------------------------------------
    static_before = len(entries["validity"])
    static_findings_before = len(findings)
    if replayed("validity.static", "validity") is None:
        static_results = static_checks.run_static_checks(
            subject.source,
            source_name=source_name,
            entrypoint=subject.entrypoint,
            tasks=list(corpus.tasks) if has_bank else None,
        )
        for result in static_results:
            entries["validity"].append(
                _static_entry(
                    result,
                    ctx,
                    subject_ref=subject_ref,
                    source_name=source_name,
                    source_digest=source_digest,
                )
            )
            code = result.code
            if code:
                if code == "RL0201":
                    _explain_writable_then_read(
                        entries["validity"][-1], result, n_responses=n_responses
                    )
                findings.append(
                    _finding(
                        code,
                        entry_id=result.entry_id,
                        rule=result.rule,
                        message=result.message,
                        level="error" if code in {"RL0201", "RL0202"} else "warning",
                        kind="fail",
                        rationale=(
                            "a static reading of the grader's own source, with the line named; the "
                            "black-box arm did not run, so this says nothing about whether a sampled "
                            "response used it"
                        ),
                        arm="static",
                        file=source_name,
                        line=result.line,
                    )
                )
                rules[result.entry_id] = contracts.Rule(
                    id=result.entry_id,
                    name=result.entry_id.split(".", 1)[1].replace("_", " "),
                    short_description=result.predicate,
                )
        seal(
            "validity.static",
            entries["validity"][static_before:],
            findings[static_findings_before:],
        )

    try:
        vut.load()
    except Exception as failure:  # the hole is the audit's answer; a failed grader is not a crash
        grader_failure = failure
    if has_bank and grader_failure is None:
        from reward_lens.product.audit.response_scores import score_response_bank

        corpus.response_scores = score_response_bank(
            grader_path=subject.grader_path,
            entrypoint=subject.entrypoint,
            corpus=corpus,
            sandbox=sandbox,
            project_dir=subject.project_dir,
        )
    if injected_seeker:
        if grader_failure is not None:
            raise contracts.UsageError(
                code="RL0003",
                message="the declared reward grader could not load for the seeker proposal",
                remediation="repair the declared reward entry before proposing a candidate",
                context={"field": "project_path"},
            )
        if outcome_execution is None or not outcome_execution.qualified:
            raise contracts.UsageError(
                code="RL0003",
                message="the declared project outcome target is not qualified for a seeker proposal",
                remediation="configure and qualify the project outcome execution before proposing",
                context={"field": "task_id"},
            )
        if contracts.digest(injected_tasks[0]) not in outcome_execution.target_items_by_task_digest:
            raise contracts.UsageError(
                code="RL0003",
                message="the proposed task is not an exact declared project outcome target",
                remediation="propose only for one task in outcome.execution.target.tasks",
                context={"field": "task_id"},
            )
        ctx.seeker_actor = seeker_actor
        ctx.seeker_tasks = injected_tasks
        ctx.seeker_intent = seeker_intent or getattr(project.config, "success", "")
        ctx.seeker_max_proposals = 1
    elif seeker_config is not None and grader_failure is None:
        from reward_lens.instruments.seeker.config import build_actor
        from reward_lens.instruments.seeker.protocol import GraderScorer, scorer_for

        # A declared plain Python file is already an accepted adapter.  Passing the function
        # returned by ``VerifierUnderTest.load`` through connect drops its response argument;
        # retain the file-backed adapter and the actual Audit containment instead.
        reward_kind = getattr(
            getattr(getattr(project, "config", None), "reward", None), "kind", "plain"
        )
        if project is None or reward_kind == "plain":
            from reward_lens.graders.python_grader import PythonGrader

            scorer = GraderScorer(
                PythonGrader.from_path(subject.grader_path, subject.entrypoint),
                outcome_check=outcome_check,
                sandbox=sandbox,
                limits=Limits(network=False),
            )
        else:
            scorer = scorer_for(vut.load(), outcome_check=outcome_check, sandbox=sandbox)
        scorer.outcome_identity = _outcome_identity(outcome_execution, outcome_loader_error)
        transport = (
            _AuditHttpTransport(
                seeker_config.endpoint or "", credential_env=seeker_config.credential_env
            )
            if seeker_config.mode == "local"
            or (seeker_config.mode == "api" and not seeker_config.recorded_proposals)
            else None
        )
        ctx.seeker_actor = build_actor(
            seeker_config,
            scorer=scorer,
            max_budget_usd=getattr(req, "max_budget_usd", None),
            offline=ctx.offline,
            transport=transport,
        )
        ctx.seeker_tasks = tuple(corpus.tasks)
        ctx.seeker_intent = getattr(project.config, "success", None) if project is not None else ""

    # --- validity: the fourth static check, which executes ---------------------------------------
    input_was_replayed = "validity.input_handling" in run_dir.partial()
    input_handling_entries = guarded(
        "validity.input_handling",
        "validity",
        "what the grader returns on malformed, empty and oversized input",
        lambda: instruments.input_handling_entry(
            subject.grader_path,
            subject.entrypoint,
            corpus,
            ctx,
            subject_ref=subject_ref,
            trainer=subject.trainer,
            project_dir=subject.project_dir,
        ),
        "run the audit where the grader can be imported and called",
        defer_seal=True,
    )
    handling = entries["validity"][-1]
    for code in (
        (handling.result or {}).get("codes", [])
        if handling.result and not input_was_replayed
        else []
    ):
        if code not in FINDING_IDS:
            continue
        findings.append(
            _finding(
                code,
                entry_id="validity.input_handling",
                message="; ".join(handling.result.get("consequences", [])) or code,
                level="error",
                kind="fail",
                rationale=(
                    "the grader was called on an input a policy can emit and returned a value the "
                    "named trainer silently converts into a score (D-65)"
                ),
                arm="white_box",
                file=source_name,
                line=None,
            )
        )
        rules["validity.input_handling"] = contracts.Rule(
            id="validity.input_handling",
            name="input handling",
            short_description=(
                "the grader returns a real score, or a refusal the trainer cannot mistake for one"
            ),
        )
    # Its derived findings are part of the instrument result.  Seal after the classification so
    # a resume never needs to re-run this classification or append a second finding.
    if "validity.input_handling" not in run_dir.partial():
        related = [finding for finding in findings if "validity.input_handling" in finding.entries]
        seal("validity.input_handling", input_handling_entries, related)

    # --- validity: D10 replay, D1 coverage -------------------------------------------------------
    guarded(
        "validity.replay_determinism",
        "validity",
        "whether the same input scored twice yields the same score",
        lambda: instruments.replay_entry(vut, corpus, ctx, subject_ref=subject_ref),
        "run the audit where the grader can be imported and called",
    )
    guarded(
        "validity.decision_coverage",
        "validity",
        "the fraction of the grader's branch arcs the response bank has ever taken",
        lambda: (
            instruments.coverage_entry(vut, corpus, ctx, subject_ref=subject_ref)
            if has_bank
            else _no_bank_coverage(ctx, subject_ref)
        ),
        "run the audit where the grader can be imported and traced",
    )

    from reward_lens.product.audit import validity_checks

    if replayed("validity.controls", "validity") is None:
        validity_control_entries = validity_checks.run(
            ctx=ctx,
            subject_ref=subject_ref,
            vut=vut,
            corpus=corpus,
            project=project,
            tasks_path=subject.tasks_path,
            sandbox=sandbox,
            outcome_execution=outcome_execution,
            has_task_set=has_tasks,
            has_response_bank=has_bank,
            grader_failure=grader_failure,
        )
        entries["validity"].extend(validity_control_entries)
        seal("validity.controls", validity_control_entries)

    # --- what this build fills, asked rather than assumed -----------------------------------------
    # Discovery runs before the absences, because the absences must skip exactly the panels that are
    # about to fill themselves. A landed panel carrying both its entries and a "this build ships
    # none of it" absence is the one thing the record must never say.
    # --- every panel this build does not fill -----------------------------------------------------
    # Composition registers a `trace` panel even when it can only report that no TraceRecord was
    # supplied.  That partial consumer must not suppress the canonical Trace-input absence: the
    # record still needs `trace.instrument_absent` to name the missing run record and its remedy.
    # Once an immutable TraceRecord was actually opened, the normal panel rule applies and no
    # canonical Trace absence is written.
    absence_skip = (
        filled if trace is not None else tuple(section for section in filled if section != "trace")
    )
    for section, panel_entries in absences.panel_absences(
        ctx, subject_ref=subject_ref, skip=absence_skip, supplied=supplied
    ).items():
        entries[section].extend(panel_entries)

    # --- panels a wave-2 build has plugged in -----------------------------------------------------
    for panel in discovered:
        section = str(panel.section)
        if section not in entries:
            entries["validity"].append(
                ctx.absence(
                    "validity",
                    "validity.panel_section",
                    "the record section a discovered panel names",
                    f"a panel names the section {section!r}, which is not a record section",
                    "register the panel under one of: " + ", ".join(_SECTIONS),
                    (f"no claim from the {section} panel",),
                    subject_ref=subject_ref,
                    state="COULD_NOT_CHECK",
                )
            )
            continue
        for instrument in panel.instruments:
            entry_id = str(getattr(instrument, "id", f"{section}.instrument"))
            panel_before = len(entries[section])
            produced, stored_findings, was_replayed = guarded_many(
                entry_id,
                section,
                entry_id,
                lambda instrument=instrument: instrument.run(subject, corpus, ctx),
                "run a build in which this instrument imports",
                declared_method=getattr(instrument, "method", None),
            )
            if was_replayed:
                # The sealed payload is the execution result.  Re-running its harvesting code
                # would be a second execution under possibly different package code.
                continue
            if seeker_config is not None and entry_id == "exploits.seeker":
                _bind_seeker_config(produced, seeker_config)
            # 8.1: the panel's own findings, harvested here so that a discovered panel's judgement
            # reaches `Assay.findings`, the verdict's reasons and the text surface under its own
            # panel. The entries are the instrument's, not the runner's absence, and a panel that
            # raises nothing is carried unchanged.
            try:
                raised = _panel_findings(instrument, produced)
            except Exception as failure:  # a broken harvest is not a broken measurement
                entries[section].append(
                    absences.could_not_check(
                        ctx,
                        section=section,
                        entry_id=f"{entry_id}.findings",
                        measurand=f"the findings {entry_id} raises over its own entries",
                        failure=failure,
                        remedy="run a build in which this panel's findings function returns",
                        subject_ref=subject_ref,
                        affected_claims=(f"no {section} finding from {entry_id}",),
                    )
                )
                seal(entry_id, entries[section][panel_before:])
                continue
            loose = _unallocated_codes(raised)
            if loose:
                entries["validity"].append(
                    _code_allocation_check(
                        ctx, entry_id=entry_id, codes=loose, subject_ref=subject_ref
                    )
                )
            findings.extend(raised)
            seal(entry_id, entries[section][panel_before:], raised)
    for module, detail in registry.import_failures.items():
        entries["validity"].append(
            ctx.absence(
                "validity",
                "validity.panel_import",
                f"the panels {module} would have contributed",
                f"{module} would not import: {detail}",
                "install what that module needs, or remove it from the build",
                ("no claim from the panels that module holds",),
                subject_ref=subject_ref,
                state="COULD_NOT_CHECK",
            )
        )

    # --- actual project outcome qualification, or one explicit unqualified hole -----------------
    # A successful loader owns the sole canonical qualification entry.  A project that does not
    # declare the execution configuration remains unqualified; Audit never invents controls or
    # converts a host callback into an outcome result.
    from reward_lens.outcome import qualify as qualify_outcome

    outcome_qualification = (
        outcome_execution.supervised.qualification
        if outcome_execution is not None
        else qualify_outcome(
            check=None,
            suite=None,
            grader_owner=subject.name,
            grader_source_digest=source_digest,
            subject_ref=subject_ref,
            provenance=ctx.provenance(),
        )
    )
    entries["validity"].append(outcome_qualification.entry)
    outcome_entry_id = outcome_qualification.entry.entry_id
    rules[outcome_entry_id] = contracts.Rule(
        id=outcome_entry_id,
        name="outcome unqualified",
        short_description="the independent outcome check has not been qualified",
    )

    # The capability floor has no caller-supplied outcome or threshold path.  It exists only when
    # the source-bound project declaration supplied fixed candidates, and every candidate is run
    # through the same independently qualified supervisor used for public Seeker targets.
    from reward_lens.product.compare.scorecard import capability_floor_assay_entry

    if outcome_execution is not None and outcome_execution.capability_floor is not None:
        floor_execution = outcome_execution.capability_floor
        try:
            resolutions = outcome_execution.resolve_capability_floor()
            entries["soundness"].append(
                capability_floor_assay_entry(
                    project=project.config,
                    project_root=project.root,
                    partition=floor_execution.partition,
                    outcomes=tuple(resolution.outcome for resolution in resolutions),
                    protocol=outcome_execution.supervised.protocol,
                    floor=floor_execution.floor,
                    subject_ref=subject_ref,
                    provenance=outcome_execution.supervised.protocol.provenance,
                    scope="evaluator_comparison",
                )
            )
        except Exception as error:
            entries["soundness"].append(
                ctx.absence(
                    "soundness",
                    "soundness.capability_floor",
                    "whether the declared held-out legitimate-solution success floor is met",
                    f"declared capability-floor execution could not complete: {type(error).__name__}: {error}",
                    "repair the declared capability-floor candidates or independent outcome execution",
                    ("held-out capability floor",),
                    subject_ref=subject_ref,
                    state="COULD_NOT_CHECK",
                    depends_on=("digest:outcome_protocol",),
                )
            )
    else:
        missing_floor = ctx.absence(
            "soundness",
            "soundness.capability_floor",
            "whether the declared held-out legitimate-solution success floor is met",
            "no source-declared held-out capability floor candidates are available",
            "declare outcome.held_out_capability.floor and matching outcome.execution.capability_floor candidates",
            ("held-out capability floor",),
            subject_ref=subject_ref,
            state="NOT_MEASURED",
            depends_on=("digest:outcome_protocol",),
        )
        missing_floor.method = contracts.Method(
            id="soundness.capability_floor",
            version="1.0.0",
            params_digest=contracts.digest({"status": "source_declaration_missing"}),
            procedure="read the source-declared held-out capability floor candidates before outcome execution",
        )
        entries["soundness"].append(missing_floor)

    if trace is not None:
        from reward_lens.product.trace.answer import answer_entry

        # Materialise this closed record once.  The entries become saved Assay evidence, and the
        # derived fiveway witness must resolve its digest-bound references against these exact
        # instances rather than a second timestamped reconstruction.
        try:
            base_entries = tuple(trace.entries())
            entries.setdefault("trace", []).extend(base_entries)
            # `trace.entries()` can include typed absences.  Stamp their stable transcript
            # metadata before taking the fiveway digest references; the final all-entry pass below
            # is idempotent, but doing it only after this point would mutate cited bytes.
            absences.stamp_codes(entries)
            evidence_entries = tuple(
                entry
                for group in entries.values()
                for entry in group
                if entry.entry_id != "trace.fiveway"
            )
            entries.setdefault("trace", []).append(
                answer_entry(
                    trace,
                    evidence_entries=evidence_entries,
                    trace_entries=base_entries,
                    provenance=ctx.provenance(),
                )
            )
        except ValueError:
            # Third-party reader records predating the trace-entry contract can still be audited;
            # their trace evidence remains in the normal panel entries, while the new derived
            # fiveway witness is withheld until its own retained trace entries are serializable.
            pass

    # Every absence gets its code and its short remedy here, once every entry is in and every
    # `needs` note with it (A-025 point 3).
    absences.stamp_codes(entries)
    record = _assemble(
        req=req,
        subject=subject,
        project=project,
        ctx=ctx,
        started=started,
        started_monotonic=started_monotonic,
        entries=entries,
        findings=findings,
        rules=rules,
        source_digest=source_digest,
        corpus=corpus,
        filled=filled,
        outcome_qualification=outcome_qualification,
        precommitted_project_subject_ref=precommitted_project_subject_ref,
    )
    if trace is not None:
        # The join is part of the statement before `_write` seals its digest.  Its producer reads
        # the saved exploit witnesses and this one closed canonical trace; it does not run an
        # instrument or manufacture row-level telemetry.
        from reward_lens.product.join import attach_join_evidence

        attach_join_evidence(
            record,
            trace_record=trace,
            audited_grader_source_digest=source_digest,
        )
    # `_assemble` stops at the statement; `_write` seals it, because the seal needs the digest of
    # the manifest and the manifest needs the bytes of the record. The sealed record is the one
    # returned, so the caller holds what is on disk and what the page carries.
    written_record = _write(record, subject=subject, project=project, req=req, run_dir=run_dir)
    _ACTIVE_RUN.set(None)
    return written_record


def _no_bank_coverage(ctx: RunContext, subject_ref: str) -> contracts.Entry:
    """D1 with no response bank is an absence, never a fraction over the audit's own probes."""
    return ctx.absence(
        "validity",
        "validity.decision_coverage",
        "the fraction of the grader's branch arcs the response bank has ever taken",
        "no response bank was supplied, so no corpus has exercised anything",
        "pass sampled responses with `--responses <file>`; D1 measures what a corpus exercised",
        ("no claim about which of the grader's decisions have ever been exercised",),
        subject_ref=subject_ref,
        depends_on=("digest:source", "digest:samples"),
    )


def _assemble(
    *,
    req: Any,
    subject: AuditSubject,
    project: Any,
    ctx: RunContext,
    started: str,
    started_monotonic: float,
    entries: dict[str, list[contracts.Entry]],
    findings: list[contracts.Finding],
    rules: dict[str, contracts.Rule],
    source_digest: str,
    corpus: instruments.Corpus,
    filled: tuple[str, ...] = (),
    outcome_qualification: Any = None,
    precommitted_project_subject_ref: str | None = None,
) -> contracts.Assay:
    actor_record = None
    actor = getattr(ctx, "seeker_actor", None)
    if actor is not None:
        actor_record = actor.record()
    # This is the same identity the store derives from the sealed record.  Omitting versions
    # here made the preliminary project version differ from `version_of(record)`, which meant a
    # correctly precommitted supervisor could be relabelled by an unrelated absence entry.
    method_ids = tuple(
        sorted(
            {
                f"{entry.method.id}@{entry.method.version}"
                if entry.method.version
                else entry.method.id
                for group in entries.values()
                for entry in group
            }
        )
    )
    if project is not None:
        version = project.version(methods=method_ids)
        version_ref = version.to_version_ref()
        digests = version.to_subject_digests()
        system_id = _ident(version.id)
        system_name = _system_name(getattr(project.config, "name", None), project.root)
    else:
        digests = contracts.models.Digests(
            source=source_digest,
            environment=None,
            scorer_config=None,
            task_distribution=None,
            samples=None,
            outcome_protocol=None,
            policy=None,
            training_semantics=None,
            instrument_method=contracts.digest({"methods": list(method_ids)}),
        )
        version_ref = contracts.models.VersionRef(
            id="unversioned",
            digest=contracts.digest({"digests": digests.to_dict(), "parents": []}),
            parents=[],
            declared_change=None,
        )
        system_id = _ident(subject.name)
        system_name = _clean(subject.grader_path.name) or DEFAULT_SYSTEM_NAME
    holes_count = sum(1 for group in entries.values() for entry in group if entry.kind == "absence")
    record = contracts.Assay(
        schema_url=contracts.SCHEMA_URL,
        assay_id="sha256:" + "0" * 64,
        created=started,
        producer=contracts.Producer(
            tool="reward-lens",
            version=_tool_version(),
            method_set=contracts.digest({"methods": list(method_ids)}),
        ),
        subject=contracts.Subject(
            reward_system=contracts.models.RewardSystemRef(id=system_id, name=system_name),
            version=version_ref,
            context=contracts.models.ContextRef(
                policy=None,
                # A-019: the four wordings the two transcripts choose between are keyed on this
                # field being null, and never on whether the audited path is a file or a
                # directory. So a run that was given a task set names it here, and a bare grader
                # leaves it null, which is the difference the wordings are asking about.
                # The field is a digest, so the task set is named by the one the record already
                # carries for it: null exactly when no task set was supplied.
                task_set=digests.task_distribution
                if getattr(subject, "tasks_path", None) is not None and subject.tasks_path.is_file()
                else None,
                configuration=corpus.origin[:200] or None,
            ),
            digests=digests,
        ),
        intent=contracts.Intent(
            success=(getattr(project.config, "success", None) if project is not None else None),
            constraints=list(getattr(project.config, "constraints", []) or [])
            if project is not None
            else [],
            non_equivalent_pairs=[],
            outcome_check=contracts.models.OutcomeCheck(
                state=(
                    "qualified"
                    if getattr(outcome_qualification, "qualified", False)
                    else "unqualified"
                ),
                kind=_outcome_kind(project),
                # The contract field identifies a supplied outcome check by its content digest.
                # An entry provenance records run mechanics, not check identity; with no check
                # supplied the honest value is therefore absent.
                provenance=None,
                measured_disagreement=None,
            ),
            attack_budget=contracts.models.AttackBudget(
                seeker_calls=0 if actor_record is None else actor_record.calls,
                usd="0.00" if actor_record is None else actor_record.spent_usd,
            ),
            partitions=[],
        ),
        measurement=contracts.models.Measurement(**entries),
        holes=[],
        rules=sorted(rules.values(), key=lambda rule: rule.id),
        findings=findings,
        join=[],
        experiments=[],
        calibration_links=[],
        tables=[],
        decision=contracts.Decision(
            state="unresolved",
            # The action is the use the project declares for this version, and the verdict is
            # unresolved *for that action*: a decision with no action names no use, and a reader
            # then has a verdict about nothing. A project that declares a trainer is going to
            # train on this version, which is what the config exposes; a bare grader declares no
            # use, so there is none to name and the field stays null.
            action=_declared_action(project),
            policy=None,
            reasons=_reasons(
                entries,
                findings,
                outcome_kind=_outcome_kind(project),
                outcome_qualified=bool(getattr(outcome_qualification, "qualified", False)),
            ),
            tradeoff=None,
            signature=None,
            regression_cases=[],
        ),
        cost=contracts.Cost(
            wall_s=round(max(0.0, time.monotonic() - started_monotonic), 3),
            cpu_s=0.0,
            usd="0.00" if actor_record is None else actor_record.spent_usd,
            api_calls=(
                actor_record.calls
                if actor_record is not None and actor_record.actor == "api"
                else 0
            ),
        ),
        embedding=contracts.Embedding(tier="A", omitted_tables=[]),
        attestation=contracts.Attestation(statement_digest=None, backend=None),
        provenance=contracts.Provenance(
            offline=ctx.offline,
            sandbox_tier=ctx.tier(),
            os=_os_name(),
            seed=int(getattr(req, "seed", 20260911)),
            reproduce=[f"reward-lens audit {req.path} --seed {getattr(req, 'seed', 20260911)}"],
        ),
        environment_excluded_from_digest={
            "python": platform.python_version(),
            "platform": sys.platform,
            "holes": holes_count,
        },
    )
    record.holes_from_entries()
    # The two sentences and the plan the run went in under, at the record's top level (A-016).
    # They are composed here, after every entry is in, because both are statements about the whole
    # record: a sentence built while the entries were still arriving would be about a record that
    # never existed. `paid_calls` is the planner's zero for the same reason it is zero there: the
    # seeker is the only arm that costs money and this build does not ship it (D-41).
    record.extensions = dict(record.extensions or {})
    record.extensions[absences.TRANSCRIPT_EXTENSION] = {
        "could_establish": absences.could_establish(record),
        "could_not": absences.could_not(record),
        "subject": _subject_clauses(subject=subject, project=project, corpus=corpus),
        "plan": {
            "panels": len(filled),
            "paid_calls": (
                actor_record.calls
                if actor_record is not None and actor_record.actor == "api"
                else 0
            ),
            "estimate_s": _estimate_s(filled),
        },
    }
    seeker_config = getattr(ctx, "seeker_config", None)
    if seeker_config is not None:
        record.extensions["https://reward-lens.dev/extensions/seeker"] = {
            "config": seeker_config.public(),
            "actor_record": None if actor_record is None else actor_record.to_dict(),
        }
    openenv_route = getattr(ctx, "openenv_route", None)
    if openenv_route is not None:
        record.extensions[OPENENV_EXTENSION] = dict(openenv_route)
    if project is not None:
        # The project's `instrument_method` is derived from the record's own entries, so the
        # subject is settled from the assembled record rather than from a guess at the method set.
        settled = project.version_of(record)
        if (
            precommitted_project_subject_ref is not None
            and settled.digest() != precommitted_project_subject_ref
        ):
            raise contracts.UsageError(
                code="RL0003",
                message="the final Audit method set differs from the subject bound to outcome execution",
                remediation="refresh the declared outcome execution subject_ref for this Audit method set",
                context={
                    "field": "outcome.execution.subject_ref",
                    "expected_subject_ref": precommitted_project_subject_ref,
                    "final_subject_ref": settled.digest(),
                },
            )
        record.subject.version = settled.to_version_ref()
        record.subject.digests = settled.to_subject_digests()
    return record


def _subject_clauses(*, subject: AuditSubject, project: Any, corpus: instruments.Corpus) -> dict:
    """The Subject line's clauses the schema's closed `subject` cannot carry (A-019).

    Two shapes, and which one is written is read off the record's own situation rather than off the
    path the reader typed: a project names the grader file and the counts it measured over, and a
    bare grader names that same file as the user gave it, the signature it declares and the shape
    that signature is in. Both name the file (A-022 corrects A-019's "module" here): the module
    name is an import detail the reader never typed, and `my_grader.py` is what they can point at
    again.
    A directory audited without a task set is a bare grader for this purpose, which is the point of
    keying on the record.
    """
    if project is not None:
        clauses = {
            "grader": subject.grader_path.name,
            "tasks": int(getattr(corpus, "n_tasks", 0) or 0),
            "responses": len(corpus.rollouts),
        }
        # The task set's file name, which the record's own `subject` cannot carry: its `task_set`
        # is a digest, and a digest does not tell a reader which file to open again. A project with
        # no task set names none, because there is no file to name (A-025 point 4).
        tasks_path = getattr(subject, "tasks_path", None)
        if tasks_path is not None:
            clauses["task_file"] = tasks_path.name
        return clauses
    # A bare grader declares nothing, so the shape is the one the audit had to assume to call it
    # at all: a plain number back from a plain call. The clause says it so a reader can contradict
    # it, which is the only way an assumption in a record is worth carrying.
    shape = "plain"
    return {
        "grader": subject.grader_path.name,
        "signature": instruments.declared_signature(subject.source, subject.entrypoint)
        or f"{subject.entrypoint}(...)",
        "shape": f"{shape} shape",
    }


def _seal(record: contracts.Assay, *, bundle_manifest_digest: str | None) -> contracts.Assay:
    """D-80: plan the embedding on the statement, then take the id over the planned record.

    The record declares its own embedding tier before it is sealed, so the id is the digest of a
    record that already says how it will be carried, and nothing downstream may edit a sealed
    record. That is why the plan is taken here and not by the renderer, and it is also why the
    bundle manifest's digest has to be in hand this early: the page carries that digest in a
    `<meta>`, so the digest changes the page's length, and the length is `embedding.html_bytes`,
    which is sealed into the record and which the renderer weighs the page against.

    Sealing is idempotent. `plan_embedding` ignores an `embedding` already on the record and
    recomputes it, and `assay_id` is excluded from its own digest, so `_write` may call this more
    than once on the same object while the manifest digest settles.
    """
    from reward_lens.render.report import plan_embedding

    record.embedding = contracts.Embedding.model_validate(
        plan_embedding(record, bundle_manifest_digest=bundle_manifest_digest).to_dict()
    )
    record.assay_id = contracts.digest(record)
    return record


def _all_absent(entries: dict[str, list[contracts.Entry]], section: str) -> bool:
    group = entries.get(section) or []
    return bool(group) and all(entry.kind == "absence" for entry in group)


#: The order the report prints the dark sections in, and so the order the decision's
#: `required_missing:` reasons take (A-025 point 1). It is not the record's own field order: the
#: report puts signal before reward_statistics, and the reasons follow the report, because a reader
#: checks this list against the sections they have just read past. A section the record gains and
#: this tuple does not name still gets its reason, appended in record order, so nothing can fall
#: out of the list by being forgotten here.
_REASON_SECTIONS: tuple[str, ...] = (
    "soundness",
    "exploits",
    "framing",
    "signal",
    "reward_statistics",
    "trace",
    "forecast",
    "calibration",
)

#: What `outcome_unqualified:` names when the subject declares no independent check at all. The
#: reason names the protocol that was not qualified, and a bare grader declares no protocol: the
#: slot says so rather than borrowing a kind the subject never claimed.
_NO_OUTCOME_DECLARED = "none_declared"


def _reason_order() -> tuple[str, ...]:
    """The printed order, then any section the record has that the printed order does not name."""
    return _REASON_SECTIONS + tuple(
        section for section in _SECTIONS if section not in _REASON_SECTIONS
    )


def _outcome_kind(project: Any) -> str | None:
    """The kind of independent check the subject declares, or None when it declares none."""
    outcome = getattr(getattr(project, "config", None), "outcome", None)
    kind = getattr(outcome, "kind", None)
    return str(kind) if kind else None


def _reasons(
    entries: dict[str, list[contracts.Entry]],
    findings: list[contracts.Finding],
    *,
    outcome_kind: str | None = None,
    outcome_qualified: bool = False,
) -> list[str]:
    """Why the decision is what it is, blocking findings first (section 7.3's vocabulary).

    A decision that named only what was missing left out what was found: the demo exits non-zero on
    a finding at level error, and a reader given only `outcome_unqualified` and `required_missing`
    could not tell which finding did it. One error-level finding is one reason. Naming it twice,
    once by rule and once by finding id, put the same finding in the list twice and made the report
    count two where it prints `1 blocking finding`: the text layout prints the count and not the
    name, so a second name bought nothing and cost the count. The name kept is the finding id
    (`RGX-local-0001`), because the envelope golden reads `blocking_finding:RGX-local-0001` and the
    golden is the specification. The rule is the stabler name across records and reads better, and
    it is still one field away in the record's own findings, which carry both; the record the
    golden pins is the one this has to be.

    `outcome_unqualified:` is its own reason kind and is never a blocking finding: it says the
    outcome check was not qualified, which is a limit of the run and not a defect of the grader.
    What it names is the protocol that was not qualified, the kind the subject declares
    (`protected_test_suite` for the demo), and not the id of the qualification entry: the entry id
    is where the record puts the hole, and a reader who sees its explicit unqualified reason knows
    which protocol would have settled the verdict.

    The order is the golden's (A-025 point 1): the blocking findings, then the unqualified
    outcome, then the sections that could not run, in the order the report prints them. The
    outcome comes before the sections because it is about the verdict and they are about the
    measurement, and a reader stops at the first line that explains the state.

    Every dark section the policy requires is one reason, and no section is waived. Calibration
    was held out of the count on the reading that no reader could fetch a certified reference
    material for this substrate, so naming it would send the reader after something that does not
    exist. A-023 withdrew that: a required panel the run could not fill costs the verdict whether
    or not the input is obtainable, and a reason a reader cannot act on is still why the decision
    is unresolved. The remedy row for calibration already says that nothing here can be calibrated
    until such a material is published, which is where that caveat belongs. The count is measured,
    not curated: the demo leaves eight dark and the transcript prints eight.
    """
    reasons: list[str] = []
    for finding in findings:
        if finding.level != "error":
            continue
        name = finding.id or finding.rule
        reason = f"blocking_finding:{name}"
        if name and reason not in reasons:
            reasons.append(reason)
    if not outcome_qualified:
        reasons.append(f"outcome_unqualified:{outcome_kind or _NO_OUTCOME_DECLARED}")
    reasons += [
        f"required_missing:{section}"
        for section in _reason_order()
        if _all_absent(entries, section)
    ]
    return reasons


def _declared_action(project: Any) -> str | None:
    """The use the project declares for the version, or null when it declares none."""
    reward = getattr(getattr(project, "config", None), "reward", None)
    return "train_on_this_version" if getattr(reward, "trainer", None) else None


def _tool_version() -> str:
    try:
        from importlib.metadata import version

        return version("reward-lens")
    except Exception:  # pragma: no cover - an editable tree without metadata
        from reward_lens import __version__

        return __version__


def record_bytes(record: contracts.Assay) -> bytes:
    """The one serialisation of a record on disk: RFC 8785 JCS over the whole record.

    `canonical_bytes` is the *digest* serialisation and strips the four fields a record cannot be
    hashed over, its own `assay_id` among them. Writing those bytes to a file, as the projectless
    branch of `_write` did, wrote a record that did not carry its own identifier and differed field
    for field from the record the SDK returned and from the record the store writes. The store's
    serialisation is the one both branches use here, so the bytes on disk are the returned record.
    """
    import rfc8785

    return rfc8785.dumps(record.to_dict())


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """The one serialisation of a manifest on disk, the record's own: RFC 8785 JCS.

    The page's claim is `contracts.digest` of this mapping, and a reader checks it by loading the
    file and digesting what comes back. JCS makes that hold for any JSON encoding of the same
    mapping, but writing the canonical bytes means the file a reader hashes and the bytes the
    digest was taken over are the same string, with nothing in between to argue about.
    """
    import rfc8785

    return rfc8785.dumps(dict(manifest))


def _manifest(*, record_name: str, payload: bytes, assay_id: str) -> dict[str, Any]:
    """What the bundle is, as the page's `<meta name="bundle-manifest-digest">` will claim it.

    D-16, and section 7's `bundle_manifest_digest`: the page names the bundle it was rendered
    from, so a reader holding a page and a directory can tell whether the two belong together.
    Paths are relative to the manifest, which sits beside them, so a bundle that is copied or
    moved whole still checks. Every artifact this pipeline writes is named here except three: the
    report, which carries this manifest's digest and so cannot also be digested by it; the
    manifest, which cannot name itself; and the store's `index.json`, which is the project's
    rather than this bundle's, spans every assay the project holds and is rewritten by the next
    audit, so naming it would make the manifest stop describing what is on disk as soon as one
    more audit ran.

    `assay_id` is nested under `assay` rather than written at the top level, and the nesting is
    load-bearing. `contracts.digest` removes `EXCLUDED_FROM_DIGEST` before hashing and
    `assay_id` is the first name on that list, so a manifest that put the id at the top level
    would leave the one field a reader most wants pinned outside the manifest's own digest.
    """
    return {
        "schema": "reward-lens/bundle-manifest/1",
        "assay": {"id": assay_id},
        "tool": {"name": "reward-lens", "version": _tool_version()},
        "artifacts": [
            {
                "path": record_name,
                "role": "record",
                "media_type": "application/json",
                "bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        ],
    }


def _name_stem(record: contracts.Assay, subject: AuditSubject) -> str:
    """What a record's name starts at: the version it measures, named as the project declares it.

    A-015: the stem is the subject version's id, not the reward system's name and not the
    directory. A project that declares `version.id: code-reward-v1` gets records called that
    whichever directory it was opened from, which is what makes two copies of one project produce
    the same name rather than two names that agree about nothing. A bare grader has no version to
    declare, so it is named for its file with the suffix dropped: `my_grader.py` is the reward
    system, `my_grader` is the stem of the bundle.
    """
    version_id = str(record.subject.version.id or "")
    if version_id and version_id != "unversioned":
        return _ident(version_id)
    return _ident(subject.grader_path.stem)


def _measured(record: contracts.Assay) -> tuple[str, ...]:
    """Every entry's method, kind, absence code and reserved-slot bit, sorted.

    The same key the store's `_check_not_a_rewrite` reads, in the model's spelling rather than the
    dict's, because the name and the refusal have to agree: the name is what keeps two records
    that the store would otherwise see as one out of one file.
    """

    def key(entry: Any) -> str:
        missing_code = ""
        if entry.kind == "absence":
            transcript = entry.extensions.get("https://reward-lens.github.io/x/transcript", {})
            if isinstance(transcript, Mapping) and isinstance(transcript.get("missing_code"), str):
                missing_code = transcript["missing_code"]
        reserved = "1" if _unbuilt_placeholder(entry) else "0"
        return f"{entry.entry_id}@{method_identity(entry.method)}@{entry.kind}@{missing_code}@{reserved}"

    return tuple(sorted(key(entry) for entry in record.entries()))


def _derived_name(
    record: contracts.Assay, *, subject: AuditSubject, stamp: str, project: Any = None
) -> str:
    """`<version id>-<date>`, and the version's short digest after it only on a real collision.

    The date alone was not a name. Two audits of one subject on one day want the same file, and
    the store, which never rewrites a record, refused the second (RL0620) after the instruments
    had already run. Two things answer that now, and the suffix is the second of them. The
    unchanged case never reaches here: `_standing_record` hands the stored record back before any
    instrument runs. What is left is the changed case, where two records of one subject on one day
    genuinely differ, and only there does the name carry what tells them apart.

    So the suffix is conditional, and the condition is asked of the store rather than assumed: the
    name is plain unless the store already holds it for a different subject version. An
    unconditional suffix put eight hex on every bundle a reader ever sees, for a collision almost
    no project has. The store has no name allocator to ask -- `Project` offers `path_for`, `names`
    and `record`, and none of them proposes a next name -- so this is the pipeline's rule. The
    digest is truncated because the name is a label, not the identity; the whole digest is inside
    the record.
    """
    name = f"{_name_stem(record, subject)}-{stamp}"
    if project is None:
        return name
    try:
        if name not in project.names():
            return name
        standing = project.open_record(name)
    except (OSError, ValueError, contracts.RewardLensError):
        # A name the store cannot read as a record is not evidence of a collision, and the write
        # path already steps over such a file rather than raising on it.
        return name
    # A-016: the suffix comes from the digest that differs. The version digest when the version
    # moved, and the instrument-method digest when it did not, which is the wider-plan case: the
    # subject is the same and what was measured of it is not, so the name has to carry the second
    # or the two records want one file. What "the instruments differ" means here is the set of
    # entry ids, for the reason the store's rewrite rule reads the same thing: every absence in a
    # section shares one placeholder method id, so the method digest does not move when a panel
    # lands and finds nothing, and the two records would then want one name.
    if standing.subject.version.digest != record.subject.version.digest:
        differing = record.subject.version.digest
    elif _measured(standing) != _measured(record):
        differing = contracts.digest({"entries": list(_measured(record))})
    else:
        return name
    short = differing.rsplit(":", 1)[-1][:NAME_DIGEST_CHARS]
    return f"{name}-{short}"


def persist_assay_bundle(
    record: contracts.Assay,
    *,
    project: Any,
    name: str,
    fallback_dir: Path,
) -> tuple[contracts.Assay, dict[str, Path]]:
    """Seal an already assembled assay and write its record, manifest, and report.

    This owns only destination and bundle semantics. Callers provide a complete ``Assay`` and an
    explicit name, so it cannot inspect a grader, infer a subject, or manufacture a measurement.
    The fixed point is the existing record/manifest/report relation: the record embeds the
    manifest digest and the manifest names the final record bytes.
    """
    from reward_lens.render.report import render_to

    record_name = f"{name}.assay.json"

    manifest: dict[str, Any] = {}
    digest: str | None = None
    for _ in range(MANIFEST_PASSES):
        _seal(record, bundle_manifest_digest=digest)
        manifest = _manifest(
            record_name=record_name, payload=record_bytes(record), assay_id=record.assay_id
        )
        settled = contracts.digest(manifest)
        if settled == digest:
            break
        digest = settled
    else:  # pragma: no cover - the fixed point is reached on the third pass
        raise RuntimeError(
            f"the bundle manifest digest did not settle in {MANIFEST_PASSES} passes; the last"
            f" was {digest}"
        )

    if project is not None:
        written = project.record(record, name=name)
    else:
        fallback_dir.mkdir(parents=True, exist_ok=True)
        written = fallback_dir / record_name
        written.write_bytes(record_bytes(record))

    named = manifest["artifacts"][0]
    on_disk = "sha256:" + hashlib.sha256(written.read_bytes()).hexdigest()
    if written.name != named["path"] or on_disk != named["sha256"]:
        raise RuntimeError(
            f"the manifest names {named['path']} at {named['sha256']}, and the record was written"
            f" to {written.name} as {on_disk}: the store's serialisation is no longer the one"
            f" `record_bytes` predicts, so the manifest would name bytes that are not on disk"
        )
    written.with_name(MANIFEST_NAME).write_bytes(manifest_bytes(manifest))
    render_to(record, written.with_name(f"{name}.assay.html"), bundle_manifest_digest=digest)
    return record, {
        "assay": written,
        "manifest": written.with_name(MANIFEST_NAME),
        "report": written.with_name(f"{name}.assay.html"),
    }


def _write(
    record: contracts.Assay,
    *,
    subject: AuditSubject,
    project: Any,
    req: Any,
    run_dir: RunDir | None = None,
) -> contracts.Assay:
    """Persist Audit's assembled record through the destination-only bundle helper."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    name = getattr(req, "name", None) or _derived_name(
        record, subject=subject, stamp=stamp, project=project
    )
    policy = getattr(req, "policy", None)
    if policy is not None:
        # A policy is part of the same canonical record that the caller asked Audit to make.
        # The acceptance finalizer applies the immutable policy before its sole bundle writer
        # seals the record, then reopens the saved JSON.  Writing an undecided record first and
        # replacing it later would violate the store's no-rewrite rule and leave the report bound
        # to different bytes than the saved decision.
        from reward_lens.acceptance import apply_and_persist

        persisted = apply_and_persist(
            record,
            policy,
            project=project,
            name=name,
            fallback_dir=subject.grader_path.parent / "assays",
        )
        if run_dir is not None:
            run_dir.finish(record_path=persisted.paths["assay"])
        return persisted.record
    written_record, paths = persist_assay_bundle(
        record,
        project=project,
        name=name,
        fallback_dir=subject.grader_path.parent / "assays",
    )
    if run_dir is not None:
        run_dir.finish(record_path=paths["assay"])
    return written_record
