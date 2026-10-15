"""SDK engine for Compare's persisted public request contract."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import sys
import time
from dataclasses import fields, is_dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from reward_lens import contracts
from reward_lens.api._record import absent_record
from reward_lens.bank import load_bank
from reward_lens.bank.select import require_bank
from reward_lens.contracts import UsageError
from reward_lens.learning.contracts import TrainerUnsupported
from reward_lens.product.audit.run import persist_assay_bundle

from .registry import CompareInputs, CompareResult, Rung, get

_BANK_EXTENSION = "https://reward-lens.github.io/extensions/compare-bank/1"
_REPLAY_EXTENSION = "https://reward-lens.github.io/extensions/compare-replay/1"
_EXECUTION_EXTENSION = "https://reward-lens.github.io/extensions/compare-execution/1"
_RUNG_METHODS = {
    Rung.RE_SCORE: "reward_statistics.evaluator_rescore@1.0.0",
    Rung.SELECTION_STRESS: "reward_statistics.selection_stress@1.0.0",
    Rung.ADVANTAGE_REPLAY: "reward_statistics.advantage_replay@1.0.0",
    Rung.SAME_BATCH_SWAP: "reward_statistics.controlled_update_response@1.0.0",
    Rung.SHORT_FORK: "reward_statistics.short_fork_response@3.1.0",
    Rung.TRANSFER: "reward_statistics.repair_transfer@3.1.0",
}


def _requested_rungs(request: Any) -> tuple[Rung, ...]:
    """The complete requested rung set, defaulting only when the request is empty."""
    return tuple(Rung(number) for number in (request.rungs or (Rung.RE_SCORE.value,)))


def _authoritative_factory_subject(
    *, source: Path, project: Any, rungs: tuple[Rung, ...]
) -> contracts.Subject:
    """Derive the subject a factory must bind before it executes.

    A factory retains its subject in its own fixture and result.  The public
    record must therefore know the same subject before the factory runs, rather
    than deriving a second value and relabelling the produced entry later.  The
    planned method set is exact for each currently public Compare rung; future
    producer rungs add their declared method here before using this boundary.
    """
    provisional = absent_record(command="compare", reproduce=[], path=source)
    if project is None:
        return provisional.subject

    from reward_lens.store.digests import method_set

    methods = set(method_set(provisional.to_dict()))
    measured_methods = {_RUNG_METHODS[rung] for rung in rungs if rung in _RUNG_METHODS}
    if measured_methods:
        absence_method = provisional.measurement.reward_statistics[0].method
        absence_name = f"{absence_method.id}@{absence_method.version}"
        methods.discard(absence_name)
        methods.update(measured_methods)
    version = project.version(methods=tuple(sorted(methods)))
    digests = version.to_subject_digests()
    return contracts.Subject(
        reward_system=contracts.models.RewardSystemRef(
            id=str(version.id),
            name=str(getattr(project.config, "name", None) or version.id),
        ),
        version=version.to_version_ref(),
        context=contracts.models.ContextRef(
            policy=None,
            task_set=digests.task_distribution,
            configuration=None,
        ),
        digests=digests,
    )


def _source_digest(path: Path) -> str:
    """Return the stable source-byte identity used for one consumed input.

    Directory names participate because a closed-reader can attach meaning to a
    particular filename.  This is deliberately the same byte-tree spelling as
    the product's established audit provenance, while remaining local to the
    Compare dispatcher rather than coupling its public contract to Audit internals.
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


def _canonical_replay_value(value: Any) -> Any:
    """Make strict-reader replay inputs JSON-ready without dropping their values."""
    if is_dataclass(value):
        return {
            field.name: _canonical_replay_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_replay_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_replay_value(item) for item in value]
    if isinstance(value, (Decimal, Path)):
        return str(value)
    if isinstance(value, Enum):
        return _canonical_replay_value(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot seal replay evidence of type {type(value).__name__}")


def _replayed_run_evidence(run_record: Any) -> dict[str, Any]:
    """The complete strict-reader facts Rung 3 uses to replay pressure."""
    return {
        "reader": run_record.reader,
        "run_id": run_record.run_id,
        "complete": run_record.complete,
        "rows": _canonical_replay_value(run_record.rows),
        "components": _canonical_replay_value(run_record.components),
        "group_estimators": _canonical_replay_value(run_record.metadata.group_estimators),
        "depends_on": _canonical_replay_value(run_record.depends_on),
    }


def _replay_run_identity(run_dir: Path, run_record: Any) -> dict[str, Any]:
    """Bind both the closed source tree and the parsed evidence Rung 3 replayed."""
    source_digest = _source_digest(run_dir)
    return {
        "source_path": str(run_dir),
        "source_digest": source_digest,
        "source_tree_digest": source_digest,
        "reader": run_record.reader,
        "run_id": run_record.run_id,
        "parsed_evidence_digest": contracts.digest(_replayed_run_evidence(run_record)),
    }


def _replay_version(request: Any, field: str):
    """Read one explicit, finite replay-version declaration for Rung 3."""
    from reward_lens.contracts import UsageError
    from reward_lens.product.compare.rung3 import ReplayVersion

    path = getattr(request, field)
    if path is None:
        raise UsageError(
            code="RL0131",
            message=f"compare rung 3 requires --{field.replace('_', '-')} with explicit weights",
            remediation="write a JSON replay version with id, weights, and optional decimal offset",
            context={"field": field},
        )
    try:
        source_path = Path(path)
        source_bytes = source_path.read_bytes()
        raw = json.loads(source_bytes)
        weights = raw["weights"]
        if not isinstance(raw, dict) or not isinstance(weights, dict):
            raise TypeError("the declaration must be an object with a weights object")
        version = ReplayVersion(
            id=raw["id"],
            weights={name: Decimal(str(weight)) for name, weight in weights.items()},
            offset=Decimal(str(raw.get("offset", "0"))),
        )
    except (OSError, ValueError, KeyError, TypeError, InvalidOperation) as error:
        raise UsageError(
            code="RL0131",
            message=f"compare rung 3 could not read a valid replay version from {path}: {error}",
            remediation="supply an existing JSON file with id, complete decimal weights, and optional offset",
            context={"field": field, "path": str(path)},
        ) from error
    expected = request.baseline if field == "baseline_config" else request.candidate
    if version.id != expected:
        raise UsageError(
            code="RL0131",
            message=f"compare rung 3 {field} id {version.id!r} does not match {expected!r}",
            remediation="use the same version identifier in the request and replay declaration",
            context={"field": field, "request_version": expected, "replay_version": version.id},
        )
    declaration = {
        "id": version.id,
        "weights": {name: str(weight) for name, weight in sorted(version.weights.items())},
        "offset": str(version.offset),
    }
    return version, {
        "source_path": str(source_path),
        "source_digest": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        "canonical_declaration_digest": contracts.digest(declaration),
        "declaration": declaration,
    }


def _run_record(request: Any):
    """Open the requested run through the registered closed run-record reader."""
    from reward_lens.contracts import UsageError
    from reward_lens.product.trace import open_trace

    if request.run_dir is None:
        raise UsageError(
            code="RL0131",
            message="compare rung 3 requires --run-dir naming a closed run record",
            remediation="supply the closed training run directory read by the run_record reader",
            context={"field": "run_dir"},
        )
    try:
        return open_trace(request.run_dir, policy="strict", reader="run_record")
    except Exception as error:
        raise UsageError(
            code="RL0131",
            message=f"compare rung 3 could not read a closed run record from {request.run_dir}: {error}",
            remediation="supply a complete run_record-compatible training directory",
            context={"field": "run_dir", "path": str(request.run_dir)},
        ) from error


def _reproduce(request: Any, rungs: Iterable[Rung]) -> str:
    argv = ["reward-lens", "compare", str(request.baseline), str(request.candidate)]
    if getattr(request, "bank", None) is not None:
        argv.extend(("--bank", str(request.bank)))
    if getattr(request, "run_dir", None) is not None:
        argv.extend(("--run-dir", str(request.run_dir)))
    for field in (
        "baseline_config",
        "candidate_config",
        "selection_config",
        "controlled_update_config",
        "short_fork_config",
    ):
        value = getattr(request, field, None)
        if value is not None:
            argv.extend(("--" + field.replace("_", "-"), str(value)))
    for rung in rungs:
        argv.extend(("--rung", str(rung.value)))
    if getattr(request, "project", None) is not None:
        argv.extend(("--project", str(request.project)))
    if getattr(request, "name", None):
        argv.extend(("--name", str(request.name)))
    argv.extend(("--seed", str(request.seed)))
    return shlex.join(argv)


def _replace_measured_entries(record: contracts.Assay, results: Iterable[CompareResult]) -> None:
    """Install every actual requested-rung entry in its record section.

    A future request may name several registered rungs. Each lower-level result remains useful,
    so the public record collects their entries by section instead of returning only the first.
    The typed registry refusal still occurs before persistence if a requested rung has not landed.
    """
    by_section: dict[str, list[contracts.Entry]] = {}
    for result in results:
        for entry in result.entries:
            by_section.setdefault(entry.section, []).append(entry)
    for section, entries in by_section.items():
        setattr(record.measurement, section, entries)


def _bank_digest(bank: Any) -> str:
    """The response bank's canonical loaded-row digest in the assay digest spelling."""
    return "sha256:" + str(bank.digest)


def _retain_bank_identity(record: contracts.Assay, bank: Any, bank_entry_ids: set[str]) -> None:
    """Seal the finite input identity even when a project already binds `samples`.

    A configured project has a version digest over its declared response file. Replacing that
    dependency would make the stored subject lie about the version it names, and `Project.record`
    correctly refuses it. For an unconstrained subject, the bank is the samples identity itself.
    In either case the complete loaded-bank and source-byte provenance is sealed in the canonical
    record extension, not left only in a mutable command-line path.
    """
    canonical = _bank_digest(bank)
    identity = {
        "canonical_digest": canonical,
        "source_path": str(bank.provenance.source_path),
        "source_digest": "sha256:" + str(bank.provenance.digest),
        "generating_command": bank.provenance.generating_command,
        "revision": bank.provenance.revision,
    }
    if record.subject.digests.samples is None:
        record.subject.digests.samples = canonical
    record.extensions = dict(record.extensions or {})
    record.extensions[_BANK_EXTENSION] = identity
    for section in contracts.Measurement.model_fields:
        entries = getattr(record.measurement, section)
        bound: list[contracts.Entry] = []
        for entry in entries:
            if entry.kind == "absence" or entry.entry_id not in bank_entry_ids:
                bound.append(entry)
                continue
            result = dict(entry.result or {})
            result["response_bank"] = identity
            method = entry.method.model_copy(
                update={
                    "params_digest": contracts.digest(
                        {
                            "method_params": entry.method.params_digest,
                            "response_bank": canonical,
                        }
                    )
                }
            )
            bound.append(entry.model_copy(update={"method": method, "result": result}))
        setattr(record.measurement, section, bound)


def _retain_replay_identity(
    record: contracts.Assay, replay_inputs: dict[str, Any], replay_entry_ids: set[str]
) -> None:
    """Seal Rung 3's run and declaration inputs on its own entry and record extension.

    A comparison can carry both a finite response bank and a closed training run.
    They describe different evidence, so this deliberately binds replay identity only
    to advantage-replay entries and never overloads the bank or project samples digest.
    """
    record.extensions = dict(record.extensions or {})
    record.extensions[_REPLAY_EXTENSION] = replay_inputs
    for section in contracts.Measurement.model_fields:
        entries = getattr(record.measurement, section)
        bound: list[contracts.Entry] = []
        for entry in entries:
            if entry.kind == "absence" or entry.entry_id not in replay_entry_ids:
                bound.append(entry)
                continue
            result = dict(entry.result or {})
            result["replay_inputs"] = replay_inputs
            method = entry.method.model_copy(
                update={
                    "params_digest": contracts.digest(
                        {
                            "method_params": entry.method.params_digest,
                            "replay_inputs": replay_inputs,
                        }
                    )
                }
            )
            bound.append(entry.model_copy(update={"method": method, "result": result}))
        setattr(record.measurement, section, bound)


def _retain_execution_identity(
    record: contracts.Assay, execution_inputs: dict[Rung, dict[str, Any]]
) -> None:
    """Seal the exact executable configuration beside the R5 measurement it produced."""
    if not execution_inputs:
        return
    identities = {
        f"rung_{rung.value}": identity
        for rung, identity in sorted(execution_inputs.items(), key=lambda item: item[0].value)
    }
    record.extensions = dict(record.extensions or {})
    record.extensions[_EXECUTION_EXTENSION] = identities
    for section in contracts.Measurement.model_fields:
        entries = getattr(record.measurement, section)
        bound: list[contracts.Entry] = []
        for entry in entries:
            identity = next(
                (
                    execution_inputs[rung]
                    for rung, entry_id in (
                        (Rung.SELECTION_STRESS, "reward_statistics.selection_stress"),
                        (Rung.SAME_BATCH_SWAP, "reward_statistics.controlled_update_response"),
                        (Rung.SHORT_FORK, "reward_statistics.short_fork_response"),
                        (Rung.TRANSFER, "reward_statistics.held_out_transfer"),
                    )
                    if entry.entry_id == entry_id and rung in execution_inputs
                ),
                None,
            )
            if entry.kind == "absence" or identity is None:
                bound.append(entry)
                continue
            result = dict(entry.result or {})
            result["execution_config"] = identity
            method = entry.method.model_copy(
                update={
                    "params_digest": contracts.digest(
                        {
                            "method_params": entry.method.params_digest,
                            "execution_config": identity,
                        }
                    )
                }
            )
            bound.append(entry.model_copy(update={"method": method, "result": result}))
        setattr(record.measurement, section, bound)


def _retain_case_evidence(record: contracts.Assay, rungs: tuple[Rung, ...]) -> None:
    """Attach the saved Case view only from the final canonical R3 entries.

    R3's two candidate witnesses receive their public subject in
    ``_settle_subject`` and the decisive entry receives its sealed replay
    inputs there.  Constructing this extension afterwards binds its pointers
    to the exact entries that will be persisted, without changing either the
    replay calculation or its measured per-candidate counters.
    """
    if Rung.ADVANTAGE_REPLAY not in rungs:
        return
    from .case_evidence import CASE_EXTENSION, build_case_evidence
    from .rung3 import R3_ENTRY_ID

    decisive = next(entry for entry in record.entries() if entry.entry_id == R3_ENTRY_ID)
    candidates = tuple(
        entry
        for entry in record.entries()
        if entry.entry_id.startswith(R3_ENTRY_ID + ".candidate.")
    )
    record.extensions = dict(record.extensions or {})
    record.extensions[CASE_EXTENSION] = build_case_evidence(decisive, candidates)


def _settle_subject(
    record: contracts.Assay,
    project: Any,
    bank: Any,
    bank_entry_ids: set[str],
    replay_inputs: dict[str, Any] | None,
    replay_entry_ids: set[str],
    authoritative_subject: contracts.Subject | None = None,
    prebound_entry_ids: frozenset[str] = frozenset(),
) -> None:
    """Bind a completed comparison to the project version its entries actually used."""
    if authoritative_subject is not None:
        record.subject = authoritative_subject
    elif project is not None:
        version = project.version_of(record)
        digests = version.to_subject_digests()
        record.subject = contracts.Subject(
            reward_system=contracts.models.RewardSystemRef(
                id=str(version.id),
                name=str(getattr(project.config, "name", None) or version.id),
            ),
            version=version.to_version_ref(),
            context=contracts.models.ContextRef(
                policy=None,
                task_set=digests.task_distribution,
                configuration=None,
            ),
            digests=digests,
        )
    if bank is not None:
        _retain_bank_identity(record, bank, bank_entry_ids)
    if replay_inputs is not None:
        _retain_replay_identity(record, replay_inputs, replay_entry_ids)
    subject_ref = record.subject.version.digest
    for section in contracts.Measurement.model_fields:
        entries = getattr(record.measurement, section)
        setattr(
            record.measurement,
            section,
            [
                entry
                if entry.entry_id in prebound_entry_ids
                else entry.model_copy(update={"subject_ref": subject_ref})
                for entry in entries
            ],
        )
    mismatched = [
        entry.entry_id
        for entry in record.entries()
        if entry.entry_id in prebound_entry_ids and entry.subject_ref != subject_ref
    ]
    if mismatched:
        raise RuntimeError(
            "prebound factory entries disagree with the authoritative subject: "
            + ", ".join(sorted(mismatched))
        )


def _assemble_record(
    request: Any,
    *,
    source: Path,
    bank: Any,
    project: Any,
    rungs: tuple[Rung, ...],
    results: tuple[CompareResult, ...],
    replay_inputs: dict[str, Any] | None,
    execution_inputs: dict[Rung, dict[str, Any]],
    authoritative_subject: contracts.Subject | None,
    prebound_entry_ids: frozenset[str],
    started: float,
) -> contracts.Assay:
    """Convert lower-level rung evidence into the one canonical public record."""
    record = absent_record(
        command="compare",
        reproduce=[_reproduce(request, rungs)],
        path=source,
        offline=bool(request.offline),
        seed=int(request.seed),
        started_monotonic=started,
    )
    _replace_measured_entries(record, results)
    bank_entry_ids = {
        entry.entry_id
        for rung, result in zip(rungs, results, strict=True)
        if rung in (Rung.RE_SCORE, Rung.SELECTION_STRESS)
        for entry in result.entries
    }
    replay_entry_ids = {
        entry.entry_id
        for rung, result in zip(rungs, results, strict=True)
        if rung is Rung.ADVANTAGE_REPLAY
        for entry in result.entries
    }
    _settle_subject(
        record,
        project,
        bank,
        bank_entry_ids,
        replay_inputs,
        replay_entry_ids,
        authoritative_subject,
        prebound_entry_ids,
    )
    from .persisted_scorecard import retain_persisted_scorecards

    retain_persisted_scorecards(
        record,
        baseline=str(request.baseline),
        candidate=str(request.candidate),
        rungs=rungs,
        results=results,
    )
    _retain_execution_identity(record, execution_inputs)
    _retain_case_evidence(record, rungs)
    method_ids = sorted({entry.method.id for entry in record.entries()})
    record.producer = contracts.Producer(
        tool="reward-lens",
        version=record.producer.version,
        method_set=contracts.digest({"methods": method_ids}),
    )
    record.holes_from_entries()
    record.decision = contracts.Decision(
        state="unresolved",
        action=None,
        policy=None,
        reasons=[
            f"required_missing:{section}"
            for section in contracts.Measurement.model_fields
            if all(entry.kind == "absence" for entry in getattr(record.measurement, section))
        ],
        tradeoff=None,
        signature=None,
        regression_cases=[],
    )
    record.cost = contracts.Cost(
        wall_s=round(max(0.0, time.monotonic() - started), 3),
        cpu_s=0.0,
        usd="0.00",
        api_calls=0,
    )
    record.environment_excluded_from_digest = {
        "python": platform.python_version(),
        "platform": sys.platform,
    }
    record.assay_id = contracts.digest(record)
    return record


def _default_name(request: Any, rungs: tuple[Rung, ...]) -> str:
    """A stable, explicit label; callers choose a distinct name for a new comparison."""
    if getattr(request, "name", None):
        return str(request.name)
    rung_label = "-".join(str(rung.value) for rung in rungs)
    return f"compare-{request.baseline}-to-{request.candidate}-rungs-{rung_label}"


def run(request, *, project=None, sandbox=None) -> contracts.Assay:
    """Run requested registered rungs and persist their combined public Assay bundle."""
    started = time.monotonic()
    rungs = _requested_rungs(request)
    implementations = tuple(get(rung) for rung in rungs)
    controlled_update_config = None
    controlled_update_result = None
    short_fork_config = None
    short_fork_input = None
    transfer_config = None
    transfer_execution = None
    selection_execution = None
    execution_inputs: dict[Rung, dict[str, Any]] = {}
    if Rung.SAME_BATCH_SWAP in rungs:
        from .execution import load_execution_config
        from .rung4 import (
            CODING_UPDATE_PROFILE,
            CodingControlledUpdateConfig,
            CpuControlledUpdateConfig,
        )

        controlled_update_config = load_execution_config(
            getattr(request, "controlled_update_config", None), field="controlled_update_config"
        )
        try:
            config_type = (
                CodingControlledUpdateConfig
                if controlled_update_config.payload.get("profile") == CODING_UPDATE_PROFILE
                else CpuControlledUpdateConfig
            )
            controlled_config = config_type.from_mapping(controlled_update_config.payload)
        except (TypeError, ValueError) as error:
            raise UsageError(
                code="RL0131",
                message=(
                    "compare controlled_update_config is not a supported bounded "
                    f"configuration: {error}"
                ),
                remediation=(
                    "supply the exact offline CPU controlled-update configuration required "
                    "by rung 4"
                ),
                context={
                    "field": "controlled_update_config",
                    "path": str(controlled_update_config.path),
                },
            ) from error
    else:
        controlled_config = None
    if Rung.SHORT_FORK in rungs:
        from .execution import load_execution_config
        from .rung5 import CODING_SHORT_FORK_PROFILE, CodingShortForkConfig, CPUShortForkConfig

        short_fork_config = load_execution_config(
            getattr(request, "short_fork_config", None), field="short_fork_config"
        )
        try:
            config_type = (
                CodingShortForkConfig
                if short_fork_config.payload.get("profile") == CODING_SHORT_FORK_PROFILE
                else CPUShortForkConfig
            )
            cpu_config = config_type.from_mapping(short_fork_config.payload)
        except (TypeError, ValueError) as error:
            raise UsageError(
                code="RL0131",
                message=f"compare short_fork_config is not a bounded CPU conformance configuration: {error}",
                remediation="supply the exact offline T0 CPU short-fork configuration required by rung 5",
                context={"field": "short_fork_config", "path": str(short_fork_config.path)},
            ) from error
    else:
        cpu_config = None
    if Rung.TRANSFER in rungs:
        from .execution import load_execution_config

        transfer_config = load_execution_config(
            getattr(request, "transfer_config", None), field="transfer_config"
        )

    bank_path = getattr(request, "bank", None)
    bank = None
    if Rung.RE_SCORE in rungs or Rung.SELECTION_STRESS in rungs:
        if bank_path is None:
            require_bank(None)
        source = Path(bank_path)
        bank = load_bank(source, name=source.stem)
    elif request.run_dir is not None:
        source = Path(request.run_dir)
    elif short_fork_config is not None:
        source = short_fork_config.path
    elif controlled_update_config is not None:
        source = controlled_update_config.path
    elif transfer_config is not None:
        source = transfer_config.path
    else:
        require_bank(None)
    if project is None and getattr(request, "project", None) is not None:
        from reward_lens.store import Project

        project = Project.open(request.project)
    authoritative_subject = (
        _authoritative_factory_subject(source=source, project=project, rungs=rungs)
        if Rung.SELECTION_STRESS in rungs
        or cpu_config is not None
        or controlled_config is not None
        or transfer_config is not None
        else None
    )
    if Rung.SELECTION_STRESS in rungs:
        from .rung2 import load_selection_stress_execution

        if project is None:
            raise UsageError(
                code="RL0131",
                message="compare rung 2 requires --project for its project-local supervised execution",
                remediation="supply the project that owns the selection config, controls, and held-out targets",
                context={"field": "project"},
            )
        selection_config = getattr(request, "selection_config", None)
        if selection_config is None:
            raise UsageError(
                code="RL0131",
                message="compare rung 2 requires --selection-config for its supervised execution",
                remediation="supply the project-local selection-stress JSON configuration",
                context={"field": "selection_config"},
            )
        selection_execution = load_selection_stress_execution(
            project.root,
            selection_config,
            expected_subject_ref=authoritative_subject.version.digest,
        )
        execution_inputs[Rung.SELECTION_STRESS] = {
            "source_path": str(Path(selection_config).resolve()),
            "source_digest": selection_execution.canonical_declaration[
                "selection_config_source_digest"
            ],
            "canonical_declaration_digest": selection_execution.canonical_declaration_digest,
            "declaration": selection_execution.canonical_declaration,
        }
    if Rung.TRANSFER in rungs:
        from .rung6 import build_supervised_rung6, load_supervised_rung6_execution

        if project is None:
            raise UsageError(
                code="RL0131",
                message="compare rung 6 requires --project for its project-local supervised execution",
                remediation="supply the project that owns the transfer declaration and its local partitions",
                context={"field": "project"},
            )
        declared_subject = transfer_config.payload.get("subject_ref")
        if declared_subject != authoritative_subject.version.digest:
            raise UsageError(
                code="RL0131",
                message="compare transfer_config.subject_ref does not match the authoritative public subject for this request",
                remediation="set transfer_config.subject_ref to the exact subject version digest for this project",
                context={
                    "field": "transfer_config.subject_ref",
                    "configured_subject_ref": declared_subject,
                    "authoritative_subject_ref": authoritative_subject.version.digest,
                },
            )
        try:
            transfer_execution = build_supervised_rung6(
                load_supervised_rung6_execution(transfer_config.path, root=project.root)
            )
        except (TypeError, ValueError) as error:
            raise UsageError(
                code="RL0131",
                message=f"compare transfer_config is not a strict local supervised transfer configuration: {error}",
                remediation="supply the exact bounded R6 declaration with separate local controls, original, and transfer partitions",
                context={"field": "transfer_config", "path": str(transfer_config.path)},
            ) from error
        transfer_identity = {**transfer_config.identity()}
        if not transfer_execution.qualification.qualification.qualified:
            # An absence cannot carry a ``result`` under the frozen record
            # schema.  Retain the actual supervisor receipts alongside the
            # sealed declaration instead, so the saved public bundle shows
            # that the route ran and why its transfer result is unavailable.
            transfer_identity["supervised_qualification"] = {
                "state": transfer_execution.qualification.qualification.state,
                "reason": transfer_execution.qualification.qualification.reason,
                "control_receipts": [
                    {
                        "receipt_digest": receipt.receipt_digest,
                        "request_digest": receipt.request_digest,
                        "run_id": receipt.run_id,
                        "verdict": receipt.verdict,
                    }
                    for receipt in transfer_execution.qualification.control_receipts
                ],
            }
        execution_inputs[Rung.TRANSFER] = transfer_identity
    run_record = _run_record(request) if Rung.ADVANTAGE_REPLAY in rungs else None
    replay_inputs = None
    if run_record is not None:
        replay_a, baseline_identity = _replay_version(request, "baseline_config")
        replay_b, candidate_identity = _replay_version(request, "candidate_config")
        replay_inputs = {
            "run": _replay_run_identity(Path(request.run_dir), run_record),
            "baseline": baseline_identity,
            "candidate": candidate_identity,
        }
    else:
        replay_a = replay_b = None
    if controlled_config is not None and controlled_update_config is not None:
        from .rung4 import (
            CodingControlledUpdateConfig,
            build_coding_controlled_update,
            build_cpu_controlled_update,
        )

        if controlled_config.subject_ref != authoritative_subject.version.digest:
            raise UsageError(
                code="RL0131",
                message=(
                    "compare controlled_update_config.subject_ref does not match the "
                    "authoritative public subject for this request"
                ),
                remediation=(
                    "set controlled_update_config.subject_ref to the exact subject version "
                    "digest for this project or projectless configured source"
                ),
                context={
                    "field": "controlled_update_config.subject_ref",
                    "configured_subject_ref": controlled_config.subject_ref,
                    "authoritative_subject_ref": authoritative_subject.version.digest,
                },
            )
        name = _default_name(request, rungs)
        artifact_root = (
            project.assays_dir
            if project is not None
            else controlled_update_config.path.parent / "assays"
        )
        artifact_dir = artifact_root / (
            "f{0}.controlled-update-{1}".format(
                name,
                controlled_update_config.source_digest.removeprefix("sha256:")[:16],
            )
        )
        factory = (
            build_coding_controlled_update
            if isinstance(controlled_config, CodingControlledUpdateConfig)
            else build_cpu_controlled_update
        )
        try:
            controlled_update_result = factory(controlled_config, artifact_dir)
        except TrainerUnsupported as error:
            raise contracts.CapabilityUnavailable(
                code="RL0702",
                message=str(error),
                remediation="install 'reward-lens[train]' 'trl==1.10.0' in an isolated environment for this pinned controlled-intervention route",
                context={
                    "family": "trl==1.10.0",
                    "capability": "controlled intervention",
                    "installed_version": error.version,
                },
            ) from error
        execution_inputs[Rung.SAME_BATCH_SWAP] = {
            **controlled_update_config.identity(),
            "artifact_dir": str(artifact_dir),
        }
    if cpu_config is not None and short_fork_config is not None:
        from .rung5 import CodingShortForkConfig, build_coding_short_fork, build_cpu_short_fork

        if cpu_config.subject_ref != authoritative_subject.version.digest:
            raise UsageError(
                code="RL0131",
                message=(
                    "compare short_fork_config.subject_ref does not match the authoritative "
                    "public subject for this request"
                ),
                remediation=(
                    "set short_fork_config.subject_ref to the exact subject version digest "
                    "for this project or projectless configured source"
                ),
                context={
                    "field": "short_fork_config.subject_ref",
                    "configured_subject_ref": cpu_config.subject_ref,
                    "authoritative_subject_ref": authoritative_subject.version.digest,
                },
            )
        name = _default_name(request, rungs)
        artifact_root = (
            project.assays_dir if project is not None else short_fork_config.path.parent / "assays"
        )
        artifact_dir = artifact_root / (
            f"{name}.short-fork-{short_fork_config.source_digest.removeprefix('sha256:')[:16]}"
        )
        factory = (
            build_coding_short_fork
            if isinstance(cpu_config, CodingShortForkConfig)
            else build_cpu_short_fork
        )
        try:
            short_fork_input = factory(cpu_config, artifact_dir)
        except TrainerUnsupported as error:
            raise contracts.CapabilityUnavailable(
                code="RL0702",
                message=str(error),
                remediation="install 'reward-lens[train]' 'trl==1.10.0' in an isolated environment for this pinned controlled-intervention route",
                context={
                    "family": "trl==1.10.0",
                    "capability": "controlled intervention",
                    "installed_version": error.version,
                },
            ) from error
        execution_inputs[Rung.SHORT_FORK] = {
            **short_fork_config.identity(),
            "artifact_dir": str(artifact_dir),
        }
    results = tuple(
        implementation(
            CompareInputs(
                bank=bank,
                version_a=replay_a if rung is Rung.ADVANTAGE_REPLAY else request.baseline,
                version_b=replay_b if rung is Rung.ADVANTAGE_REPLAY else request.candidate,
                run_record=(
                    run_record
                    if rung is Rung.ADVANTAGE_REPLAY
                    else controlled_update_result
                    if rung is Rung.SAME_BATCH_SWAP
                    else short_fork_input
                    if rung is Rung.SHORT_FORK
                    else transfer_execution
                    if rung is Rung.TRANSFER and transfer_execution is not None
                    else None
                ),
                selector=(
                    selection_execution.selector
                    if rung is Rung.SELECTION_STRESS and selection_execution is not None
                    else None
                ),
                outcome_check=(
                    selection_execution.outcome_provider
                    if rung is Rung.SELECTION_STRESS and selection_execution is not None
                    else None
                ),
                ladder=(
                    selection_execution.ladder
                    if rung is Rung.SELECTION_STRESS and selection_execution is not None
                    else ()
                ),
                bootstrap_resamples=(
                    selection_execution.bootstrap_resamples
                    if rung is Rung.SELECTION_STRESS and selection_execution is not None
                    else 0
                ),
                sandbox=(
                    selection_execution.sandbox
                    if rung is Rung.SELECTION_STRESS and selection_execution is not None
                    else None
                ),
                limits=(
                    selection_execution.limits
                    if rung is Rung.SELECTION_STRESS and selection_execution is not None
                    else None
                ),
            )
        )
        for rung, implementation in zip(rungs, implementations, strict=True)
    )
    record = _assemble_record(
        request,
        source=source,
        bank=bank,
        project=project,
        rungs=rungs,
        results=results,
        replay_inputs=replay_inputs,
        execution_inputs=execution_inputs,
        authoritative_subject=authoritative_subject,
        prebound_entry_ids=frozenset(
            entry_id
            for produced, entry_id in (
                (selection_execution, "reward_statistics.selection_stress"),
                (controlled_update_result, "reward_statistics.controlled_update_response"),
                (short_fork_input, "reward_statistics.short_fork_response"),
                (transfer_execution, "reward_statistics.held_out_transfer"),
            )
            if produced is not None
        ),
        started=started,
    )
    persisted, _ = persist_assay_bundle(
        record,
        project=project,
        name=_default_name(request, rungs),
        # A closed run directory is an input tree whose complete source bytes have
        # already been sealed above.  A projectless bundle must never land inside
        # it, or an identical later request would digest its own old output.
        fallback_dir=source.parent / "assays",
    )
    return persisted
