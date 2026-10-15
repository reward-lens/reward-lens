"""Build the E1 assay from caller-supplied release bytes and analysis records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from reward_lens.record.convert.trl_parquet import CompletionsTable, read_completions

_ROLLOUTS = Path(".e1-cache/extract/kl0.0_rollouts.parquet")
_ANALYSIS = Path("e1_analysis")
_ANALYSIS_FILES = ("COMPOSITION.md", "REGISTERED.md", "RECORD_CONTENTS.md", "GATE2.md")
_REWARD_LINE = re.compile(r"# Reward weights: \[(?P<names>[^]]+)\]")
_OBJECTIVE_LINE = re.compile(r"objective is `(?P<terms>[^`]+)`")


@dataclass(frozen=True)
class Assay:
    source_root: Path
    rollout_path: Path
    derivation_receipt: Mapping[str, object] | None
    rollouts: CompletionsTable
    analysis: Mapping[str, str]
    citations: tuple[str, ...]
    reward_functions: tuple[str, ...]
    optimized_weights: Mapping[str, float]
    contrast_threshold: tuple[None, None]
    registered_threshold: tuple[None, None, None]
    reproduce: Mapping[str, str]
    absences: Mapping[str, str]
    native_grid_support: int
    trace: Mapping[str, object]

    def canonical(self) -> dict[str, object]:
        """The public canonical form is the product contract, never a package-local summary."""
        return self.contract_assay().to_dict()

    def source_digests(self) -> dict[str, str]:
        """SHA-256 identities for every release byte this assay actually reads."""
        paths = (
            self.rollout_path,
            *(self.source_root / _ANALYSIS / name for name in _ANALYSIS_FILES),
        )
        if self.derivation_receipt is not None:
            paths = (*paths, self.source_root / "receipt.json")
        return {
            str(path.relative_to(self.source_root)): "sha256:"
            + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }

    def contract_assay(self):
        """Return the schema-validated product Assay carrying this reconstruction as evidence."""
        from reward_lens import contracts
        from reward_lens.api._record import absent_record
        from reward_lens.contracts.models import ContextRef, Digests, RewardSystemRef, VersionRef
        from reward_lens.render.report import plan_embedding

        from .counts import counts

        source_digests = self.source_digests()
        rollout_key = str(self.rollout_path.relative_to(self.source_root))
        public_join = self.derivation_receipt is not None
        subject_ref = contracts.digest({"e1_release_inputs": source_digests})
        record = absent_record(
            command="e1",
            reproduce=list(self.reproduce.values()),
            path=self.source_root,
            offline=True,
            seed=0,
        )
        method = contracts.Method(
            id="reward_statistics.e1_reconstruction",
            version="1.1.0" if public_join else "1.0.0",
            params_digest=contracts.digest(
                {"reader": self.trace["reader"], "source_digests": source_digests}
            ),
            procedure=(
                "read the caller-supplied parquet and derivation receipt, verify output and analysis digests; "
                "derive the registered census from the reconstructed table and E1 analysis"
                if public_join
                else "read the pinned rollout parquet through the shipped record converter and derive the registered census from the parsed E1 analysis"
            ),
        )
        provenance = contracts.EntryProvenance(
            started=record.created, duration_s=0.0, sandbox_tier="T0", offline=True
        )
        measured = counts(self).as_dict()
        entry = contracts.Entry(
            entry_id="reward_statistics.e1_reconstruction",
            section="reward_statistics",
            kind="witness",
            measurand="the source-derived E1 reward composition and support census",
            method=method,
            scope="evaluator_comparison",
            subject_ref=subject_ref,
            depends_on=[
                "digest:source",
                "digest:samples",
                "digest:scorer_config",
                "digest:training_semantics",
            ],
            state="complete",
            provenance=provenance,
            limitations=(
                [
                    "source identities and upstream join checks are caller-attested in receipt.json; this reader verifies output and analysis digests and recomputes counts, without rereading upstream eval archives"
                ]
                if public_join
                else [
                    "available evidence is rollout parquet plus registered analysis; raw evaluation archives are not materialised on this host"
                ]
            )
            + [
                "the reconstruction is reward-side evidence and does not establish an applied training update"
            ],
            result={
                "counts": measured,
                "native_grid_support": self.native_grid_support,
                "citations": list(self.citations),
                "reproduce": dict(self.reproduce),
                "trace": dict(self.trace),
            },
            witness=contracts.Witness(
                inputs=source_digests,
                procedure=(
                    "read the caller-supplied derivation receipt, verify its output and analysis digests, then recompute counts"
                    if public_join
                    else "read the pinned source bytes and recompute the counts"
                ),
                observed="the row-derived census and the registered analysis records agree",
                artifact_identity=subject_ref,
                path=str(self.source_root),
            ),
        )
        record.subject = contracts.Subject(
            reward_system=RewardSystemRef(
                id="e1-reconstruction-inputs" if public_join else "aisi-olmo-e1",
                name="Caller-supplied E1 reconstruction"
                if public_join
                else "AISI OLMo reward-hacking E1",
            ),
            version=VersionRef(
                id="receipt-inputs" if public_join else "kl0.0-seed2",
                digest=subject_ref,
                parents=[],
                declared_change=["weights"],
            ),
            context=ContextRef(
                policy=None,
                task_set=source_digests[rollout_key],
                configuration=(
                    "caller-supplied parquet and derivation receipt"
                    if public_join
                    else "published rollout artifact"
                ),
            ),
            digests=Digests(
                source=subject_ref,
                environment=None,
                scorer_config=source_digests[str(_ANALYSIS / "COMPOSITION.md")],
                task_distribution=None,
                samples=source_digests[rollout_key],
                outcome_protocol=None,
                policy=None,
                training_semantics=source_digests[str(_ANALYSIS / "RECORD_CONTENTS.md")],
                instrument_method=method.params_digest,
            ),
        )
        record.measurement.reward_statistics = [entry]
        record.measurement.trace = [
            contracts.absence(
                "trace",
                f"trace.{name}",
                f"E1 {name.replace('_', ' ')}",
                missing,
                remedy,
                (claim,),
                subject_ref=subject_ref,
                provenance=provenance,
                depends_on=("digest:training_semantics",),
                state="NOT_MEASURED",
                scope="reconstructed_training_pressure",
                limitations=("the available source does not record this training-side quantity",),
            )
            for name, missing, remedy, claim in (
                (
                    "applied_update_weight",
                    "no per-rollout applied update weights",
                    "record signed applied update weights for every rollout",
                    "no applied-update attribution",
                ),
                (
                    "gradient",
                    "no training gradients",
                    "record the training gradients with parameter and optimizer context",
                    "no gradient attribution",
                ),
                (
                    "effective_sample_size",
                    "no importance ratios, completion masks, or token weights",
                    "record importance ratios, masks, and token weighting",
                    "no effective-sample-size claim",
                ),
            )
        ]
        record.decision.reasons = [
            reason
            for reason in record.decision.reasons
            if reason != "required_missing:reward_statistics"
        ]
        record.holes_from_entries()
        record.producer.method_set = contracts.digest(
            {"methods": sorted(item.method.id for item in record.entries())}
        )
        record.embedding = contracts.Embedding(
            tier="A", omitted_tables=[], record_bytes=0, html_bytes=0
        )
        record.embedding = contracts.Embedding(
            **plan_embedding(record, bundle_manifest_digest="sha256:" + "0" * 64).to_dict()
        )
        record.assay_id = contracts.digest(record)
        contracts.validate_record(record.to_dict())
        return record


@dataclass(frozen=True)
class AssayBundle:
    root: Path
    assay_json: Path
    report_html: Path
    manifest: Path


def _source_root(source_root: Path | str | None = None) -> Path:
    configured = (
        source_root if source_root is not None else os.environ.get("REWARD_LENS_E1_ASSAY_ROOT")
    )
    if configured is None or str(configured).strip() == "":
        raise FileNotFoundError(
            "E1 reconstruction needs an explicit public artifact root: pass source_root "
            "or set REWARD_LENS_E1_ASSAY_ROOT"
        )
    return Path(configured).expanduser().resolve()


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"recorded E1 artifact is missing: {path}")
    return path


def _analysis(root: Path) -> dict[str, str]:
    directory = root / _ANALYSIS
    if not directory.is_dir():
        raise FileNotFoundError(f"recorded E1 analysis directory is missing: {directory}")
    return {
        name: _require(directory / name).read_text(encoding="utf-8") for name in _ANALYSIS_FILES
    }


def _reward_functions(composition: str) -> tuple[str, ...]:
    match = _REWARD_LINE.search(composition)
    if match is None:
        raise ValueError("e1_analysis/COMPOSITION.md does not declare the reward function list")
    return tuple(part.strip() for part in match.group("names").split(","))


def _optimized_weights(composition: str) -> dict[str, float]:
    match = _OBJECTIVE_LINE.search(composition)
    if match is None:
        raise ValueError("e1_analysis/COMPOSITION.md does not declare the optimized objective")
    weights: dict[str, float] = {}
    for term in (term.strip() for term in match.group("terms").split("+")):
        weight, name = (piece.strip() for piece in term.split("·", maxsplit=1))
        weights[name] = float(weight)
    return weights


def _table(path: Path) -> CompletionsTable:
    import pyarrow.parquet as parquet

    columns = tuple(parquet.ParquetFile(path).schema_arrow.names)
    reward_functions = ("thinking_format", "training_passed")
    return read_completions(
        path,
        reward_functions=reward_functions,
        extra_columns=tuple(column for column in columns if column not in reward_functions),
    )


def _native_grid_support(
    rows: tuple[Mapping[str, object], ...], registered: tuple[None, None, None]
) -> int:
    cells: dict[tuple[object, object], set[tuple[float, float]]] = {}
    for row in rows:
        cells.setdefault((row["source_eval_file"], row["sample_id"]), set()).add(
            (float(row["thinking_format"]), float(row["training_passed"]))
        )
    return sum(len(group) >= len(registered) for group in cells.values())


def _public_join(
    root: Path, derivation_receipt: Path | str | None
) -> tuple[Path, Mapping[str, object] | None]:
    configured = (
        derivation_receipt
        if derivation_receipt is not None
        else os.environ.get("REWARD_LENS_E1_DERIVATION_RECEIPT")
    )
    if configured is None:
        return _require(root / _ROLLOUTS), None
    path = Path(configured).expanduser().resolve()
    if derivation_receipt is None and path != root / "receipt.json":
        # An explicit source_root may be a small fixture in a process whose
        # environment points at a different public assay root.
        return _require(root / _ROLLOUTS), None
    if path != root / "receipt.json":
        raise ValueError(
            "E1 public join receipt must be named receipt.json in the explicit source root"
        )
    receipt = json.loads(_require(path).read_text(encoding="utf-8"))
    if receipt.get("schema") != "reward-lens-e1-public-join/1":
        raise ValueError("E1 public join receipt has an unexpected schema")
    name = receipt.get("output")
    if not isinstance(name, str) or Path(name).name != name or name != "public_join_v1.parquet":
        raise ValueError("E1 public join receipt names an unexpected output")
    output = _require(root / name)
    if hashlib.sha256(output.read_bytes()).hexdigest() != receipt.get("output_sha256"):
        raise ValueError("E1 public join output SHA-256 disagrees with receipt")
    for name, expected in receipt.get("analysis_sha256", {}).items():
        if name not in {f"e1_analysis/{item}" for item in _ANALYSIS_FILES}:
            raise ValueError(f"E1 public join receipt cites an unexpected analysis file: {name}")
        if hashlib.sha256(_require(root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"E1 analysis SHA-256 disagrees with receipt: {name}")
    if len(receipt.get("analysis_sha256", {})) != len(_ANALYSIS_FILES):
        raise ValueError("E1 public join receipt omits analysis digests")
    if not isinstance(receipt.get("inputs_sha256"), dict) or len(receipt["inputs_sha256"]) < 2:
        raise ValueError("E1 public join receipt omits public source digests")
    return output, receipt


def build(
    source_root: Path | str | None = None, *, derivation_receipt: Path | str | None = None
) -> Assay:
    """Read an explicit E1 release root without network or model weights."""
    root = _source_root(source_root)
    analysis = _analysis(root)
    rollout_path, receipt = _public_join(root, derivation_receipt)
    rollouts = _table(rollout_path)
    functions = _reward_functions(analysis["COMPOSITION.md"])
    optimized = _optimized_weights(analysis["COMPOSITION.md"])
    contrast = (None, None)
    registered = contrast + (None,)
    extra = (
        f" --derivation-receipt {shlex.quote(str(root / 'receipt.json'))}"
        if receipt is not None
        else ""
    )
    reproduce = {
        field: f"python -m reward_lens.examples.e1 --source-root {shlex.quote(str(root))}{extra} --field {field}"
        for field in (
            "groups",
            "groups_with_contrast",
            "accepted_hacks",
            "solutions",
            "functions_summed",
            "advantage_components",
            "registered_analysis_support",
        )
    }
    trace = {
        "reader": "reward_lens.record.convert.trl_parquet.read_completions",
        "source": str(rollout_path),
        "reward_semantics": "e1_analysis/COMPOSITION.md: logged reward is the unweighted composition",
        "objective_semantics": "e1_analysis/COMPOSITION.md: advantages use the declared weighted objective",
        "record_boundary": "e1_analysis/RECORD_CONTENTS.md: no applied training-update telemetry",
        "gate": "e1_analysis/GATE2.md: source reconstruction gate",
    }
    if receipt is not None:
        trace.update(
            {
                "derivation": "reward-lens-e1-public-join/1",
                "source_dataset": receipt.get("source_dataset"),
                "source_revision": receipt.get("source_revision"),
                "receipt": str(root / "receipt.json"),
            }
        )
    assay = Assay(
        source_root=root,
        rollout_path=rollout_path,
        derivation_receipt=receipt,
        rollouts=rollouts,
        analysis=analysis,
        citations=tuple(f"e1_analysis/{name}" for name in _ANALYSIS_FILES),
        reward_functions=functions,
        optimized_weights=optimized,
        contrast_threshold=contrast,
        registered_threshold=registered,
        reproduce=reproduce,
        absences={
            "applied_update_weight": "not measured: the release records no per-rollout applied update weight",
            "gradient": "not measured: the release records no training gradient",
            "effective_sample_size": "not measured: the release records no importance-ratio or token-mask basis",
        },
        native_grid_support=_native_grid_support(rollouts.rows, registered),
        trace=trace,
    )
    if receipt is not None:
        from .counts import counts

        measured = counts(assay).as_dict()
        differences = {
            key: (receipt.get("counts", {}).get(key), value)
            for key, value in {**measured, "native_grid_support": assay.native_grid_support}.items()
            if receipt.get("counts", {}).get(key) != value
        }
        if receipt.get("rows") != len(rollouts.rows) or differences:
            raise ValueError(
                f"E1 public join receipt counts disagree with reconstructed rows: {differences}"
            )
    return assay


def load_golden() -> dict[str, int]:
    return json.loads(Path(__file__).with_name("counts.golden.json").read_text(encoding="utf-8"))[
        "counts"
    ]


def persist_assay_bundle(assay: Assay, destination: Path | str) -> AssayBundle:
    """Persist one JCS assay record, reopen it through the SDK, and render its HTML report."""
    import rfc8785

    from reward_lens import api, contracts
    from reward_lens.render.report import render_to

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    record = assay.contract_assay()
    assay_json = root / "assay.json"
    assay_json.write_bytes(rfc8785.dumps(record.to_dict()))
    reopened = api.open_record(assay_json)
    if reopened.assay_id != record.assay_id or contracts.digest(reopened) != record.assay_id:
        raise AssertionError("persisted E1 assay did not reopen with its canonical digest")
    manifest = root / "manifest.json"
    manifest_body = {
        "kind": "reward-lens.e1.bundle/v1",
        "assay_id": record.assay_id,
        "assay_jcs_sha256": "sha256:" + hashlib.sha256(assay_json.read_bytes()).hexdigest(),
        "source_digests": assay.source_digests(),
    }
    manifest.write_bytes(rfc8785.dumps(manifest_body))
    manifest_digest = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
    report_html = render_to(reopened, root / "report.html", bundle_manifest_digest=manifest_digest)
    return AssayBundle(root=root, assay_json=assay_json, report_html=report_html, manifest=manifest)


def save_report(assay: Assay, destination: Path | str) -> Path:
    """Compatibility wrapper returning the persisted canonical assay path."""
    return persist_assay_bundle(assay, destination).assay_json
