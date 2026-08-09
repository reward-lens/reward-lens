"""The project store: files are the source of truth, and the index is a cache of them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import rfc8785
import yaml
from pydantic import ValidationError

from reward_lens.contracts import Assay, ProjectConfig, UsageError, digest
from reward_lens.contracts.models import Absence, Entry, EntryProvenance

from . import invalidation
from .digests import method_set
from .errors import (
    BadConfigField,
    DependencyDigestMismatch,
    SubjectVersionRewritten,
)
from .versions import RewardSystemVersion

__all__ = ["CONFIG_NAME", "Project", "Reuse"]

CONFIG_NAME = "rewardlens.yaml"
RECORD_SUFFIX = ".assay.json"
SECTIONS: tuple[str, ...] = (
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
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refused_field(exc: UsageError) -> str:
    """The field a contracts RL0003 refusal named, for the store to name in its own error.

    The refusal carries the field in its context. Where it does not, the first location of the
    `ValidationError` it kept as its cause is the same answer arrived at the long way, and a
    refusal with neither is about the file rather than a field in it.
    """
    named = (getattr(exc, "context", None) or {}).get("field")
    if isinstance(named, str) and named:
        return named
    cause = exc.__cause__
    if isinstance(cause, ValidationError):
        errors = cause.errors()
        if errors:
            located = ".".join(str(part) for part in errors[0]["loc"])
            if located:
                return located
    return CONFIG_NAME


@dataclass(frozen=True)
class Reuse:
    """What a rerun carries forward, what it must measure again, and the sentence that says so."""

    changed: frozenset[str]
    reused_entry_ids: tuple[str, ...]
    stale_entry_ids: tuple[str, ...]
    absences: tuple[Entry, ...]
    unchanged: bool
    says: str
    version: RewardSystemVersion

    def apply(self, previous: Assay, *, created: str | None = None, started: str | None = None
              ) -> Assay:
        """A new assay that references the eligible old entries and marks the rest stale."""
        data = previous.to_dict()
        keep = set(self.reused_entry_ids)
        measurement: dict[str, list[Any]] = {section: [] for section in SECTIONS}
        for entry in previous.entries():
            if entry.entry_id in keep:
                measurement[entry.section].append(_dump(entry))
        for entry in self.absences:
            measurement[entry.section].append(_dump(entry))
        data["measurement"] = measurement
        data["holes"] = []
        data["created"] = created or _now()
        data["subject"] = dict(data["subject"])
        data["subject"]["digests"] = self.version.digests.to_dict()
        data["subject"]["version"] = _dump(self.version.to_version_ref())
        decision = dict(data["decision"])
        decision["state"] = "unresolved"
        decision["policy"] = None
        decision["signature"] = None
        if self.stale_entry_ids:
            decision["reasons"] = [f"required_stale:{i}" for i in self.stale_entry_ids]
        data["decision"] = decision
        assay = Assay.model_validate(data)
        assay.holes_from_entries()
        sealed = assay.to_dict()
        sealed["assay_id"] = digest(sealed)  # the id is the digest of what the record now holds
        return Assay.model_validate(sealed)


def _dump(model) -> dict:
    return model.model_dump(mode="json", by_alias=True, exclude_unset=True)


def _measured(data: dict) -> tuple[str, ...]:
    """Every (entry id, method identity) pair a record holds, sorted: what it measured and how.

    The method identity is the method's rule id and its parameter digest, which is what tells one
    measurement of a thing from another measurement of the same thing. Not `version`: that is the
    method's release string, and a procedure whose wording was corrected has not measured anything
    differently. Read off the dict rather than the model because this runs on files in the
    directory that the store has not validated and will not.
    """
    measurement = data.get("measurement") or {}
    out: list[str] = []
    for group in measurement.values():
        if not isinstance(group, list):
            continue
        for entry in group:
            if not isinstance(entry, dict):
                continue
            method = entry.get("method")
            if not isinstance(method, dict):
                method = {}
            out.append(
                f"{entry.get('entry_id', '')}@{method.get('id', '')}@"
                f"{method.get('params_digest', '')}"
            )
    return tuple(sorted(out))


class Project:
    """A reward system on disk: `rewardlens.yaml`, `assays/`, and nothing that is not a file."""

    def __init__(self, root: Path, config: ProjectConfig) -> None:
        self.root = Path(root)
        self.config = config

    # --- opening and creating ---------------------------------------------------------------

    @classmethod
    def open(cls, path: Path | str) -> "Project":
        root = Path(path)
        config_path = root / CONFIG_NAME
        if not config_path.is_file():
            raise BadConfigField(
                field=CONFIG_NAME,
                message=f"{root} is not a reward-lens project: it has no {CONFIG_NAME}",
                remediation=f"run reward-lens init {root} to write a {CONFIG_NAME}",
                path=root,
            )
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise BadConfigField(
                field=CONFIG_NAME,
                message=f"{config_path} is not readable YAML: {exc}",
                remediation="fix the YAML syntax the parser named",
                path=config_path,
            ) from exc
        if not isinstance(data, dict):
            raise BadConfigField(
                field=CONFIG_NAME,
                message=f"{config_path} holds {type(data).__name__}, not a mapping of fields",
                remediation="a project file is a mapping with reward, tasks and outcome",
                path=config_path,
            )
        try:
            config = ProjectConfig.model_validate(data)
        except UsageError as exc:
            if exc.code != "RL0003":
                raise
            field = _refused_field(exc)
            carried = dict(getattr(exc, "context", None) or {})
            detail = carried.get("detail") or exc.message
            raise BadConfigField(
                field=field,
                message=f"{config_path}: {field} is wrong: {detail}",
                remediation=f"correct {field} in {CONFIG_NAME}",
                path=config_path,
                extra=carried,
            ) from exc
        except ValidationError as exc:
            first = exc.errors()[0]
            field = ".".join(str(part) for part in first["loc"]) or CONFIG_NAME
            raise BadConfigField(
                field=field,
                message=f"{config_path}: {field} is wrong: {first['msg']}",
                remediation=f"correct {field} in {CONFIG_NAME}",
                path=config_path,
            ) from exc
        return cls(root, config)

    @classmethod
    def init(cls, path: Path | str, config: ProjectConfig) -> "Project":
        root = Path(path)
        config_path = root / CONFIG_NAME
        if config_path.exists():
            raise BadConfigField(
                field=CONFIG_NAME,
                message=f"{root} already holds a {CONFIG_NAME}; init never overwrites one",
                remediation="edit the file, or init somewhere else",
                path=config_path,
            )
        root.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (root / "assays").mkdir(exist_ok=True)
        return cls(root, config)

    # --- versions ---------------------------------------------------------------------------

    def version(self, *, methods: tuple[str, ...] = ()) -> RewardSystemVersion:
        """The version this project currently declares. Recomputed; never cached across a reload."""
        return RewardSystemVersion.from_config(self.config, self.root, methods=methods)

    def version_of(self, record: Assay | dict) -> RewardSystemVersion:
        """This project's version as that record's own method set makes it comparable.

        A record is compared against the project only through this: `instrument_method` is a
        digest of the methods that ran, so the project's value for it is derived from the record's
        own entries (`digests.method_set`), never from an assumption that nothing ran.
        """
        data = record if isinstance(record, dict) else record.to_dict()
        return self.version(methods=method_set(data))

    # --- records ----------------------------------------------------------------------------

    @property
    def assays_dir(self) -> Path:
        return self.root / "assays"

    def path_for(self, name: str) -> Path:
        return self.assays_dir / f"{name}{RECORD_SUFFIX}"

    def record(self, assay: Assay, *, name: str) -> Path:
        """Write `assays/<name>.assay.json`. Canonical bytes, and never over an older record."""
        data = assay.to_dict()
        self._check_dependencies(data)
        payload = rfc8785.dumps(data)
        path = self.path_for(name)
        self._check_not_a_rewrite(data, payload, name)
        if path.exists():
            if path.read_bytes() == payload:
                return path  # D-28: re-recording the same thing is a no-op that says nothing new
            raise UsageError(
                code="RL0001",
                message=(
                    f"{path.name} already holds a different record; an old measurement is never "
                    "rewritten"
                ),
                remediation="record the new assay under a new name and let it reference the old",
                context={"name": name, "path": str(path)},
            )
        self.assays_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self._write_index(self._scan())
        return path

    def _check_dependencies(self, data: dict) -> None:
        """Every digest the record carries, against this project. Nothing is corrected in place."""
        declared = data["subject"]["digests"]
        version = self.version_of(data)
        current = version.digests.to_dict()
        for digest_name, expected in current.items():
            if expected is None:
                continue  # the project does not constrain what an experiment bound
            found = declared.get(digest_name)
            if found != expected:
                raise DependencyDigestMismatch(
                    digest=digest_name, expected=expected, found=found
                )
        expected_version = version.digest()
        found_version = data["subject"]["version"]["digest"]
        if found_version != expected_version:
            raise DependencyDigestMismatch(
                digest="subject.version", expected=expected_version, found=found_version
            )
        computed = digest(data)
        if data.get("assay_id") != computed:
            raise DependencyDigestMismatch(
                digest="assay_id", expected=computed, found=data.get("assay_id")
            )

    def _check_not_a_rewrite(self, data: dict, payload: bytes, name: str) -> None:
        """One subject version measured one way, one record: a second, different one is refused.

        A-016. The rule used to key on the version digest alone, which read every second record of
        a version as a rewrite of the first. It is not one when the two measured different things.
        A panel that lands between two audits widens the plan, and the second record then holds
        entries the first could not have had; refusing it would leave a build that ships more
        instruments unable to record what they found.

        So the identity is the pair: the version, and what was measured of it, which this reads as
        the set of (entry id, method identity) pairs the record holds. Not `producer.method_set`,
        which was the first spelling and does not work: the runner gives every absence in a section
        the one placeholder method `<section>.absence`, so a panel that lands and returns an absence
        of its own leaves that digest exactly where it was and would read as a rewrite of the record
        that had no such panel at all. And not the entry ids alone, which was the second spelling
        and stops one step short: re-measuring one entry under a method whose parameters have moved
        writes the same ids and is not the same measurement, and refusing it would leave a project
        unable to record what the new parameters found. Same version, same ids measured the same
        way, different bytes is still a rewrite and is still refused, which is the case the rule was
        written for; the reuse check in `product/audit/run.py` reads the same key, so a record the
        audit hands back unmeasured is exactly a record this would have refused.
        """
        version = data["subject"]["version"]["digest"]
        measured = _measured(data)
        for row in self._scan():
            path = self.path_for(row["name"])
            try:
                existing = path.read_bytes()
            except OSError:
                continue
            if existing == payload:
                continue  # the same record under another name is the same measurement
            try:
                other = json.loads(existing.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue  # not a record this store wrote; the file is not a measurement of anything
            if (
                other.get("subject", {}).get("version", {}).get("digest") == version
                and _measured(other) == measured
            ):
                raise SubjectVersionRewritten(
                    version=version, existing=row["name"], name=name
                )

    def names(self) -> tuple[str, ...]:
        """Every recorded name, oldest first."""
        return tuple(row["name"] for row in self._index())

    def open_record(self, name: str) -> Assay:
        path = self.path_for(name)
        return Assay.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def latest(self) -> Assay | None:
        rows = self._index()
        if not rows:
            return None
        return self.open_record(rows[-1]["name"])

    # --- the index, which is a cache ---------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self.assays_dir / "index.json"

    def _scan(self) -> list[dict]:
        if not self.assays_dir.is_dir():
            return []
        found = sorted(self.assays_dir.glob(f"*{RECORD_SUFFIX}"), key=lambda p: (p.stat().st_mtime, p.name))
        return [{"name": p.name[: -len(RECORD_SUFFIX)], "mtime": p.stat().st_mtime} for p in found]

    def _write_index(self, rows: list[dict]) -> None:
        self.assays_dir.mkdir(parents=True, exist_ok=True)
        self._index_path.write_bytes(rfc8785.dumps({"records": rows}))

    def _index(self) -> list[dict]:
        """The index if it still describes the files, else the files, which are the truth."""
        on_disk = self._scan()
        try:
            cached = json.loads(self._index_path.read_text(encoding="utf-8"))["records"]
        except (OSError, ValueError, KeyError, TypeError):
            cached = None
        if cached is not None and [r.get("name") for r in cached] == [r["name"] for r in on_disk]:
            return cached
        if on_disk:
            self._write_index(on_disk)
        return on_disk

    # --- the dependency graph -----------------------------------------------------------------

    def stale(self, assay: Assay, changed: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """`(stale, standing)` entry ids for this request, per the five rows of section 6.1."""
        return invalidation.partition(assay, changed)

    def decision_stale(self, assay: Assay, changed: Iterable[str]) -> bool:
        return invalidation.decision_stale(assay, changed)

    def changed_since(self, previous: Assay) -> frozenset[str]:
        """Which of the nine digests moved between that record and this project as it stands."""
        current = self.version_of(previous).digests.to_dict()
        declared = previous.to_dict()["subject"]["digests"]
        return frozenset(
            name
            for name, value in current.items()
            if value is not None and declared.get(name) != value
        )

    def reuse(
        self,
        previous: Assay,
        changed: Iterable[str] | None = None,
        *,
        started: str | None = None,
    ) -> Reuse:
        """What this rerun may carry forward from `previous`, and what it must measure again."""
        changes = self.changed_since(previous) if changed is None else frozenset(changed)
        stale_ids, standing_ids = self.stale(previous, changes)
        started = started or _now()
        version = self.version_of(previous)
        by_id = {entry.entry_id: entry for entry in previous.entries()}
        absences = tuple(
            _absent_for_now(by_id[entry_id], changes, started, version.digest())
            for entry_id in stale_ids
        )
        total = len(stale_ids) + len(standing_ids)
        if not changes:
            says = (
                f"reused all {total} entries of {previous.assay_id}: the dependency set is unchanged"
            )
        else:
            says = (
                f"reused {len(standing_ids)} of {total} entries of {previous.assay_id}; "
                f"{len(stale_ids)} went stale because {', '.join(sorted(changes))} changed"
            )
        return Reuse(
            changed=changes,
            reused_entry_ids=standing_ids,
            stale_entry_ids=stale_ids,
            absences=absences,
            unchanged=not changes,
            says=says,
            version=version,
        )


def _absent_for_now(
    before: Entry, changed: frozenset[str], started: str, subject_ref: str
) -> Entry:
    """The stale entry, marked not measured for this request. Real provenance, no placeholders."""
    what = ", ".join(sorted(changed)) or "the dependency set"
    return Entry(
        entry_id=before.entry_id,
        section=before.section,
        kind="absence",
        measurand=before.measurand,
        method=before.method,
        scope=before.scope,
        subject_ref=subject_ref,
        depends_on=list(before.depends_on),
        state="absent",
        provenance=EntryProvenance(
            started=started,
            duration_s=0.0,
            sandbox_tier="T0",
            offline=True,
            actor="reward_lens.store.reuse",
        ),
        limitations=[f"carried forward from an earlier assay, invalidated by {what}"],
        absence=Absence(
            state="NOT_MEASURED",
            missing_access=f"a measurement of this under the changed {what}",
            affected_claims=[before.measurand],
            remedy="re-run this instrument against the current version",
        ),
    )
