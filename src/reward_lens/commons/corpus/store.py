"""A small file-backed store whose cases bind to real Project records."""

from __future__ import annotations

import json
from pathlib import Path

import rfc8785

from reward_lens.contracts import Assay, digest
from reward_lens.store import Project

from .case import Case, RecordSource

__all__ = ["CorpusStore"]


class CorpusStore:
    """Persist cases as canonical JSON and validate their source on every reopen."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def cases_dir(self) -> Path:
        return self.root / "cases"

    def path_for(self, case_id: str) -> Path:
        # Case validates ids before writing; this second check protects callers of path_for itself.
        if "/" in case_id or "\\" in case_id or case_id in {"", ".", ".."}:
            raise ValueError("case id is not a safe corpus path component")
        return self.cases_dir / f"{case_id}.case.json"

    def save(self, case: Case, *, project: Project, record_name: str) -> Case:
        """Bind and persist ``case`` to the named, already-persisted project assay.

        There is intentionally no API that accepts a record dictionary or a caller-supplied
        digest.  The store reopens the project record, then writes only its path and verified id.
        """
        if not isinstance(case, Case):
            raise TypeError("corpus save requires a Case")
        if not isinstance(project, Project):
            raise TypeError("corpus save requires a Project")
        assay_path = project.path_for(record_name)
        if not assay_path.is_file():
            raise ValueError(f"project record {record_name!r} is not persisted")
        assay = project.open_record(record_name)
        source = RecordSource.from_assay(assay_path, assay)
        stored = case.with_source(source)
        payload = rfc8785.dumps({"case": stored.to_dict(), "source": source.to_dict()})
        destination = self.path_for(stored.id)
        if destination.exists():
            if destination.read_bytes() == payload:
                return stored
            raise ValueError(f"corpus case {stored.id!r} already exists and is immutable")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return stored

    def reopen(self, case_id: str) -> Case:
        path = self.path_for(case_id)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"corpus case {case_id!r} is not persisted") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError(f"corpus case {case_id!r} is not valid JSON") from None
        if not isinstance(document, dict) or set(document) != {"case", "source"}:
            raise ValueError(f"corpus case {case_id!r} has an invalid persisted shape")
        if not isinstance(document["case"], dict) or not isinstance(document["source"], dict):
            raise ValueError(f"corpus case {case_id!r} has invalid case or source data")
        case = Case.from_dict(document["case"])
        if case.id != case_id:
            raise ValueError(
                f"corpus path id {case_id!r} disagrees with stored case id {case.id!r}"
            )
        source = RecordSource.from_dict(document["source"])
        self._validate_source(source)
        return case.with_source(source)

    def all(self) -> tuple[Case, ...]:
        if not self.cases_dir.is_dir():
            return ()
        return tuple(
            self.reopen(path.name.removesuffix(".case.json"))
            for path in sorted(self.cases_dir.glob("*.case.json"))
        )

    def search(self, query: str, *, actor_context):
        """Search through the public safe-context route exposed by :mod:`.search`."""
        from .search import search

        return search(query, actor_context=actor_context, corpus=self)

    @staticmethod
    def _validate_source(source: RecordSource) -> None:
        try:
            assay = Assay.model_validate_json(source.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"case source assay is no longer persisted at {source.path}") from None
        except (UnicodeDecodeError, ValueError):
            raise ValueError(f"case source assay at {source.path} is invalid") from None
        actual = digest(assay)
        if assay.assay_id != actual or source.assay_id != actual:
            raise ValueError(f"case source assay digest changed at {source.path}")
