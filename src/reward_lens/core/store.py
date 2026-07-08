"""The evidence store (section 2.1.2).

Append-only, file-backed, trivially inspectable. Envelopes are JSON Lines in ``evidence.jsonl``;
bulk arrays are content-addressed ``.npy`` sidecars under ``payloads/``. There is no database
server; the files are the interface, so the store is diffable and a human can read it. Cards,
the Atlas, papers, and safety cases are views over this store and never compute fresh numbers,
which is what guarantees a card and a paper cite identical values (I5, liability 4).

The store is a DAG: a derived Evidence names its parents in provenance, and the store refuses to
append a derived Evidence whose parents it cannot resolve (I5). That refusal is the mechanism
behind "every result must compose": you cannot record a forecast that consumed a KUI number the
store never saw.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterator

from reward_lens.core.config import get_settings
from reward_lens.core.errors import ProvenanceError
from reward_lens.core.evidence import Evidence, evidence_from_envelope
from reward_lens.core.types import EvidenceID, TrustLevel


class EvidenceStore:
    """A directory-backed, append-only store of Evidence.

    Not thread-safe across processes (files are the interface; use one writer), but guarded by a
    lock within a process. The in-memory index maps id to envelope for O(1) lookup; it is built
    once on construction by streaming the JSONL.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else get_settings().resolved_store()
        self.path.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.path / "evidence.jsonl"
        self.payloads = self.path / "payloads"
        self._lock = threading.RLock()
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        if not self.jsonl.exists():
            return
        with self.jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                env = json.loads(line)
                self._index[env["id"]] = env

    # -- write ---------------------------------------------------------------

    def append(self, evidence: Evidence[Any]) -> EvidenceID:
        """Append an Evidence, returning its id. Idempotent on content-derived ids.

        Enforces the DAG invariant: if the Evidence declares parents, every parent must already
        be in the store, else `ProvenanceError`. Re-appending an id that is already present is a
        no-op (the content is identical by construction), so replaying a study is safe.
        """
        with self._lock:
            for parent in evidence.provenance.parents:
                if parent not in self._index:
                    raise ProvenanceError(
                        f"Evidence {evidence.id} declares parent {parent} not present in the "
                        f"store; a derived quantity whose parents are missing is an error (I5). "
                        f"Append the parent measurements first."
                    )
            if evidence.id in self._index:
                return evidence.id
            env = evidence.envelope(sidecar_dir=self.payloads)
            with self.jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(env, ensure_ascii=False) + "\n")
                fh.flush()
            self._index[evidence.id] = env
            return evidence.id

    # -- read ----------------------------------------------------------------

    def __contains__(self, ev_id: str) -> bool:
        return ev_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def get(self, ev_id: str) -> Evidence[Any]:
        """Load an Evidence by id, reconstructing its typed payload."""
        env = self._index.get(ev_id)
        if env is None:
            raise KeyError(f"no evidence with id {ev_id}")
        return evidence_from_envelope(env, sidecar_dir=self.payloads)

    def __iter__(self) -> Iterator[Evidence[Any]]:
        for ev_id in list(self._index):
            yield self.get(ev_id)

    def find(
        self,
        observable: str | None = None,
        signal: str | None = None,
        dataset: str | None = None,
        readout: str | None = None,
        study: str | None = None,
        min_trust: TrustLevel | None = None,
        latest: bool = False,
    ) -> list[Evidence[Any]]:
        """Query the store with simple structural filters.

        ``signal`` matches an Evidence whose subject names that model fingerprint. ``latest``
        collapses to the most recently created Evidence per (observable, subject) key, which is
        the common "give me the current value" query. All filtering is over the in-memory index;
        for ad hoc analysis use `frame` to get a pandas DataFrame of envelopes.
        """
        out: list[dict[str, Any]] = []
        for env in self._index.values():
            if observable is not None and env["observable"] != observable:
                continue
            subj = env["subject"]
            if signal is not None and signal not in subj.get("signals", []):
                continue
            if dataset is not None and subj.get("dataset") != dataset:
                continue
            if readout is not None and subj.get("readout") != readout:
                continue
            if study is not None and env["provenance"].get("study") != study:
                continue
            if min_trust is not None and env["trust"] < int(min_trust):
                continue
            out.append(env)
        if latest:
            keyed: dict[tuple[str, str], dict[str, Any]] = {}
            for env in out:
                key = (env["observable"], json.dumps(env["subject"], sort_keys=True))
                cur = keyed.get(key)
                if cur is None or env["created_at"] > cur["created_at"]:
                    keyed[key] = env
            out = list(keyed.values())
        out.sort(key=lambda e: e["created_at"])
        return [evidence_from_envelope(e, sidecar_dir=self.payloads) for e in out]

    def frame(self) -> Any:
        """Return a pandas DataFrame of flattened envelopes for ad hoc queries."""
        import pandas as pd

        rows = []
        for env in self._index.values():
            rows.append(
                {
                    "id": env["id"],
                    "observable": env["observable"],
                    "version": env["observable_version"],
                    "signals": ",".join(env["subject"].get("signals", [])),
                    "dataset": env["subject"].get("dataset"),
                    "readout": env["subject"].get("readout"),
                    "trust": TrustLevel(env["trust"]).name,
                    "gauge": env["gauge"],
                    "calibrated": env.get("calibration") is not None,
                    "study": env["provenance"].get("study"),
                    "gpu_seconds": env["provenance"].get("cost", {}).get("gpu_seconds", 0.0),
                    "created_at": env["created_at"],
                }
            )
        return pd.DataFrame(rows)

    # -- DAG -----------------------------------------------------------------

    def parents(self, evidence: Evidence[Any]) -> list[Evidence[Any]]:
        """Return the immediate parent Evidence, raising if any is unresolvable."""
        out = []
        for pid in evidence.provenance.parents:
            if pid not in self._index:
                raise ProvenanceError(f"parent {pid} of {evidence.id} is not in the store")
            out.append(self.get(pid))
        return out

    def ancestors(self, evidence: Evidence[Any]) -> list[Evidence[Any]]:
        """Return the transitive closure of parents (the full derivation DAG of a quantity)."""
        seen: dict[str, Evidence[Any]] = {}
        frontier = list(evidence.provenance.parents)
        while frontier:
            pid = frontier.pop()
            if pid in seen:
                continue
            if pid not in self._index:
                raise ProvenanceError(f"ancestor {pid} of {evidence.id} is not in the store")
            anc = self.get(pid)
            seen[pid] = anc
            frontier.extend(anc.provenance.parents)
        return list(seen.values())


_DEFAULT_STORE: EvidenceStore | None = None


def default_store() -> EvidenceStore:
    """Return the process-wide default store (under the configured store path)."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = EvidenceStore()
    return _DEFAULT_STORE


def set_default_store(store: EvidenceStore) -> None:
    global _DEFAULT_STORE
    _DEFAULT_STORE = store


__all__ = ["EvidenceStore", "default_store", "set_default_store"]
