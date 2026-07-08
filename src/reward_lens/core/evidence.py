"""Evidence: the universal measurement return type (R1, section 2.1.2).

Every measurement API in the kernel returns ``Evidence[T]``, never a bare float. The Evidence
carries the typed value, its uncertainty, its gauge status, its calibration reference, its
trust level (computed by the gates, never set by a caller), and its provenance including the
parent Evidence it was derived from. This is the atom of the store and the reason a card and a
paper can be guaranteed to cite the same number.

The value payload ``T`` is a typed dataclass (or a primitive). It is serialized by the
`ValueCodec` below: primitives and small arrays inline into the JSON envelope, bulk arrays go to
content-addressed ``.npy`` sidecars so the store stays a diffable directory of files while large
tensors do not bloat the JSONL. Payload dataclasses register themselves with `register_payload`
so they round-trip exactly.
"""

from __future__ import annotations

import base64
import importlib
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

import numpy as np

from reward_lens.core.gates import CalibrationRef, compute_trust
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import (
    EvidenceID,
    GaugeStatus,
    SubjectRef,
    TrustLevel,
    content_hash,
)

T = TypeVar("T")

# Arrays with more elements than this go to a sidecar rather than inlining into the envelope.
_INLINE_ARRAY_MAX = 64


@dataclass(frozen=True)
class Uncertainty:
    """The uncertainty of a measurement (section 2.1.2).

    ``n`` is the nominal row count; ``n_effective`` is the lineage-aware effective sample size
    (section 2.10.1) which, on a dataset of clones, is far smaller than ``n`` and is the
    structural death of v1's fake-n failure class. ``seed_spread`` is the cross-seed standard
    deviation where a quantity is measured over multiple seeds. ``method`` names how the
    interval was produced ("bootstrap-bca", "analytic", "conformal", and crucially
    "bootstrap-CLONE-INFLATED" when a caller opted into resampling across clones).
    """

    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float | None = None
    n: int | None = None
    n_effective: float | None = None
    seed_spread: float | None = None
    method: str = "none"

    def __canonical__(self) -> dict[str, Any]:
        return {
            "ci_low": _num(self.ci_low),
            "ci_high": _num(self.ci_high),
            "ci_level": self.ci_level,
            "n": self.n,
            "n_effective": _num(self.n_effective),
            "seed_spread": _num(self.seed_spread),
            "method": self.method,
        }


def _num(x: float | None) -> float | str | None:
    """JSON cannot represent NaN/Inf portably; encode them as tagged strings."""
    if x is None:
        return None
    if isinstance(x, float):
        if np.isnan(x):
            return "__nan__"
        if np.isposinf(x):
            return "__inf__"
        if np.isneginf(x):
            return "__-inf__"
    return float(x)


def _unnum(x: float | str | None) -> float | None:
    if isinstance(x, str):
        return {"__nan__": float("nan"), "__inf__": float("inf"), "__-inf__": float("-inf")}[x]
    return x


# ---------------------------------------------------------------------------
# Value codec
# ---------------------------------------------------------------------------

_PAYLOAD_REGISTRY: dict[str, type] = {}


def register_payload(cls: type) -> type:
    """Register a dataclass as an Evidence value payload so it round-trips exactly.

    Decorate any dataclass used as ``Evidence.value``. The codec tags the encoded form with the
    fully qualified name and reconstructs the instance on read. Payloads must be dataclasses;
    their fields may be primitives, lists, dicts, numpy arrays, or other registered payloads.
    """
    if not is_dataclass(cls):
        raise TypeError(f"payload {cls.__name__} must be a dataclass")
    key = f"{cls.__module__}.{cls.__qualname__}"
    _PAYLOAD_REGISTRY[key] = cls
    return cls


class ValueCodec:
    """Encode/decode Evidence value payloads to a JSON-compatible envelope plus array sidecars.

    ``encode`` returns a JSON-serializable object; any array larger than the inline threshold is
    written to ``sidecar_dir`` as a content-addressed ``.npy`` file and referenced by name.
    ``decode`` inverts this, reconstructing registered dataclasses and loading sidecars.
    """

    def encode(self, value: Any, sidecar_dir: Any = None) -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return _num(value)
        if isinstance(value, np.floating):
            return _num(float(value))
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (list, tuple)):
            return {"__seq__": [self.encode(v, sidecar_dir) for v in value]}
        if isinstance(value, dict):
            return {"__map__": {str(k): self.encode(v, sidecar_dir) for k, v in value.items()}}
        if isinstance(value, np.ndarray):
            return self._encode_array(value, sidecar_dir)
        if is_dataclass(value) and not isinstance(value, type):
            key = f"{type(value).__module__}.{type(value).__qualname__}"
            return {
                "__type__": key,
                "fields": {
                    f.name: self.encode(getattr(value, f.name), sidecar_dir) for f in fields(value)
                },
            }
        raise TypeError(f"cannot encode value of type {type(value).__name__}")

    def decode(self, obj: Any, sidecar_dir: Any = None) -> Any:
        if obj is None or isinstance(obj, (bool, int)):
            return obj
        if isinstance(obj, float):
            return obj
        if isinstance(obj, str):
            return _unnum(obj) if obj.startswith("__") and obj.endswith("__") else obj
        if isinstance(obj, dict):
            if "__seq__" in obj:
                return [self.decode(v, sidecar_dir) for v in obj["__seq__"]]
            if "__map__" in obj:
                return {k: self.decode(v, sidecar_dir) for k, v in obj["__map__"].items()}
            if "__ndarray__" in obj:
                return self._decode_array(obj["__ndarray__"], sidecar_dir)
            if "__type__" in obj:
                return self._decode_dataclass(obj, sidecar_dir)
        return obj

    def _encode_array(self, arr: np.ndarray, sidecar_dir: Any) -> dict[str, Any]:
        arr = np.ascontiguousarray(arr)
        if arr.size <= _INLINE_ARRAY_MAX or sidecar_dir is None:
            return {
                "__ndarray__": {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
                }
            }
        from pathlib import Path

        sidecar_dir = Path(sidecar_dir)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        digest = content_hash(
            {"dtype": str(arr.dtype), "shape": list(arr.shape), "bytes": arr.tobytes()}, "arr"
        ).split(":")[1]
        name = f"{digest}.npy"
        path = sidecar_dir / name
        if not path.exists():
            np.save(path, arr)
        return {"__ndarray__": {"sidecar": name, "dtype": str(arr.dtype), "shape": list(arr.shape)}}

    def _decode_array(self, spec: dict[str, Any], sidecar_dir: Any) -> np.ndarray:
        if "b64" in spec:
            raw = base64.b64decode(spec["b64"])
            return np.frombuffer(raw, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"]).copy()
        from pathlib import Path

        if sidecar_dir is None:
            raise ValueError("array sidecar referenced but no sidecar_dir supplied")
        return np.asarray(np.load(Path(sidecar_dir) / spec["sidecar"]))

    def _decode_dataclass(self, obj: dict[str, Any], sidecar_dir: Any) -> Any:
        key = obj["__type__"]
        cls = _PAYLOAD_REGISTRY.get(key)
        decoded = {k: self.decode(v, sidecar_dir) for k, v in obj["fields"].items()}
        if cls is None:
            # Try to import the module so the decorator runs, then retry.
            module_name = key.rsplit(".", 1)[0]
            try:
                importlib.import_module(module_name)
            except ImportError:
                pass
            cls = _PAYLOAD_REGISTRY.get(key)
        if cls is None:
            return decoded  # best-effort: return the field dict if the type is unavailable
        return cls(**decoded)


_CODEC = ValueCodec()


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence(Generic[T]):
    """The universal typed measurement return value (section 2.1.2).

    Construct via `make_evidence`, which computes the content-derived id and the gate-computed
    trust level. The trust level is not a constructor argument on purpose: it is a function of
    the calibration reference and the registration status, and allowing a caller to set it would
    defeat the gates.
    """

    id: EvidenceID
    observable: str
    observable_version: str
    subject: SubjectRef
    value: T
    uncertainty: Uncertainty
    gauge: GaugeStatus
    calibration: CalibrationRef | None
    trust: TrustLevel
    provenance: Provenance
    created_at: str

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def envelope(self, sidecar_dir: Any = None) -> dict[str, Any]:
        """The JSON-serializable store envelope for this Evidence."""
        return {
            "id": self.id,
            "observable": self.observable,
            "observable_version": self.observable_version,
            "subject": self.subject.__canonical__(),
            "value": _CODEC.encode(self.value, sidecar_dir),
            "uncertainty": self.uncertainty.__canonical__(),
            "gauge": self.gauge.value,
            "calibration": self.calibration.__canonical__() if self.calibration else None,
            "trust": int(self.trust),
            "provenance": self.provenance.__canonical__(),
            "created_at": self.created_at,
        }


def make_evidence(
    *,
    observable: str,
    observable_version: str,
    subject: SubjectRef,
    value: T,
    uncertainty: Uncertainty | None = None,
    gauge: GaugeStatus = GaugeStatus.INVARIANT,
    calibration: CalibrationRef | None = None,
    provenance: Provenance | None = None,
    registered: bool = False,
    adjudicated: bool = False,
    created_at: str | None = None,
) -> "Evidence[T]":
    """Build an Evidence, computing its content id and gate-derived trust level.

    The id hashes the observable, subject, value, gauge, calibration, and provenance (excluding
    the wall-clock timestamp) so identical measurements from identical inputs share an id, which
    is what makes the store a deduplicating DAG. Trust is computed by `compute_trust`; passing
    ``registered=True`` (the study runner does this) yields REGISTERED, a calibration reference
    yields at least CALIBRATED, and the two together with ``adjudicated=True`` yield ADJUDICATED.
    """
    unc = uncertainty or Uncertainty()
    prov = provenance or Provenance()
    trust = compute_trust(calibration=calibration, registered=registered, adjudicated=adjudicated)
    id_material = {
        "observable": observable,
        "observable_version": observable_version,
        "subject": subject.__canonical__(),
        "value": _CODEC.encode(value, None),
        "uncertainty": unc.__canonical__(),
        "gauge": gauge.value,
        "calibration": calibration.__canonical__() if calibration else None,
        "trust": int(trust),
        "provenance": prov.__canonical__(),
    }
    ev_id = EvidenceID(content_hash(id_material, "ev"))
    ts = created_at or datetime.now(timezone.utc).isoformat()
    return Evidence(
        id=ev_id,
        observable=observable,
        observable_version=observable_version,
        subject=subject,
        value=value,
        uncertainty=unc,
        gauge=gauge,
        calibration=calibration,
        trust=trust,
        provenance=prov,
        created_at=ts,
    )


def evidence_from_envelope(env: dict[str, Any], sidecar_dir: Any = None) -> "Evidence[Any]":
    """Reconstruct an Evidence from its store envelope."""
    from reward_lens.core.provenance import Cost

    subj = env["subject"]
    subject = SubjectRef(
        signals=tuple(subj.get("signals", [])),
        dataset=subj.get("dataset"),
        readout=subj.get("readout"),
        frame=subj.get("frame"),
        interventions=tuple(subj.get("interventions", [])),
        extra=subj.get("extra", {}),
    )
    u = env["uncertainty"]
    unc = Uncertainty(
        ci_low=_unnum(u.get("ci_low")),
        ci_high=_unnum(u.get("ci_high")),
        ci_level=u.get("ci_level"),
        n=u.get("n"),
        n_effective=_unnum(u.get("n_effective")),
        seed_spread=_unnum(u.get("seed_spread")),
        method=u.get("method", "none"),
    )
    p = env["provenance"]
    c = p.get("cost", {})
    prov = Provenance(
        git_sha=p.get("git_sha", "unknown"),
        config_hash=p.get("config_hash"),
        seeds=tuple(p.get("seeds", [])),
        cost=Cost(
            gpu_seconds=c.get("gpu_seconds", 0.0),
            tokens=c.get("tokens", 0),
            wall_seconds=c.get("wall_seconds", 0.0),
        ),
        oracle_calls=tuple(p.get("oracle_calls", [])),
        parents=tuple(p.get("parents", [])),
        study=p.get("study"),
        extra=p.get("extra", {}),
    )
    cal = env.get("calibration")
    calibration = (
        CalibrationRef(
            scorecard_entry=cal["scorecard_entry"],
            organism_family=cal["organism_family"],
            regime_match=cal.get("regime_match", "exact"),
            operating_point=cal.get("operating_point"),
        )
        if cal
        else None
    )
    return Evidence(
        id=EvidenceID(env["id"]),
        observable=env["observable"],
        observable_version=env["observable_version"],
        subject=subject,
        value=_CODEC.decode(env["value"], sidecar_dir),
        uncertainty=unc,
        gauge=GaugeStatus(env["gauge"]),
        calibration=calibration,
        trust=TrustLevel(env["trust"]),
        provenance=prov,
        created_at=env["created_at"],
    )


__all__ = [
    "Uncertainty",
    "Evidence",
    "make_evidence",
    "evidence_from_envelope",
    "register_payload",
    "ValueCodec",
]
