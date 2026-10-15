"""Identity types, enums, and addressing primitives for the kernel.

Everything comparable across runs gets a stable, content-derived id (section 2.1.1). Ids are
BLAKE2b-128 digests over canonical serializations, carried with human-readable prefixes so a
glance at a string tells you what kind of thing it names. Content derivation is what lets the
evidence store be a DAG: two runs that computed the same thing from the same inputs land on the
same id, and a derived quantity can point at the leaf measurements it consumed.

The enums here are load-bearing policy, not decoration. `Capability` is the declared contract
that replaces v1's duck-typed `hasattr` discovery (R3). `TrustLevel` is the ladder the three
gates climb (section 1.3); it is an `IntEnum` so "the highest applicable rung" is a max and card
rendering can sort by it. `GaugeStatus` is the typing that makes a raw-coordinate cross-model
number impossible to mistake for an invariant one (I3, gate 2).
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, NewType

# ---------------------------------------------------------------------------
# Content-derived identity
# ---------------------------------------------------------------------------

_HASH_BYTES = 16  # 128-bit digest, per section 2.1.1


def canonical_bytes(obj: Any) -> bytes:
    """Serialize an object to canonical bytes for hashing.

    The canonical form is JSON with sorted keys, no insignificant whitespace, and a small
    set of extensions for objects JSON does not natively handle (bytes, sets, tuples via the
    default list coercion, and objects exposing ``__canonical__`` or ``_asdict``). Floats are
    emitted with ``repr`` semantics via json, which is stable within a platform; ids that must
    survive across platforms should hash structural content (shapes, names, integer counts),
    not raw float payloads, and the callers in this codebase do exactly that.
    """

    def _default(o: Any) -> Any:
        if isinstance(o, bytes):
            return {"__bytes__": o.hex()}
        if isinstance(o, (set, frozenset)):
            return {"__set__": sorted(_default(x) if not _json_native(x) else x for x in o)}
        if hasattr(o, "__canonical__"):
            return o.__canonical__()
        if hasattr(o, "_asdict"):  # namedtuple-like
            return o._asdict()
        if hasattr(o, "__dict__"):
            return {k: v for k, v in sorted(vars(o).items()) if not k.startswith("_")}
        raise TypeError(f"cannot canonicalize {type(o).__name__}")

    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default
    ).encode("utf-8")


def _json_native(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None


def content_hash(obj: Any, prefix: str) -> str:
    """Return ``"{prefix}:{hexdigest}"`` for the canonical serialization of ``obj``.

    The prefix is the human-readable tag (``mfp``, ``ds``, ``ev`` and so on). The digest is
    BLAKE2b truncated to 128 bits, which is collision-safe at the scale of any evidence store
    and short enough to read.
    """
    digest = hashlib.blake2b(canonical_bytes(obj), digest_size=_HASH_BYTES).hexdigest()
    return f"{prefix}:{digest}"


def hash_bytes(data: bytes, prefix: str) -> str:
    """Content hash of raw bytes (streamed file content, tensor buffers)."""
    digest = hashlib.blake2b(data, digest_size=_HASH_BYTES).hexdigest()
    return f"{prefix}:{digest}"


# Stable identifier NewTypes. These are ``str`` at runtime; the NewType is documentation and a
# mypy guard so a DatasetID is never accidentally passed where a ModelFP is expected.
ModelFP = NewType("ModelFP", str)  # "mfp:..."   weights+config+tokenizer hash (section 2.2.5)
DatasetID = NewType("DatasetID", str)  # "ds:..."    dataset card hash (content + builder version)
DirectionID = NewType("DirectionID", str)  # "dir:..." persisted direction/probe hash
FrameID = NewType("FrameID", str)  # "frame:..." gauge frame hash
EvidenceID = NewType("EvidenceID", str)  # "ev:..."   assigned at store append
StudyID = NewType("StudyID", str)  # "study:name@vN#hash"
OrganismID = NewType("OrganismID", str)  # "org:..."


# ---------------------------------------------------------------------------
# Enums: capabilities, trust, gauge
# ---------------------------------------------------------------------------


class Capability(enum.Flag):
    """Declared capabilities of a reward signal (R3, section 2.3.2).

    Observables declare `requires: Capability`; the runner checks compatibility before any
    GPU work and fails with a precise message. This replaces v1's `hasattr(adapter, ...)`
    duck typing, where a missing method surfaced as a deep AttributeError or, worse, a
    silently skipped code path.
    """

    NONE = 0
    SCORES = enum.auto()
    PREFIX_SCORES = enum.auto()
    ACTIVATIONS = enum.auto()
    GRADIENTS = enum.auto()
    HVP = enum.auto()
    LINEAR_READOUT = enum.auto()
    MULTI_READOUT = enum.auto()
    STEP_SCORES = enum.auto()
    DISTRIBUTIONAL = enum.auto()
    SPAN_TYPES = enum.auto()
    GENERATIVE = enum.auto()
    PAIRED_MODELS = enum.auto()

    def missing_from(self, available: "Capability") -> "Capability":
        """Return the subset of ``self`` not present in ``available`` (empty if satisfied)."""
        return self & ~available


class TrustLevel(enum.IntEnum):
    """The trust ladder the three gates compute (section 1.3).

    Ordered so that comparisons and ``max`` express "the highest applicable rung". The level
    is never set by a caller; it is computed from whether the Evidence carries a calibration
    reference, whether it was produced under a frozen study, and whether it survived its kill
    criteria and review. See `reward_lens.core.gates.compute_trust`.
    """

    EXPLORATORY = 0  # default; anything computed ad hoc
    CALIBRATED = 1  # the observable has a scorecard entry covering this signal family + regime
    REGISTERED = 2  # computed under a frozen Study whose predictions predate the run
    ADJUDICATED = 3  # registered + calibrated + survived its own kill criteria and review

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


class GaugeStatus(enum.Enum):
    """How an Observable's value transforms under the reward gauge group (I3, gate 2).

    INVARIANT quantities are safe to compare across signals directly. COVARIANT quantities
    (directions, angles, subspace overlaps) require a shared `Frame` to compare and the
    comparison APIs take a frame argument with no default. RAW_ONLY quantities are computable
    and scientifically interesting (E19 proved this) but are typed as raw coordinates and
    rendered as such, never mistaken for invariant.
    """

    INVARIANT = "invariant"
    COVARIANT = "covariant"
    RAW_ONLY = "raw_only"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# ---------------------------------------------------------------------------
# Addressing: Site and Span
# ---------------------------------------------------------------------------

SitePoint = Literal["resid_pre", "resid_post", "attn_out", "mlp_out", "head_out", "embed"]


@dataclass(frozen=True, order=True)
class Site:
    """A location in a network (section 2.1.1).

    ``layer`` indexes the transformer block; ``point`` names the read/write surface within it;
    ``head`` selects an attention head where ``point == "head_out"`` and is None otherwise. The
    type is frozen and ordered so it can key the activation cache and sort deterministically in
    reports.
    """

    layer: int
    point: SitePoint = "resid_post"
    head: int | None = None

    def __canonical__(self) -> dict[str, Any]:
        return {"layer": self.layer, "point": self.point, "head": self.head}

    def __str__(self) -> str:
        h = f".h{self.head}" if self.head is not None else ""
        return f"L{self.layer}.{self.point}{h}"


@dataclass(frozen=True)
class Span:
    """A typed token interval ``[start, end)`` (section 2.1.1, extended in section 2.4).

    The ``kind`` tag is what makes span-level patching and attribution meaningful: a receipt
    span, an error step, a critique sentence, a verdict token. Core defines the primitive; the
    data plane's `spans.py` defines the vocabulary of kinds and the character-to-token mapping
    that produces exact spans.
    """

    start: int
    end: int
    kind: str = "text"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"span end {self.end} precedes start {self.start}")

    def __len__(self) -> int:
        return self.end - self.start

    def __canonical__(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "kind": self.kind, "meta": self.meta}


# ---------------------------------------------------------------------------
# SubjectRef: what an Evidence is about
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectRef:
    """The subject of a measurement (section 2.1.2).

    Names the signal(s) by fingerprint, the dataset, the readout, the frame (for covariant
    quantities), and any interventions applied, by fingerprint. Recording intervention
    fingerprints here is what makes an erased-model card structurally unable to masquerade as a
    base-model card: the interventions are part of the subject's identity.
    """

    signals: tuple[ModelFP, ...] = ()
    dataset: DatasetID | None = None
    readout: str | None = None
    frame: FrameID | None = None
    interventions: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "signals": list(self.signals),
            "dataset": self.dataset,
            "readout": self.readout,
            "frame": self.frame,
            "interventions": list(self.interventions),
            "extra": self.extra,
        }


__all__ = [
    "canonical_bytes",
    "content_hash",
    "hash_bytes",
    "ModelFP",
    "DatasetID",
    "DirectionID",
    "FrameID",
    "EvidenceID",
    "StudyID",
    "OrganismID",
    "Capability",
    "TrustLevel",
    "GaugeStatus",
    "SitePoint",
    "Site",
    "Span",
    "SubjectRef",
]
