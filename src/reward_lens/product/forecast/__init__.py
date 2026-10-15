"""The checkpoint-bounded, append-only forecast issue/resolve ledger.

The API request carries a JSON ``record`` path (owned by the API declaration).
The document contains actual issue inputs; this module never manufactures a
forecast, baseline, checkpoint, outcome, or fit provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from reward_lens.forecast.evidence import (
    BASELINES,
    BaselineProvenance,
    ForecastRefusal,
    ForecastSpec,
    IssuedForecast,
    ResolvedForecast,
)
from reward_lens.forecast.evidence import (
    issue as seal_issue,
)
from reward_lens.forecast.evidence import (
    resolve as seal_resolve,
)

CAMPAIGN_EVIDENCE_ID = "ev:3990df7af1488c08ad62786d879b0b0f"
CAMPAIGN_SOURCE_SHA256 = "ae67d15dae7e3e6ec5737bd2b96476bca51a4d5fe4f361a31771eb329c157ee8"
_CAMPAIGN_RESOURCE = "resources/campaign/RESULTS.md"
_LEDGER_EXTENSION = "https://reward-lens.dev/forecast-ledger/v1"
_REQUIRED_T1_RESOLUTION_FIELDS = (
    "resolved_at",
    "wealth_by_baseline",
    "wealth_history_by_baseline",
    "first_crossing_by_baseline",
    "headline_state",
    "log10_wealth_hardest",
)


def _recorded_time(value: Any, *, field: str) -> datetime:
    """Parse a local tool-recorded time; this is not external event authentication."""
    if not isinstance(value, str):
        raise ForecastRefusal(f"FORECAST_RESOLUTION_MISMATCH {field} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ForecastRefusal(
            f"FORECAST_RESOLUTION_MISMATCH {field} is not canonical ISO-8601"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.isoformat() != value:
        raise ForecastRefusal(
            f"FORECAST_RESOLUTION_MISMATCH {field} must be timezone-aware canonical ISO-8601"
        )
    if parsed > datetime.now(UTC):
        raise ForecastRefusal(
            f"FORECAST_RESOLUTION_MISMATCH future recorded resolution time field={field}"
        )
    return parsed


def _campaign_from_evidence() -> dict[str, str]:
    """Load and authenticate the bundled historical campaign snapshot."""
    try:
        source = files("reward_lens.product.forecast").joinpath(_CAMPAIGN_RESOURCE).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise ForecastRefusal(
            "FORECAST_CAMPAIGN_EVIDENCE bundled campaign snapshot is missing"
        ) from exc
    source_sha256 = hashlib.sha256(source).hexdigest()
    if source_sha256 != CAMPAIGN_SOURCE_SHA256:
        raise ForecastRefusal(
            "FORECAST_CAMPAIGN_EVIDENCE bundled campaign snapshot checksum mismatch"
        )
    text = source.decode("utf-8")
    if CAMPAIGN_EVIDENCE_ID not in text:
        raise ForecastRefusal(
            "FORECAST_CAMPAIGN_EVIDENCE registered campaign evidence id is absent"
        )
    # The supplied provenance states that no counted reference-class base rate
    # exists. The public historical row therefore reports the comparator refusal.
    brier_matches = re.findall(
        r"claim value=([0-9.]+) ev="
        + re.escape(CAMPAIGN_EVIDENCE_ID)
        + r" field=metrics\.brier_directional",
        text,
    )
    n_matches = re.findall(
        r"claim value=([0-9]+) ev="
        + re.escape(CAMPAIGN_EVIDENCE_ID)
        + r" field=metrics\.n_directional",
        text,
    )
    if not brier_matches or not n_matches:
        raise ForecastRefusal(
            "FORECAST_CAMPAIGN_EVIDENCE cannot reconstruct campaign directional score"
        )
    return {
        "evidence_id": CAMPAIGN_EVIDENCE_ID,
        "source_sha256": source_sha256,
        "brier": str(Decimal(brier_matches[0])),
        "n": n_matches[0],
        "climatology": "refused:no_counted_base_rate",
    }


def _root(project: Any) -> Path:
    candidate = getattr(project, "root", None) if project is not None else None
    return Path(str(candidate)) if candidate else Path.cwd()


def _path(project: Any) -> Path:
    return _root(project) / ".reward-lens" / "forecast-ledger.jsonl"


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _seal_row(row: Mapping[str, Any], previous_digest: str | None) -> dict[str, Any]:
    payload = dict(row)
    payload["previous_digest"] = previous_digest
    payload["row_digest"] = _digest(payload)
    return payload


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(row).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ForecastRefusal(
                f"FORECAST_RESOLUTION_MISMATCH ledger line={number} is invalid JSON"
            ) from exc
        supplied = row.pop("row_digest", None)
        if supplied != _digest(row):
            raise ForecastRefusal(
                f"FORECAST_RESOLUTION_MISMATCH ledger line={number} row digest mismatch"
            )
        if row.get("previous_digest") != previous:
            raise ForecastRefusal(
                f"FORECAST_RESOLUTION_MISMATCH ledger line={number} chain mismatch"
            )
        row["row_digest"] = supplied
        rows.append(row)
        previous = supplied
    if rows:
        header = rows[0]
        if header.get("kind") != "header":
            raise ForecastRefusal("FORECAST_RESOLUTION_MISMATCH ledger header is missing")
        registered = header.get("spec")
        if not isinstance(registered, Mapping):
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH ledger header has no registered spec"
            )
        if header.get("spec_digest") != _spec_from_mapping(registered).digest():
            raise ForecastRefusal("FORECAST_RESOLUTION_MISMATCH ledger header spec digest mismatch")
    return rows


def _spec_from_mapping(value: Mapping[str, Any] | None) -> ForecastSpec:
    if not value:
        return ForecastSpec.default()
    return ForecastSpec(
        targets=tuple(value.get("targets", ("T1", "T2", "T3"))),
        threshold=Decimal(str(value.get("threshold", "0.5"))),
        cadence=int(value.get("cadence", 20)),
        horizon=int(value.get("horizon", 20)),
        baselines=tuple(value.get("baselines", BASELINES)),
        quantiles=tuple(
            Decimal(str(x)) for x in value.get("quantiles", ("0.1", "0.25", "0.5", "0.75", "0.9"))
        ),
        n_min=int(value.get("n_min", 20)),
        alpha=Decimal(str(value.get("alpha", "0.05"))),
        null_seed_set=tuple(int(x) for x in value.get("null_seed_set", range(1000))),
        null_constructions=tuple(value.get("null_constructions", BASELINES)),
        null_multiplicity=int(value.get("null_multiplicity", 4)),
    )


def _record_document(request: Any) -> Mapping[str, Any]:
    record = getattr(request, "record", None)
    if record is None:
        raise ForecastRefusal(
            "FORECAST_FUTURE_INPUT record is required: issue accepts only a checkpoint-bounded JSON record"
        )
    path = Path(str(record))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastRefusal(f"FORECAST_FUTURE_INPUT record={path} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ForecastRefusal("FORECAST_FUTURE_INPUT record must be a JSON object")
    return payload


def _provenance(value: Mapping[str, Any]) -> dict[str, BaselineProvenance]:
    raw = value.get("baseline_provenance")
    if not isinstance(raw, Mapping):
        raise ForecastRefusal(
            "FORECAST_BASELINE_LEAKAGE baseline_provenance is required for all four baselines"
        )
    try:
        return {
            name: BaselineProvenance(cutoff=int(row["cutoff"]), fit_digest=str(row["fit_digest"]))
            for name, row in raw.items()
            if isinstance(row, Mapping)
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ForecastRefusal(
            "FORECAST_BASELINE_LEAKAGE baseline provenance must contain cutoff and fit_digest"
        ) from exc


def _issued_from_input(spec: ForecastSpec, row: Mapping[str, Any], claim: str) -> IssuedForecast:
    required = (
        "target",
        "checkpoint",
        "source_name",
        "source_cutoff",
        "forecast",
        "baselines",
        "model_digest",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ForecastRefusal(f"FORECAST_FUTURE_INPUT record issue missing {','.join(missing)}")
    baselines = row["baselines"]
    if not isinstance(baselines, Mapping):
        raise ForecastRefusal("FORECAST_BASELINE_LEAKAGE baselines must be an object")
    return seal_issue(
        checkpoint=int(row["checkpoint"]),
        spec=spec,
        target=str(row["target"]),
        source_name=str(row["source_name"]),
        source_cutoff=int(row["source_cutoff"]),
        feature_cutoff=int(row.get("feature_cutoff", row["source_cutoff"])),
        model_selection_cutoff=(
            None
            if row.get("model_selection_cutoff") is None
            else int(row["model_selection_cutoff"])
        ),
        forecast=row["forecast"],
        baselines=baselines,
        model_digest=str(row["model_digest"]),
        baseline_provenance=_provenance(row),
        claim=str(row.get("claim", claim)),
    )


def _header(
    spec: ForecastSpec, campaign: Mapping[str, Any], previous: str | None
) -> dict[str, Any]:
    campaign = dict(campaign)
    return _seal_row(
        {
            "kind": "header",
            "spec": spec.canonical(),
            "spec_digest": spec.digest(),
            "campaign": campaign,
        },
        previous,
    )


def _issues(rows: list[dict[str, Any]]) -> dict[str, IssuedForecast]:
    result: dict[str, IssuedForecast] = {}
    for row in rows:
        if row.get("kind") == "issue":
            issued = IssuedForecast.from_issued_bytes(str(row["issued_bytes"]).encode("utf-8"))
            if issued.sealed_digest != row.get("sealed_digest") or issued.sealed_digest in result:
                raise ForecastRefusal("FORECAST_RESOLUTION_MISMATCH issued row identity is invalid")
            result[issued.sealed_digest] = issued
    return result


def _resolved(
    rows: list[dict[str, Any]], issues: Mapping[str, IssuedForecast]
) -> list[ResolvedForecast]:
    settled: list[ResolvedForecast] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("kind") != "resolve":
            continue
        digest = str(row.get("sealed_digest"))
        if digest not in issues or digest in seen:
            raise ForecastRefusal("FORECAST_RESOLUTION_MISMATCH duplicate or orphan resolution")
        issued = issues[digest]
        raw = row.get("outcome")
        outcome: bool | Decimal = (
            str(raw).lower() == "true" if issued.target == "T1" else Decimal(str(raw))
        )
        scored = seal_resolve(issued, outcome=outcome)
        if str(scored.score) != str(row.get("score")):
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH recorded score differs from sealed issue"
            )
        seen.add(digest)
        settled.append(scored)
    return settled


def _t1_issue_rows(
    rows: list[dict[str, Any]], issues: Mapping[str, IssuedForecast]
) -> list[dict[str, Any]]:
    ordered = [
        row
        for row in rows
        if row.get("kind") == "issue" and issues[str(row["sealed_digest"])].target == "T1"
    ]
    for sequence, row in enumerate(ordered):
        if row.get("t1_sequence") != sequence:
            raise ForecastRefusal(
                "FORECAST_FUTURE_BET T1 issue sequence is missing or non-monotone"
            )
        stakes = row.get("t1_stakes")
        if not isinstance(stakes, Mapping) or set(stakes) != set(BASELINES):
            raise ForecastRefusal(
                "FORECAST_FUTURE_BET T1 issue lacks four sealed predictable stakes"
            )
        _recorded_time(row.get("issued_at"), field="issued_at")
    return ordered


def _t1_state(rows: list[dict[str, Any]], issues: Mapping[str, IssuedForecast]) -> dict[str, Any]:
    """Replay only issuance-sequenced T1 settlements and verify each sealed stake."""
    from reward_lens.forecast.evidence import T1EProcess, t1_headline_state

    ordered = _t1_issue_rows(rows, issues)
    by_digest = {str(row["sealed_digest"]): row for row in ordered}
    processes = {name: T1EProcess() for name in BASELINES}
    settled = 0
    previous_resolution_time: datetime | None = None
    evidence: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("kind") != "resolve":
            continue
        digest = str(row.get("sealed_digest"))
        issued = issues.get(digest)
        if issued is None or issued.target != "T1":
            continue
        if settled >= len(ordered) or digest != str(ordered[settled]["sealed_digest"]):
            raise ForecastRefusal(
                "FORECAST_FUTURE_BET T1 settlement is not in immutable issuance order"
            )
        missing = [field for field in _REQUIRED_T1_RESOLUTION_FIELDS if field not in row]
        if missing:
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH required T1 chronology field missing="
                + ",".join(missing)
            )
        resolved_time = _recorded_time(row["resolved_at"], field="resolved_at")
        issued_time = _recorded_time(ordered[settled]["issued_at"], field="issued_at")
        if resolved_time < issued_time:
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH recorded resolution time predates its issue"
            )
        if previous_resolution_time is not None and resolved_time <= previous_resolution_time:
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH recorded resolution times are not strictly monotone"
            )
        scored = seal_resolve(issued, outcome=str(row.get("outcome")).lower() == "true")
        if str(scored.score) != str(row.get("score")):
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH recorded T1 score differs from sealed issue"
            )
        stakes = by_digest[digest]["t1_stakes"]
        for name, process in processes.items():
            process.update(
                forecast_brier=scored.score,
                baseline_brier=scored.baseline_scores[name],
                bet=Decimal(str(stakes[name])),
                resolution_index=settled,
                resolved_at=str(row["resolved_at"]),
            )
        snapshot = {name: str(process.wealth) for name, process in processes.items()}
        headline_state = t1_headline_state(processes)
        expected_log = headline_state["min_current_log10_wealth"]
        if (
            "wealth_history_by_baseline" in row
            and row.get("wealth_history_by_baseline") != headline_state["wealth_history"]
        ):
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH T1 wealth history differs from sealed chronology"
            )
        if "first_crossing_by_baseline" in row and row.get("first_crossing_by_baseline") != {
            "index": headline_state["first_crossing_index"],
            "time": headline_state["first_crossing_time"],
        }:
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH T1 first crossing differs from sealed chronology"
            )
        if "headline_state" in row and row.get("headline_state") != headline_state:
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH T1 headline state differs from sealed chronology"
            )
        if "wealth_by_baseline" in row and row.get("wealth_by_baseline") != snapshot:
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH T1 wealth snapshot differs from sealed chronology"
            )
        if "log10_wealth_hardest" in row and str(row.get("log10_wealth_hardest")) != expected_log:
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH T1 evidence snapshot differs from sealed chronology"
            )
        evidence[digest] = {
            "log10_wealth_hardest": expected_log,
            "current_wealth": snapshot,
            "first_crossing": {
                "index": headline_state["first_crossing_index"],
                "time": headline_state["first_crossing_time"],
            },
            "headline_state": headline_state,
        }
        settled += 1
        previous_resolution_time = resolved_time
    headline_state = t1_headline_state(processes)
    return {
        "processes": processes,
        "settled": settled,
        "ordered": ordered,
        "evidence": evidence,
        "current_wealth": headline_state["current_wealth"],
        "wealth_history": headline_state["wealth_history"],
        "first_crossing": {
            "index": headline_state["first_crossing_index"],
            "time": headline_state["first_crossing_time"],
        },
        "headline_state": headline_state,
    }


def _assay(command: str, request: Any, *, project: Any) -> Any:
    import time

    from reward_lens.api import _record

    return _record.absent_record(
        command=command,
        reproduce=[f"reward-lens forecast {command.split('_')[-1]}"],
        path=_root(project),
        offline=bool(getattr(request, "offline", True)),
        seed=int(getattr(request, "seed", 20260911)),
        started_monotonic=time.monotonic(),
    )


def issue(request: Any, *, project: Any = None, sandbox: Any = None) -> Any:
    """Seal every actual, checkpoint-bounded issue in the supplied record."""
    del sandbox
    document = _record_document(request)
    spec = _spec_from_mapping(
        document.get("spec") if isinstance(document.get("spec"), Mapping) else None
    )
    raw_issues = document.get("issues")
    if isinstance(raw_issues, Mapping):
        raw_issues = [raw_issues]
    if (
        not isinstance(raw_issues, list)
        or not raw_issues
        or not all(isinstance(row, Mapping) for row in raw_issues)
    ):
        raise ForecastRefusal("FORECAST_FUTURE_INPUT record requires a non-empty issues list")
    if sum(str(row.get("target")) == "T1" for row in raw_issues) > 1:
        raise ForecastRefusal(
            "FORECAST_FUTURE_BET issue record may contain only one T1 forecast; settle before the next issue"
        )
    path = _path(project)
    rows = _load(path)
    if rows and (rows[0].get("spec_digest") != spec.digest()):
        raise ForecastRefusal(
            "FORECAST_RESOLUTION_MISMATCH ledger spec digest differs from issued record"
        )
    previous = rows[-1]["row_digest"] if rows else None
    pending: list[dict[str, Any]] = []
    if not rows:
        header = _header(spec, _campaign_from_evidence(), previous)
        pending.append(header)
        rows.append(header)
        previous = header["row_digest"]
    prior = _issues(rows)
    for input_row in raw_issues:
        issued = _issued_from_input(spec, input_row, str(getattr(request, "claim", "")))
        if issued.sealed_digest in prior:
            raise ForecastRefusal(
                f"FORECAST_RESOLUTION_MISMATCH duplicate issue sealed_digest={issued.sealed_digest}"
            )
        issue_row: dict[str, Any] = {
            "kind": "issue",
            "sealed_digest": issued.sealed_digest,
            "issued_bytes": issued.issued_bytes.decode("utf-8"),
            "issued_at": datetime.now(UTC).isoformat(),
        }
        if issued.target == "T1":
            state = _t1_state(rows, prior)
            if state["settled"] != len(state["ordered"]):
                raise ForecastRefusal(
                    "FORECAST_FUTURE_BET issue requires every earlier T1 issue to settle first"
                )
            issue_row["t1_sequence"] = len(state["ordered"])
            issue_row["t1_stakes"] = {
                name: str(process.next_bet()) for name, process in state["processes"].items()
            }
        sealed = _seal_row(issue_row, previous)
        pending.append(sealed)
        rows.append(sealed)
        previous = sealed["row_digest"]
        prior[issued.sealed_digest] = issued
    # Validate the entire request before touching the append-only ledger. A
    # malformed later issue must not strand an earlier issue or a bare header.
    for row in pending:
        _append(path, row)
    return _assay("forecast_issue", request, project=project)


def resolve(request: Any, *, project: Any = None, sandbox: Any = None) -> Any:
    """Score one frozen issue, refusing mutated, absent, and duplicate resolutions."""
    del sandbox
    path = _path(project)
    rows = _load(path)
    issues = _issues(rows)
    digest = str(getattr(request, "forecast_id", ""))
    issued = issues.get(digest)
    if issued is None:
        raise ForecastRefusal(
            f"FORECAST_RESOLUTION_MISMATCH sealed_digest={digest} is not in {path}"
        )
    resolved = _resolved(rows, issues)
    if any(item.issued.sealed_digest == digest for item in resolved):
        raise ForecastRefusal(
            f"FORECAST_RESOLUTION_MISMATCH sealed_digest={digest} is already resolved"
        )
    raw = getattr(request, "outcome", None)
    if raw is None:
        raise ForecastRefusal("FORECAST_RESOLUTION_MISMATCH outcome is required")
    outcome: bool | Decimal
    if issued.target == "T1":
        if str(raw).lower() not in {"true", "false"}:
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH outcome must be true or false for T1"
            )
        outcome = str(raw).lower() == "true"
    else:
        try:
            outcome = Decimal(str(raw))
        except InvalidOperation as error:
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH outcome must be a fraction"
            ) from error
        if not outcome.is_finite() or not Decimal(0) <= outcome <= Decimal(1):
            raise ForecastRefusal(
                "FORECAST_RESOLUTION_MISMATCH outcome must be a fraction in [0, 1]"
            )
    scored = seal_resolve(issued, outcome=outcome)
    chronology: dict[str, Any] = {}
    resolved_at = datetime.now(UTC).isoformat()
    if issued.target == "T1":
        state = _t1_state(rows, issues)
        if state["settled"] >= len(state["ordered"]) or digest != str(
            state["ordered"][state["settled"]]["sealed_digest"]
        ):
            raise ForecastRefusal(
                "FORECAST_FUTURE_BET T1 resolution must settle the earliest issued unresolved forecast"
            )
        stakes = state["ordered"][state["settled"]]["t1_stakes"]
        for name, process in state["processes"].items():
            process.update(
                forecast_brier=scored.score,
                baseline_brier=scored.baseline_scores[name],
                bet=Decimal(str(stakes[name])),
                resolution_index=state["settled"],
                resolved_at=resolved_at,
            )
        from reward_lens.forecast.evidence import t1_headline_state

        headline_state = t1_headline_state(state["processes"])
        chronology = {
            "wealth_by_baseline": headline_state["current_wealth"],
            "wealth_history_by_baseline": headline_state["wealth_history"],
            "first_crossing_by_baseline": {
                "index": headline_state["first_crossing_index"],
                "time": headline_state["first_crossing_time"],
            },
            "headline_state": headline_state,
            "log10_wealth_hardest": headline_state["min_current_log10_wealth"],
        }
    row = _seal_row(
        {
            "kind": "resolve",
            "sealed_digest": digest,
            "outcome": str(raw).lower(),
            "score": str(scored.score),
            "baseline_scores": {name: str(value) for name, value in scored.baseline_scores.items()},
            "hardest_difference": str(scored.hardest_difference),
            "resolved_at": resolved_at,
            **chronology,
        },
        rows[-1]["row_digest"],
    )
    _append(path, row)
    return _assay("forecast_resolve", request, project=project)


def render_ledger(project: Any = None) -> str:
    """Validate and render campaign plus local rows, with local rows worst score first."""
    rows = _load(_path(project))
    if not rows:
        return "forecast ledger is empty"
    issues = _issues(rows)
    settled = _resolved(rows, issues)
    chronology = _t1_state(rows, issues)
    evidence = chronology["evidence"]
    campaign = rows[0].get("campaign", {})
    if not isinstance(campaign, Mapping):
        raise ForecastRefusal(
            "FORECAST_CAMPAIGN_EVIDENCE ledger header campaign evidence is invalid"
        )
    campaign_baseline = str(campaign.get("climatology", "refused:no_counted_base_rate"))
    campaign_brier = str(campaign.get("brier", "unavailable"))
    resolution_times = {
        str(row["sealed_digest"]): str(row["resolved_at"])
        for row in rows
        if row.get("kind") == "resolve"
    }
    lines = [
        "run | issue_step | horizon | target | outcome | score | persistence | climatology | drift | covariance | hardest_baseline | hardest_diff | min_current_log10_wealth | model_digest | feature_cutoff | spec_digest | resolved_at | censored",
    ]
    for scored in sorted(settled, key=lambda row: row.score, reverse=True):
        issued = scored.issued
        lines.append(
            " | ".join(
                (
                    issued.source_name,
                    str(issued.checkpoint),
                    str(rows[0]["spec"]["horizon"]),
                    issued.target,
                    str(scored.outcome).lower(),
                    str(scored.score),
                    *(str(scored.baseline_scores[name]) for name in BASELINES),
                    min(BASELINES, key=lambda name: scored.baseline_scores[name]),
                    str(scored.hardest_difference),
                    str(evidence.get(issued.sealed_digest, {}).get("log10_wealth_hardest", "-")),
                    issued.model_digest,
                    str(issued.feature_cutoff),
                    issued.spec_digest,
                    resolution_times[issued.sealed_digest],
                    "false",
                )
            )
        )
    lines.append(
        f"campaign | - | - | T1 | - | {campaign_brier} | - | {campaign_baseline} | - | - | - | - | - | {campaign.get('evidence_id', 'campaign-results')} | - | {rows[0]['spec_digest']} | - | false"
    )
    headline = chronology["headline_state"]
    lines.append(
        "headline | "
        f"qualified={str(headline['qualified']).lower()} | "
        f"criterion={headline['criterion']} | "
        f"crossed={','.join(headline['crossed_baselines']) or '-'} | "
        f"min_current_log10_wealth={headline['min_current_log10_wealth']}"
    )
    return "\n".join(lines)


def ledger_machine_state(project: Any = None) -> dict[str, Any]:
    """Return replayed T1 crossing history and diagnostics for machine consumers."""
    rows = _load(_path(project))
    if not rows:
        return {
            "first_crossing": {"index": {}, "time": {}},
            "current_wealth": {},
            "wealth_history": {},
            "headline_state": {"qualified": False, "criterion": "ledger_is_empty"},
        }
    chronology = _t1_state(rows, _issues(rows))
    return {
        "first_crossing": chronology["first_crossing"],
        "current_wealth": chronology["current_wealth"],
        "wealth_history": chronology["wealth_history"],
        "headline_state": chronology["headline_state"],
    }


def ledger(request: Any, *, project: Any = None, sandbox: Any = None) -> Any:
    del sandbox
    state = ledger_machine_state(project)
    record = _assay("forecast_ledger", request, project=project)
    record.extensions = {_LEDGER_EXTENSION: state}
    from reward_lens import contracts

    record.assay_id = contracts.digest(record)
    return record


__all__ = ["issue", "ledger", "ledger_machine_state", "render_ledger", "resolve"]
