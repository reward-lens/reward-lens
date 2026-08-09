"""The honest record a verb returns when its instrument is not in this build.

Ten sections, ten absences, one hole apiece. A hole is a first-class state of the record, so a
verb whose engine has not landed answers rather than raising, and the answer says exactly what is
missing and what would fix it.

Nothing here invents a measurement, and nothing here invents a subject. Two fields the frozen
schema requires non-null have no honest value in a build that reads nothing:
`subject.version.digest` and `subject.digests.instrument_method`. Each gets the digest of a small
mapping that says truthfully what it is the name of, and every entry carries the limitation that
says so out loud. The alternative, a zero digest or a plausible hash of the path, is a lie that
validates, which is the failure mode A-004 exists to close.
"""

from __future__ import annotations

import platform
import re
import sys
import time
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path
from typing import Any, Iterable

from reward_lens import contracts

__all__ = ["MISSING_ACCESS", "SECTIONS", "absent_record"]

#: The one string the fleet agreed on for "this build ships no instrument for this".
MISSING_ACCESS = "instrument not in this build"

_NO_SUBJECT = (
    "no subject was read: this build holds no instrument, so subject.version.digest names the "
    "unread request rather than the reward system's source"
)

#: Section -> (entry id suffix, what the section measures, the claim the absence costs).
SECTIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "validity",
        "instrument_absent",
        "whether the grader measures what it is claimed to measure",
        "no validity claim about this reward system",
    ),
    (
        "soundness",
        "instrument_absent",
        "whether the score survives the transformations that must not change it",
        "no soundness claim about this reward system",
    ),
    (
        "reach",
        "instrument_absent",
        "what the graded process can reach that the score depends on",
        "no claim about what the graded process can reach",
    ),
    (
        "exploits",
        "instrument_absent",
        "responses that score well without satisfying the intent",
        "no claim that this reward system resists exploitation",
    ),
    (
        "framing",
        "instrument_absent",
        "how the choice of tasks and comparisons frames the score",
        "no claim about the framing of the comparison",
    ),
    (
        "reward_statistics",
        "instrument_absent",
        "the distribution of the reward signal over the task set",
        "no claim about the distribution of the reward",
    ),
    (
        "signal",
        "instrument_absent",
        "how much of the score separates better responses from worse",
        "no claim that the score carries usable signal",
    ),
    (
        "trace",
        "instrument_absent",
        "what a training run under this reward system actually optimised",
        "no claim about what training optimised",
    ),
    (
        "forecast",
        "instrument_absent",
        "what this reward system is expected to do to a policy",
        "no forecast about this reward system",
    ),
    (
        "calibration",
        "instrument_absent",
        "how the measured quantities line up with the outcomes later observed",
        "no claim that the measurements are calibrated",
    ),
)

_REMEDY = (
    "install a reward-lens build whose {section} instruments are present, then run the same "
    "command again; this build ships none of them"
)

_ABSENCE_TAKES_PROVENANCE = "provenance" in signature(contracts.absence).parameters


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tool_version() -> str:
    try:
        from importlib.metadata import version

        return version("reward-lens")
    except Exception:
        from reward_lens import __version__

        return __version__


def _os_name() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    return "windows"


def _ident(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:/-]", "-", text).strip("-")[:200]
    return cleaned if cleaned and cleaned[0].isalnum() else "unknown"


def _entry_provenance(started: str, offline: bool) -> contracts.EntryProvenance:
    return contracts.EntryProvenance(
        started=started, duration_s=0.0, sandbox_tier="T0", offline=offline
    )


def _absence_entry(
    section: str,
    suffix: str,
    measurand: str,
    claim: str,
    *,
    subject_ref: str,
    started: str,
    offline: bool,
) -> contracts.Entry:
    """One honest absence, built through the contracts so that the record stays theirs."""
    provenance = _entry_provenance(started, offline)
    extra: dict[str, Any] = {"subject_ref": subject_ref}
    if _ABSENCE_TAKES_PROVENANCE:
        extra["provenance"] = provenance
    entry = contracts.absence(
        section,
        f"{section}.{suffix}",
        measurand,
        MISSING_ACCESS,
        _REMEDY.format(section=section),
        (claim,),
        state="NOT_MEASURED",
        **extra,
    )
    entry.provenance = provenance
    entry.limitations = [_NO_SUBJECT]
    return entry


def absent_record(
    *,
    command: str,
    reproduce: Iterable[str],
    path: Path | None = None,
    offline: bool = True,
    seed: int = 20260911,
    started_monotonic: float | None = None,
) -> contracts.Assay:
    """A record that measures nothing and says so, section by section."""
    started = _now()
    subject_name = _ident(path.name) if path is not None else "unknown"
    version_digest = contracts.digest(
        {"unread_subject": {"command": command, "path": str(path) if path else None}}
    )
    instrument_method = contracts.digest({"instruments": []})
    subject = contracts.Subject(
        reward_system=contracts.models.RewardSystemRef(id=subject_name, name=subject_name),
        version=contracts.models.VersionRef(
            id="unread", digest=version_digest, parents=[], declared_change=None
        ),
        context=contracts.models.ContextRef(policy=None, task_set=None, configuration=None),
        digests=contracts.models.Digests(
            source=None,
            environment=None,
            scorer_config=None,
            task_distribution=None,
            samples=None,
            outcome_protocol=None,
            policy=None,
            training_semantics=None,
            instrument_method=instrument_method,
        ),
    )
    entries = {
        section: [
            _absence_entry(
                section,
                suffix,
                measurand,
                claim,
                subject_ref=version_digest,
                started=started,
                offline=offline,
            )
        ]
        for section, suffix, measurand, claim in SECTIONS
    }
    method_ids = sorted(entry.method.id for group in entries.values() for entry in group)
    elapsed = 0.0 if started_monotonic is None else max(0.0, time.monotonic() - started_monotonic)
    record = contracts.Assay(
        schema_url=contracts.SCHEMA_URL,
        assay_id="sha256:" + "0" * 64,
        created=started,
        producer=contracts.Producer(
            tool="reward-lens",
            version=_tool_version(),
            method_set=contracts.digest({"methods": method_ids}),
        ),
        subject=subject,
        intent=contracts.Intent(
            success=None,
            constraints=[],
            non_equivalent_pairs=[],
            outcome_check=contracts.models.OutcomeCheck(
                state="unqualified", kind=None, provenance=None, measured_disagreement=None
            ),
            attack_budget=contracts.models.AttackBudget(seeker_calls=0, usd="0.00"),
            partitions=[],
        ),
        measurement=contracts.models.Measurement(**entries),
        holes=[],
        rules=[],
        findings=[],
        join=[],
        experiments=[],
        calibration_links=[],
        tables=[],
        decision=contracts.Decision(
            state="unresolved",
            action=None,
            policy=None,
            reasons=["required_missing:subject"]
            + [f"required_missing:{section}" for section, *_ in SECTIONS],
            tradeoff=None,
            signature=None,
            regression_cases=[],
        ),
        cost=contracts.Cost(wall_s=float(elapsed), cpu_s=0.0, usd="0.00", api_calls=0),
        embedding=contracts.Embedding(tier="A", omitted_tables=[]),
        attestation=contracts.Attestation(statement_digest=None, backend=None),
        provenance=contracts.Provenance(
            offline=offline,
            sandbox_tier="T0",
            os=_os_name(),
            seed=seed,
            reproduce=list(reproduce),
        ),
        environment_excluded_from_digest={
            "python": platform.python_version(),
            "platform": sys.platform,
        },
    )
    record.holes_from_entries()
    record.assay_id = contracts.digest(record)
    return record
