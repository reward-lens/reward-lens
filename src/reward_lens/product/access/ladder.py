"""The access ladder: what this machine and this input can measure, and what each rung would take.

D-20. A rung never carries a status. It carries the tokens it *needs*, and one function measures
every token on this machine and this input; a rung is available when each token it names is
present, and otherwise carries the first missing token's reason and what bringing it would unlock.

That is why the wave-1 panel and the target panel of section 5.6 are the same code. Nothing here
says "the seeker is not in this build": the seeker rung needs the `seeker` token, the token is
measured by looking for the module, and when that packet lands the rung falls through to its own
reasons, a key and a cap or a local endpoint, with no line of this file edited.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reward_lens import execution

from . import endpoints

__all__ = [
    "ROWS",
    "RUNGS",
    "Row",
    "Rung",
    "Resolution",
    "api_key",
    "api_key_variables",
    "reference_set",
    "resolve",
    "rung_ids",
    "sandbox_probe",
]

NOT_IN_BUILD = "not in this build"

#: What an available rung says when only part of it held, so that the flag survives a `Capability`.
MEASURED = "measured on this machine"
PARTLY = ", in part"

#: What the interfaces say wave 1's audit fills, used when the audit engine is in the build and has
#: not declared `PANELS_FILLED` itself (interfaces section 10).
INTERFACE_WAVE_1_PANELS: dict[str, str] = {"validity": "partial", "reach": "partial"}

#: Where an API actor's key is read from when the seeker packet is not in the build to name its own
#: providers. The seeker owns that list; this is the fallback the ladder's table holds until then,
#: and it is the library's own credential set (`reward_lens.execution.limits.SECRET_NAMES`).
API_KEY_VARIABLES: tuple[str, ...] = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")

#: The variable that points the reference-set lookup at a directory of its own. Without it the
#: lookup is the kernel's home convention (`reward_lens.core.config`): `REWARD_LENS_HOME`, or the
#: user's `.reward_lens` directory, with `reference` under it.
REFERENCE_VARIABLE = "REWARD_LENS_REFERENCE_SET"
REFERENCE_HOME_VARIABLE = "REWARD_LENS_HOME"


# --- rows and rungs -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One line of the printed panel, or two when its rungs disagree."""

    key: str
    #: The label to print when every rung of the row is unavailable for one shared reason. Without
    #: it the row prints its rungs' own fragments joined, which is what a mixed row always does.
    collapsed: str | None = None
    #: An input row is printed only when the caller gave a project to scope the report to.
    scope: str = "machine"


@dataclass(frozen=True)
class Rung:
    """One capability: what it is called, what it needs, and what bringing that would unlock."""

    id: str
    row: str
    fragment: str
    needs: tuple[str, ...]
    unlocks: tuple[str, ...]
    cost: str = "0.00"
    #: The fragment to print when the rung is available but only partly, as a static arm is.
    partial_fragment: str | None = None


ROWS: tuple[Row, ...] = (
    Row("panels"),
    Row("verbs"),
    Row("seeker", collapsed="the seeker, either arm"),
    Row("calibration"),
    Row("input", scope="input"),
)

RUNGS: tuple[Rung, ...] = (
    # The report's panels, in the order the text formatter prints them (interfaces section 9).
    Rung("panel.validity", "panels", "validity", ("panel:validity",),
         ("the validity panel: the static checks, replay and decision coverage",),
         partial_fragment="validity (the static checks and replay)"),
    Rung("panel.soundness", "panels", "soundness", ("panel:soundness",),
         ("the soundness panel: whether the score means what the reward system claims",)),
    Rung("panel.reach", "panels", "reach", ("panel:reach",),
         ("the reach panel: what a policy can get at",),
         partial_fragment="reach (static exposure only)"),
    Rung("panel.exploits", "panels", "exploits", ("panel:exploits",),
         ("the exploits panel: what a seeker found and what it cost",)),
    Rung("panel.framing", "panels", "framing", ("panel:framing",),
         ("the framing panel: what the reward leaves out",)),
    Rung("panel.signal", "panels", "signal", ("panel:signal",),
         ("the signal panel: whether the score separates anything",)),
    Rung("panel.cost", "panels", "cost", ("panel:cost",),
         ("the cost panel: what a run of this reward system costs",)),
    # The other verbs.
    Rung("verb.trace", "verbs", "trace", ("verb:trace",),
         ("`reward-lens trace`: a run record read back as evidence",)),
    Rung("verb.compare", "verbs", "compare", ("verb:compare",),
         ("`reward-lens compare`: two reward systems on the six-rung ladder",)),
    Rung("verb.forecast", "verbs", "forecast", ("verb:forecast_issue",),
         ("`reward-lens forecast`: a pre-registered claim and its ledger",)),
    Rung("verb.improve", "verbs", "improve", ("verb:improve",),
         ("`reward-lens improve`: the repair a finding implies",)),
    # The seeker's two arms. Each needs the engine first and then its own actor.
    Rung("seeker.api", "seeker", "seeker with an API model", ("seeker", "key:api"),
         ("the seeker's black-box arm with an API model as the actor",),
         cost="2.00"),
    Rung("seeker.local", "seeker", "seeker with a local model", ("seeker", "endpoint:local"),
         ("the seeker's black-box arm with a local model as the actor, at no cost",)),
    # Metrology.
    Rung("calibration", "calibration", "calibration and limits of detection", ("reference",),
         ("a limit of detection for this substrate, and a calibrated reading rather than a number",)),
    # The input, when one was given.
    Rung("input.project", "input", "the project this report is scoped to", ("input:project",),
         ("a report scoped to this reward system rather than to the machine alone",)),
)


# --- measuring the tokens -------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """One measured fact a rung can depend on."""

    present: bool
    reason: str = ""
    partial: bool = False
    unlocks: tuple[str, ...] = field(default_factory=tuple)


def _audit_module() -> Any:
    """`reward_lens.product.audit` if this build holds it, consulting `sys.modules` first."""
    module = sys.modules.get("reward_lens.product.audit")
    if module is not None:
        return module
    try:
        if importlib.util.find_spec("reward_lens.product.audit") is None:
            return None
        return importlib.import_module("reward_lens.product.audit")
    except Exception:
        return None


def audit_panels() -> dict[str, str]:
    """Which panels the audit engine fills, and whether each is full or partial.

    Asked of the engine, never assumed: `PANELS_FILLED` when it declares one, the interfaces'
    wave-1 list when the engine is in the build and has not, and nothing at all when it is not.
    """
    module = _audit_module()
    if module is None:
        return {}
    declared = getattr(module, "PANELS_FILLED", None)
    if declared is None:
        return dict(INTERFACE_WAVE_1_PANELS)
    if isinstance(declared, dict):
        return {str(name): str(state) for name, state in declared.items()}
    return {str(name): "full" for name in declared}


def _verb_engine(name: str) -> bool:
    from reward_lens.api import _dispatch

    return _dispatch.engine(name) is not None


def _importable(module: str) -> bool:
    if module in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _seeker_module() -> Any:
    """`reward_lens.product.seeker` if this build holds it, consulting `sys.modules` first."""
    module = sys.modules.get("reward_lens.product.seeker")
    if module is not None:
        return module
    if not _importable("reward_lens.product.seeker"):
        return None
    try:
        return importlib.import_module("reward_lens.product.seeker")
    except Exception:
        return None


def api_key_variables() -> tuple[str, ...]:
    """The environment variables an API actor's key would be in, on this build.

    Asked of the seeker when the seeker is here, because the providers are the seeker's to declare:
    `API_KEY_VARIABLES` or `KEY_VARIABLES` if it names them outright, else the key variable of each
    entry in its `PROVIDERS`. Without that packet the answer is this module's documented names.
    """
    module = _seeker_module()
    if module is not None:
        for attribute in ("API_KEY_VARIABLES", "KEY_VARIABLES"):
            declared = getattr(module, attribute, None) or ()
            named = tuple(one for one in declared if isinstance(one, str) and one)
            if named:
                return named
        providers = getattr(module, "PROVIDERS", None) or ()
        entries = list(providers.values()) if isinstance(providers, dict) else list(providers)
        named = []
        for entry in entries:
            if isinstance(entry, str):
                continue
            for attribute in ("key_variable", "api_key_variable", "env_var"):
                value = getattr(entry, attribute, None)
                if isinstance(value, str) and value:
                    named.append(value)
                    break
        if named:
            return tuple(named)
    return API_KEY_VARIABLES


def api_key() -> tuple[str | None, tuple[str, ...]]:
    """Which of those variables carries a key on this machine, and every one that was looked in."""
    looked_in = api_key_variables()
    for name in looked_in:
        if os.environ.get(name, "").strip():
            return name, looked_in
    return None, looked_in


def reference_set() -> tuple[Path, tuple[Path, ...]]:
    """Where a certified reference material is looked for on this machine, and what is there.

    A reference material is a file the machine either has or has not, so the calibration rung is
    settled by looking rather than by a constant: `REWARD_LENS_REFERENCE_SET` when it points
    somewhere, else `reference` under the kernel's home.
    """
    declared = os.environ.get(REFERENCE_VARIABLE, "").strip()
    if declared:
        root = Path(declared).expanduser()
    else:
        home = os.environ.get(REFERENCE_HOME_VARIABLE, "").strip()
        root = Path(home or os.path.join(os.path.expanduser("~"), ".reward_lens")) / "reference"
    try:
        found = tuple(sorted(one for one in root.glob("*.json") if one.is_file()))
    except OSError:  # pragma: no cover - an unreadable directory answers as an absent one does
        found = ()
    return root, found


def _open_project(project: Path | str | None) -> tuple[bool, str]:
    if project is None:
        return False, "no project was given"
    opener = None
    try:
        from reward_lens.store import Project as opener  # noqa: N813
    except Exception:
        opener = None
    if opener is None:
        return False, f"the store is {NOT_IN_BUILD}, so a project cannot be read"
    try:
        opener.open(project)
    except Exception as refusal:
        return False, getattr(refusal, "message", str(refusal))
    return True, ""


def measure(project: Path | str | None = None) -> dict[str, Token]:
    """Every token a rung can name, measured here and now."""
    panels = audit_panels()
    tokens: dict[str, Token] = {}

    for rung in RUNGS:
        for need in rung.needs:
            if need in tokens:
                continue
            kind, _, argument = need.partition(":")
            if kind == "panel":
                state = panels.get(argument)
                tokens[need] = Token(
                    present=state is not None,
                    reason=NOT_IN_BUILD,
                    partial=state == "partial",
                )
            elif kind == "verb":
                tokens[need] = Token(present=_verb_engine(argument), reason=NOT_IN_BUILD)
            elif kind == "seeker":
                tokens[need] = Token(
                    present=_importable("reward_lens.product.seeker"), reason=NOT_IN_BUILD
                )
            elif kind == "extra":
                marker = _EXTRA_MARKERS.get(argument)
                tokens[need] = Token(
                    present=bool(marker) and _importable(marker),
                    reason=f"pip install 'reward-lens[{argument}]'"
                    + (f"  ({marker})" if marker else ""),
                )
            elif kind == "key":
                found, looked_in = api_key()
                tokens[need] = Token(
                    present=found is not None,
                    reason=(
                        "a key and a cap: "
                        + " or ".join(looked_in)
                        + " in the environment, then --seeker api --max-budget-usd 2.00"
                    ),
                )
            elif kind == "endpoint":
                found = endpoints.scan()
                tokens[need] = Token(
                    present=bool(found),
                    reason="an OpenAI-compatible endpoint; none found here",
                    unlocks=tuple(
                        f"{endpoints.actor_name(one)} as the actor" for one in found
                    ),
                )
            elif kind == "reference":
                root, materials = reference_set()
                tokens[need] = Token(
                    present=bool(materials),
                    # The panel prints a rung's reason and not its unlocks, and this sentence is
                    # the wave-1 golden's line for the calibration rung, which `fleet/golden/` owns.
                    # So where a reference material goes is named in `unlocks`, which the JSON form
                    # carries, and the sentence stays as the golden has it.
                    reason="no reference material for this substrate yet",
                    unlocks=(
                        ()
                        if materials
                        else (
                            f"a certified reference material in {root}, or {REFERENCE_VARIABLE} "
                            "pointing at a directory that holds one",
                        )
                    ),
                )
            elif kind == "input":
                readable, why = _open_project(project)
                tokens[need] = Token(present=readable, reason=why)
            else:  # pragma: no cover - a token nobody declared cannot be measured
                tokens[need] = Token(present=False, reason=f"nothing measures {need}")
    return tokens


#: Extra -> the distribution whose presence decides it. `judge` declares nothing yet (pyproject
#: keeps the name installable and the list empty), so nothing can mark it installed.
_EXTRA_MARKERS: dict[str, str] = {"trace": "pyarrow", "train": "torch", "sigstore": "sigstore"}

EXTRAS: tuple[str, ...] = ("trace", "judge", "train", "sigstore")


def installed_extras() -> tuple[str, ...]:
    """The declared extras whose marker distribution is importable here."""
    return tuple(
        name for name in EXTRAS if (_EXTRA_MARKERS.get(name) and _importable(_EXTRA_MARKERS[name]))
    )


# --- resolving the ladder --------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    rung: Rung
    status: str
    reason: str
    unlocks: tuple[str, ...]
    partial: bool

    @property
    def label(self) -> str:
        if self.partial and self.rung.partial_fragment:
            return self.rung.partial_fragment
        return self.rung.fragment


@dataclass(frozen=True)
class Resolution:
    """The measured ladder: every rung, and the panel's lines already grouped."""

    rungs: tuple[Resolved, ...]
    available: tuple[str, ...]
    unavailable: tuple[tuple[str, str], ...]


def _rows_for(project: Path | str | None) -> tuple[Row, ...]:
    return tuple(row for row in ROWS if row.scope != "input" or project is not None)


def rung_ids(project: Path | str | None = None) -> tuple[str, ...]:
    keys = {row.key for row in _rows_for(project)}
    return tuple(rung.id for rung in RUNGS if rung.row in keys)


def resolve(project: Path | str | None = None) -> Resolution:
    """Measure every token, settle every rung, and group the rungs into the panel's lines."""
    tokens = measure(project)
    rows = _rows_for(project)
    keys = {row.key for row in rows}

    settled: dict[str, Resolved] = {}
    for rung in RUNGS:
        if rung.row not in keys:
            continue
        missing = next((need for need in rung.needs if not tokens[need].present), None)
        if missing is None:
            extra = tuple(part for need in rung.needs for part in tokens[need].unlocks)
            partly = any(tokens[need].partial for need in rung.needs)
            settled[rung.id] = Resolved(
                rung=rung,
                status="available",
                reason=MEASURED + (PARTLY if partly else ""),
                unlocks=rung.unlocks + extra,
                partial=partly,
            )
            continue
        # What a rung would gain from a token it already has is still worth naming: the endpoint
        # that is here but cannot be used yet is the seeker's actor the moment the engine lands.
        extra = tuple(part for need in rung.needs for part in tokens[need].unlocks)
        settled[rung.id] = Resolved(
            rung=rung,
            status="unavailable",
            reason=tokens[missing].reason,
            unlocks=rung.unlocks + extra,
            partial=False,
        )

    available: list[str] = []
    unavailable: list[tuple[str, str]] = []
    for row in rows:
        members = [settled[rung.id] for rung in RUNGS if rung.row == row.key and rung.id in settled]
        if not members:
            continue
        held = [one for one in members if one.status == "available"]
        missing = [one for one in members if one.status != "available"]
        if held:
            available.append(", ".join(one.label for one in held))
        if not missing:
            continue
        groups: dict[str, list[Resolved]] = {}
        for one in missing:
            groups.setdefault(one.reason, []).append(one)
        if row.collapsed and len(groups) == 1 and len(missing) == len(members):
            unavailable.append((row.collapsed, next(iter(groups))))
            continue
        for reason, group in groups.items():
            unavailable.append((", ".join(one.label for one in group), reason))

    return Resolution(
        rungs=tuple(settled[rung.id] for rung in RUNGS if rung.id in settled),
        available=tuple(available),
        unavailable=tuple(unavailable),
    )


# --- the sandbox rung ------------------------------------------------------------------------------


def sandbox_probe() -> Any:
    """`execution.probe()`, with a probe that could not run turned into RL0401 naming it (D-37).

    `doctor` exits 0 whenever it produced its report; a probe that raised means there is no report
    to produce, and the tier that a record would carry is exactly the thing that could not be
    established. So this is the one path out of `doctor` that is not a report.
    """
    from reward_lens.execution.errors import SandboxTierUnavailable

    try:
        return execution.probe()
    except Exception as failure:
        raise SandboxTierUnavailable(
            tier="auto",
            probe="reward_lens.execution.probe()",
            reason=f"{type(failure).__name__}: {failure}",
        ) from failure


#: Tier -> what it is, in the D-37 vocabulary, and the mechanism the next rung up would add.
TIER_DESCRIPTION: dict[str, str] = {
    "T0": "a new session, a 0700 working directory and an allowlisted environment",
    "L0": "rlimits and PR_SET_NO_NEW_PRIVS",
    "L1": "Landlock",
    "L2": "Landlock + seccomp",
    "L3": "Landlock + seccomp inside bubblewrap --unshare-all",
    "M0": "rlimits and a resident-set supervisor",
    "M1": "sandbox-exec with a deny-by-default Seatbelt profile",
    "W0": "a job object",
    "W1": "a job object inside an AppContainer with zero capabilities",
}

TIER_MECHANISM: dict[str, str] = {
    "L0": "rlimits",
    "L1": "Landlock",
    "L2": "seccomp",
    "L3": "bubblewrap",
    "M1": "sandbox-exec",
    "W1": "an AppContainer",
}


def sandbox_line(probe: Any) -> tuple[str, int]:
    """The tier label and description, and how long the whole ladder took to probe."""
    ladder = execution.tier_ladder(probe.os)
    description = TIER_DESCRIPTION.get(probe.tier_held, probe.tier_held)
    index = ladder.index(probe.tier_held) if probe.tier_held in ladder else len(ladder) - 1
    if index + 1 < len(ladder):
        nxt = ladder[index + 1]
        result = probe.tiers.get(nxt)
        mechanism = TIER_MECHANISM.get(nxt, nxt)
        why = result.reason if result is not None else "it was not probed"
        description = f"{description}; {mechanism} not layered ({why})"
    total = sum(getattr(result, "ms", 0.0) for result in probe.tiers.values())
    return f"{probe.tier_held}: {description}", int(round(total))
