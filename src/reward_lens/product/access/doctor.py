"""`doctor`, `dry_run`, and the two forms the panel is printed in (D-20, section 5.6).

The report is what was asked for, so producing it is success: `doctor` exits 0 whenever it produced
one, and the one path out that is not a report is a probe that could not run (RL0401). `audit
--dry-run` prints this same panel scoped to the planned run, and executes nothing.
"""

from __future__ import annotations

import inspect
import platform as _platform
import sys
from pathlib import Path
from typing import Any

from reward_lens.api import Capabilities, Capability, InstallInfo, Plan

from . import ladder, matrix

__all__ = [
    "INTERFACE_PANELS",
    "doctor",
    "dry_run",
    "render_json",
    "render_text",
]

#: The ten panels of the interfaces' panel rule (section 9), in the order the formatter prints them.
#: What `dry_run` reports when the audit engine is not in the build and there is no measured plan.
INTERFACE_PANELS: tuple[str, ...] = (
    "validity",
    "soundness",
    "reach",
    "exploits",
    "framing",
    "signal",
    "cost",
    "trace",
    "forecast",
    "calibration",
)

_GAP = 6


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("reward-lens")
    except Exception:  # pragma: no cover - an uninstalled checkout
        from reward_lens import __version__

        return __version__


def _install() -> InstallInfo:
    return InstallInfo(
        version=_version(),
        python=_platform.python_version(),
        platform=f"{_platform.system().lower()} {_platform.machine()}",
        extras=ladder.installed_extras(),
    )


def doctor(*, project: Path | str | None = None) -> Capabilities:
    """What this machine and this input can measure, what they cannot, and what each would take."""
    probe = ladder.sandbox_probe()
    resolution = ladder.resolve(project)
    return Capabilities(
        capabilities=[
            Capability(
                id=one.rung.id,
                status=one.status,
                reason=one.reason,
                unlocks=list(one.unlocks),
                cost=one.rung.cost,
            )
            for one in resolution.rungs
        ],
        sandbox=probe,
        install=_install(),
    )


def render_json(capabilities: Capabilities) -> list[dict[str, Any]]:
    """One object per capability, `{id, status, reason, unlocks, cost}`, so an agent branches on a
    field rather than on a sentence (section 5.6)."""
    return [
        {
            "id": one.id,
            "status": one.status,
            "reason": one.reason,
            "unlocks": list(one.unlocks),
            "cost": one.cost,
        }
        for one in capabilities
    ]


def _limits_line() -> str:
    from reward_lens.execution import DEFAULT_LIMITS as limits

    wall = int(limits.wall_s) if float(limits.wall_s).is_integer() else limits.wall_s
    gigabytes = limits.memory_bytes / 2**30
    memory = f"{int(gigabytes)} GB" if gigabytes.is_integer() else f"{gigabytes:.1f} GB"
    network = "no network" if not limits.network else "network allowed"
    return f"{wall} s wall, {memory}, {limits.processes} procs, {network}"


def _network_line(probe: Any) -> str:
    from reward_lens.execution.limits import egress_mechanism

    mechanism = egress_mechanism(probe.tier_held, landlock_abi=probe.landlock_abi)
    if mechanism:
        return "offline is available and is the default for local graders"
    return (
        f"offline cannot be enforced at {probe.tier_held} on this machine: nothing here filters "
        "egress, so a run that needs it says so in the record"
    )


def render_text(capabilities: Capabilities, *, plan: Plan | None = None) -> str:
    """The panel of section 5.6. With a plan, the same panel plus what the planned run would do."""
    probe = capabilities.sandbox
    # Render the capabilities handed over, never a second measurement: the report is the argument.
    resolution = _resolution_of(capabilities)
    install = capabilities.install
    extras = ", ".join(install.extras) if install.extras else "none"
    local = "yes" if _audits_local_graders() else "no"

    lines: list[str] = [""]
    lines.append("  Install")
    lines.append(
        f"    reward-lens {install.version}  ·  python {install.python}  ·  "
        f"{install.platform}  ·  {sys.prefix}"
    )
    lines.append(f"    extras installed: {extras}          audit on local graders: {local}")
    lines.append("")
    lines.append("  Sandbox")
    label, milliseconds = ladder.sandbox_line(probe)
    lines.append(f"    {label}     probed in {milliseconds} ms, held")
    lines.append(f"    per call: {_limits_line()}")
    lines.append("")
    if resolution.available:
        lines.append("  What this machine can measure now")
        lines.extend(f"    ✔ {one}" for one in resolution.available)
        lines.append("")
    if resolution.unavailable:
        lines.append("  What it cannot, and what each would take")
        width = max(len(label) for label, _ in resolution.unavailable)
        for label, reason in resolution.unavailable:
            lines.append(f"    ○ {label.ljust(width)}{' ' * _GAP}{reason}")
        lines.append("")
    if plan is not None:
        lines.append("  Plan")
        panels = ", ".join(plan.panels) if plan.panels else "no panel"
        lines.append(f"    {plan.command}: {panels}")
        lines.append(f"    {plan.paid_calls} paid calls, ${plan.estimate_usd} estimated")
        for note in plan.notes:
            lines.append(f"    {note}")
        lines.append("")
    lines.append(f"  Network   {_network_line(probe)}")
    first = "for everything in the first list" if resolution.available else "and nothing is in the first list"
    lines.append(f"  Cost      $0.00 {first}")
    lines.append("")
    return "\n".join(lines)


def _audits_local_graders() -> bool:
    """Both halves are measured: a family that conformed, and an audit engine with a panel to fill."""
    return bool(matrix.runs_local_graders()) and bool(ladder.audit_panels())


def _resolution_of(capabilities: Capabilities) -> ladder.Resolution:
    """Group the capabilities that were handed over into the panel's lines.

    The panel is a rendering of this report and never a second measurement, so a caller holding a
    `Capabilities` from an hour ago prints what it says rather than what the machine says now.
    """
    settled = {one.id: one for one in capabilities}
    rungs: list[ladder.Resolved] = []
    for rung in ladder.RUNGS:
        found = settled.get(rung.id)
        if found is None:
            continue
        rungs.append(
            ladder.Resolved(
                rung=rung,
                status=found.status,
                reason=found.reason,
                unlocks=tuple(found.unlocks),
                partial=ladder.PARTLY in found.reason,
            )
        )
    available: list[str] = []
    unavailable: list[tuple[str, str]] = []
    for row in ladder.ROWS:
        members = [one for one in rungs if one.rung.row == row.key]
        if not members:
            continue
        held = [one for one in members if one.status == "available"]
        missing = [one for one in members if one.status != "available"]
        if held:
            available.append(", ".join(one.label for one in held))
        if not missing:
            continue
        groups: dict[str, list[ladder.Resolved]] = {}
        for one in missing:
            groups.setdefault(one.reason, []).append(one)
        if row.collapsed and len(groups) == 1 and len(missing) == len(members):
            unavailable.append((row.collapsed, next(iter(groups))))
            continue
        for reason, group in groups.items():
            unavailable.append((", ".join(one.label for one in group), reason))
    return ladder.Resolution(
        rungs=tuple(rungs), available=tuple(available), unavailable=tuple(unavailable)
    )


def dry_run(request: Any) -> Plan:
    """What an audit would do before it does it. Executes nothing and spends nothing (D-20).

    The project is opened first, so a path that is not a reward-lens project is RL0003 here rather
    than a plan for a run that could never start. The audit engine's own `plan()` is used when this
    build holds one; without it the plan is the ten panels of the interfaces, named as such.
    """
    from reward_lens.api import _dispatch

    project = None
    opener = _dispatch.load("reward_lens.store:Project")
    if opener is not None:
        project = opener.open(request.path)

    engine = _dispatch.load("reward_lens.product.audit:plan")
    if engine is not None:
        keywords: dict[str, Any] = {}
        try:
            accepted = inspect.signature(engine).parameters
        except (TypeError, ValueError):  # pragma: no cover - a callable with no signature
            accepted = {}
        if "project" in accepted:
            keywords["project"] = project
        if "sandbox" in accepted:
            keywords["sandbox"] = None
        return engine(request, **keywords)

    return Plan(
        command="audit",
        panels=INTERFACE_PANELS,
        paid_calls=0,
        estimate_usd="0.00",
        notes=(
            f"the audit engine is {ladder.NOT_IN_BUILD}: these are the ten panels of the "
            "interfaces, not a measured plan",
            f"reward-lens audit {request.path} would produce a record of honest absences",
            "nothing was executed and nothing was spent to produce this plan",
        ),
    )
