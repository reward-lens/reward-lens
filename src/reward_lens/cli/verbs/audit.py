"""`reward-lens audit`: what this reward accepts, reaches, pays for, and teaches."""

from __future__ import annotations

import click

from .. import registry
from . import _shared


def run(*, out, **opts) -> int:
    from reward_lens import api

    seeker = opts.get("seeker", "off")
    _shared.budget_guard(out, seeker=seeker, max_budget_usd=opts.get("max_budget_usd"), command="audit")
    path = _shared.ids.validate_path_argument(opts["path"], field="path")
    # D-31, and it has to come after D-33's budget guard: a cap that cannot cover the work is a
    # refusal (exit 6) and there is nothing to ask about, while a cap that can cover it leaves a
    # real decision to make. A paid seeker spends money, so it is asked for rather than assumed.
    if seeker in ("api", "agent") and not opts.get("dry_run"):
        _shared.refuse_without_yes(
            out,
            decision=f"run the {seeker} seeker against {path}, which makes paid calls",
            argv=_paid_argv(path, opts),
            command="audit",
        )
    out.event("started", command="audit")
    request = api.AuditRequest(path=path, seeker=seeker, dry_run=bool(opts.get('dry_run')), **{k: v for k, v in ((n, opts.get(n)) for n in ('name', 'resume', 'max_budget_usd')) if v is not None})
    plan = api.dry_run(request)
    record = api.audit(request)
    artifacts, project = _shared.artifacts_for_record(record, path)
    return _shared.deliver_record(
        out,
        record,
        command="audit",
        artifacts=artifacts,
        project=project,
        plan=plan,
        reused=api.last_reuse(),
    )


def _paid_argv(path, opts) -> list:
    """The same command with `--yes`, so the caller can read it back and run it."""
    argv = ["reward-lens", "audit", str(path), "--seeker", str(opts.get("seeker"))]
    if opts.get("max_budget_usd") is not None:
        argv += ["--max-budget-usd", str(opts["max_budget_usd"])]
    return argv + ["--yes"]


@click.command("audit", short_help=registry.SUMMARIES["audit"], epilog=_shared.EPILOG)
@click.argument("path", required=False, default=".")
@_shared.common
@_shared.measuring
@_shared.verb("audit")
def audit(**opts) -> int:
    """what this reward accepts, reaches, pays for, and teaches."""
    return run(**opts)
