"""`reward-lens audit`: what this reward accepts, reaches, pays for, and teaches."""

from __future__ import annotations

import click

from .. import registry
from . import _shared


def run(*, out, **opts) -> int:
    from reward_lens import api

    seeker = opts.get("seeker", "off")
    _shared.budget_guard(
        out, seeker=seeker, max_budget_usd=opts.get("max_budget_usd"), command="audit"
    )
    path = _shared.ids.validate_path_argument(opts["path"], field="path")
    # D-31, and it has to come after D-33's budget guard: a cap that cannot cover the work is a
    # refusal (exit 6) and there is nothing to ask about, while a cap that can cover it leaves a
    # real decision to make. A paid seeker spends money, so it is asked for rather than assumed.
    if seeker == "api" and not opts.get("dry_run"):
        _shared.refuse_without_yes(
            out,
            decision=f"run the {seeker} seeker against {path}, which makes paid calls",
            argv=_paid_argv(path, opts),
            command="audit",
        )
    out.event("started", command="audit")
    request = api.AuditRequest(
        path=path,
        seeker=seeker,
        dry_run=bool(opts.get("dry_run")),
        **{
            k: v
            for k, v in (
                (n, opts.get(n))
                for n in (
                    "tasks",
                    "responses",
                    "outcome",
                    "name",
                    "resume",
                    "max_budget_usd",
                    "offline",
                    "sandbox",
                    "require_tier",
                    "run_dir",
                    "detection_reference",
                    "policy",
                    "seeker_config",
                    "seed",
                )
            )
            if v is not None
        },
        continue_run=bool(opts.get("continue_run")),
        only=tuple(opts.get("only", ())),
        non_interactive=bool(opts.get("non_interactive")),
        reward_functions=tuple(opts.get("reward_functions", ())),
        extra_columns=tuple(opts.get("extra_columns", ())),
        weights=tuple(opts.get("weights", ())),
        group_column=opts.get("group_column"),
        group_size=opts.get("group_size"),
    )
    plan = api.dry_run(request)
    try:
        record = api.audit(request)
    except KeyboardInterrupt:
        return _resumable_stop(out, code="RL0130", exit_code=130)
    except Exception as stopped:
        if type(stopped).__name__ != "BudgetExhausted":
            raise
        return _resumable_stop(out, code="RL0501", exit_code=6, stopped=stopped)
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


def _resumable_stop(out, *, code: str, exit_code: int, stopped=None) -> int:
    """Render a durable stop and leave the exact continuation as stderr's final line."""
    from reward_lens import errors
    from reward_lens.product.audit.run import active_run

    active = active_run()
    if active is None:
        raise KeyboardInterrupt()
    context = {"command": "reward-lens", "run": active.run_id}
    if stopped is not None:
        context.update(
            {name: str(getattr(stopped, name, "")) for name in ("cap", "spent", "asked")}
        )
    error = errors.make(code, **context)
    error.exit_code = exit_code
    out.fail(error)
    click.echo(" ".join(active.resume_command()), err=True)
    return exit_code


def _paid_argv(path, opts) -> list:
    """The same command with `--yes`, so the caller can read it back and run it."""
    argv = ["reward-lens", "audit", str(path), "--seeker", str(opts.get("seeker"))]
    if opts.get("max_budget_usd") is not None:
        argv += ["--max-budget-usd", str(opts["max_budget_usd"])]
    if opts.get("seeker_config") is not None:
        argv += ["--seeker-config", str(opts["seeker_config"])]
    return argv + ["--yes"]


@click.command("audit", short_help=registry.SUMMARIES["audit"], epilog=_shared.EPILOG)
@click.argument("path", required=False, default=".")
@click.option(
    "--run-dir",
    type=click.Path(path_type=str),
    default=None,
    help="a readable training artifact opened once as the canonical AuditCorpus.trace",
)
@click.option(
    "--tasks",
    type=click.Path(path_type=str),
    default=None,
    help="JSONL task records for response-bank measurement",
)
@click.option(
    "--responses",
    type=click.Path(path_type=str),
    default=None,
    help="JSONL responses keyed to --tasks for executed response-bank measurement",
)
@click.option(
    "--outcome",
    type=click.Path(path_type=str),
    default=None,
    help="declared outcome file for this Audit request",
)
@click.option("--sandbox", type=str, default=None, help="declared Audit sandbox mode")
@click.option("--require-tier", type=str, default=None, help="minimum required execution tier")
@click.option(
    "--detection-reference",
    type=click.Path(path_type=str),
    default=None,
    help="registered JSON bundle with disjoint detector calibration and assessment traces",
)
@click.option(
    "--reward-function",
    "reward_functions",
    multiple=True,
    metavar="NAME",
    help="a TRL reward-function column; repeat to declare the complete partition",
)
@click.option(
    "--extra-column",
    "extra_columns",
    multiple=True,
    metavar="NAME",
    help="a TRL log_extra column; repeat to declare the complete partition",
)
@click.option(
    "--weight",
    "weights",
    multiple=True,
    metavar="NAME=VALUE",
    help="a finite declared weight for every declared reward-function column",
)
@click.option(
    "--group-column",
    type=str,
    default=None,
    metavar="NAME",
    help="the declared log_extra column that names source groups",
)
@click.option(
    "--group-size",
    type=int,
    default=None,
    metavar="K",
    help="the positive source-declared cardinality for each group",
)
@click.option(
    "--policy",
    type=click.Path(path_type=str),
    default=None,
    help="immutable acceptance policy evaluated and saved with this Audit record",
)
@click.option(
    "--seeker-config",
    type=click.Path(path_type=str),
    default=None,
    help="closed JSON actor configuration for an opted-in seeker",
)
@_shared.common
@_shared.measuring
@_shared.verb("audit")
def audit(**opts) -> int:
    """what this reward accepts, reaches, pays for, and teaches."""
    return run(**opts)
