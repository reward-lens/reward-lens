"""`reward-lens open`: open a report, or the workbench."""

from __future__ import annotations

import os
import pathlib
import webbrowser
from typing import TYPE_CHECKING

import click

from .. import ids, registry
from . import _shared

if TYPE_CHECKING:
    from reward_lens.product.serve import WorkbenchServer


def _report_path(path: pathlib.Path, supplied: pathlib.Path | None) -> pathlib.Path:
    return supplied if supplied is not None else path.with_suffix(".html")


def launch_workbench(*, path: pathlib.Path, report: pathlib.Path | None) -> "WorkbenchServer":
    """Create the read-only loopback surface after the SDK has trusted the record."""
    from reward_lens.product.serve import WorkbenchServer

    report_path = _report_path(path, report)
    try:
        report_bytes = report_path.read_bytes()
    except OSError as error:
        from reward_lens import contracts

        raise contracts.UsageError(
            code="RL0621",
            message=f"no rendered report at {report_path}",
            remediation="render the record first, or pass its existing HTML with --report",
            context={"path": str(report_path), "detail": str(error)},
        ) from error
    from reward_lens import api

    api.open_record(path)
    server = WorkbenchServer(
        record_bytes=path.read_bytes(), report_bytes=report_bytes, run_directory=path.parent
    )
    server.start()
    return server


def open_browser_if_available(server: "WorkbenchServer", *, no_browser: bool) -> None:
    if not no_browser and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        webbrowser.open(server.url)


def run(*, out, path, workbench=False, report=None, no_browser=False, **opts) -> int:
    checked = ids.validate_path_argument(path, field="path")
    if workbench:
        server = launch_workbench(path=checked, report=report)
        # URL output is the first visible effect; browser launch is deliberately later.
        out.result(server.url)
        open_browser_if_available(server, no_browser=no_browser)
        try:
            assert server._thread is not None  # the CLI owns this long-lived local process
            server._thread.join()
        finally:
            server.close()
        return 0
    from reward_lens import api

    record = api.open_record(checked)
    return _shared.deliver_record(out, record, command="open")


@click.command("open", short_help=registry.SUMMARIES["open"], epilog=_shared.EPILOG)
@click.argument("path")
@click.option("--workbench", is_flag=True, help="serve this record in the local workbench")
@click.option(
    "--report", type=click.Path(path_type=pathlib.Path), help="existing one-file HTML report"
)
@click.option("--no-browser", is_flag=True, help="print the local URL without opening a browser")
@_shared.common
@_shared.verb("open")
def open_(**opts) -> int:
    """Open a report, or the workbench."""
    return run(**opts)
