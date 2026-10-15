"""A small authenticated loopback server for the reward-lens workbench.

The workbench is deliberately a reader of sealed record/report bytes.  It has
no route that starts an audit or imports a reward callable, so a web request
cannot execute a grader by accident.
"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from reward_lens import contracts

__all__ = ["A009_STATES", "DisplayState", "WorkbenchServer"]


@dataclass(frozen=True)
class DisplayState:
    """One of A-009's displayable states and its non-statistical next action."""

    ident: str
    screen: str
    label: str
    next_action: str


# A-009's 7 + 3 + 2 + 2 states. These are authenticated API resources and
# render at `/state/<id>` through the shared workbench client. P-WB-2 retains
# ownership of the detailed compare/history/queue data routes.
A009_STATES: tuple[DisplayState, ...] = (
    DisplayState("case-empty", "case", "No record", "Run an audit or open a saved record."),
    DisplayState(
        "case-loading", "case", "Loading record", "Wait for the record to finish loading."
    ),
    DisplayState("case-loaded", "case", "Record loaded", "Read the evidence and its limits."),
    DisplayState("case-invalid", "case", "Record cannot be read", "Fix the named record error."),
    DisplayState("case-stale", "case", "Evidence is stale", "Rerun the changed dependency."),
    DisplayState("case-decision", "case", "Decision recorded", "Read the decision and policy."),
    DisplayState("case-tradeoff", "case", "Tradeoff remains", "Compare both stated options."),
    DisplayState(
        "compare-scorecard", "compare", "Versions on one scorecard", "Choose two records."
    ),
    DisplayState(
        "compare-tradeoff", "compare", "Comparison has a tradeoff", "Read both consequences."
    ),
    DisplayState(
        "compare-unreachable", "compare", "Rung cannot be reached", "Read the named scope limit."
    ),
    DisplayState(
        "history-graph", "history", "Version history", "Inspect freshness and rerun cost."
    ),
    DisplayState(
        "history-single", "history", "One version only", "Create another measured version."
    ),
    DisplayState("queue-running", "queue", "Run in progress", "Read spend and the next step."),
    DisplayState("queue-idle", "queue", "No active run", "Read the last failure or start a run."),
)


def state_for(ident: str) -> DisplayState:
    """Resolve a state by its public A-009 identifier without deriving it."""
    for state in A009_STATES:
        if state.ident == ident:
            return state
    raise KeyError(ident)


_CSP = (
    "default-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; "
    "object-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'"
)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


class WorkbenchServer:
    """Serve immutable record and report bytes on a new loopback port."""

    host = "127.0.0.1"

    def __init__(
        self,
        *,
        record_bytes: bytes,
        report_bytes: bytes,
        run_directory: Path | None = None,
        asset_root: Path | None = None,
    ) -> None:
        self.record_bytes = bytes(record_bytes)
        self.report_bytes = bytes(report_bytes)
        try:
            record = json.loads(self.record_bytes)
        except ValueError as error:
            raise ValueError("the workbench only serves a sealed JSON record") from error
        # The server does not have a permissive reader. Bytes must already be a
        # record the SDK would accept before they can reach a browser.
        contracts.validate_record(record, raw=self.record_bytes)
        self._record = record
        self.run_directory = Path(run_directory) if run_directory is not None else None
        self.asset_root = Path(asset_root) if asset_root is not None else _default_asset_root()
        if not (self.asset_root / "index.html").is_file():
            raise FileNotFoundError(
                "the installed workbench bundle is missing; rebuild the reward-lens distribution "
                "with `npm --prefix frontend run build:workbench` before launching it"
            )
        self._launch_token = secrets.token_urlsafe(32)
        self._sessions: set[str] = set()
        self._token_used = False
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("the workbench has not started")
        return int(self._server.server_address[1])

    @property
    def token(self) -> str:
        return self._launch_token

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?token={self.token}"

    def start(self) -> "WorkbenchServer":
        if self._server is not None:
            return self
        owner = self

        class Handler(_Handler):
            workbench = owner

        self._server = _Server((self.host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        if self._server is None:
            self.start()
        assert self._server is not None
        self._server.serve_forever()

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def __enter__(self) -> "WorkbenchServer":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _queue_snapshot(self, since: str | None) -> dict[str, Any]:
        """The zero-dependency polling fallback for the same directory as the CLI."""
        current = 0
        if self.run_directory and self.run_directory.exists():
            current = max(
                (
                    item.stat().st_mtime_ns
                    for item in self.run_directory.rglob("*")
                    if item.is_file()
                ),
                default=0,
            )
        try:
            requested = int(since or "0")
        except ValueError:
            requested = 0
        return {
            "changed": current > requested,
            "since": requested,
            "updated_ns": current,
            **self._queue_status(),
        }

    def _queue_status(self) -> dict[str, Any]:
        """Read, but never manufacture, the durable CLI run state.

        `RunDir` owns execution state.  The workbench only has a sealed assay,
        so a record without a durable run id is an idle local queue rather than
        an invented running job.  A corrupt or disappeared run is displayable
        as failed, preserving the path and error for the user to inspect.
        """
        provenance = self._record.get("provenance")
        run_id = provenance.get("run_id") if isinstance(provenance, dict) else None
        if not isinstance(run_id, str) or not run_id:
            return {
                "state": "idle",
                "run_id": None,
                "spent_usd": None,
                "next_action": "No active run is named by this saved record.",
            }
        try:
            from reward_lens.store.runs import RunDir

            run = RunDir.find(run_id)
            manifest = run.manifest
        except Exception as error:  # an on-disk run can disappear after an assay was sealed
            return {
                "state": "failed",
                "run_id": run_id,
                "spent_usd": None,
                "next_action": "Resume or inspect the named local run before relying on its status.",
                "reason": str(error),
            }
        execution = manifest.get("execution_identity")
        budget = execution.get("budget_usd") if isinstance(execution, dict) else None
        if manifest.get("state") == "running":
            return {
                "state": "running",
                "run_id": run_id,
                "spent_usd": None,
                "budget_usd": budget,
                "next_action": "Read the durable run plan and wait for its next sealed partial.",
            }
        return {
            "state": "idle",
            "run_id": run_id,
            "spent_usd": None,
            "budget_usd": budget,
            "next_action": "The named local run is finished; open its saved record or start a new run.",
        }

    def _record_catalog(self) -> dict[str, tuple[dict[str, Any], Path | None]]:
        """Reopen only sealed assays in the selected local project directory.

        The route selector is an assay identity or an existing filename stem,
        never a path supplied by the browser.  This keeps workbench reads
        contained under the CLI-selected directory and makes a stale or broken
        neighbouring file an absence rather than a server-side parse escape.
        """
        rows: dict[str, tuple[dict[str, Any], Path | None]] = {"demo": (self._record, None)}
        if self.run_directory is None:
            return rows
        roots = [self.run_directory]
        if self.run_directory.name == "assays":
            roots.append(self.run_directory.parent)
        else:
            roots.append(self.run_directory / "assays")
        for root in roots:
            if not root.is_dir():
                continue
            for candidate in sorted(root.glob("*.assay.json")):
                try:
                    raw = candidate.read_bytes()
                    loaded = json.loads(raw)
                    contracts.validate_record(loaded, raw=raw)
                except (OSError, UnicodeDecodeError, ValueError, TypeError):
                    continue
                if not isinstance(loaded, dict):
                    continue
                if loaded["assay_id"] != contracts.digest(loaded):
                    continue
                assay_id = loaded.get("assay_id")
                if isinstance(assay_id, str):
                    rows.setdefault(assay_id, (loaded, candidate))
                rows.setdefault(candidate.name.removesuffix(".assay.json"), (loaded, candidate))
                # ``demo`` is the record selected by the CLI.  When the same
                # sealed record is present in its actual run directory, retain
                # that path so scorecard consumption goes through api.open_record.
                if loaded.get("assay_id") == self._record.get("assay_id"):
                    rows["demo"] = (loaded, candidate)
        return rows

    def _selected_record(self, selector: str) -> dict[str, Any]:
        if not selector or "/" in selector or "\\" in selector:
            raise KeyError(selector)
        return self._record_catalog()[selector][0]

    def _opened_selected_record(self, selector: str) -> Any:
        """Reopen a selected file through the public SDK before projection.

        The catalog is intentionally only a local selector index. A typed
        scorecard reader receives an Assay from ``api.open_record`` so every
        request reuses the same sealed-byte validation path as the supported
        CLI, rather than treating the browser's JSON projection as evidence.
        """
        if not selector or "/" in selector or "\\" in selector:
            raise KeyError(selector)
        _record, path = self._record_catalog()[selector]
        if path is None:
            raise ValueError(
                "The selected record has no saved local path to reopen. "
                "Open a comparison saved in the selected project directory."
            )
        from reward_lens import api

        return api.open_record(path)

    @staticmethod
    def _record_summary(record: dict[str, Any]) -> dict[str, Any]:
        subject = record.get("subject") if isinstance(record.get("subject"), dict) else {}
        version = subject.get("version") if isinstance(subject.get("version"), dict) else {}
        return {
            "assay_id": record.get("assay_id"),
            "created": record.get("created"),
            "version_id": version.get("id"),
            "version_digest": version.get("digest"),
            "parents": version.get("parents") if isinstance(version.get("parents"), list) else [],
        }

    @staticmethod
    def _render_scorecard_value(value: Any) -> tuple[str, str, str | None]:
        """Render a typed canonical value without calculating a replacement."""
        from reward_lens.product.compare.scorecard import Absence, CapabilityFloor, Estimate

        if isinstance(value, Absence):
            return (f"NOT MEASURED: {value.reason}", "NOT MEASURED", value.remedy)
        if isinstance(value, CapabilityFloor):
            held = "held" if value.held else "not held"
            return (
                f"{value.value} {value.unit}; 95% interval [{value.interval[0]}, {value.interval[1]}]; "
                f"n={value.n}; capability floor {value.floor} is {held}",
                "CAPABILITY FLOOR",
                None,
            )
        if isinstance(value, Estimate):
            return (
                f"{value.value} {value.unit}; 95% interval [{value.interval[0]}, {value.interval[1]}]; n={value.n}",
                "MEASURED",
                None,
            )
        raise TypeError("persisted scorecard reader returned an unsupported value")

    def _scorecard_column(self, selector: str, rung: str | None) -> dict[str, Any]:
        """Turn one reader-validated rung into one display column."""
        from reward_lens.product.compare.persisted_scorecard import read_persisted_scorecards

        record = self._opened_selected_record(selector)
        scorecards = read_persisted_scorecards(record)
        if rung is None:
            if len(scorecards) != 1:
                raise ValueError(
                    "This saved comparison has multiple rungs. Select one with the `rung` query parameter."
                )
            scorecard = scorecards[0]
        else:
            try:
                selected_rung = int(rung)
            except ValueError as error:
                raise ValueError("The requested comparison rung is invalid.") from error
            matches = [item for item in scorecards if int(item.rung) == selected_rung]
            if len(matches) != 1:
                raise ValueError("The requested comparison rung is not saved in this record.")
            scorecard = matches[0]
        rows = []
        for row in scorecard.rows:
            rendered, state, remedy = self._render_scorecard_value(row.value)
            rows.append(
                {
                    "id": row.id,
                    "label": row.label,
                    "value": rendered,
                    "state": state,
                    "remedy": remedy,
                }
            )
        tradeoff = None
        if scorecard.tradeoff is not None:
            tradeoff = {
                "option_a": scorecard.tradeoff.option_a,
                "option_b": scorecard.tradeoff.option_b,
                "competing_consequences": list(scorecard.tradeoff.competing_consequences),
            }
        return {
            "version": self._record_summary(record.to_dict()),
            "rung": str(int(scorecard.rung)),
            # The canonical reader has already verified that every fixed row
            # names one common scope. The container intentionally carries only
            # the rung and ordered rows, so expose that validated row scope.
            "scope": scorecard.rows[0].scope,
            "rows": rows,
            "tradeoff": tradeoff,
        }

    def _compare_projection(
        self, selector: str, other_selector: str | None, rung: str | None
    ) -> dict[str, Any]:
        left = self._selected_record(selector)
        right = self._selected_record(other_selector) if other_selector else left

        def unavailable(reason, remedy):
            return {
                "state": "unavailable",
                "reason": reason,
                "remedy": remedy,
            }

        if not other_selector:
            return {
                "selected": self._record_summary(left),
                "other": self._record_summary(right),
                "scorecard": unavailable(
                    "Choose a second saved comparison record for the side-by-side scorecard.",
                    "Add a `right` selection naming another sealed comparison in this project.",
                ),
            }
        try:
            columns = [
                self._scorecard_column(selector, rung),
                self._scorecard_column(other_selector, rung),
            ]
        except (ValueError, TypeError, contracts.RewardLensError) as error:
            return {
                "selected": self._record_summary(left),
                "other": self._record_summary(right),
                "scorecard": unavailable(
                    str(error),
                    "Run and save the requested comparison rung, then reopen both sealed records.",
                ),
            }
        if columns[0]["rung"] != columns[1]["rung"] or columns[0]["scope"] != columns[1]["scope"]:
            return {
                "selected": self._record_summary(left),
                "other": self._record_summary(right),
                "scorecard": unavailable(
                    "The selected records do not carry the same rung and scope.",
                    "Choose records with the same saved rung before comparing their scorecard rows.",
                ),
            }
        return {
            "selected": self._record_summary(left),
            "other": self._record_summary(right),
            "scorecard": {
                "state": "available",
                "reason": None,
                "remedy": None,
                "columns": columns,
            },
        }

    def _history_projection(self, selector: str) -> dict[str, Any]:
        selected = self._selected_record(selector)
        unique: dict[str, dict[str, Any]] = {}
        for record, _path in self._record_catalog().values():
            assay_id = record.get("assay_id")
            if isinstance(assay_id, str):
                unique[assay_id] = record
        project = self._project_for_history()
        nodes = [self._history_node(record, project) for record in unique.values()]
        if len(nodes) < 2:
            graph = {
                "state": "unavailable",
                "reason": "No version graph is available because this workspace has one saved version.",
                "remedy": "Save a measurement of another reward-system version before comparing freshness.",
                "nodes": nodes,
            }
        else:
            graph = {
                "state": "available",
                "reason": None,
                "remedy": None,
                "nodes": nodes,
            }
        return {"selected": self._record_summary(selected), "graph": graph}

    def _project_for_history(self) -> Any | None:
        """Open the selected project only when its on-disk config is adjacent.

        A loose record remains a supported Workbench input.  It can show a
        one-record history but cannot claim freshness against a project it did
        not name.
        """
        if self.run_directory is None:
            return None
        candidates = (self.run_directory, self.run_directory.parent)
        for root in candidates:
            if not (root / "rewardlens.yaml").is_file():
                continue
            try:
                from reward_lens.store import Project

                return Project.open(root)
            except Exception:
                return None
        return None

    def _history_node(self, record: dict[str, Any], project: Any | None) -> dict[str, Any]:
        node = self._record_summary(record)
        node["rerun_cost"] = (
            "NOT MEASURED: this saved record has no registered rerun-cost estimate."
        )
        if project is None:
            node["freshness"] = "unavailable"
            node["freshness_reason"] = (
                "No selected reward-lens project is available to check this record."
            )
            return node
        try:
            current = project.version_of(record)
            declared = record["subject"]["version"]["digest"]
            current_digests = current.digests.to_dict()
            recorded_digests = record["subject"]["digests"]
            changed = sorted(
                key
                for key, value in current_digests.items()
                if value is not None and recorded_digests.get(key) != value
            )
            if declared == current.digest() and not changed:
                node["freshness"] = "fresh"
                node["freshness_reason"] = (
                    "The stored version and all declared dependency digests match this project."
                )
            else:
                node["freshness"] = "stale"
                names = ", ".join(changed) if changed else "the reward-system version"
                node["freshness_reason"] = f"This evidence is stale because {names} changed."
        except Exception as error:
            node["freshness"] = "unavailable"
            node["freshness_reason"] = f"Freshness could not be checked: {error}"
        return node


class _Handler(BaseHTTPRequestHandler):
    """The request boundary. It contains no engine or grader call by design."""

    protocol_version = "HTTP/1.1"
    workbench: WorkbenchServer

    def log_message(self, _format: str, *_args: object) -> None:
        # Tokens must not reach stderr through the standard request log.
        return

    def _headers(self, *, content_type: str, length: int | None = None) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if length is not None:
            self.send_header("Content-Length", str(length))

    def _deny(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self._headers(content_type="application/json; charset=utf-8", length=len(body))
        self.end_headers()
        self.wfile.write(body)

    def _same_origin(self) -> bool:
        expected = f"127.0.0.1:{self.workbench.port}"
        if self.headers.get("Host") != expected:
            self._deny(HTTPStatus.FORBIDDEN, "request host is not this loopback workbench")
            return False
        # Top-level navigation normally has no Origin. Any supplied Origin is
        # security-relevant and must be this exact origin.
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{expected}":
            self._deny(HTTPStatus.FORBIDDEN, "request origin is not this loopback workbench")
            return False
        return True

    def _authenticated(self, path: str, query: dict[str, list[str]]) -> bool:
        token = query.get("token", [None])[0]
        if token is not None:
            if self.workbench._token_used or not secrets.compare_digest(
                token, self.workbench.token
            ):
                self._deny(
                    HTTPStatus.UNAUTHORIZED, "launch token is invalid or has already been used"
                )
                return False
            self.workbench._token_used = True
            session = secrets.token_urlsafe(32)
            self.workbench._sessions.add(session)
            self.send_response(HTTPStatus.SEE_OTHER)
            self._headers(content_type="text/plain; charset=utf-8", length=0)
            self.send_header(
                "Set-Cookie", f"reward_lens_session={session}; HttpOnly; SameSite=Strict; Path=/"
            )
            # A one-time launch can open a named workbench route directly.
            # Keep only non-secret selections through the redirect: the token
            # remains out of browser history and referrer-bearing document URLs.
            retained = [
                (name, value)
                for name, values in query.items()
                if name != "token"
                for value in values
            ]
            self.send_header("Location", path + (f"?{urlencode(retained)}" if retained else ""))
            self.end_headers()
            return False
        cookie = SimpleCookie(self.headers.get("Cookie"))
        session = cookie.get("reward_lens_session")
        if session is None or session.value not in self.workbench._sessions:
            self._deny(HTTPStatus.UNAUTHORIZED, "open the one-time launch URL first")
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler protocol
        if not self._same_origin():
            return
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if not self._authenticated(parsed.path, query):
            return
        if parsed.path == "/":
            return self._index()
        if parsed.path.startswith("/state/"):
            try:
                state_for(parsed.path.removeprefix("/state/"))
            except KeyError:
                self._deny(HTTPStatus.NOT_FOUND, "workbench state not found")
            else:
                self._index()
            return
        if parsed.path.startswith("/api/states/"):
            try:
                state = state_for(parsed.path.removeprefix("/api/states/"))
            except KeyError:
                self._deny(HTTPStatus.NOT_FOUND, "workbench state not found")
            else:
                payload = json.dumps(
                    {
                        "id": state.ident,
                        "screen": state.screen,
                        "label": state.label,
                        "next_action": state.next_action,
                    },
                    sort_keys=True,
                ).encode("utf-8")
                self._bytes(payload, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/record/demo":
            return self._bytes(self.workbench.record_bytes, "application/json; charset=utf-8")
        if parsed.path == "/api/report/demo":
            return self._bytes(
                self.workbench.report_bytes,
                "text/html; charset=utf-8",
                attachment="report.assay.html",
            )
        if parsed.path.startswith("/api/compare/"):
            try:
                payload = self.workbench._compare_projection(
                    parsed.path.removeprefix("/api/compare/"),
                    query.get("right", [None])[0],
                    query.get("rung", [None])[0],
                )
            except KeyError:
                self._deny(HTTPStatus.NOT_FOUND, "selected comparison record is not available")
            else:
                self._bytes(
                    json.dumps(payload, sort_keys=True).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            return
        if parsed.path.startswith("/api/history/"):
            try:
                payload = self.workbench._history_projection(
                    parsed.path.removeprefix("/api/history/")
                )
            except KeyError:
                self._deny(HTTPStatus.NOT_FOUND, "selected history record is not available")
            else:
                self._bytes(
                    json.dumps(payload, sort_keys=True).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            return
        if parsed.path == "/api/queue":
            payload = json.dumps(
                self.workbench._queue_snapshot(query.get("since", [None])[0]), sort_keys=True
            ).encode("utf-8")
            return self._bytes(payload, "application/json; charset=utf-8")
        if parsed.path == "/api/events":
            return self._events(query.get("since", [None])[0])
        if parsed.path.startswith("/assets/"):
            return self._asset(parsed.path.removeprefix("/assets/"))
        self._deny(HTTPStatus.NOT_FOUND, "workbench route not found")

    def _bytes(self, payload: bytes, content_type: str, *, attachment: str | None = None) -> None:
        self.send_response(HTTPStatus.OK)
        self._headers(content_type=content_type, length=len(payload))
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        self.wfile.write(payload)

    def _events(self, since: str | None) -> None:
        payload = json.dumps(self.workbench._queue_snapshot(since), sort_keys=True).encode("utf-8")
        body = b"event: queue\ndata: " + payload + b"\n\n"
        self.send_response(HTTPStatus.OK)
        self._headers(content_type="text/event-stream; charset=utf-8", length=len(body))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _index(self) -> None:
        index = self.workbench.asset_root / "index.html"
        # Vite's workbench artifact is built with a portable `./assets/` base.
        # That is correct for its build directory but would make `/state/<id>`
        # fetch `/state/assets/...`.  The local HTTP server owns route mounting,
        # so it makes package assets absolute without mutating the frozen build
        # config or an offline report artifact.
        payload = index.read_bytes().replace(b'"./assets/', b'"/assets/')
        self._bytes(payload, "text/html; charset=utf-8")

    def _asset(self, relative: str) -> None:
        candidate = (self.workbench.asset_root / "assets" / relative).resolve()
        root = (self.workbench.asset_root / "assets").resolve()
        if root not in candidate.parents or not candidate.is_file():
            self._deny(HTTPStatus.NOT_FOUND, "workbench asset not found")
            return
        content_type = (
            "text/javascript; charset=utf-8"
            if candidate.suffix == ".js"
            else "text/css; charset=utf-8"
        )
        self._bytes(candidate.read_bytes(), content_type)


def _default_asset_root() -> Path:
    # Tracked package data means a source checkout and an installed wheel use
    # the same bundle rather than an untracked frontend build directory.
    return Path(__file__).resolve().parent / "assets"
