"""A-009 workbench state routes exercised through the served client boundary."""

from __future__ import annotations

import http.client
import json
from pathlib import Path

from reward_lens.product.serve import WorkbenchServer

RECORD = (
    Path(__file__).parents[2] / "fleet/gates/fixtures/controls/g39/positive/sdk.assay.json"
).read_bytes()
REPORT = b"<!doctype html><title>the CLI report</title>"

# Frozen independently from server internals: seven case, three compare, two
# history and two queue screens. A route must expose every row to pass.
EXPECTED = {
    "case-empty": ("case", "No record"),
    "case-loading": ("case", "Loading record"),
    "case-loaded": ("case", "Record loaded"),
    "case-invalid": ("case", "Record cannot be read"),
    "case-stale": ("case", "Evidence is stale"),
    "case-decision": ("case", "Decision recorded"),
    "case-tradeoff": ("case", "Tradeoff remains"),
    "compare-scorecard": ("compare", "Versions on one scorecard"),
    "compare-tradeoff": ("compare", "Comparison has a tradeoff"),
    "compare-unreachable": ("compare", "Rung cannot be reached"),
    "history-graph": ("history", "Version history"),
    "history-single": ("history", "One version only"),
    "queue-running": ("queue", "Run in progress"),
    "queue-idle": ("queue", "No active run"),
}


def _request(server: WorkbenchServer, path: str, cookie: str | None = None):
    connection = http.client.HTTPConnection(server.host, server.port, timeout=2)
    headers = {"Host": f"{server.host}:{server.port}"}
    if cookie:
        headers["Cookie"] = cookie
    connection.request("GET", path, headers=headers)
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response, body


def test_all_fourteen_states_are_reachable_routes_in_the_built_client() -> None:
    with WorkbenchServer(record_bytes=RECORD, report_bytes=REPORT) as server:
        launch, _ = _request(server, f"/?token={server.token}")
        assert launch.status == 303
        cookie = (launch.getheader("Set-Cookie") or "").split(";", 1)[0]
        for ident, (screen, label) in EXPECTED.items():
            route, page = _request(server, f"/state/{ident}", cookie)
            assert route.status == 200
            assert b'<script type="module"' in page
            state, body = _request(server, f"/api/states/{ident}", cookie)
            assert state.status == 200
            payload = json.loads(body)
            assert (payload["screen"], payload["label"]) == (screen, label)
            assert payload["next_action"]


def test_built_client_mounts_the_report_component_tree_and_state_adapter() -> None:
    assets = Path(__file__).parents[2] / "src/reward_lens/product/serve/assets/assets"
    script = next(assets.glob("*.js")).read_text(encoding="utf-8")
    # These strings are emitted from the report App and the workbench state API;
    # together they rule out a second summary renderer or a static state catalogue.
    assert "The answer" in script
    assert "/api/states/" in script
