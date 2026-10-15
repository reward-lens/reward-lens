"""P-WB-2 routes read durable server projections; they do not calculate results."""

from __future__ import annotations

import http.client
import importlib
import json
import shutil
from pathlib import Path

from reward_lens import api
from reward_lens.product.serve import WorkbenchServer

RECORD = (
    Path(__file__).parents[2] / "fleet/gates/fixtures/controls/g39/positive/sdk.assay.json"
).read_bytes()
REPORT = b"<!doctype html><title>the CLI report</title>"


def _project(destination: Path) -> Path:
    from reward_lens.examples import REGISTRY

    module_name, _, attribute = REGISTRY["code-reward"].partition(":")
    getattr(importlib.import_module(module_name), attribute)(destination)
    return destination


def _bank(destination: Path, *, after: tuple[str, str]) -> Path:
    destination.write_text(
        "\n".join(
            (
                json.dumps({"name": destination.stem}),
                json.dumps(
                    {
                        "task_id": "one",
                        "responses": [{"text": "a", "scores": {"before": "1", "after": after[0]}}],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "two",
                        "responses": [{"text": "b", "scores": {"before": "3", "after": after[1]}}],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _replay_config(destination: Path, version: str, *, offset: str) -> Path:
    destination.write_text(
        json.dumps({"id": version, "weights": {"length_reward": "1"}, "offset": offset}),
        encoding="utf-8",
    )
    return destination


def _two_multi_rung_records(root: Path) -> tuple[object, object]:
    """Create two actual saved comparisons that each carry rungs one and three."""
    bank = _bank(root / "replay-bank.jsonl", after=("2", "5"))
    fixture = Path(__file__).parents[1] / "fixtures/grpo_run/short"
    run = root.parent / "replay-run"
    shutil.copytree(fixture, run)
    baseline = _replay_config(root / "before.json", "before", offset="0")
    candidate = _replay_config(root / "after.json", "after", offset="-3")
    first = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            bank=bank,
            project=root,
            rungs=(1, 3),
            run_dir=run,
            baseline_config=baseline,
            candidate_config=candidate,
            name="first-multi-rung",
        )
    )
    _replay_config(candidate, "after", offset="-2")
    second = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            bank=bank,
            project=root,
            rungs=(1, 3),
            run_dir=run,
            baseline_config=baseline,
            candidate_config=candidate,
            name="second-multi-rung",
        )
    )
    return first, second


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


def _launch(server: WorkbenchServer) -> str:
    response, _ = _request(server, f"/?token={server.token}")
    assert response.status == 303
    return (response.getheader("Set-Cookie") or "").split(";", 1)[0]


def test_compare_deep_link_consumes_the_one_time_token_without_losing_selections() -> None:
    with WorkbenchServer(record_bytes=RECORD, report_bytes=REPORT) as server:
        response, _ = _request(
            server,
            f"/state/compare-scorecard?token={server.token}&left=first&right=second&rung=1",
        )

    assert response.status == 303
    assert (
        response.getheader("Location") == "/state/compare-scorecard?left=first&right=second&rung=1"
    )


def test_compare_and_history_are_authenticated_record_projections_not_browser_calculations(
    tmp_path: Path,
) -> None:
    assays = tmp_path / "assays"
    assays.mkdir()
    selected = assays / "selected.assay.json"
    selected.write_bytes(RECORD)
    with WorkbenchServer(record_bytes=RECORD, report_bytes=REPORT, run_directory=assays) as server:
        cookie = _launch(server)
        compare, compare_body = _request(server, "/api/compare/demo", cookie)
        history, history_body = _request(server, "/api/history/demo", cookie)

    assert compare.status == 200
    compare_payload = json.loads(compare_body)
    assert compare_payload["selected"]["assay_id"]
    assert compare_payload["scorecard"]["state"] == "unavailable"
    assert "second saved comparison" in compare_payload["scorecard"]["reason"]
    assert history.status == 200
    history_payload = json.loads(history_body)
    assert history_payload["selected"]["version_id"] == "v1"
    assert history_payload["graph"]["state"] == "unavailable"
    assert history_payload["graph"]["reason"]


def test_compare_route_reopens_two_saved_scorecards_and_renders_their_actual_six_rows(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path / "project")
    first = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            bank=_bank(root / "first.jsonl", after=("2", "5")),
            project=root,
            rungs=(1,),
            name="first-scorecard",
        )
    )
    second = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            bank=_bank(root / "second.jsonl", after=("4", "7")),
            project=root,
            rungs=(1,),
            name="second-scorecard",
        )
    )
    saved = root / "assays" / "first-scorecard.assay.json"
    with WorkbenchServer(
        record_bytes=saved.read_bytes(), report_bytes=REPORT, run_directory=root / "assays"
    ) as server:
        cookie = _launch(server)
        response, body = _request(
            server,
            "/api/compare/demo?right=second-scorecard&rung=1",
            cookie,
        )

    assert response.status == 200
    payload = json.loads(body)
    assert payload["selected"]["assay_id"] == first.assay_id
    assert payload["other"]["assay_id"] == second.assay_id
    assert payload["scorecard"]["state"] == "available"
    columns = payload["scorecard"]["columns"]
    assert len(columns) == 2
    assert all(
        column["rung"] == "1" and column["scope"] == "evaluator_comparison" for column in columns
    )
    assert all(len(column["rows"]) == 6 for column in columns)
    assert columns[0]["rows"][0]["state"] == "NOT MEASURED"
    assert columns[1]["rows"][0]["state"] == "NOT MEASURED"
    assert columns[0]["rows"][1]["remedy"]


def test_compare_route_keeps_a_missing_second_or_scorecard_as_a_named_refusal(
    tmp_path: Path,
) -> None:
    from reward_lens import contracts

    root = _project(tmp_path / "project")
    saved = root / "assays" / "plain.assay.json"
    saved.parent.mkdir()
    plain = json.loads(RECORD)
    plain["assay_id"] = contracts.digest(plain)
    saved.write_text(json.dumps(plain), encoding="utf-8")
    with WorkbenchServer(
        record_bytes=saved.read_bytes(), report_bytes=REPORT, run_directory=saved.parent
    ) as server:
        cookie = _launch(server)
        missing_second, missing_second_body = _request(server, "/api/compare/demo", cookie)
        absent, absent_body = _request(server, "/api/compare/demo?right=plain", cookie)

    assert missing_second.status == 200
    assert json.loads(missing_second_body)["scorecard"]["state"] == "unavailable"
    absent_payload = json.loads(absent_body)
    assert absent_payload["scorecard"]["state"] == "unavailable"
    assert "persisted compare scorecard" in absent_payload["scorecard"]["reason"]


def test_history_excludes_a_neighbor_whose_saved_assay_id_no_longer_matches(tmp_path: Path) -> None:
    from reward_lens import contracts

    assays = tmp_path / "assays"
    assays.mkdir()
    selected = json.loads(RECORD)
    selected["assay_id"] = contracts.digest(selected)
    selected_path = assays / "selected.assay.json"
    selected_path.write_text(json.dumps(selected), encoding="utf-8")

    neighbor = json.loads(RECORD)
    neighbor["subject"]["version"]["id"] = "neighbor-before-edit"
    neighbor["assay_id"] = contracts.digest(neighbor)
    neighbor["subject"]["version"]["id"] = "neighbor-after-edit"
    neighbor_path = assays / "tampered.assay.json"
    tampered_bytes = json.dumps(neighbor).encode()
    neighbor_path.write_bytes(tampered_bytes)
    assert neighbor["assay_id"] != contracts.digest(neighbor)

    with WorkbenchServer(
        record_bytes=selected_path.read_bytes(), report_bytes=REPORT, run_directory=assays
    ) as server:
        cookie = _launch(server)
        history, body = _request(server, "/api/history/demo", cookie)
        missing, _ = _request(server, "/api/history/tampered", cookie)

    assert history.status == 200
    graph = json.loads(body)["graph"]
    assert graph["state"] == "unavailable"
    assert [node["assay_id"] for node in graph["nodes"]] == [selected["assay_id"]]
    assert missing.status == 404
    assert neighbor_path.read_bytes() == tampered_bytes


def test_compare_multi_rung_selection_keeps_invalid_and_missing_rungs_as_refusals(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path / "project")
    first, second = _two_multi_rung_records(root)
    saved = root / "assays" / "first-multi-rung.assay.json"
    with WorkbenchServer(
        record_bytes=saved.read_bytes(), report_bytes=REPORT, run_directory=root / "assays"
    ) as server:
        cookie = _launch(server)
        selected, selected_body = _request(
            server, "/api/compare/demo?right=second-multi-rung&rung=3", cookie
        )
        missing, missing_body = _request(
            server, "/api/compare/demo?right=second-multi-rung", cookie
        )
        invalid, invalid_body = _request(
            server, "/api/compare/demo?right=second-multi-rung&rung=not-a-rung", cookie
        )

    assert selected.status == missing.status == invalid.status == 200
    selected_payload = json.loads(selected_body)
    assert selected_payload["selected"]["assay_id"] == first.assay_id
    assert selected_payload["other"]["assay_id"] == second.assay_id
    assert selected_payload["scorecard"]["state"] == "available"
    assert all(column["rung"] == "3" for column in selected_payload["scorecard"]["columns"])
    assert all(len(column["rows"]) == 6 for column in selected_payload["scorecard"]["columns"])
    assert json.loads(missing_body)["scorecard"]["state"] == "unavailable"
    assert "multiple rungs" in json.loads(missing_body)["scorecard"]["reason"]
    assert json.loads(invalid_body)["scorecard"]["state"] == "unavailable"
    assert "rung is invalid" in json.loads(invalid_body)["scorecard"]["reason"]


def test_route_modules_render_server_states_and_refusals_without_a_second_scorecard() -> None:
    root = Path(__file__).parents[2]
    compare = (root / "frontend/workbench/routes/compare/index.tsx").read_text(encoding="utf-8")
    history = (root / "frontend/workbench/routes/history/index.tsx").read_text(encoding="utf-8")
    queue = (root / "frontend/workbench/routes/queue/index.tsx").read_text(encoding="utf-8")
    entry = (root / "frontend/workbench/src/main.tsx").read_text(encoding="utf-8")

    assert "/api/compare/" in compare
    assert "NOT MEASURED" in compare
    assert "Rung" in compare and "Scope" in compare
    assert "/api/history/" in history
    assert "Evidence is stale" in history
    assert "No version graph" in history
    assert "/api/queue" in queue
    assert "No active run" in queue
    assert "Queue status is unavailable" in queue
    assert "CompareRoute" in entry and "HistoryRoute" in entry and "QueueRoute" in entry
    assert 'selection("rung")' in entry
    assert 'params.set("rung", rung)' in compare


def test_state_routes_keep_packaged_assets_at_the_loopback_root() -> None:
    with WorkbenchServer(record_bytes=RECORD, report_bytes=REPORT) as server:
        cookie = _launch(server)
        response, page = _request(server, "/state/compare-scorecard", cookie)

    assert response.status == 200
    assert b'src="/assets/' in page
    assert b'href="/assets/' in page
