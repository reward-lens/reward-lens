"""The loopback workbench boundary (P-WB-1)."""

from __future__ import annotations

import http.client
import importlib
import json
from pathlib import Path

import pytest

from reward_lens import contracts
from reward_lens.cli.verbs.open import launch_workbench
from reward_lens.product.serve import WorkbenchServer

RECORD = (
    Path(__file__).parents[2] / "fleet/gates/fixtures/controls/g39/positive/sdk.assay.json"
).read_bytes()
REPORT = b"<!doctype html><title>the CLI report</title>"
REFERENCE = Path(__file__).parents[2] / "schema/assay/1.0/fixtures/reference.json"


def _sealed_reference_bytes() -> bytes:
    """Derive a valid saved assay from the shipped full schema reference."""
    document = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert document["assay_id"] == "sha256:" + "0" * 64
    document["assay_id"] = contracts.digest(document)
    return json.dumps(document, separators=(",", ":")).encode("utf-8")


def _request(server: WorkbenchServer, path: str, *, headers: dict[str, str] | None = None):
    connection = http.client.HTTPConnection(server.host, server.port, timeout=2)
    sent = {"Host": f"{server.host}:{server.port}"}
    sent.update(headers or {})
    connection.request("GET", path, headers=sent)
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response, body


def _launch(server: WorkbenchServer) -> str:
    response, _ = _request(server, f"/?token={server.token}")
    assert response.status == 303
    assert response.getheader("Location") == "/"
    cookie = response.getheader("Set-Cookie")
    assert cookie is not None
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    return cookie.split(";", 1)[0]


def test_workbench_server_is_authenticated_loopback_and_sets_security_headers() -> None:
    with WorkbenchServer(record_bytes=RECORD, report_bytes=REPORT) as server:
        assert server.host == "127.0.0.1"
        assert server.port > 0
        cookie = _launch(server)
        response, body = _request(server, "/", headers={"Cookie": cookie})
        assert response.status == 200
        assert b"reward-lens workbench" in body
        assert response.getheader("Cache-Control") == "no-store"
        assert "default-src 'self'" in (response.getheader("Content-Security-Policy") or "")


def test_host_origin_replay_and_fragment_requests_are_refused() -> None:
    with WorkbenchServer(record_bytes=RECORD, report_bytes=REPORT) as server:
        response, _ = _request(server, f"/?token={server.token}", headers={"Host": "example.test"})
        assert response.status == 403
        response, _ = _request(
            server,
            f"/?token={server.token}",
            headers={"Origin": "https://example.test"},
        )
        assert response.status == 403
        cookie = _launch(server)
        response, _ = _request(server, f"/?token={server.token}")
        assert response.status == 401
        # Fragments never cross HTTP. The server therefore sees an unauthenticated
        # root request and cannot accept a fragment as a secret.
        response, _ = _request(server, "/", headers={})
        assert response.status == 401
        response, _ = _request(server, "/", headers={"Cookie": cookie})
        assert response.status == 200


def test_download_is_exact_and_queue_has_sse_and_polling(tmp_path: Path) -> None:
    run = tmp_path / "runs"
    run.mkdir()
    (run / "active.json").write_text("{}", encoding="utf-8")
    with WorkbenchServer(record_bytes=RECORD, report_bytes=REPORT, run_directory=run) as server:
        cookie = _launch(server)
        response, body = _request(server, "/api/report/demo", headers={"Cookie": cookie})
        assert response.status == 200
        assert body == REPORT
        assert (
            response.getheader("Content-Disposition") == 'attachment; filename="report.assay.html"'
        )
        response, body = _request(server, "/api/record/demo", headers={"Cookie": cookie})
        assert response.status == 200
        assert body == RECORD
        response, body = _request(server, "/api/events?since=0", headers={"Cookie": cookie})
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream; charset=utf-8"
        assert body.startswith(b"event: queue\ndata: ")
        response, body = _request(server, "/api/queue?since=0", headers={"Cookie": cookie})
        assert response.status == 200
        assert json.loads(body)["changed"] is True


def test_malformed_record_is_refused_before_any_request_handler_runs() -> None:
    with pytest.raises(ValueError, match="sealed JSON"):
        WorkbenchServer(record_bytes=b"not json", report_bytes=REPORT)


def test_launch_helper_honours_no_browser_and_uses_existing_report(tmp_path: Path) -> None:
    record = tmp_path / "demo.assay.json"
    report = tmp_path / "demo.assay.html"
    record.write_bytes(_sealed_reference_bytes())
    report.write_bytes(REPORT)
    server = launch_workbench(path=record, report=report)
    try:
        assert server.url.startswith("http://127.0.0.1:")
    finally:
        server.close()


def test_launch_helper_refuses_the_original_unsealed_fixture(tmp_path: Path) -> None:
    record = tmp_path / "unsealed.assay.json"
    report = tmp_path / "unsealed.assay.html"
    record.write_bytes(RECORD)
    report.write_bytes(REPORT)
    with pytest.raises(contracts.RecordInvalid) as refused:
        launch_workbench(path=record, report=report)
    assert refused.value.code == "RL0604"
    assert refused.value.context["declared_assay_id"] == "sha256:" + "0" * 64
    assert refused.value.context["original_bytes"] == RECORD
    assert record.read_bytes() == RECORD


def test_cli_prints_the_url_before_it_considers_opening_a_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "demo.assay.json"
    record.write_bytes(RECORD)
    events: list[str] = []

    class Thread:
        def join(self) -> None:
            events.append("join")
            raise KeyboardInterrupt

    class Server:
        url = "http://127.0.0.1:43123/?token=one-use"
        _thread = Thread()

        def close(self) -> None:
            events.append("close")

    class Out:
        def result(self, value: str) -> None:
            assert value == Server.url
            events.append("url")

    command = importlib.import_module("reward_lens.cli.verbs.open")
    monkeypatch.setattr(command, "launch_workbench", lambda **_: Server())
    monkeypatch.setattr(
        command, "open_browser_if_available", lambda *_args, **_kwargs: events.append("browser")
    )
    with pytest.raises(KeyboardInterrupt):
        command.run(out=Out(), path=record, workbench=True, report=None, no_browser=False)
    assert events == ["url", "browser", "join", "close"]
