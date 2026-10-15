from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner


def _issue_row(target: str) -> dict[str, object]:
    provenance = {
        name: {"cutoff": 4, "fit_digest": "sha256:" + ("a" * 64)}
        for name in ("persistence", "climatology", "drift", "covariance")
    }
    if target == "T1":
        forecast: object = 0.8
        baselines: object = {
            "persistence": 0.6,
            "climatology": 0.4,
            "drift": 0.7,
            "covariance": 0.5,
        }
    else:
        forecast = {"values": [0.1, 0.2, 0.3, 0.4, 0.5]}
        baselines = {name: {"values": [0.1, 0.2, 0.3, 0.4, 0.5]} for name in provenance}
    return {
        "target": target,
        "checkpoint": 4,
        "source_name": "controlled-run",
        "source_cutoff": 4,
        "feature_cutoff": 4,
        "model_selection_cutoff": 4,
        "forecast": forecast,
        "baselines": baselines,
        "baseline_provenance": provenance,
        "model_digest": "sha256:" + ("f" * 64),
    }


def test_product_journey_issues_resolves_and_renders_real_record(tmp_path: Path) -> None:
    from reward_lens.product.forecast import issue, render_ledger, resolve

    record = tmp_path / "prospective.json"
    record.write_text(
        json.dumps({"issues": [_issue_row("T1"), _issue_row("T2"), _issue_row("T3")]})
    )
    request = SimpleNamespace(record=record, claim="held-out forecast", offline=True, seed=7)
    issue(request, project=SimpleNamespace(root=tmp_path))
    ledger_path = tmp_path / ".reward-lens" / "forecast-ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["header", "issue", "issue", "issue"]
    digests = [row["sealed_digest"] for row in rows if row["kind"] == "issue"]
    for digest, outcome in zip(digests, ("true", "0.35", "0.25"), strict=True):
        resolve(
            SimpleNamespace(forecast_id=digest, outcome=outcome, offline=True, seed=7),
            project=SimpleNamespace(root=tmp_path),
        )
    rendered = render_ledger(SimpleNamespace(root=tmp_path))
    assert "campaign | - | - | T1 | - | 0.26" in rendered
    assert "controlled-run" in rendered and "persistence" in rendered and "log10_wealth" in rendered
    assert rendered.index("controlled-run") < rendered.index("campaign |")
    assert (
        rows[0]["campaign"]["source_sha256"]
        == "ae67d15dae7e3e6ec5737bd2b96476bca51a4d5fe4f361a31771eb329c157ee8"
    )
    with pytest.raises(ValueError, match="already resolved"):
        resolve(
            SimpleNamespace(forecast_id=digests[0], outcome="true", offline=True, seed=7),
            project=SimpleNamespace(root=tmp_path),
        )
    tampered = ledger_path.read_text().replace('"kind":"issue"', '"kind":"tampered"', 1)
    ledger_path.write_text(tampered)
    with pytest.raises(ValueError, match="row digest mismatch"):
        render_ledger(SimpleNamespace(root=tmp_path))


def test_forecast_issue_request_preserves_existing_construction_and_types_record_path() -> None:
    from reward_lens.api.requests import ForecastIssueRequest

    assert ForecastIssueRequest(claim="existing SDK call").record is None
    assert ForecastIssueRequest(claim="typed", record="prospective.json").record == Path(
        "prospective.json"
    )


def test_forecast_cli_public_journey_accepts_record_and_outcome(tmp_path, monkeypatch) -> None:
    from reward_lens.cli.main import cli

    monkeypatch.chdir(tmp_path)
    record = tmp_path / "prospective.json"
    record.write_text(json.dumps({"issues": [_issue_row("T1")]}))
    runner = CliRunner()
    issue_result = runner.invoke(cli, ["forecast", "issue", "real claim", "--record", str(record)])
    assert issue_result.exit_code == 2, issue_result.output
    rows = [
        json.loads(line)
        for line in Path(".reward-lens/forecast-ledger.jsonl").read_text().splitlines()
    ]
    digest = next(row["sealed_digest"] for row in rows if row["kind"] == "issue")
    resolve_result = runner.invoke(cli, ["forecast", "resolve", digest, "--outcome", "true"])
    assert resolve_result.exit_code == 2, resolve_result.output
    ledger_result = runner.invoke(cli, ["forecast", "ledger", "--format", "text"])
    assert ledger_result.exit_code == 2, ledger_result.output
    assert "campaign | - | - | T1 | - | 0.26" in ledger_result.output
    assert "controlled-run" in ledger_result.output


def test_forecast_cli_resolve_exposes_the_declared_outcome_field(tmp_path, monkeypatch) -> None:
    from reward_lens.cli.main import cli

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["forecast", "resolve", "sha256:missing", "--outcome", "true"])
    assert result.exit_code == 4
    assert "RL0002" in result.output
    assert "outcome" not in result.output.lower() or "required" not in result.output.lower()
    help_result = CliRunner().invoke(cli, ["forecast", "resolve", "--help"])
    assert help_result.exit_code == 0
    assert "--outcome" in help_result.output
    issue_help = CliRunner().invoke(cli, ["forecast", "issue", "--help"])
    assert issue_help.exit_code == 0
    assert "--record" in issue_help.output


def test_t1_chronology_seals_stakes_and_refuses_later_issue_or_resolution(tmp_path: Path) -> None:
    from reward_lens.forecast.evidence import ForecastRefusal
    from reward_lens.product.forecast import issue, resolve

    project = SimpleNamespace(root=tmp_path)
    first = tmp_path / "first.json"
    first.write_text(json.dumps({"issues": [_issue_row("T1")]}))
    issue(SimpleNamespace(record=first, claim="one", offline=True, seed=7), project=project)
    path = tmp_path / ".reward-lens" / "forecast-ledger.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    first_issue = next(row for row in rows if row["kind"] == "issue")
    assert first_issue["t1_sequence"] == 0
    assert set(first_issue["t1_stakes"]) == {"persistence", "climatology", "drift", "covariance"}
    second = tmp_path / "second.json"
    second.write_text(json.dumps({"issues": [_issue_row("T1")]}))
    with pytest.raises(ForecastRefusal, match="settle first"):
        issue(SimpleNamespace(record=second, claim="two", offline=True, seed=7), project=project)
    resolve(
        SimpleNamespace(
            forecast_id=first_issue["sealed_digest"], outcome="true", offline=True, seed=7
        ),
        project=project,
    )
    issue(SimpleNamespace(record=second, claim="two", offline=True, seed=7), project=project)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    second_issue = [row for row in rows if row["kind"] == "issue"][-1]
    assert second_issue["t1_sequence"] == 1
    assert any(Decimal(stake) > 0 for stake in second_issue["t1_stakes"].values())
    # The only later-issued unresolved T1 is sequence 1; a forged sequence cannot settle out of order.
    second_issue["t1_sequence"] = 9
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n"
    )
    with pytest.raises(ForecastRefusal, match="row digest mismatch"):
        resolve(
            SimpleNamespace(
                forecast_id=second_issue["sealed_digest"], outcome="true", offline=True, seed=7
            ),
            project=project,
        )


def test_t1_replay_preserves_crossing_history_after_current_wealth_falls(tmp_path: Path) -> None:
    import reward_lens.product.forecast as product
    from reward_lens.forecast.evidence import ForecastRefusal
    from reward_lens.product.forecast import issue, ledger_machine_state, render_ledger, resolve

    project = SimpleNamespace(root=tmp_path)
    for index, outcome in enumerate(("true",) * 15 + ("false",)):
        record = tmp_path / f"prospective-{index}.json"
        issue_row = _issue_row("T1")
        issue_row["forecast"] = 1
        issue_row["baselines"] = {
            name: 0 for name in ("persistence", "climatology", "drift", "covariance")
        }
        record.write_text(json.dumps({"issues": [issue_row]}))
        issue(
            SimpleNamespace(record=record, claim=f"history-{index}", offline=True, seed=7),
            project=project,
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / ".reward-lens" / "forecast-ledger.jsonl")
            .read_text()
            .splitlines()
        ]
        digest = next(row["sealed_digest"] for row in reversed(rows) if row["kind"] == "issue")
        resolve(
            SimpleNamespace(forecast_id=digest, outcome=outcome, offline=True, seed=7),
            project=project,
        )

    state = ledger_machine_state(project)
    assert state["headline_state"]["qualified"] is True
    assert all(Decimal(value) < 20 for value in state["current_wealth"].values())
    assert set(state) == {"first_crossing", "current_wealth", "wealth_history", "headline_state"}
    assert set(state["first_crossing"]["index"].values()) == {14}
    assert all(value is not None for value in state["first_crossing"]["time"].values())
    assert all(len(history) == 16 for history in state["wealth_history"].values())
    rendered = render_ledger(project)
    assert rendered.count("controlled-run") == 16
    assert "headline | qualified=true" in rendered
    assert "'current_wealth'" not in rendered

    path = tmp_path / ".reward-lens" / "forecast-ledger.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    tampered = next(row for row in reversed(rows) if row["kind"] == "resolve")
    tampered["headline_state"]["qualified"] = False
    previous = None
    resealed = []
    for row in rows:
        payload = dict(row)
        payload.pop("row_digest")
        sealed = product._seal_row(payload, previous)
        resealed.append(sealed)
        previous = sealed["row_digest"]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in resealed) + "\n"
    )
    with pytest.raises(ForecastRefusal, match="headline state differs"):
        ledger_machine_state(project)


def test_t1_replay_refuses_missing_required_chronology_fields(tmp_path: Path) -> None:
    import reward_lens.product.forecast as product
    from reward_lens.forecast.evidence import ForecastRefusal

    project = SimpleNamespace(root=tmp_path)
    record = tmp_path / "prospective.json"
    record.write_text(json.dumps({"issues": [_issue_row("T1")]}))
    product.issue(
        SimpleNamespace(record=record, claim="required", offline=True, seed=7), project=project
    )
    path = tmp_path / ".reward-lens" / "forecast-ledger.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    digest = next(row["sealed_digest"] for row in rows if row["kind"] == "issue")
    product.resolve(
        SimpleNamespace(forecast_id=digest, outcome="true", offline=True, seed=7), project=project
    )

    required = (
        "resolved_at",
        "wealth_by_baseline",
        "wealth_history_by_baseline",
        "first_crossing_by_baseline",
        "headline_state",
        "log10_wealth_hardest",
    )
    pristine = path.read_text()
    for field in required:
        changed = [json.loads(line) for line in pristine.splitlines()]
        next(row for row in changed if row["kind"] == "resolve").pop(field)
        previous = None
        resealed = []
        for row in changed:
            payload = dict(row)
            payload.pop("row_digest")
            sealed = product._seal_row(payload, previous)
            resealed.append(sealed)
            previous = sealed["row_digest"]
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in resealed)
            + "\n"
        )
        with pytest.raises(ForecastRefusal, match="required T1 chronology field"):
            product.ledger_machine_state(project)


def test_t1_replay_refuses_future_recorded_resolution_time(tmp_path: Path) -> None:
    import reward_lens.product.forecast as product
    from reward_lens.forecast.evidence import ForecastRefusal

    project = SimpleNamespace(root=tmp_path)
    record = tmp_path / "prospective.json"
    record.write_text(json.dumps({"issues": [_issue_row("T1")]}))
    product.issue(
        SimpleNamespace(record=record, claim="time", offline=True, seed=7), project=project
    )
    path = tmp_path / ".reward-lens" / "forecast-ledger.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    digest = next(row["sealed_digest"] for row in rows if row["kind"] == "issue")
    product.resolve(
        SimpleNamespace(forecast_id=digest, outcome="true", offline=True, seed=7), project=project
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    next(row for row in rows if row["kind"] == "resolve")["resolved_at"] = (
        "2099-01-01T00:00:00+00:00"
    )
    previous = None
    resealed = []
    for row in rows:
        payload = dict(row)
        payload.pop("row_digest")
        sealed = product._seal_row(payload, previous)
        resealed.append(sealed)
        previous = sealed["row_digest"]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in resealed) + "\n"
    )
    with pytest.raises(ForecastRefusal, match="future recorded resolution time"):
        product.ledger_machine_state(project)


def test_t1_replay_refuses_reordered_recorded_resolution_times(tmp_path: Path) -> None:
    import reward_lens.product.forecast as product
    from reward_lens.forecast.evidence import ForecastRefusal

    project = SimpleNamespace(root=tmp_path)
    for claim in ("first", "second"):
        record = tmp_path / f"{claim}.json"
        record.write_text(json.dumps({"issues": [_issue_row("T1")]}))
        product.issue(
            SimpleNamespace(record=record, claim=claim, offline=True, seed=7), project=project
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / ".reward-lens" / "forecast-ledger.jsonl")
            .read_text()
            .splitlines()
        ]
        digest = next(row["sealed_digest"] for row in reversed(rows) if row["kind"] == "issue")
        product.resolve(
            SimpleNamespace(forecast_id=digest, outcome="true", offline=True, seed=7),
            project=project,
        )

    path = tmp_path / ".reward-lens" / "forecast-ledger.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    issues = [row for row in rows if row["kind"] == "issue"]
    resolves = [row for row in rows if row["kind"] == "resolve"]
    issues[1]["issued_at"] = issues[0]["issued_at"]
    resolves[1]["resolved_at"] = resolves[0]["resolved_at"]
    previous = None
    resealed = []
    for row in rows:
        payload = dict(row)
        payload.pop("row_digest")
        sealed = product._seal_row(payload, previous)
        resealed.append(sealed)
        previous = sealed["row_digest"]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in resealed) + "\n"
    )
    with pytest.raises(ForecastRefusal, match="not strictly monotone"):
        product.ledger_machine_state(project)


def test_forecast_ledger_json_and_sdk_expose_one_machine_state(tmp_path: Path, monkeypatch) -> None:
    from reward_lens import api
    from reward_lens.cli.main import cli
    from reward_lens.product.forecast import issue, resolve

    monkeypatch.chdir(tmp_path)
    record = tmp_path / "prospective.json"
    record.write_text(json.dumps({"issues": [_issue_row("T1")]}))
    project = SimpleNamespace(root=tmp_path)
    issue(SimpleNamespace(record=record, claim="machine", offline=True, seed=7), project=project)
    rows = [
        json.loads(line)
        for line in (tmp_path / ".reward-lens" / "forecast-ledger.jsonl").read_text().splitlines()
    ]
    digest = next(row["sealed_digest"] for row in rows if row["kind"] == "issue")
    resolve(
        SimpleNamespace(forecast_id=digest, outcome="true", offline=True, seed=7), project=project
    )

    sdk = api.forecast_ledger(api.ForecastLedgerRequest(project=tmp_path))
    state = sdk.extensions["https://reward-lens.dev/forecast-ledger/v1"]
    assert set(state) == {"first_crossing", "current_wealth", "wealth_history", "headline_state"}
    assert set(state["current_wealth"]) == {"persistence", "climatology", "drift", "covariance"}

    result = CliRunner().invoke(cli, ["forecast", "ledger", "--format", "json"])
    payload = json.loads(result.output)
    assert payload["headline_state"]["criterion"] == "all_four_fixed_processes_ever_crossed_20"
    assert payload["first_crossing"] == state["first_crossing"]


def test_bundled_campaign_snapshot_refuses_tampering_or_absence(monkeypatch) -> None:
    import reward_lens.product.forecast as product
    from reward_lens.forecast.evidence import ForecastRefusal

    class Resource:
        def __init__(self, payload: bytes | None) -> None:
            self.payload = payload

        def joinpath(self, _name: str) -> "Resource":
            return self

        def read_bytes(self) -> bytes:
            if self.payload is None:
                raise FileNotFoundError
            return self.payload

    monkeypatch.setattr(product, "files", lambda _package: Resource(b"tampered"))
    with pytest.raises(ForecastRefusal, match="checksum mismatch"):
        product._campaign_from_evidence()
    monkeypatch.setattr(product, "files", lambda _package: Resource(None))
    with pytest.raises(ForecastRefusal, match="snapshot is missing"):
        product._campaign_from_evidence()
