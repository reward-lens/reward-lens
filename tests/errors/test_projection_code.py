"""RL0640 is one catalogue-backed refusal shared by every projection view."""

from __future__ import annotations

from reward_lens.errors import CATALOGUE, explain, make, render_two_line
from reward_lens.render.projections import ProjectionFieldAbsent, View


def test_rl0640_catalogue_row_is_a_usage_refusal_with_view_and_field_context() -> None:
    spec = CATALOGUE["RL0640"]

    assert spec.exit_code == 4
    assert spec.context_keys == ("view", "field")
    assert "projection" in spec.title


def test_projection_field_absent_uses_the_registered_message_and_remedy() -> None:
    error = ProjectionFieldAbsent(
        view=View.CASE,
        field="candidate_costs",
        pointer="/extensions/https:~1~1reward-lens.github.io~1case~1v1/candidate_costs",
    )
    registered = make("RL0640", view="case", field="candidate_costs")

    assert error.code == "RL0640"
    assert error.exit_code == error.exit_status == 4
    assert error.message == registered.message
    assert error.remediation == registered.remediation
    assert error.context == {
        "view": "case",
        "field": "candidate_costs",
        "pointer": "/extensions/https:~1~1reward-lens.github.io~1case~1v1/candidate_costs",
    }
    assert render_two_line(error).splitlines() == [
        "error: the case projection needs saved field candidate_costs, but the Assay does not carry it",
        "help:  record the required projection evidence before requesting the case projection: run: reward-lens audit .",
        "       reward-lens explain RL0640",
    ]


def test_rl0640_explain_is_context_free_and_keeps_the_exit_four_contract() -> None:
    text = explain("RL0640")

    assert "RL0640" in text
    assert "saved Assay" in text
    assert "{view}" not in text
    assert "{field}" not in text
    assert "reward-lens exits 4" in text
