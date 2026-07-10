"""``reward_lens.studies`` — the science layer's engine (gate 3, section 2.14).

A Study is the unit of confirmatory work: a frozen spec plus a thin analysis function (R9). This
package provides the spec schema, freezing (which stamps the git sha and makes predictions
uneditable after the run), the runner (which produces REGISTERED Evidence and adjudicates it
against the frozen predictions), the theorem scoreboard, and the report renderer. The sixteen
sciences live under the top-level ``studies/`` directory as specs and analysis functions; none is
a subsystem.

Torch-free: the engine orchestrates Evidence and the gates, so it imports without torch. The
analysis functions a study resolves may of course touch models.
"""

from __future__ import annotations

from reward_lens.studies.freeze import FrozenStudy, freeze
from reward_lens.studies.report import render_report
from reward_lens.studies.runner import StudyRun, run_study
from reward_lens.studies.scoreboard import DEFAULT_ROWS, Scoreboard, ScoreboardRow
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)

__all__ = [
    "StudySpec",
    "Hypothesis",
    "Prediction",
    "KillCriterion",
    "SubjectQuery",
    "StudyResult",
    "freeze",
    "FrozenStudy",
    "run_study",
    "StudyRun",
    "Scoreboard",
    "ScoreboardRow",
    "DEFAULT_ROWS",
    "render_report",
]
