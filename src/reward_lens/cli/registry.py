"""The command table, and the root help generated from it.

One source. `COMMANDS` is what the lazy group resolves, `GROUPS` and `SUMMARIES` are what the root
help prints, and `root_help()` renders the second from the first. Nothing here imports a verb, and
nothing here imports anything outside the standard library, because this module is on the `--help`
path (D-30).

The count: D-19's heading says sixteen and its own listing carries fourteen plus `version` and
`completion`, both marked "(unlisted in root help)". The frozen root help carries the fourteen, and
`fleet/CLI_COMMANDS.json` names them as the contracted set, so the fourteen are what this table
holds. `--version` is a root flag instead, which is the half of appendix G defect 1 that matters.
"""

from __future__ import annotations

PROG = "reward-lens"
TAGLINE = "Build better rewards. Test what they teach."
USAGE = f"Usage: {PROG} <command> [options]"
DOCS = "https://reward-lens.github.io/docs/"

#: Every contracted command, in the order the root help lists them.
ORDER: tuple[str, ...] = (
    "audit",
    "trace",
    "compare",
    "forecast",
    "improve",
    "open",
    "export",
    "import",
    "runs",
    "init",
    "doctor",
    "explain",
    "describe",
    "mcp",
)

#: name -> "module:attribute". The module is imported when the name is invoked, and not before.
COMMANDS: dict[str, str] = {
    "audit": "reward_lens.cli.verbs.audit:audit",
    "trace": "reward_lens.cli.verbs.trace:trace",
    "compare": "reward_lens.cli.verbs.compare:compare",
    "forecast": "reward_lens.cli.verbs.forecast:forecast",
    "improve": "reward_lens.cli.verbs.improve:improve",
    "open": "reward_lens.cli.verbs.open:open_",
    "export": "reward_lens.cli.verbs.export:export",
    "import": "reward_lens.cli.verbs.import_cmd:import_",
    "runs": "reward_lens.cli.verbs.runs:runs",
    "init": "reward_lens.cli.verbs.init:init",
    "doctor": "reward_lens.cli.verbs.doctor:doctor",
    "explain": "reward_lens.cli.verbs.explain:explain",
    "describe": "reward_lens.cli.verbs.describe:describe",
    "mcp": "reward_lens.cli.verbs.mcp:mcp",
}

SUMMARIES: dict[str, str] = {
    "audit": "what this reward accepts, reaches, pays for, and teaches",
    "trace": "what a training run optimised, and what it selected",
    "compare": "what changes if the reward changes",
    "forecast": "freeze a prediction, resolve it, score the ledger",
    "improve": "start from a goal and run the loop",
    "open": "open a report, or the workbench",
    "export": "write a revision, a bundle, a badge, or a SARIF file",
    "import": "validate a report and attach it to a project",
    "runs": "list, show, continue or remove the runs on this machine",
    "init": "connect a reward system, or start from an example",
    "doctor": "what can be measured here, and what it would cost",
    "explain": "what an error code or a finding means",
    "describe": "the command tree as versioned JSON, for tools and agents",
    "mcp": "run the agent server on stdio",
}

GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Measure", ("audit", "trace", "compare", "forecast", "improve")),
    ("Records", ("open", "export", "import", "runs")),
    ("Set up", ("init", "doctor", "explain", "describe", "mcp")),
)

START_HERE: tuple[str, ...] = (
    f"{PROG} init --example code-reward ./demo",
    f"{PROG} audit ./demo",
)

#: What each verb does to the world, reported by `describe` so an agent can plan before it acts.
SIDE_EFFECTS: dict[str, str] = {
    "audit": "reads the project, writes a record and a report, executes the grader in a sandbox",
    "trace": "reads a run record, writes a record and a report",
    "compare": "reads two records or two versions, writes a record and a report",
    "forecast": "reads and writes the project's forecast ledger",
    "improve": "reads the project, executes the grader, writes records and a report",
    "open": "reads a record, opens a viewer",
    "export": "reads a record, writes files",
    "import": "reads a record, writes into the project",
    "runs": "reads and removes run directories on this machine",
    "init": "writes files into a destination directory",
    "doctor": "reads this machine and this project, writes nothing",
    "explain": "writes nothing, reads nothing, works offline",
    "describe": "writes nothing",
    "mcp": "serves on stdio, and the tools it exposes carry their own side effects",
}

#: The shape of what each verb puts on stdout under `--format json`.
OUTPUT_SHAPES: dict[str, str] = {
    "audit": "assay-result/1.0",
    "trace": "assay-result/1.0",
    "compare": "assay-result/1.0",
    "forecast": "assay-result/1.0",
    "improve": "assay-result/1.0",
    "open": "assay-result/1.0",
    "export": "assay-result/1.0",
    "import": "assay-result/1.0",
    "runs": "run-list/1.0",
    "init": "assay-result/1.0",
    "doctor": "capabilities/1.0",
    "explain": "error-explanation/1.0",
    "describe": "command-tree/1.0",
    "mcp": "jsonrpc on stdio",
}

_NAME_WIDTH = 12

#: What A-027 appends to a verb this build cannot run, after its summary and not in a column of
#: its own: the list is read aloud, and a column would make the mark look like a second summary.
ABSENT_MARK = "  (not in this build)"


def root_help() -> str:
    """The frozen root help, generated from this table.

    `fleet/golden/target/root-help.txt` is a transcript: its first line is the invocation and the
    rest is this text. The test compares against the rest.

    A-027: a verb whose engine is not in this build carries `ABSENT_MARK`, so the list a reader is
    sent to by RL0703 says which verbs that is. The target transcript is the end state, where every
    engine has landed and no line takes the mark; the wave-1 transcript is this build.
    """
    from . import availability  # here, not at the top: `availability` reads this module.

    absent = availability.marked_absent()
    lines = [TAGLINE, "", USAGE, ""]
    for heading, names in GROUPS:
        lines.append(heading)
        for name in names:
            mark = ABSENT_MARK if name in absent else ""
            lines.append(f"  {name:<{_NAME_WIDTH}}{SUMMARIES[name]}{mark}")
        lines.append("")
    lines.append("Start here")
    for example in START_HERE:
        lines.append(f"  {example}")
    lines.append("")
    lines.append(f"Per-command options: {PROG} audit --help")
    lines.append(f"Docs: {DOCS}")
    return "\n".join(lines) + "\n"
