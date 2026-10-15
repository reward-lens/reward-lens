---
name: reward-lens
description: "when you write or change a reward function, rubric, grader or environment, run `reward-lens audit`, paste the exploit table into your reply, and do not hand back a grader with an open reach finding."
license: MIT
compatibility: "reward-lens 3.1.0; a local project with the reward-lens CLI, and the optional MCP server for seeker.propose"
metadata:
  disable-model-invocation: true
allowed-tools:
  - "Bash(reward-lens audit:*)"
  - "mcp__reward_lens__seeker_propose"
---

# reward-lens audit

Use this skill when a request creates or changes a reward function, rubric,
grader, or environment. Run the audit before saying the grader is ready.

1. Run `reward-lens audit <project>` with the project policy and its declared
   offline/sandbox constraints. Do not add a paid provider or a network route
   merely to finish an audit.
2. Read the saved record and its exploit/reach findings. Put the actual exploit
   table in the reply. A missing or inconclusive finding stays missing or
   inconclusive; it is not a pass.
3. Do not hand back a grader with an open reach finding. State the precise
   repair or refusal required to close it.

For a candidate response that the project asks an independent actor to score,
call the installed MCP tool `seeker.propose` with the project's bounded,
offline request and the actor identity. It returns a typed proposal, score,
outcome, and saved path. Do not invent a score, use a different grader, or
silently select a paid/remote actor. If the project has not supplied the task,
response, and explicit budget conditions, say that the proposal cannot be
made.

`reward-lens audit` writes records and can execute a grader, so it is a
side-effecting route. Inspect `reward-lens describe` or command help before
choosing flags, and obtain the user's explicit argument for any write outside
the project or any spend.
