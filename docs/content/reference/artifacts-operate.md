# Artifacts and operate

**Can a report say more than the evidence store can back?** No, and that is the whole point of `reward_lens.artifacts`: cards, atlases, and safety cases are views over the store, not fresh computations, so a claim on a card exists only if a piece of evidence in the store supports it. `reward_lens.operate` is the command-line and MCP surface that drives all of it.

## Cards and the atlas

`build_card` assembles a single model's card by reading the store, never by recomputing a number. `Atlas` is the population-scale view: a standard summary, a leaderboard, and a sweep that plans on CPU and gates its own execution behind hardware.

::: reward_lens.artifacts.card.build_card
    options:
      heading_level: 3

::: reward_lens.artifacts.atlas.Atlas
    options:
      heading_level: 3

## The claims checker

`check_text` and `check_files` scan a manuscript for claim tags of the form `[[claim value=… ev=… field=… tol=…]]` and verify each one against the store, failing when the evidence is absent, the field is missing, or the stored value is further from the claimed one than the tolerance allows. On the command line this is what makes `reward-lens claims` exit non-zero on an unbacked number. See [cards and claims](../how-to/cards-and-claims.md).

::: reward_lens.artifacts.claims.check_text
    options:
      heading_level: 3

::: reward_lens.artifacts.claims.check_files
    options:
      heading_level: 3

## The safety case

`assemble_safety_case` is the strictest artifact: it refuses to assemble unless every component it rests on is both calibrated and registered, raising `SafetyCaseRefusal` rather than producing a case the [trust ladder](../discipline/trust-ladder.md) cannot support.

::: reward_lens.artifacts.safety_case.assemble_safety_case
    options:
      heading_level: 3

## The command line and MCP

`reward_lens.operate` exposes the `reward-lens` command. The CPU-pure verbs read and check the store: `card`, `scoreboard`, `claims`, `atlas export`, and `study freeze`. The verbs that need a model are gated: `score`, `serve`, `audit`, `study run`, and `organism train` print the exact dispatch they would make and exit with code 3 rather than fabricate a result. The same module carries a minimal, in-process MCP server, honest in its own docstring about being transport-less. The full walkthrough is on [the command line](../cli.md).
