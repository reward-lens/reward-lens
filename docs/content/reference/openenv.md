# OpenEnv adapter

`reward.kind: openenv` binds a local OpenEnv 0.6.0 client through explicit project bridges. It is
a `stateful_tool_environment` adapter, not a new evidence family, and it never qualifies an outcome.

The binding pins package `openenv`, version `0.6.0`, and wheel SHA-256
`dd1033006d17c3c0147a02f73fc6deef881e15dc5501aea28930765708292641`. Its endpoint must be a
loopback `ws://` or `wss://` URL. Every project provides the client factory in `reward.entry`, then
`reward.openenv.reset` and `reward.openenv.action`; all are `module:callable` references.

The action bridge returns an OpenEnv `Action`, never a raw dictionary. A score calls the real
`EnvClient.sync()` context manager, makes one reset, then exactly one typed step. Only a finite,
non-boolean `StepResult.reward` scores. The envelope retains `done`; a false value records
`one step only; episode remained open` and does not lead to another action.

Malformed bindings and unresolved bridges are RL0003. An unavailable OpenEnv optional package is
RL0704 with extra `openenv`. Protocol and bridge exceptions after construction produce an unscored
`grader_error` envelope.
