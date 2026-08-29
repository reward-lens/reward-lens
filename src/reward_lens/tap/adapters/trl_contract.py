"""The recording contract's emitter, over the TRL tap. BLK-048.

The contract is the design's rung 0 and its stated insurance, and before this
module no object wrote it. The temptation, and the named way this packet fails,
is a parallel parquet writer in the experiment directory: it would satisfy a
closure proof and violate the standing rule to fix the library and never the
caller, and it would leave the next run with the same gap.

So this subclasses `TRLTap` and writes through it. The rollout tables still go
out through `reward_lens.record.writer.RecordWriter`; the contract's own fields
go beside them through `reward_lens.record.contract.ContractWriter`, into the
same run directory. `reward_lens.tap.adapters.trl_signature` documents the
interface this is written against and `verify()` there fails if it drifts.

**Where each grain lands, and why some of it cannot go on the trajectory.**
`_features_for` is the only per-rollout scalar map the record model has and it
is `Mapping[str, float]`. Contract fields that are floats go there and reach the
released parquet as columns. Fields that are strings, objects or arrays cannot,
so they go into the contract record at their own grain. That split is a fact
about the record model, not a preference, and it is documented here rather than
discovered by whoever writes the next emitter.

**What the emitter refuses.** `emit` verifies field by field against the key set
the tap itself produced, so "every rollout carries every rollout field" is
checked against the rollouts the run actually emitted rather than against a row
count. A writer that stopped populating a field after the first group fails,
which is the case a count passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from reward_lens.record.contract import (
    BANNED_COLUMN_NAMES,
    ContractRecord,
    ContractRefusal,
    ContractSchema,
    ContractWriter,
)
from reward_lens.tap.adapters.trl import TRLTap

__all__ = ["ContractSources", "ContractTRLTap", "EmitReport", "MappingSources"]


class ContractSources(Protocol):
    """Everything the tap does not have, supplied by the run.

    The tap sees prompts, completions, scores and the trainer's configuration.
    It does not see the oracle's label, the hardware identifier, the fork
    lineage, the publication map or the bank. Those are the run's, and they
    arrive here rather than being invented by the emitter.
    """

    def run_fields(self) -> Mapping[str, Any]: ...
    def step_fields(self, step: int) -> Mapping[str, Any]: ...
    def group_fields(self, step: int, group: int) -> Mapping[str, Any]: ...
    def rollout_fields(self, step: int, group: int, gen: int) -> Mapping[str, Any]: ...
    def token_fields(self, step: int, group: int, gen: int) -> Mapping[str, Any]: ...
    def artifact_fields(self) -> Mapping[str, Mapping[str, Any]]: ...
    def quantity_fields(self) -> Mapping[str, Mapping[str, Any]]: ...


@dataclass
class MappingSources:
    """A `ContractSources` backed by plain dictionaries, for a rehearsal."""

    run: dict = field(default_factory=dict)
    step: dict = field(default_factory=dict)
    group: dict = field(default_factory=dict)
    rollout: dict = field(default_factory=dict)
    token: dict = field(default_factory=dict)
    artifact: dict = field(default_factory=dict)
    quantity: dict = field(default_factory=dict)

    def run_fields(self):
        return self.run

    def step_fields(self, step):
        return self.step.get(step, {})

    def group_fields(self, step, group):
        return self.group.get((step, group), {})

    def rollout_fields(self, step, group, gen):
        return self.rollout.get((step, group, gen), {})

    def token_fields(self, step, group, gen):
        return self.token.get((step, group, gen), {})

    def artifact_fields(self):
        return self.artifact

    def quantity_fields(self):
        return self.quantity


@dataclass(frozen=True)
class EmitReport:
    root: Path
    contract_dir: Path
    rollouts: int
    steps: int
    fields_checked: int
    pending_fields: tuple[str, ...]


def rollout_key(step: int, group: int, gen: int) -> str:
    """One rollout, addressed by where it is rather than by a hash.

    `BLK-009`'s finding is that the tap put the group into a content-addressed
    `Group.id` and the generation index inside a trajectory id, so neither is
    recoverable without inverting a hash. The contract addresses a rollout by
    the triple, in the open.
    """
    return f"{int(step)}:{int(group)}:{int(gen)}"


class ContractTRLTap(TRLTap):
    """`TRLTap`, plus the recording contract."""

    def __init__(self, *, contract: ContractSchema, sources: ContractSources, **kw: Any) -> None:
        super().__init__(**kw)
        self.contract = contract
        self.sources = sources
        self.record = ContractRecord(schema=contract)
        self._emitting_step: int = 0
        self._rollouts: list[str] = []

    def declare_config(self, **config: Any) -> None:
        """Declare the run configuration without a trainer.

        In a real run `attach` reads this off `GRPOConfig`. The dress rehearsal
        has no trainer and still has to exercise the tap's own grouping path,
        which needs `num_generations`, so the geometry is declared here rather
        than reached for through a private attribute by the caller.
        """
        self._config.update(config)

    # ------------------------------------------------------------- the seams
    def _groups_for(self, b):  # noqa: D102 - see trl_signature
        self._emitting_step = int(b.index)
        return super()._groups_for(b)

    def _features_for(self, row, weights, **kw):
        """The tap's per-rollout float map, plus this run's contract floats.

        Everything non-float for the same rollout is put into the contract
        record here too, because this is the one call that knows the triple.
        """
        out = dict(super()._features_for(row, weights, **kw))
        g = kw.get("group_ordinal")
        j = kw.get("generation_index")
        if g is None or j is None:
            # BLK-009: without the grouping the rollout has no address, and a
            # contract field written against an unknown rollout is worse than an
            # absent one. Refuse rather than guess.
            raise ContractRefusal(
                "the recording contract needs the prompt-group id and the "
                "generation index on every rollout and the tap could not "
                "provide them for this step. BLK-009 owns the capture; the "
                "grouping provenance is in `unverified_grouping_steps`"
            )
        key = rollout_key(self._emitting_step, g, j)
        self._rollouts.append(key)
        for name, value in self.sources.rollout_fields(self._emitting_step, g, j).items():
            self.record.put("rollout", key, name, value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[name] = float(value)
        for name, value in self.sources.token_fields(self._emitting_step, g, j).items():
            self.record.put("token", key, name, value)
        for banned in BANNED_COLUMN_NAMES & set(out):
            raise ContractRefusal(f"a rollout feature is named `{banned}`; no column anywhere is")
        return out

    def _turns_for(
        self,
        prompt: str,
        completion: str,
        completion_ids: "tuple[int, ...] | None" = None,
    ):
        """Unchanged in shape. The token arrays ride on the contract record.

        `Turn` already declares `token_ids`, `logprobs_sampling`,
        `logprobs_train` and `loss_mask`, and the tap sets only index, role and
        text. Populating them here would need more than the ids, so contract row
        10 is written at token grain in the contract record instead, keyed on the
        same triple. That is recorded rather than hidden: a `Turn` carrying
        logprobs of the wrong length with no `token_ids` constructs cleanly, so
        half-populating it is a trap.

        **The signature is the fix.** This override took two positional
        arguments while `TRLTap._groups_for` calls it with three, so `finish()`
        raised `TypeError: _turns_for() takes 3 positional arguments but 4 were
        given` on any run that captured completion ids, which is every real run.
        BLK-048's dress rehearsal was red at HEAD for this and a verifier had
        already patched it by hand. The docstring above used to say the ids do
        not arrive on this call; they do, and saying otherwise is what kept the
        wrong signature looking deliberate.
        """
        return super()._turns_for(prompt, completion, completion_ids)

    # ------------------------------------------------------------- the emit
    def collect(self, run: Any) -> ContractRecord:
        """Every grain the seams do not reach, gathered from the run's sources."""
        for name, value in self.sources.run_fields().items():
            self.record.put("run", "run", name, value)
        for step in run.steps:
            for name, value in self.sources.step_fields(int(step.index)).items():
                self.record.put("step", int(step.index), name, value)
            for gi, group in enumerate(step.groups):
                for name, value in self.sources.group_fields(int(step.index), gi).items():
                    self.record.put("group", f"{int(step.index)}:{gi}", name, value)
        for path, fields in self.sources.artifact_fields().items():
            for name, value in fields.items():
                self.record.put("artifact", path, name, value)
        for qid, fields in self.sources.quantity_fields().items():
            for name, value in fields.items():
                self.record.put("quantity", qid, name, value)
        return self.record

    def expected_keys(self, run: Any) -> dict:
        """The key set the assertion runs over, taken from the RUN.

        This is the difference between "every rollout carries every field" and
        "the row count looks right". The rollout keys come out of the emitted
        record, so a writer that stopped after the first group is a failure here
        and a pass under a count.
        """
        steps = [int(s.index) for s in run.steps]
        groups, rollouts = [], []
        for s in run.steps:
            for gi, g in enumerate(s.groups):
                groups.append(f"{int(s.index)}:{gi}")
                for j, _t in enumerate(g.trajectories):
                    rollouts.append(rollout_key(int(s.index), gi, j))
        return {
            "run": ["run"],
            "step": steps,
            "group": groups,
            "rollout": rollouts,
            "token": rollouts,
            "artifact": list(self.sources.artifact_fields()),
            "quantity": list(self.sources.quantity_fields()),
        }

    def emit(
        self,
        root: str | Path,
        *,
        kind: str = "train",
        allow_pending: bool = False,
        write_record: bool = True,
    ) -> EmitReport:
        """Finish through the tap, persist, then refuse field by field."""
        run = self.finish(kind=kind)
        if write_record:
            from reward_lens.record.writer import RecordWriter

            RecordWriter(root).write(run)
        self.collect(run)
        expect = self.expected_keys(run)
        self.record.verify(expect=expect, allow_pending=allow_pending)
        d = ContractWriter(root=root, run_id=str(run.id)).write(self.record)
        return EmitReport(
            root=Path(root),
            contract_dir=d,
            rollouts=len(expect["rollout"]),
            steps=len(expect["step"]),
            fields_checked=len(self.contract),
            pending_fields=tuple(f.name for f in self.contract.pending()),
        )
