"""Specific invalidation: the five rows of section 6.1, as one rule.

An entry is stale when it is about the subject that changed **and** its declared dependency set
touches a digest the change perturbs. Nothing else. That rule reproduces every cell of the table,
including the last one: a forecast entry that named a different subject is not about this subject,
so it stands whatever happens to this grader's source.

Invalidation is transitive. The only derivation section 6.1 states is that the exact samples are
drawn from the task distribution by the policy, so `close()` carries a change to either of those
through to the samples. The five rows are already closed under it, and a test says so.
"""

from __future__ import annotations

from typing import Iterable

from .digests import DIGEST_NAMES

__all__ = ["CHANGE_KINDS", "DERIVES_FROM", "PERTURBS", "close", "partition", "resolve"]

#: The five changes of the section 6.1 table, in table order.
CHANGE_KINDS: tuple[str, ...] = (
    "judge_revision",
    "target_checkpoint",
    "outcome_label_version",
    "acceptance_policy",
    "grader_source",
)

#: Which of the nine digests each change perturbs. A judge is named by the scorer configuration,
#: never by the grader's source; a checkpoint is the policy and the samples it produced; the
#: acceptance policy perturbs no measurement at all, which is why its row invalidates the decision
#: and nothing else.
PERTURBS: dict[str, frozenset[str]] = {
    "judge_revision": frozenset({"scorer_config"}),
    "target_checkpoint": frozenset({"policy", "samples"}),
    "outcome_label_version": frozenset({"outcome_protocol"}),
    "acceptance_policy": frozenset(),
    "grader_source": frozenset({"source"}),
}

#: digest -> the digests it is derived from.
DERIVES_FROM: dict[str, tuple[str, ...]] = {"samples": ("task_distribution", "policy")}

#: The two changes that take the decision with them: one is the decision's own rule, the other is
#: the labels every qualification rests on.
DECIDES: frozenset[str] = frozenset({"acceptance_policy", "outcome_label_version"})


def close(digests: Iterable[str]) -> frozenset[str]:
    """The digests reachable from `digests` by derivation."""
    out = set(digests)
    growing = True
    while growing:
        growing = False
        for derived, sources in DERIVES_FROM.items():
            if derived not in out and any(source in out for source in sources):
                out.add(derived)
                growing = True
    return frozenset(out)


def _name(token: str) -> str:
    return token[len("digest:") :] if token.startswith("digest:") else token


def resolve(changed: Iterable[str]) -> frozenset[str]:
    """The digests a set of changes perturbs. A change is a table row or a digest by name."""
    perturbed: set[str] = set()
    for token in changed:
        name = _name(token)
        if name in PERTURBS:
            perturbed |= PERTURBS[name]
        elif name in DIGEST_NAMES:
            perturbed |= close({name})
        else:
            raise ValueError(
                f"{token!r} is not one of the five changes of section 6.1 "
                f"({', '.join(CHANGE_KINDS)}) and not one of the nine digests "
                f"({', '.join(DIGEST_NAMES)})"
            )
    return frozenset(perturbed)


def partition(assay, changed: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(stale, standing)` entry ids, in record order, for this request."""
    tokens = {"digest:" + name for name in resolve(changed)}
    subject = assay.subject.version.digest
    stale: list[str] = []
    standing: list[str] = []
    for entry in assay.entries():
        about_this_subject = entry.subject_ref == subject
        touched = bool(set(entry.depends_on) & tokens)
        (stale if about_this_subject and touched else standing).append(entry.entry_id)
    return tuple(stale), tuple(standing)


def decision_stale(assay, changed: Iterable[str]) -> bool:
    """Row 4 invalidates the decision alone; row 3 takes it too; otherwise it follows its reasons."""
    kinds = {_name(token) for token in changed}
    if kinds & DECIDES:
        return True
    stale, _ = partition(assay, changed)
    rests_on = {reason.split(":", 1)[1] for reason in assay.decision.reasons if ":" in reason}
    return bool(rests_on & set(stale))
