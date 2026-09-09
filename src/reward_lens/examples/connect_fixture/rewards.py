"""The three-function reward file of section 5.9, and the one the connect transcript is read off.

This is written the way an adopter's `rewards.py` is written: two TRL reward functions that a
`GRPOTrainer` would be handed in `reward_funcs`, and one plain scorer. Nothing here imports TRL,
and nothing here knows about reward-lens. Annotations are real objects rather than strings,
because the connector reads the return annotation and so does `verifiers`' own group test.

Owned by P-CONNECT.
"""

__all__ = ["correctness_reward", "format_reward", "length_penalty"]


def correctness_reward(prompts, completions, completion_ids, **kwargs) -> list[float]:
    """1.0 where the completion ends in the reference answer, else 0.0."""
    answers = kwargs.get("answer") or [""] * len(completions)
    return [
        1.0 if answer and completion.strip().endswith(answer) else 0.0
        for completion, answer in zip(completions, answers)
    ]


def format_reward(prompts, completions, **kwargs) -> list[float]:
    """1.0 where the completion carries the fenced block the task asked for."""
    return [1.0 if completion.count("```") >= 2 else 0.0 for completion in completions]


def length_penalty(completion) -> float:
    """A plain single-item scorer: shorter is better, saturating at 400 characters."""
    return max(0.0, 1.0 - len(completion) / 400.0)
