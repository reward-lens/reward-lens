"""A `@vf.reward` pair, individual and group, with nothing from `verifiers` imported.

`Rubric` decides which it has by parameter names and return annotation, and binds the arguments by
name subset. These two callables are what the connector's discriminator and the framework's own
functions are run against, side by side, on the same object.

Owned by P-CONNECT.
"""

__all__ = ["group_reward", "individual_reward"]


def individual_reward(prompt, completion, answer, state) -> float:
    """Individual: singular parameter names, one number back."""
    return 1.0 if answer and answer in completion else 0.0


def group_reward(prompts, completions, answers) -> list[float]:
    """Group: plural parameter names, a list back. Both halves of `_is_group_func` fire."""
    return [
        1.0 if answer and answer in completion else 0.0
        for completion, answer in zip(completions, answers)
    ]
