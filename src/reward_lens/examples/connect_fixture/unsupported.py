"""The two shapes 4.0 refuses: verl's `compute_score` and a `step()` environment.

D-65 names both as unsupported with their shapes, because an adapter that half-works is worse
than a refusal. They ship as fixtures so the refusal is exercised against a real signature rather
than asserted in prose.

Owned by P-CONNECT.
"""

__all__ = ["StepEnvironment", "compute_score"]


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """verl's reward entry point. Its parameter names are how it is told apart."""
    return 1.0 if solution_str.strip() == str(ground_truth).strip() else 0.0


class StepEnvironment:
    """The `step() -> {reward}` environment shape: SkyRL, Tinker and OpenEnv all wear it."""

    def __init__(self) -> None:
        self.turns = 0

    def reset(self):
        self.turns = 0
        return {"observation": "", "info": {}}

    def step(self, action):
        self.turns += 1
        return {"observation": action, "reward": 0.0, "done": self.turns >= 3, "info": {}}
