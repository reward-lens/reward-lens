"""Examples that ship in the wheel (D-63): the code-reward demo with its real planted defect under
code_reward/, and the connector fixture under connect_fixture/. Shared parent; each subpackage has
one owner. This file is controller-owned: REGISTRY is a lazy pointer table, like the rung registry,
so that `reward-lens init --example <name>` resolves a name without importing every example.
"""

REGISTRY: dict[str, str] = {
    "code-reward": "reward_lens.examples.code_reward:write",   # P-EXAMPLE, wave 1
}
