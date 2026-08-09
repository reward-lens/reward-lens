"""The rung registry: controller-owned for the whole build (section 10.2, W0.7).

Each rung of ``compare`` (section 7.5) is registered here by number as a lazy ``module:function``
pointer once its packet integrates. Nothing here imports a rung module; the pointer is resolved on
demand by the compare verb. A rung absent from this table is not runnable and ``doctor`` says so.
"""

RUNGS: dict[int, str] = {
    # 1: "reward_lens.product.compare.rung1:run",   # P-RUNG1, wave 3
    # 2: "reward_lens.product.compare.rung2:run",   # P-RUNG2, wave 3
    # 3: "reward_lens.product.compare.rung3:run",   # P-RUNG3, wave 3
    # 4: "reward_lens.product.compare.rung4:run",   # P-LEARN-SWAP, wave 4
    # 5: "reward_lens.product.compare.rung5:run",   # P-LEARN-FORK, wave 4
    # 6: "reward_lens.product.compare.rung6:run",   # P-RUNG6, wave 4
}

SCOPE_LABEL: dict[int, str] = {
    1: "evaluator_comparison",
    2: "selection_stress",
    3: "reconstructed_training_pressure",
    4: "controlled_update_response",
    5: "short_fork_behavior",
    6: "held_out_transfer",
}
