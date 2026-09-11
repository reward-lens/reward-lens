# Retained v3 compatibility fixes

The 3.1.0 source retains six behavior fixes that were verified against the published 3.0.0 line and the current source snapshot. The private port ledger that coordinated the transfer is intentionally not published. The evidence below points to source and public behavior tests instead.

1. Base-install test collection excludes white-box tests when Torch is absent. `tests/conftest.py` scans test source before import, so a base environment does not import model-only tests during collection.
2. `reward_lens.tap.adapters.trl.TRLTap` attaches a callback even when `transformers` is unavailable. `tests/compat/test_v3_reconciliation.py` uses a trainer-shaped fixture.
3. `reward_lens.measure.base.capability_name` names `Flag` combinations without calling `int()` on a `Flag`. The compatibility test covers empty, atomic, and composite sets.
4. The public-surface checker retains its reachability rules for deliberately private and dual-use module paths. Its broad optional-dependency run is recorded as incomplete in `TEST_STATUS.md`; this evidence preserves the rule without claiming a base-install pass.
5. `tests/acceptance/test_debt_m.py` uses floors and ratios for backend-sensitive window counts instead of exact totals. The compatibility test reads that assertion source without importing optional runtime code.
6. `reward_lens.policy.vllm.EngineBoundary` inherits both `RuntimeError` and `AttributeError`, so unavailable serving-engine attributes make `hasattr` return `False`. The compatibility test checks the class and its `__getattr__` boundary from source.

These tests preserve behavior. They do not claim that all unfinished product work has been reconciled or released.
