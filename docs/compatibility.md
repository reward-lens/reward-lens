# Adapter compatibility

`reward-lens doctor` probes the adapter families available in the current Python environment. Its compatibility matrix comes from `reward_lens.product.access.matrix.build()`. A row marked `ran` means a local adapter was constructed and accepted a valid input while refusing a malformed one. The family test suites exercise scoring and evidence records separately. Neither result verifies a live provider.

The table below records a run in the development environment on 2026-10-13. This environment has the OpenEnv 0.6.0 extra and lacks the `train` extra. Run `reward-lens doctor` on the machine that will perform an audit to see its current status.

| Family | Local input probe | Fixture | Framework | Install route for the 3.1.0 release | Live profile |
| --- | --- | --- | --- | --- | --- |
| Local deterministic | ran | packaged `code_reward/my_grader.py` | built in | `pip install reward-lens==3.1.0` | unverified |
| Stateful tool environment | ran | local OpenEnv contract fixture | OpenEnv 0.6.0 | `pip install 'reward-lens[openenv]==3.1.0'` | unverified |
| LLM judge | ran | local judge JSON protocol fixture | local protocol v1 | `pip install reward-lens==3.1.0` | unverified |
| Neural reward model | missing extra | in-memory tiny CPU scoring head | PyTorch local CPU | `pip install 'reward-lens[train]==3.1.0'` | unverified |
| Composite | ran | one-component packaged Python grader | built in | `pip install reward-lens==3.1.0` | unverified |
| Remote black box | ran | local remote wire-protocol fixture | remote protocol v1 | `pip install reward-lens==3.1.0` | unverified |

Version 3.1.0 is published. The table pins each distribution install to that
version. In a source checkout, use `python -m pip install -e .`,
`python -m pip install -e '.[openenv]'`, or
`python -m pip install -e '.[train]'` for the corresponding routes.

Each successful row reports support for `manifest` and `evidence_envelope` records. The row also exposes the adapter's measured capability flags, determinism class, network policy, access level, and replay mode. A `missing extra` row names the install route. A `did not run` row carries the probe error. The matrix does not promote fixture responses to measurements of a named commercial model or to independent task outcomes.

For a machine-readable snapshot, call `reward-lens doctor --format json` and read its `compatibility_matrix` field. The existing capability list remains a separate part of that response. In Python:

```python
from reward_lens.product.access.matrix import build

for family, row in build().items():
    print(family, row["conformance"], row["live"], row.get("install_route", ""))
```
