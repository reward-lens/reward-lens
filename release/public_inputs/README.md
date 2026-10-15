# Public regression inputs

This tool rebuilds the release tests' E1 table and three campaign banks from
hash-pinned public records. It downloads about 1.38 GB of E1 inputs and a 7 MB
RewardBench 2 parquet. Downloads and generated data belong in a new scratch
directory outside the source tree. No model, paid API, or training run is used.

`sources.json` pins dataset revisions, hashes, and expected counts.
`vendor-provenance.json` identifies the small historical generator import closure;
its source files retain their original bytes and MIT license. The four analysis
records retain their recorded hashes. This source revision provides a maintained
public location for these previously local generator and analysis files.

```bash
python release/public_inputs/build_inputs.py eval-metadata > /absolute/scratch/eval-metadata.json
python release/public_inputs/build_inputs.py fetch-e1 /absolute/scratch/eval-metadata.json /absolute/scratch/public-e1
python release/public_inputs/build_inputs.py build-e1 /absolute/scratch/public-e1/data/train-00000-of-00001.parquet /absolute/scratch/public-e1/eval_logs /absolute/scratch/e1 --eval-metadata /absolute/scratch/eval-metadata.json
python release/public_inputs/build_inputs.py fetch-rb2 /absolute/scratch/rb2
python release/public_inputs/build_inputs.py build-campaign /absolute/scratch/campaign-cache --rb2-parquet /absolute/scratch/rb2/data/test-00000-of-00001.parquet
```

The E1 output is a new public parquet/Inspect-score join, with its own digest and
receipt. It does not claim the historical ignored extraction's byte identity.
The join requires all keys to match one to one and nine scorer correspondences
to agree; it then checks the original seven count assertions and native-grid
support. The three campaign banks must reproduce their original byte and content
hashes. The builder refuses absent or inconsistent inputs.

Set `REWARD_LENS_E1_ASSAY_ROOT`, `REWARD_LENS_E1_DERIVATION_RECEIPT`, and
`REWARD_LENS_CAMPAIGN_CACHE` to the generated directories/receipt before running
the release matrix. Missing required E1/bank inputs fail the tests. Historical
scientific suites that already require separate campaign evidence archives retain
their explicit availability guards; these generated banks do not supply those
scientific results. Neither raw datasets nor generated caches enter source history
or the product wheel.
