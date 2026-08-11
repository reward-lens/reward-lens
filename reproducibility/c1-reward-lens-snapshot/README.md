# Historical C1 dependency

This installable snapshot preserves the reward-lens substrate used by the C1 study. It reports the historical package version 3.0.0. The maintained product at the repository root targets unreleased 3.1.0.

The historical source identifier is `607774d19f79140234260da712a18487eca5b45c`. That private history is not part of this public repository. The new public export commit and this subdirectory jointly identify the installable dependency; GRPO Reward Trace pins both.

[The selected-file manifest](provenance/reward-lens-export.json) records the original Git blobs and SHA-256 hashes for the selected Python package and its runtime data (the unrelated X8 private-dossier analysis is excluded). The MIT license and attribution are retained in this snapshot in [LICENSE](LICENSE). Packaging and public-provenance changes are listed separately from executable changes. Every selected runtime file and package-data file has been checked against the historical blob and built wheel. The CPU claim-routing fixture does not import reward-lens and cannot by itself validate this dependency; separate import checks and their limits are recorded with the study. GPU behavior has not been revalidated by this source publication.

Use the immutable installation command in [GRPO Reward Trace](https://github.com/reward-lens/grpo-reward-trace). Install this snapshot in its own environment. It and the current product use the same distribution and import names, so they are alternative installations.

This directory is retained in Git and excluded from the maintained product's wheel and source distribution. It has no independent release or tag.
