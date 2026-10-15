# Pinned CPU validation model

The all-extras release profile uses the public
[`trl-internal-testing/tiny-Qwen3ForCausalLM` model](https://huggingface.co/trl-internal-testing/tiny-Qwen3ForCausalLM)
at the immutable revision recorded in `model-files.json`. It exercises real model
loading, policy scoring, capture coordinates, and the retained CPU conformance
checks. This fixture is separate from `release/public_inputs`.

Run the provisioner with a new cache owned by the validation run, then keep model
loading offline:

```sh
python release/validation_model/provision_model.py --hf-home "$validation_model_home"
export HF_HOME="$validation_model_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python release/validation_model/provision_model.py --hf-home "$validation_model_home" --offline
```

Set `validation_model_home` to a new temporary directory before these commands.
The provisioner checks the complete selected file set, per-file sizes, SHA-256,
and upstream Git or LFS identities. The selected files total 20,792,513 bytes;
the download bound is 25 MB. Configuration and weight hashes also anchor the
retained model-identity assertions independently of Transformers private fields.

The pinned model card declares no license. This repository records public-source
provenance and downloads this external fixture for validation; it does not
redistribute model or tokenizer weights. Do not commit a downloaded cache or add
it to a wheel, sdist, public-input artifact, or release asset. The fixture is not
needed for ordinary base installation or first use. Missing or incompatible
required model inputs block all-extras validation.
