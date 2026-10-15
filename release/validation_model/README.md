# Pinned CPU validation fixtures

The all-extras release profile uses the public
[`trl-internal-testing/tiny-Qwen3ForCausalLM` model](https://huggingface.co/trl-internal-testing/tiny-Qwen3ForCausalLM)
at the immutable revision recorded in `model-files.json`. It exercises real model
loading, policy scoring, capture coordinates, and the retained CPU conformance
checks. This fixture is separate from `release/public_inputs`.

Run the provisioner with a new cache owned by the validation run, then keep model
loading offline:

```sh
python release/validation_model/provision_model.py --hf-home "$validation_model_home"
python release/validation_model/provision_model.py --hf-home "$validation_model_home" --fixture gpt2-tokenizer
python release/validation_model/provision_model.py --hf-home "$validation_model_home" --fixture migration-llama-head
python release/validation_model/provision_model.py --hf-home "$validation_model_home" --fixture migration-qwen3
python release/validation_model/provision_model.py --hf-home "$validation_model_home" --fixture migration-gemma2-config
export HF_HOME="$validation_model_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python release/validation_model/provision_model.py --hf-home "$validation_model_home" --offline
python release/validation_model/provision_model.py --hf-home "$validation_model_home" --fixture gpt2-tokenizer --offline
python release/validation_model/provision_model.py --hf-home "$validation_model_home" --fixture migration-llama-head --offline
python release/validation_model/provision_model.py --hf-home "$validation_model_home" --fixture migration-qwen3 --offline
python release/validation_model/provision_model.py --hf-home "$validation_model_home" --fixture migration-gemma2-config --offline
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

`make_micro_trunk()` also requires the GPT-2 tokenizer in the offline all-extras
profile. Its default `gpt2` lookup uses the immutable revision in
`gpt2-tokenizer-files.json`; other caller-supplied tokenizer names keep their
normal resolution. The five pinned tokenizer and configuration files total
2,854,566 bytes. The manifest records their SHA-256, Git blob identities, and
the MIT declaration in the pinned
[model card](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/README.md).
The tokenizer is downloaded into the validation cache and stays out of source,
the wheel, the sdist, and release assets. GPT-2 model weights are not fetched.

The migration acceptance checks also need three pinned public subjects. The
[`Skywork-Reward-Llama-3.1-8B`](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B)
fixture fetches only shard 4, 117,473,824 bytes, which contains the published
`score.weight` vector. The
[`Skywork-Reward-V2-Qwen3-0.6B`](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)
fixture fetches the complete 1,192,137,232-byte checkpoint and eight small
configuration and tokenizer files. The
[`GRM-Gemma2-2B-rewardmodel-ft`](https://huggingface.co/Ray2333/GRM-gemma2-2B-rewardmodel-ft)
fixture fetches its 1,071-byte config and no weights. Their manifests pin full
revisions, sizes, SHA-256, and Git or LFS identities. The model cards declare
Apache-2.0 for Qwen3 and Gemma-2; the Llama model card links the Skywork
Community License. These are CI-only downloads. No weights or model config are
distributed with reward-lens.

CI verifies these three fixtures before running acceptance with
`REWARD_LENS_REQUIRE_REAL_SUBJECTS=1` and offline model loading. The real Llama
head, Qwen3 forward and adapter comparison, Gemma-2 config policy, and family
inventory must then execute. A developer without those optional local cache
files still sees the original explicit skip reason; absence cannot produce a
passing release CI job.
