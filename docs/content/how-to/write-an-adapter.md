# Write an adapter for your model

You have a reward model from a family `reward-lens` does not ship an adapter for, and you want the tools to work on it.

## First, just try to load it

The generic adapter auto-detects most `AutoModelForSequenceClassification` models with a linear reward head. Try before you write anything:

```python
from reward_lens import RewardModel

rm = RewardModel.from_pretrained("your-org/your-reward-model")
print(rm.score("Explain photosynthesis.", "Plants convert light into chemical energy."))
```

If that returns a float, you are done: no adapter needed. Write one only when the auto-detector cannot find the reward head or cannot navigate the layers.

## The adapter is one class

An adapter tells `reward-lens` how to walk a specific model's module tree. Subclass `ModelAdapter` and implement its abstract methods. The one that matters most is `get_reward_head_params`, which hands back the reward direction \(w_r\) and its bias; everything else is a short accessor into the module tree.

```python
import torch
import torch.nn as nn
from typing import Any, Optional
from reward_lens.model_adapters import ModelAdapter

class MyModelAdapter(ModelAdapter):
    """Adapter for the MyModel reward-model family."""

    # The one method every tool depends on: the reward direction w_r and its bias.
    def get_reward_head_params(self, model: nn.Module) -> tuple[torch.Tensor, float]:
        head = model.reward_head                       # your linear head, weight (1, d_model)
        w_r = head.weight.data.squeeze().float()       # (d_model,)
        bias = float(head.bias.data.item()) if head.bias is not None else 0.0
        return w_r, bias

    # Where the transformer blocks live, and how many.
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.transformer.blocks
    def n_layers(self, model: nn.Module) -> int:
        return len(model.transformer.blocks)
    def n_heads(self, model: nn.Module) -> int:
        return model.config.num_attention_heads

    # The two sublayers inside one block.
    def get_attn_module(self, layer: nn.Module) -> Optional[nn.Module]:
        return layer.attn
    def get_mlp_module(self, layer: nn.Module) -> Optional[nn.Module]:
        return layer.mlp

    # How this architecture packages each forward output, often a tuple.
    def extract_layer_output(self, output: Any) -> torch.Tensor:
        return output[0] if isinstance(output, tuple) else output
    def extract_attn_output(self, output: Any) -> torch.Tensor:
        return output[0] if isinstance(output, tuple) else output
    def extract_mlp_output(self, output: Any) -> torch.Tensor:
        return output

    # The token embedding, and the scalar reward off the final output.
    def get_embedding(self, model: nn.Module) -> nn.Module:
        return model.transformer.embed_tokens
    def extract_reward(self, output: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return output.logits.squeeze(-1)

    # Optional: expose o_proj to enable per-head patching. Return None to skip.
    def get_attn_o_proj(self, layer: nn.Module) -> Optional[nn.Module]:
        return layer.attn.o_proj
```

Swap the attribute names (`model.transformer.blocks`, `layer.attn`, `model.reward_head`) for wherever your architecture actually keeps them. That is the whole job.

## Wire it into the dispatch

There is no registry. `get_adapter(model, model_name)` is a hardcoded if-chain that matches on the model's class name and config `model_type`. Add a branch for your family, before the generic fallback:

```python
# in reward_lens/model_adapters/__init__.py, inside get_adapter(...)
    if "mymodel" in class_name or model_type == "mymodel":
        return MyModelAdapter()
```

Reinstall (`pip install -e .`) and `RewardModel.from_pretrained("your-org/your-reward-model")` picks it up automatically.

To try it without editing the library, construct `RewardModel` directly with your adapter:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from reward_lens import RewardModel

name = "your-org/your-reward-model"
device = torch.device("cuda")
hf = AutoModelForSequenceClassification.from_pretrained(
    name, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()
tok = AutoTokenizer.from_pretrained(name)

rm = RewardModel(hf, tok, MyModelAdapter(), device=device)
```

!!! note "Per-head analysis is opt-in"
    `get_attn_o_proj` is the one optional method with teeth. Return the attention output projection and per-head tools like `ActivationPatcher.patch_all_heads` light up; return `None`, the default, and analysis stays at the sublayer level. Skip it until you need heads.

Got a working adapter? It is worth contributing back so the next person with that model family gets it for free. See [contributing](../contributing/index.md).
