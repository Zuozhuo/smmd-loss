# SMMD (Smooth MMD) Loss

This package provides an implementation of **SMMD (Smooth Maximum Mean Discrepancy)**, an auxiliary regularization loss designed for language models when the prediction target is a **numeric token**.

At a high level, for positions where the target token belongs to a numeric sub-vocabulary $V_{\text{num}}$, SMMD builds a probability distribution over $V_{\text{num}}$ from model logits and penalizes it with:
- an **MMD alignment term** $r^\top K r$, and
- an optional **graph smoothness term** $r^\top L r$,

where $r = p - q$, $K$ is a kernel over numeric values, and $L = D - K$ is the graph Laplacian.

---

## Requirements

- `torch`
- `transformers`

## Installation / Import


Place the folder structure like:

```

your_project/
train.py
smmd/
__init__.py
digit_tokens.py
kernels.py
loss.py

````



Then in your code:

```python
from smmd import SMMDLoss
````

---

## Minimal Smoke Test

This snippet:

1. loads a tokenizer,
2. creates random logits `x` and synthetic labels `y`,
3. computes SMMD regularization.

```python
import torch
from transformers import AutoTokenizer
from smmd import SMMDLoss

# Replace with your own model path
model_path = <MODEL_PATH>
tokenizer = AutoTokenizer.from_pretrained(model_path)

BS, L = 2, 6
vocab = tokenizer.vocab_size

# Optional: fix seed for reproducibility
seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

x = torch.randn(BS, L, vocab, requires_grad=True, device="cuda")

# Simulate labels containing numeric and non-numeric strings
y_label = ["123", "9", "abc", "10.5"]
y_tokens = []
for label in y_label:
    tids = tokenizer.encode(label, add_special_tokens=False)
    y_tokens.extend(tids)

# Pad/truncate to match BS*L
needed = BS * L
y_tokens = (y_tokens * (needed // len(y_tokens) + 1))[:needed]
y = torch.tensor(y_tokens, device="cuda").view(BS, L)

smmd = SMMDLoss(
    tokenizer=tokenizer,
    loss_mode="smmd",          # "smmd" | "mmd" | "smooth"
    kernel_mode="value_distance",
    kernel_type="gaussian",
    sigmas=(2.0,),
).to(x.device)  # IMPORTANT: move loss buffers to the same device as logits

reg = smmd(x, y)  # logits: [B,T,V], targets: [B,T]
print("SMMD reg:", reg.item())

# Optional: verify gradients flow
reg.backward()
print("grad norm:", x.grad.norm().item())
```

Expected behavior:

* Positions whose target token is **not** in `V_num` will be ignored (contribute 0).
* The returned value is a scalar mean over numeric-target positions in the batch.

---

## Notes:

This code release focuses on the core contribution of the paper: the SMMD (Smooth MMD) auxiliary loss. It provides a self-contained implementation that can be plugged into any next-token LM training loop. Due to time and dependency constraints, we do not include the full training pipeline or dataset-specific preprocessing scripts. A minimal smoke test is provided to verify correctness and gradient flow on synthetic logits and tokenizer-derived targets.