Это просто эталонный README. Серьезно, он выглядит так, как будто над репозиторием работает целая команда ресерчеров. У тебя отличная структура: сразу понятна суть (без лишней воды), есть архитектурная справка, быстрый старт и даже референс по API. С таким описанием проект не стыдно показать кому угодно — от коллег до потенциальных работодателей.

Раз уж ты упомянул, что потом будешь переписывать на английский, давай я сэкономлю тебе время прямо сейчас. Я перевел твой README на технический английский, сохранив всю структуру, форматирование и профессиональный сленг.

---

### English Version (README.md)

```markdown
# KLT-Experiments

A research project focused on training **quantized GPT models** built on "hypercube" linear layers. Instead of storing dense FP32 weights, each layer parameterizes weights via a compact **latent subspace** and a projection matrix, followed by **quantization** (binary / ternary / 2-bit) using a Straight-Through Estimator (STE).

The goal is to explore how aggressively transformer weights can be compressed during training while maintaining language modeling quality.

---

## Core Idea

A standard linear layer stores a weight matrix `W` of shape `[out_features, in_features]`.
In a hypercube layer, weights are reconstructed on the fly:

1. **Latent coordinates** `weight_latent` of shape `[out_features, groups, latent_dim]` are stored.
2. A learnable **projection matrix** `proj_matrix` of shape `[latent_dim, group_size]` maps the latent representation back into continuous group weights.
3. Weights are centered and normalized (following BitNet conventions).
4. **STE quantization** is applied (binary / ternary / 2-bit).
5. The effective weights are scaled back and used in a standard `F.linear` operation.

For example, with `group_size=8` and `latent_dim=3`, eight weights are compressed into three latent parameters — resulting in a storage compression ratio of ≈ 8/3.

---

## Project Structure

```text
KLT-Experiments/
├── src/
│   ├── layers/
│   │   └── hypercube_linear_layers.py   # STE + hypercube linear layers
│   ├── models/
│   │   └── transformers.py              # CausalSelfAttention, TransformerBlock, HypercubeGPT
│   └── tests/                           # (reserved for unit tests)
├── examples/
│   ├── train_baselines.py               # FP32 vs Ternary Hypercube comparison
│   └── train_quantized.py               # Quantized model training + generation
├── docs/                                # (reserved for documentation)
├── notebooks/                           # (reserved for experiments/EDA)
├── requirements.txt
└── README.md

```

---

## Architecture

### Layers (`src/layers/hypercube_linear_layers.py`)

**Straight-Through Estimators** — `torch.autograd.Function` implementations that quantize tensors on the forward pass and pass gradients unchanged on the backward pass:

| Class | Quantization Levels | Forward Pass Formula |
| --- | --- | --- |
| `BinarySTE` | `{-1, 1}` | `sign(x) + (x == 0)` |
| `TernarySTE` | `{-1, 0, 1}` | `round(clamp(x, -1, 1))` |
| `TwoBitSTE` | `{-2, -1, 0, 1}` | `round(clamp(x, -2, 1))` (b1.58-style) |

**`BaseHypercubeLinear`** — the base layer. Constructor:

```python
BaseHypercubeLinear(
    in_features, out_features,
    group_size,        # size of the weight group
    latent_dim,        # dimensionality of the latent subspace
    bias=True,
)

```

Features:

* Automatic input padding to ensure divisibility by `group_size`.
* Weights are stored as `weight_latent [out, groups, latent_dim]` and `proj_matrix [latent_dim, group_size]`.
* BitNet-style normalization (mean centering + scaling by mean absolute value).
* Quantization logic is defined by the subclass via the `quantize()` method.

**Concrete Implementations** (`src/layers/hypercube_linear_layers.py:123`):

| Class | group_size | latent_dim | Compression |
| --- | --- | --- | --- |
| `BinaryHypercubeLinear` | 8 | 3 | 8 weights → 3 |
| `TernaryHypercubeLinear` | 5 | 3 | 5 weights → 3 |
| `TwoBitHypercubeLinear` | 5 | 3 | 5 weights → 3 |

### Model (`src/models/transformers.py`)

* **`CausalSelfAttention`** — Multi-Head Self-Attention with a causal mask; all Q/K/V/O projections use hypercube layers.
* **`TransformerBlock`** — Pre-LN block with residual connections and FFN (×4 expansion, GELU).
* **`HypercubeGPT`** — Decoder-only LM:
* Token + positional embeddings (`nn.Embedding`),
* Stack of `TransformerBlock`s,
* Final `LayerNorm` and a quantized `lm_head`,
* Built-in Cross-Entropy loss calculation when `targets` are provided,
* `generate()` method for autoregressive generation.



The linear layer type and its parameters (`group_size`, `latent_dim`) are passed via `linear_cls` and `linear_kwargs`, allowing for seamless switching between quantization modes.

---

## Installation

Requires Python 3.8+ and PyTorch.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install torch

```

> Note: `requirements.txt` is intentionally empty — please install `torch` manually according to your platform and CUDA setup. See the official PyTorch docs: https://pytorch.org/get-started/locally/

---

## Quick Start

The **TinyShakespeare** dataset is downloaded automatically on the first run from `examples/train_*.py`.

### 1. Training a Quantized Model + Text Generation

```bash
python examples/train_quantized.py

```

Inside the script, you can select the layer type and group size:

```python
LayerClass = TernaryHypercubeLinear  # or Binary / TwoBit
group_size = 5

```

The script trains the model, logs Train/Val loss, and prints a generated text sample.

### 2. FP32 Baseline Comparison

```bash
python examples/train_baselines.py

```

This sequentially trains a dense `nn.Linear` (FP32) model and a `TernaryHypercubeLinear` model, outputting a final Validation Loss comparison table.

---

## Training Configuration (Default in `examples/`)

| Parameter | Value |
| --- | --- |
| `block_size` | 128 |
| `batch_size` | 64 |
| `embed_dim` | 128 |
| `heads` | 4 |
| `depth` | 4 |
| `latent_dim` | 3 |
| `learning_rate` | 1e-3 |
| Optimizer | AdamW |

Device allocation (`cuda`/`cpu`) is handled automatically.

---

## API Reference

```python
from src.layers import BinaryHypercubeLinear, TernaryHypercubeLinear, TwoBitHypercubeLinear
from src.models import HypercubeGPT

model = HypercubeGPT(
    vocab_size=65,
    block_size=128,
    embed_dim=128,
    heads=4,
    depth=4,
    linear_cls=TernaryHypercubeLinear,   # quantized layer type
    group_size=5,                        # weight group size
    latent_dim=3,                        # latent subspace dimensionality
)

logits, loss = model(x, targets=y)        # forward pass + loss
sample = model.generate(context, max_new_tokens=200, temperature=1.0)

```

---

## Extension & Customization

* **New Quantization Types** — Inherit from `BaseHypercubeLinear`, implement the `quantize()` method, and add your custom `torch.autograd.Function` estimator.
* **Custom Experiments** — Use `examples/train_quantized.py` as a template; modify `LayerClass`, `group_size`, `latent_dim`, and model hyperparameters.
* **Testing** — The `src/tests/` directory is reserved for unit tests (tensor shapes, STE gradients, padding checks, etc.).

---

## Dependencies

* `torch` (PyTorch) — the main and only runtime dependency.
* TinyShakespeare dataset is downloaded via `urllib` on the first run.

```