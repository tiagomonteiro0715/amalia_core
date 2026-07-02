# AMALIA

AMALIA is a decoder-only transformer architecture inherited from EuroLLM-9B, implemented in plain PyTorch (no external attention kernels).

## Local usage (with uv)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone the repo and run the example:

```bash
git clone https://github.com/tiagomonteiro0715/amalia-core.git
cd amalia-core
uv sync
uv run main.py
```

`uv sync` installs the dependencies from `pyproject.toml`/`uv.lock` into a local `.venv`, and `uv run` executes inside it without needing to activate it manually.

## Usage in Google Colab

```python
!pip install uv
!uv pip install --system amalia

import torch
from amalia import AmaliaConfig, AmaliaForCausalLM

# Initialize the model with random weights
config = AmaliaConfig()
model = AmaliaForCausalLM(config)

# Run a forward pass on random token ids
input_ids = torch.randint(0, config.vocab_size, (1, 16))
logits = model(input_ids)

print(logits.shape)  # torch.Size([1, 16, 128000])
```

## Project structure

```
amalia/
├── amalia/               # the installable library
│   ├── __init__.py       # public API surface
│   ├── config.py         # hyperparameters
│   └── architecture.py   # the model itself
├── main.py                # runnable usage example
├── pyproject.toml         # package metadata + build config
├── release.bat             # version bump / GitHub tag / PyPI publish script
└── LICENSE
```

The library is split into a `config` module and an `architecture` module rather than one file because the two change for different reasons: hyperparameters get tuned/swapped constantly, while the model code (the actual math) is comparatively stable. Keeping them separate means you can hand `AmaliaConfig` around (e.g. save it as JSON alongside a checkpoint) without dragging PyTorch `nn.Module` code with it.

### `amalia/config.py`

- **`AmaliaConfig`** — a plain `@dataclass` holding every architecture hyperparameter (vocab size, hidden size, number of layers/heads, RoPE θ, etc.), with sensible defaults matching the AMALIA spec. It's a dataclass instead of a bigger config framework (like HuggingFace's `PretrainedConfig`) because there's no need for serialization/versioning machinery yet — a dataclass gives typed fields, a free `__init__`, and a readable `repr()` for zero extra code.
  - `head_dim` is a `@property` (`hidden_size // num_attention_heads`) instead of a stored field, because it's fully determined by the other two — storing it separately would let it silently drift out of sync if someone edited `hidden_size` after construction.

### `amalia/architecture.py`

Built bottom-up: small standalone pieces first, composed into progressively bigger `nn.Module`s. Each class exists because it's a distinct, reusable computation with its own shape/semantics — splitting them keeps each one testable and readable in isolation instead of one monolithic `forward`.

- **`RMSNorm`** — root-mean-square layer normalization (like LayerNorm but without re-centering on the mean). It's its own class because it's used twice per decoder layer (before attention, before the MLP) plus once at the very end of the model — defining it once avoids repeating the same three lines everywhere.
- **`RotaryEmbedding`** — precomputes the inverse-frequency table used by RoPE (Rotary Position Embeddings) from `head_dim` and `rope_theta`, then produces `cos`/`sin` tensors for a given sequence length. It's a module (not a plain function) so `inv_freq` can live as a registered buffer — it moves with the model when you call `.to(device)`, but isn't a trainable parameter.
- **`rotate_half`** / **`apply_rotary_pos_emb`** — free functions, not methods, because they're pure tensor math with no state, applied identically to both queries and keys. Keeping them as standalone functions (rather than duplicating the logic inside `GroupedQueryAttention`) makes the RoPE math easy to unit-test on its own.
- **`repeat_kv`** — expands the key/value heads so they broadcast against the (larger) number of query heads. It's a separate function because Grouped Query Attention (GQA) is the one part of the attention block that isn't "standard" multi-head attention, so isolating it makes the GQA-specific logic obvious at a glance.
- **`GroupedQueryAttention`** — the attention block: projects `x` into Q/K/V (Q has more heads than K/V, per GQA), applies RoPE, repeats K/V to match Q's head count, then calls PyTorch's built-in `F.scaled_dot_product_attention` with `is_causal=True`. This is the "no external flash-attention library" requirement in practice — `scaled_dot_product_attention` ships in core PyTorch and picks an efficient fused kernel under the hood, without adding a dependency like `flash-attn`.
- **`SwiGLU`** — the feed-forward block: `down_proj(silu(gate_proj(x)) * up_proj(x))`, the gated activation used by Llama-family models (in place of a plain ReLU/GELU MLP) because it consistently improves quality for the same parameter budget.
- **`DecoderLayer`** — one transformer block: pre-norm residual attention, then pre-norm residual MLP. "Pre-norm" (normalize *before* the sub-layer, not after) is what makes it feasible to stack 42 of these without training instability.
- **`AmaliaModel`** — the backbone: token embedding → 42 stacked `DecoderLayer`s → final `RMSNorm`. It stops at hidden states and deliberately has no output head, so the same backbone could in principle back other heads later (e.g. a classification head) without change.
- **`AmaliaForCausalLM`** — wraps `AmaliaModel` with an `lm_head` (a `Linear` projecting hidden states to vocabulary logits) and is the class you actually instantiate for language modeling. The head is a separate `Linear`, not tied to the embedding weights, because the spec calls for untied embeddings (`tie_word_embeddings = False`).

The forward pass, end to end (`AmaliaForCausalLM.forward`): embed token ids → compute RoPE `cos`/`sin` once for the sequence length → run through all 42 decoder layers → final norm → project to vocab logits. There's no KV-cache and no generation loop — this module only defines the architecture's forward pass, not an inference/serving stack, which is intentionally out of scope for what was asked.

### `amalia/__init__.py`

Re-exports `AmaliaConfig` and `AmaliaForCausalLM` (via `__all__`) so callers can do `from amalia import AmaliaConfig, AmaliaForCausalLM` instead of reaching into the submodules directly. This is the package's public API — everything else (`RMSNorm`, `GroupedQueryAttention`, etc.) is still importable, but isn't part of the "supported" surface.

### `main.py`

A minimal runnable example: build a config, build the model with random weights, run one forward pass on random token ids, print the output shape. It exists purely to prove the wiring works end to end (`uv run main.py`) and to double as copy-paste-able usage code.
