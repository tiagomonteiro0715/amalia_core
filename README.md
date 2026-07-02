# AMALIA

<p align="center">
  <img src="https://img.shields.io/pypi/v/amalia" alt="PyPI"/>
  <img src="https://img.shields.io/github/license/tiagomonteiro0715/amalia-core" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+"/>
  <img src="https://img.shields.io/github/last-commit/tiagomonteiro0715/amalia-core" alt="Last Commit"/>
  <img src="https://img.shields.io/github/stars/tiagomonteiro0715/amalia-core" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/tiagomonteiro0715/amalia-core" alt="Forks"/>
</p>

In 2026, the Portuguese government announced **AMALIA**, a sovereign large language model for European Portuguese, described in the paper *"AMALIA: A Fully Open Large Language Model for European Portuguese"* (PROPOR 2026). AMALIA builds directly on EuroLLM-9B, with two targeted architectural changes: the context window is extended from 4K to 32K tokens, and RoPE's base frequency (θ) is increased from 10,000 to 1,000,000.

This repository is an independent, open-source implementation of that architecture in plain PyTorch (no external attention kernels) — decoder-only, Grouped Query Attention, SwiGLU, RoPE.

> **Scope & disclaimer:** this repository implements the *architecture only* — the model definition and a forward pass. It ships with randomly-initialized weights, not a trained checkpoint, and does not include a training pipeline (see [Paper](#paper) for why). It has no official affiliation with the Portuguese government or the paper's authors.

## Table of contents

- [Local usage (with uv)](#local-usage-with-uv)
- [Usage in Google Colab](#usage-in-google-colab)
- [Performance](#performance)
- [Project structure](#project-structure)
- [Paper](#paper)
- [Built with](#built-with)
- [Related projects](#related-projects)
- [Contact](#contact)
- [License](#license)

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

Try it directly in your browser, no local setup required: [Open in Colab](https://colab.research.google.com/drive/18A-UVi2Ci2oWOBYINYLowC7y3GQJqDPg?usp=sharing)

```python
!pip install uv
!uv pip install --system amalia

import torch
from amalia import AmaliaConfig, AmaliaForCausalLM

# Initialize the model with random weights (bf16 by default)
config = AmaliaConfig()
model = AmaliaForCausalLM(config).to(config.dtype)

# Move to GPU and compile if one is available (e.g. a Colab GPU runtime)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
if device == "cuda":
    model = torch.compile(model)

print(model)

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}")

# Run a forward pass on random token ids
input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
logits = model(input_ids)

print(logits.shape)  # torch.Size([1, 16, 128000])
```

## Performance

- **bf16 by default** — `AmaliaConfig.dtype` defaults to `torch.bfloat16`, matching how Llama/EuroLLM-family checkpoints actually ship. It halves memory/bandwidth versus fp32, needs no loss scaling (unlike fp16), and runs fine even without hardware acceleration (CPU, older GPUs) — so it's a safe default everywhere, not just on modern GPUs. `RMSNorm` internally upcasts to fp32 for its variance computation before casting back, since bf16 is too imprecise (~3 decimal digits) for that reduction directly.
- **`torch.compile` on CUDA** — fuses ops via PyTorch's Inductor backend for a further speedup, with no architecture changes needed (the forward pass has no data-dependent control flow to trip up the compiler). It's applied at the call site (`torch.compile(model)`), gated on `torch.cuda.is_available()`, because Inductor/Triton support for CPU-only or Windows setups is inconsistent — so local CPU development stays fast and reliable, while a CUDA runtime (like a Colab GPU) actually gets compiled.

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

- **`RMSNorm`** — root-mean-square layer normalization (like LayerNorm but without re-centering on the mean). It's its own class because it's used twice per decoder layer (before attention, before the MLP) plus once at the very end of the model — defining it once avoids repeating the same three lines everywhere. It upcasts to `float32` internally for the variance computation (then casts back to the input dtype) so it stays numerically stable when the rest of the model runs in bf16.
- **`RotaryEmbedding`** — precomputes the inverse-frequency table used by RoPE (Rotary Position Embeddings) from `head_dim` and `rope_theta`, then produces `cos`/`sin` tensors for a given sequence length. It's a module (not a plain function) so `inv_freq` can live as a registered buffer — it moves with the model when you call `.to(device)`, but isn't a trainable parameter.
- **`rotate_half`** / **`apply_rotary_pos_emb`** — free functions, not methods, because they're pure tensor math with no state, applied identically to both queries and keys. Keeping them as standalone functions (rather than duplicating the logic inside `GroupedQueryAttention`) makes the RoPE math easy to unit-test on its own.
- **`repeat_kv`** — expands the key/value heads so they broadcast against the (larger) number of query heads. It's a separate function because Grouped Query Attention (GQA) is the one part of the attention block that isn't "standard" multi-head attention, so isolating it makes the GQA-specific logic obvious at a glance.
- **`GroupedQueryAttention`** — the attention block: projects `x` into Q/K/V (Q has more heads than K/V, per GQA), applies RoPE, repeats K/V to match Q's head count, then calls PyTorch's built-in `F.scaled_dot_product_attention` with `is_causal=True`. This is the "no external flash-attention library" requirement in practice — `scaled_dot_product_attention` ships in core PyTorch and picks an efficient fused kernel under the hood, without adding a dependency like `flash-attn`.
- **`SwiGLU`** — the feed-forward block: `down_proj(silu(gate_proj(x)) * up_proj(x))`, the gated activation used by Llama-family models (in place of a plain ReLU/GELU MLP) because it consistently improves quality for the same parameter budget.
- **`DecoderLayer`** — one transformer block: pre-norm residual attention, then pre-norm residual MLP. "Pre-norm" (normalize *before* the sub-layer, not after) is what makes it feasible to stack 42 of these without training instability.
- **`AmaliaModel`** — the backbone: token embedding → 42 stacked `DecoderLayer`s → final `RMSNorm`. It stops at hidden states and deliberately has no output head, so the same backbone could in principle back other heads later (e.g. a classification head) without change.
- **`AmaliaForCausalLM`** — wraps `AmaliaModel` with an `lm_head` (a `Linear` projecting hidden states to vocabulary logits) and is the class you actually instantiate for language modeling. The head is a separate `Linear`, not tied to the embedding weights, because the spec calls for untied embeddings (`tie_word_embeddings = False`).

The forward pass, end to end (`AmaliaForCausalLM.forward`): embed token ids → compute RoPE `cos`/`sin` once for the sequence length → run through all 42 decoder layers → final norm → project to vocab logits. There's no KV-cache and no generation loop — this module only defines the architecture's forward pass, not an inference/serving stack, which is intentionally out of scope for what was asked.

#### Why `AmaliaForCausalLM` and not just `AmaliaModel`

`AmaliaModel` and `AmaliaForCausalLM` are kept separate — instead of putting `lm_head` straight into `AmaliaModel` and using only one class — for a few concrete reasons:

- **`AmaliaModel`'s output isn't usable on its own.** It stops at hidden states, shape `[batch, seq, hidden_size=4096]`. Nothing consumes a bag of 4096-dim vectors directly — you always need a task-specific head to turn them into something useful (next-token logits, a classification score, an embedding for retrieval, etc.). `AmaliaForCausalLM` is what adds the *causal language modeling* head — `lm_head`, a `Linear(4096 → 128000)` — that turns hidden states into next-token logits.
- **The backbone is reusable, the head isn't.** This is the standard split used by every Llama-family implementation (and HuggingFace's `*Model` vs `*ForCausalLM`/`*ForSequenceClassification` convention): the 42 decoder layers, embeddings, and final norm are the expensive, task-agnostic part. If AMALIA were later reused for something other than language modeling — e.g. a classifier head, an embedding model — you'd build a new thin wrapper (`AmaliaForSequenceClassification`, say) around the *same* `AmaliaModel`, instead of duplicating all 42 layers or hacking a second head into one class.
- **Untied embeddings need a separate `Linear`.** Because the spec sets `tie_word_embeddings = False`, `lm_head` is its own `nn.Linear` with its own weights (not a matmul against `embed_tokens.weight`). That's naturally a "head on top of the backbone" concern, not something `AmaliaModel` itself should own.

In short: `AmaliaModel` = the transformer backbone (what's shared), `AmaliaForCausalLM` = backbone + task head (what you actually run). You always instantiate `AmaliaForCausalLM` in practice; `AmaliaModel` exists as the reusable piece underneath it, not as something meant to be used standalone today.

### `amalia/__init__.py`

Re-exports `AmaliaConfig` and `AmaliaForCausalLM` (via `__all__`) so callers can do `from amalia import AmaliaConfig, AmaliaForCausalLM` instead of reaching into the submodules directly. This is the package's public API — everything else (`RMSNorm`, `GroupedQueryAttention`, etc.) is still importable, but isn't part of the "supported" surface.

### `main.py`

A minimal runnable example: build a config, build the model with random weights in bf16, move it to GPU and `torch.compile` it if CUDA is available, run one forward pass on random token ids, print the output shape. It exists purely to prove the wiring works end to end (`uv run main.py`) and to double as copy-paste-able usage code.

## Paper

This implementation is based on:

> Simplício, A., Vinagre, G., Ramos, M. M., et al. (2026). **AMALIA: A Fully Open Large Language Model for European Portuguese**. In *Proceedings of the 17th International Conference on Computational Processing of Portuguese (PROPOR 2026)*, Salvador, Brazil, pp. 380–391. [ACL Anthology](https://aclanthology.org/2026.propor-1.38/)

```bibtex
@inproceedings{simplicio-etal-2026-amalia,
    title     = "{AMALIA}: A Fully Open Large Language Model for {E}uropean {P}ortuguese",
    author    = "Simplício, Afonso and Vinagre, Gonçalo and Ramos, Miguel Moura and others",
    booktitle = "Proceedings of the 17th International Conference on Computational Processing of {P}ortuguese ({PROPOR} 2026)",
    year      = "2026",
    address   = "Salvador, Brazil",
    pages     = "380--391",
    publisher = "Association for Computational Linguistics"
}
```

The paper's Section 8 (future work) mentions the authors' plans for RL with verifiable rewards and further context extension — a training pipeline is intentionally out of scope for this repository.

## Built with

- [PyTorch](https://pytorch.org) — the entire architecture, no other ML framework or external attention kernel
- [uv](https://github.com/astral-sh/uv) — dependency management, packaging, and the local dev workflow

## Related projects

- [Pessoa](https://github.com/tiagomonteiro0715/pessoa) — local, LLM-agnostic AI agent infrastructure with a Portuguese persona
- [The Math Behind Artificial Intelligence](https://github.com/tiagomonteiro0715/The-Math-Behind-Artificial-Intelligence-A-Guide-to-AI-Foundations) — a guide to AI's mathematical foundations
- [FreeCodeCamp author profile](https://www.freecodecamp.org/news/author/tiagomonteiro) — articles and deep dives on AI and programming

## Contact

**Tiago Monteiro**

- Email: monteiro.t@northeastern.edu
- GitHub: [@tiagomonteiro0715](https://github.com/tiagomonteiro0715)
- FreeCodeCamp: [Author profile](https://www.freecodecamp.org/news/author/tiagomonteiro)

## License

MIT License. See [LICENSE](LICENSE).

---

<p align="center">
  If this project was useful or interesting to you, please star the repo.<br>
  It helps others find it.
</p>
