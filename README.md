# AMALIA

<p align="center">
  <img src="https://img.shields.io/pypi/v/amalia" alt="PyPI"/>
  <img src="https://img.shields.io/github/license/tiagomonteiro0715/amalia-core" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+"/>
  <img src="https://img.shields.io/github/last-commit/tiagomonteiro0715/amalia-core" alt="Last Commit"/>
  <img src="https://img.shields.io/github/stars/tiagomonteiro0715/amalia-core" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/tiagomonteiro0715/amalia-core" alt="Forks"/>
</p>

In 2026, the Portuguese government announced **AMALIA**, a sovereign large language model for European Portuguese, described in the paper *"AMALIA: A Fully Open Large Language Model for European Portuguese"* (PROPOR 2026). 

AMALIA builds directly on EuroLLM-9B, with two architectural changes: the context window is extended from 4K to 32K tokens, and RoPE's base frequency (θ) is increased from 10,000 to 1,000,000.

This repository is an independent, open-source implementation of that architecture in PyTorch.

> **Scope & disclaimer:** this repository implements the *architecture only* - the model definition and a forward pass. It ships with randomly-initialized weights, not a trained checkpoint, and does not include a training pipeline (see [Paper](#paper) for why). It has no official affiliation with the Portuguese government or the paper's authors.

## Table of contents

- [Local usage (with uv)](#local-usage-with-uv)
- [Usage in Google Colab](#usage-in-google-colab)
- [Performance](#performance)
- [Project structure](#project-structure)
- [Paper](#paper)
- [Verification](#verification)
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

- **bf16 by default** - `AmaliaConfig.dtype` defaults to `torch.bfloat16`, matching how Llama/EuroLLM-family checkpoints ship. It halves memory/bandwidth versus fp32, needs no loss scaling (unlike fp16), and runs fine even without hardware acceleration (CPU, older GPUs).
- **`torch.compile` on CUDA** - fuses ops via PyTorch's Inductor backend for a further speedup, with no architecture changes needed. It's applied at the call site (`torch.compile(model)`), gated on `torch.cuda.is_available()`, because Inductor/Triton support for CPU-only or Windows setups is inconsistent - so local CPU development stays fast and reliable, while a CUDA runtime (like a Colab GPU) actually gets compiled.

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

The library is split into a `config` module and an `architecture` module rather than one file because the two change for different reasons: hyperparameters get tuned/swapped constantly, while the model code (the actual math) is comparatively stable. 

Keeping them separate means you can hand `AmaliaConfig` around (e.g. save it as JSON alongside a checkpoint) without dragging PyTorch `nn.Module` code with it.

### `amalia/config.py`

- **`AmaliaConfig`** - a plain `@dataclass` holding every architecture hyperparameter with  defaults matching the AMALIA spec. It's a dataclass instead of a bigger config framework (like HuggingFace's `PretrainedConfig`) because there's no need for serialization/versioning machinery yet.

  - `head_dim` is a `@property` (`hidden_size // num_attention_heads`) instead of a stored field, because it's fully determined by the other two. Storing it separately would let it drift out of sync if someone edited `hidden_size` after construction.

### `amalia/architecture.py`

The system was build by creating small, modular pieces of code and then combine them to build larger modules.

- **`RMSNorm`** - root-mean-square layer normalization. It's its own class because it's used twice per decoder layer (before attention, before the MLP) plus once at the very end of the model. It upcasts to `float32` internally for the variance computation (then casts back to the input dtype) so it stays numerically stable when the rest of the model runs in bf16.
- **`RotaryEmbedding`** - precomputes the inverse-frequency table used by RoPE (Rotary Position Embeddings) from `head_dim` and `rope_theta`, then produces `cos`/`sin` tensors for a given sequence length. It's a module (not a plain function) so `inv_freq` can live as a registered buffer.
- **`rotate_half`** / **`apply_rotary_pos_emb`** - free functions, not methods, because they're pure tensor math with no state, applied identically to both queries and keys.
- **`repeat_kv`** - expands the key/value heads so they broadcast against the (larger) number of query heads. It's a separate function because Grouped Query Attention (GQA) is the one part of the attention block that isn't "standard" multi-head attention.
- **`GroupedQueryAttention`** - the attention block: projects `x` into Q/K/V (Q has more heads than K/V, per GQA), applies RoPE, repeats K/V to match Q's head count, then calls PyTorch's built-in `F.scaled_dot_product_attention` with `is_causal=True`. This is the "no external flash-attention library" requirement in practice. `scaled_dot_product_attention` ships in core PyTorch and picks an efficient fused kernel under the hood, without adding a dependency like `flash-attn`.
- **`SwiGLU`** - the feed-forward block: `down_proj(silu(gate_proj(x)) * up_proj(x))`, the gated activation used by Llama-family models because it consistently improves quality for the same parameter budget.
- **`DecoderLayer`** - one transformer block: pre-norm residual attention, then pre-norm residual MLP. "Pre-norm" (normalize *before* the sub-layer) is what makes it feasible to stack 42 of these without training instability.
- **`AmaliaModel`** - The unification of the token embedding with the decoding layer and RMS Norm.
- - **`AmaliaForCausalLM`** - wraps `AmaliaModel` with an `lm_head` (a `Linear` projecting hidden states to vocabulary logits) and is the class you actually inicialize for language modeling. The head is a separate `Linear`, not tied to the embedding weights, because the spec calls for untied embeddings (`tie_word_embeddings = False`).

#### Why `AmaliaForCausalLM` and not just `AmaliaModel`

`AmaliaModel` and `AmaliaForCausalLM` are kept separate for many reasons. 

The short answer is that AmaliaModel is the raw text-processing brain, while AmaliaForCausalLM is the specific tool that uses that brain to predict the next word.

The long answer is that:

- **`AmaliaModel`'s output isn't usable on its own.** It stops at hidden states, shape `[batch, seq, hidden_size=4096]`. Nothing consumes a bag of 4096-dim vectors directly. `AmaliaForCausalLM` is what adds the *causal language modeling* head  that turns hidden states into next-token logits.
- **The backbone is reusable, the head isn't.** This is the standard split used by HuggingFace's `*Model` vs `*ForCausalLM`/`*ForSequenceClassification` convention: the decoder layers, embeddings, and final norm are the expensive, task-agnostic part. If AMALIA were later reused for something other than language modeling you'd build a new thin wrapper (`AmaliaForSequenceClassification`, say) around the *same* `AmaliaModel`, instead of duplicating the decoder layer.

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

The paper's Section 8 (future work) mentions the authors' plans for RL with verifiable rewards and further context extension - a training pipeline is intentionally out of scope for this repository.

## Verification

Instantiating `AmaliaForCausalLM` with the default `AmaliaConfig` and summing `p.numel() for p in model.parameters()` gives a parameter count that matches the paper's reported numbers:

| Component | This implementation | Paper |
|---|---|---|
| Embedding | 524,288,000 (0.524B) | 0.524B |
| Non-embedding (42 decoder layers + final norm) | 8,103,743,488 (8.104B) | 8.105B |
| LM head | 524,288,000 (0.524B) | 0.524B |
| **Total** | **9,152,319,488 (9.152B)** | **9.153B** |

The ~0.001B differences are rounding in the paper's summary table, not a mismatch — down to the exact parameter, this implementation reproduces AMALIA's stated size.

## Built with

- [PyTorch](https://pytorch.org) - the entire architecture, no other ML framework or external attention kernel
- [uv](https://github.com/astral-sh/uv) - dependency management, packaging, and the local dev workflow

## Related projects

- [Pessoa](https://github.com/tiagomonteiro0715/pessoa) - local, LLM-agnostic AI agent infrastructure with a Portuguese persona
- [The Math Behind Artificial Intelligence](https://github.com/tiagomonteiro0715/The-Math-Behind-Artificial-Intelligence-A-Guide-to-AI-Foundations) - a guide to AI's mathematical foundations
- [FreeCodeCamp author profile](https://www.freecodecamp.org/news/author/tiagomonteiro) - articles and deep dives on AI and programming

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
