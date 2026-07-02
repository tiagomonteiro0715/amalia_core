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
from amalia_core import AmaliaConfig, AmaliaForCausalLM

# Initialize the model with random weights
config = AmaliaConfig()
model = AmaliaForCausalLM(config)

# Run a forward pass on random token ids
input_ids = torch.randint(0, config.vocab_size, (1, 16))
logits = model(input_ids)

print(logits.shape)  # torch.Size([1, 16, 128000])
```
