# AMALIA

AMALIA is a decoder-only transformer architecture inherited from EuroLLM-9B, implemented in plain PyTorch (no external attention kernels).

## Usage in Google Colab

```python
# Clone the repo and install dependencies
!git clone https://github.com/<your-username>/amalia.git
%cd amalia
!pip install torch

import sys
sys.path.insert(0, "amalia-core")

import torch
from config import AmaliaConfig
from architecture import AmaliaForCausalLM

# Initialize the model with random weights
config = AmaliaConfig()
model = AmaliaForCausalLM(config)

# Run a forward pass on random token ids
input_ids = torch.randint(0, config.vocab_size, (1, 16))
logits = model(input_ids)

print(logits.shape)  # torch.Size([1, 16, 128000])
```
