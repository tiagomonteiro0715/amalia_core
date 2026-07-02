import sys
from pathlib import Path
import torch

from config import AmaliaConfig
from architecture import AmaliaForCausalLM


def main():
    config = AmaliaConfig()
    model = AmaliaForCausalLM(config)

    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    logits = model(input_ids)

    print(logits.shape)


if __name__ == "__main__":
    main()
