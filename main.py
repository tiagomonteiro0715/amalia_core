import torch

from amalia import AmaliaConfig, AmaliaForCausalLM


def main():
    config = AmaliaConfig()
    model = AmaliaForCausalLM(config)

    print(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    logits = model(input_ids)

    print(logits.shape)


if __name__ == "__main__":
    main()
