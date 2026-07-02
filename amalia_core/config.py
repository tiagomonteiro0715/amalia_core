from dataclasses import dataclass


@dataclass
class AmaliaConfig:
    vocab_size: int = 128_000
    hidden_size: int = 4096
    intermediate_size: int = 12_288
    num_hidden_layers: int = 42
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32_768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
