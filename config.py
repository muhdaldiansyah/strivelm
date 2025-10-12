from dataclasses import dataclass

@dataclass
class Config:
    # data & training
    batch_size: int = 32
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    dropout: float = 0.0
    max_iters: int = 500
    eval_interval: int = 100
    lr: float = 3e-4

    # runtime
    seed: int = 1337
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    ckpt_path: str = "checkpoints/out.pt"
    dataset: str = "input.txt"
