import torch, os

def load_text(path: str) -> str:
    """Load dataset as plain text; create a tiny default if missing."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("To be, or not to be, that is the question.\n")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_vocab(text: str):
    """Return (chars, encode, decode)."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    def encode(s: str) -> torch.Tensor:
        return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    def decode(t: torch.Tensor) -> str:
        return "".join(itos[int(i)] for i in t)
    return chars, encode, decode

def split_data(t: torch.Tensor, ratio: float = 0.9):
    """Train/val split; ensure val non-empty, else mirror train for simplicity."""
    n = max(1, int(ratio * len(t)))
    if len(t) - n > 0:
        return t[:n], t[n:]
    # fallback: use train as val if data too small
    return t[:], t[:]

def get_batch(src: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Return a batch (x, y) on device. Works even for very short texts."""
    # clamp block_size to actual data length
    effective_block_size = min(block_size, len(src) - 1)
    if effective_block_size < 1:
        effective_block_size = 1

    # number of possible starting indices
    L = len(src) - effective_block_size
    if L < 1:
        L = 1
        ix = torch.zeros(batch_size, dtype=torch.long)
    else:
        ix = torch.randint(0, L, (batch_size,))

    x = torch.stack([src[i:i + effective_block_size] for i in ix])
    y = torch.stack([src[i + 1:i + effective_block_size + 1] for i in ix])
    return x.to(device), y.to(device)
