import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """Pre-LN Transformer block with fused QKV + SDPA and a 4x GELU MLP."""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0, "n_embd must be divisible by n_head"
        self.h = n_head
        self.hd = d_model // n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        y = self.ln1(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)  # (B,T,C) each
        def shp(t):
            return t.view(B, T, self.h, self.hd).transpose(1, 2)  # (B,h,T,hd)
        q, k, v = map(shp, (q, k, v))
        # SDPA handles causal masking
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True
        )  # (B,h,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,C)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x

class GPT(nn.Module):
    """Minimal GPT: token + learned pos emb, N blocks, weight tying."""
    def __init__(self, vocab_size: int, block_size: int,
                 n_layer: int, n_head: int, n_embd: int,
                 dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, dropout) for _ in range(n_layer))
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok.weight

        # init
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            if targets is not None:
                targets = targets[:, -self.block_size:]
            T = self.block_size
        tok = self.tok(idx)  # (B,T,C)
        pos = self.pos(torch.arange(T, device=idx.device))  # (T,C)
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.lnf(x)
        logits = self.head(x)  # (B,T,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, steps: int = 200, temp: float = 1.0, top_k: int = None):
        """Autoregressive sampling with optional temperature and top-k."""
        for _ in range(steps):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-8, temp)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
