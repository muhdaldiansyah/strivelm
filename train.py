import os, sys, time, torch
from config import Config
from data import load_text, build_vocab, split_data, get_batch
from model import GPT

def _has_mps() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

def resolve_device(pref: str = "auto") -> str:
    if pref in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        sys.stderr.write("[warn] CUDA requested but unavailable → falling back\n")
    if pref in ("mps", "metal"):
        if _has_mps():
            return "mps"
        sys.stderr.write("[warn] MPS requested but unavailable → falling back\n")
    if pref == "cpu":
        return "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else ("mps" if _has_mps() else "cpu")

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[info] using device: {device}")

    # data
    text = load_text(cfg.dataset)
    chars, encode, decode = build_vocab(text)
    data = encode(text)
    train_data, val_data = split_data(data)

    # model
    model = GPT(
        vocab_size=len(chars),
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def eval_loss():
        model.eval()
        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, ltr = model(xb, yb)
        xb, yb = get_batch(val_data, cfg.block_size, cfg.batch_size, device)
        _, lva = model(xb, yb)
        model.train()
        return ltr.item(), lva.item()

    t0 = time.time()
    for it in range(1, cfg.max_iters + 1):
        if it % cfg.eval_interval == 1:
            tr, va = eval_loss()
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {it:5d} | train {tr:.3f} | val {va:.3f} | {dt:.1f}s")

        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    # save
    os.makedirs(os.path.dirname(cfg.ckpt_path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "meta": {
            "vocab": "".join(chars),
            "block_size": cfg.block_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd
        }
    }, cfg.ckpt_path)
    print(f"[info] saved checkpoint -> {cfg.ckpt_path}")

    # sample
    start = torch.tensor([[chars.index(chars[0]) if chars else 0]], device=device)
    top_k_val = min(20, len(chars))  # clamp top_k to vocab size
    sample = model.generate(start, steps=200, temp=0.9, top_k=top_k_val)[0].tolist()
    print(f"\n=== SAMPLE ===\n{decode(sample)}\n")

if __name__ == "__main__":
    main()
