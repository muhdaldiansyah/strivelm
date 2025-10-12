import argparse, os, sys, torch
from model import GPT

def load_ckpt(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if "model" not in ckpt or "meta" not in ckpt:
        raise KeyError("Invalid checkpoint: missing 'model' or 'meta'")
    req = {"vocab", "block_size", "n_layer", "n_head", "n_embd"}
    if not req.issubset(set(ckpt["meta"].keys())):
        raise KeyError(f"Invalid checkpoint meta, required keys: {sorted(req)}")
    return ckpt

def main():
    ap = argparse.ArgumentParser(description="StriveLM inference")
    ap.add_argument("--ckpt", default="checkpoints/out.pt", help="path to checkpoint")
    ap.add_argument("--start", default="T", help="starting character")
    ap.add_argument("--steps", type=int, default=400, help="number of tokens to generate")
    ap.add_argument("--temp", type=float, default=0.9, help="sampling temperature")
    ap.add_argument("--top_k", type=int, default=50, help="top-k filtering (0=disabled)")
    args = ap.parse_args()

    try:
        ckpt = load_ckpt(args.ckpt)
        meta = ckpt["meta"]
        vocab = list(meta["vocab"])
        stoi = {c: i for i, c in enumerate(vocab)}
        itos = {i: c for i, c in enumerate(vocab)}
        def decode(t: torch.Tensor) -> str:
            return "".join(itos[int(i)] for i in t)

        model = GPT(
            vocab_size=len(vocab),
            block_size=meta["block_size"],
            n_layer=meta["n_layer"],
            n_head=meta["n_head"],
            n_embd=meta["n_embd"]
        ).eval()
        model.load_state_dict(ckpt["model"])

        if args.start not in stoi:
            sys.stderr.write(f"[warn] start char '{args.start}' not in vocab, using '{vocab[0]}'\n")
        start_id = stoi.get(args.start, 0)
        start = torch.tensor([[start_id]], dtype=torch.long)

        # clamp top_k to vocab size
        top_k_val = args.top_k if args.top_k > 0 else None
        if top_k_val and top_k_val > len(vocab):
            top_k_val = len(vocab)
            sys.stderr.write(f"[warn] top_k clamped to vocab size: {len(vocab)}\n")

        out = model.generate(start, steps=args.steps, temp=args.temp, top_k=top_k_val)[0].tolist()
        print(decode(out))
    except Exception as e:
        print(f"[inference error] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
