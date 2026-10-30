import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from ..model.config import FridayConfig
from ..model.model import FridayModel


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    m = raw["model"]
    return FridayConfig(**m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="checkpoints/pretrain.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    import yaml as _y
    raw = _y.safe_load(open(args.config))
    cfg = load_config(args.config)
    model = FridayModel(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    lr = args.lr if args.lr is not None else raw.get("training", {}).get("lr", 3e-4)
    wd = raw.get("training", {}).get("weight_decay", 0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # training loop placeholder: expects tokenized .pt with input_ids
    data = torch.load(args.data)
    ids = data["input_ids"]
    bs = 4
    step = 0
    model.train()
    while step < args.steps:
        for i in range(0, len(ids) - 1, bs):
            x = ids[i:i + bs, :-1].to(device)
            y = ids[i:i + bs, 1:].to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.4f}")
            if step >= args.steps:
                break
    torch.save({"model": model.state_dict(), "config": cfg}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
