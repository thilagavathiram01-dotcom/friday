import argparse
import yaml
import torch
from ..model.config import FridayConfig
from ..model.model import FridayModel


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return FridayConfig(**raw["model"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="checkpoints/sft.pt")
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = FridayModel(cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
    data = torch.load(args.data)
    input_ids = data["input_ids"]
    labels = data["labels"]
    step = 0
    model.train()
    while step < args.steps:
        for i in range(0, len(input_ids), 4):
            x = input_ids[i:i + 4].to(device)
            y = labels[i:i + 4].to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"sft step {step} loss {loss.item():.4f}")
            if step >= args.steps:
                break
    torch.save({"model": model.state_dict(), "config": cfg}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
