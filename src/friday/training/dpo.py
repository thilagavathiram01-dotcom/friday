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
    ap.add_argument("--out", default="checkpoints/dpo.pt")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = FridayModel(cfg)
    ref = FridayModel(cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    ref.load_state_dict(model.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ref.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-7)
    data = torch.load(args.data)
    # data: chosen_ids, rejected_ids, prompt_ids
    print("dpo training placeholder; implement preference loss here")
    torch.save({"model": model.state_dict(), "config": cfg}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
