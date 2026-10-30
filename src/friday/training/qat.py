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
    ap.add_argument("--out", default="checkpoints/qat.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = FridayModel(cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    # QAT: use torchao for 4-bit group-wise quantization-aware training
    try:
        from torchao.quantization.qat import QATConfig
        print("torchao available; apply QATConfig(group_size=32) to linear layers")
    except Exception as e:
        print("torchao not available:", e)
    torch.save({"model": model.state_dict(), "config": cfg}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
