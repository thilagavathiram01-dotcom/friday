import argparse
import torch
from ..model.model import FridayModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    state = torch.load(args.ckpt, map_location="cpu")
    cfg = state["config"]
    model = FridayModel(cfg)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    data = torch.load(args.data)
    ids = data["input_ids"]
    with torch.no_grad():
        logits = model(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1))
    print("perplexity:", torch.exp(loss).item())


if __name__ == "__main__":
    main()
