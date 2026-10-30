import argparse
import torch
from ..model.config import FridayConfig
from ..model.model import FridayModel


def load(ckpt):
    state = torch.load(ckpt, map_location="cpu")
    cfg = state["config"] if "config" in state else None
    model = FridayModel(cfg)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model, cfg


def generate(model, cfg, prompt_ids, max_new=128, temperature=0.7):
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids], dtype=torch.long)
            logits = model(x)[:, -1, :]
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
            else:
                nxt = int(logits.argmax(-1).item())
            ids.append(nxt)
            if nxt == cfg.eos_token_id:
                break
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max_new", type=int, default=64)
    args = ap.parse_args()
    model, cfg = load(args.ckpt)
    out = generate(model, cfg, [cfg.bos_token_id], max_new=args.max_new)
    print(out)
