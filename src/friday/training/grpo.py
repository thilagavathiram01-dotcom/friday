import argparse
import yaml
import torch
from ..model.config import FridayConfig
from ..model.model import FridayModel


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return FridayConfig(**raw["model"])


def reward_fn(completion, ground_truth):
    # verifiable math reward: exact match of final boxed answer
    import re
    m = re.search(r"\\boxed\{(.+?)\}", completion)
    pred = m.group(1).strip() if m else completion.strip().split()[-1]
    return 1.0 if pred == str(ground_truth).strip() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="checkpoints/grpo.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--group", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = FridayModel(cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ref = FridayModel(cfg)
    ref.load_state_dict(model.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    data = torch.load(args.data)
    prompts = data["prompts"]
    answers = data["answers"]

    model.train()
    step = 0
    while step < args.steps:
        for pi in range(len(prompts)):
            # placeholder: real impl samples G completions via model.generate, scores, updates with GRPO
            step += 1
            if step >= args.steps:
                break
        print(f"grpo step {step}")
    torch.save({"model": model.state_dict(), "config": cfg}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
