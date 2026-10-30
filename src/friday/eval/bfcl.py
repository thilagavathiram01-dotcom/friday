import argparse
import json
from ..tools.tool_call_parser import parse_tool_call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()
    preds = [json.loads(l) for l in open(args.preds)]
    gold = [json.loads(l) for l in open(args.gold)]
    exact = 0
    for p, g in zip(preds, gold):
        call = parse_tool_call(p["output"]) if isinstance(p, dict) else None
        if call and call["name"] == g["name"] and call["arguments"] == g["arguments"]:
            exact += 1
    print("exact_match:", exact / max(1, len(gold)))


if __name__ == "__main__":
    main()
