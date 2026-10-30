import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="friday-q4.gguf")
    ap.add_argument("--format", default="gguf", choices=["gguf", "executorch"])
    args = ap.parse_args()
    print(f"Quantize {args.ckpt} -> {args.out} ({args.format})")
    print("Use llama.cpp convert + quantize for GGUF; ExecuTorch for mobile.")


if __name__ == "__main__":
    main()
