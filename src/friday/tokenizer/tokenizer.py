from transformers import AutoTokenizer


def get_tokenizer(name="meta-llama/Llama-3.2-1B"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
