import torch


def make_lm_dataset(texts, tokenizer, seq_len=2048):
    ids = []
    for t in texts:
        ids.extend(tokenizer.encode(t))
    import torch
    data = torch.tensor(ids, dtype=torch.long)
    n = (len(data) // seq_len) * seq_len
    data = data[:n].view(-1, seq_len)
    return {"input_ids": data}


def make_sft_dataset(prompts, responses, tokenizer, seq_len=2048):
    input_ids = []
    labels = []
    for p, r in zip(prompts, responses):
        p_ids = tokenizer.encode(p)
        r_ids = tokenizer.encode(r) + [2]
        seq = p_ids + r_ids
        lab = [-100] * len(p_ids) + r_ids
        if len(seq) > seq_len:
            seq = seq[:seq_len]
            lab = lab[:seq_len]
        pad = seq_len - len(seq)
        seq = seq + [0] * pad
        lab = lab + [-100] * pad
        input_ids.append(seq)
        labels.append(lab)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}
