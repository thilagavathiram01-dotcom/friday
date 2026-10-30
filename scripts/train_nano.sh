#!/bin/sh
set -e
CONFIG=configs/nano.yaml
python -m friday.training.pretrain --config $CONFIG --data data/train.pt --out checkpoints/friday-nano-pretrain.pt --steps 1000
python -m friday.training.sft --config $CONFIG --ckpt checkpoints/friday-nano-pretrain.pt --data data/sft.pt --out checkpoints/friday-nano-sft.pt --steps 500
python -m friday.training.grpo --config $CONFIG --ckpt checkpoints/friday-nano-sft.pt --data data/grpo.pt --out checkpoints/friday-nano-grpo.pt --steps 200
