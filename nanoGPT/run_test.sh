#!/bin/bash
git pull
pip install ../thlc --break-system-packages
nohup torchrun --rdzv-backend=c10d --rdzv-endpoint=109.248.175.95 --nnodes 2 --nproc-per-node=1 train.py config/train_gpt2.py &
