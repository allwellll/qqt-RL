#!/usr/bin/env bash
set -euo pipefail

# 用极小 batch 跑一次真实 PPO 更新；强制 CPU，避免占用正在训练的 GPU。
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export QQT_ALLOW_CPU=1

python -m jax_bomb.bun_train \
  --arch mlp \
  --hidden 32 \
  --num-envs 2 \
  --num-steps 4 \
  --iters 1 \
  --minibatch 8 \
  --epochs 1 \
  --seed 20260928 \
  --levels web/assets/maps/levels.json
