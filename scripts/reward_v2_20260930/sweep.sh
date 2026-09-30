#!/usr/bin/env bash
# Reward V2 short sweep: 4 arms x 1 GPU, warm start from overnight_tf_v2 it4000.
set -uo pipefail
PY=/mnt/jpfs/afs/wangyaqi/code_room/qqt-gpu-sim/.venv/bin/python
cd /mnt/jpfs/afs/wangyaqi/code_room/qqt-reward-v2
OUT=runs/reward_v2_20260930/sweep
LOAD=/mnt/jpfs/afs/wangyaqi/code_room/qqt-RL/runs/overnight_tf_v2/phase2_it4000.pt
ITERS=${ITERS:-300}
COMMON="--arch transformer --embed 192 --depth 4 --num-envs 1024 --num-steps 128 \
  --minibatch 2048 --epochs 2 --seed 20260930 --flee-bot-ratio 0.3 \
  --bun-curriculum danger_arena=1 --bun-reward-profile danger_arena \
  --load $LOAD --iters $ITERS --save-every 100"
declare -A ARMS=(
  [control]=""
  [v2]="--bun-base-bomb-reward 0.06 --bun-tactical-bomb-placement-reward 0.6 --bun-forced-kill-reward 2.0 --bun-tactical-bomb-resolution-reward 1.0"
  [v2_threat04]="--bun-base-bomb-reward 0.06 --bun-tactical-bomb-placement-reward 0.6 --bun-forced-kill-reward 2.0 --bun-tactical-bomb-resolution-reward 1.0 --bun-enemy-threat-reward 0.4"
  [v2_threat08]="--bun-base-bomb-reward 0.03 --bun-tactical-bomb-placement-reward 0.6 --bun-forced-kill-reward 2.0 --bun-tactical-bomb-resolution-reward 1.0 --bun-enemy-threat-reward 0.8"
)
GPU=0
for arm in control v2 v2_threat04 v2_threat08; do
  mkdir -p $OUT/$arm
  ( start=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GPU JAXBOMB_RULE=bun $PY -m jax_bomb.jax_train $COMMON ${ARMS[$arm]} \
      --save $OUT/$arm/final.pt > $OUT/$arm/train.log 2>&1
    echo "rc=$? wall_seconds=$(( $(date +%s) - start ))" >> $OUT/$arm/train.log ) &
  GPU=$((GPU+1))
done
wait
