#!/usr/bin/env bash
# Longer run of the gated arm (v2_threat04) vs matched control, lr 3e-4 and 1e-4.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${TRAIN_PY:-python3}
cd "$REPO"
OUT=runs/reward_v2_20260930/long
LOAD=$REPO/runs/overnight_tf_v2/phase2_it4000.pt
ITERS=${ITERS:-1500}
COMMON="--arch transformer --embed 192 --depth 4 --num-envs 1024 --num-steps 128 \
  --minibatch 2048 --epochs 2 --seed 20261002 --flee-bot-ratio 0.3 \
  --bun-curriculum danger_arena=1 --bun-reward-profile danger_arena \
  --load $LOAD --iters $ITERS --save-every 100"
V2="--bun-base-bomb-reward 0.06 --bun-tactical-bomb-placement-reward 0.6 --bun-forced-kill-reward 2.0 --bun-tactical-bomb-resolution-reward 1.0 --bun-enemy-threat-reward 0.4"
run() {  # name gpu extra...
  local name=$1 gpu=$2; shift 2
  mkdir -p $OUT/$name
  ( start=$(date +%s)
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 CUDA_VISIBLE_DEVICES=$gpu JAXBOMB_RULE=bun \
      $PY -m jax_bomb.jax_train $COMMON "$@" --save $OUT/$name/final.pt > $OUT/$name/train.log 2>&1
    echo "rc=$? wall_seconds=$(( $(date +%s) - start ))" >> $OUT/$name/train.log ) &
}
run control_lr3e4 0 --lr 3e-4
run thr04_lr3e4 1 --lr 3e-4 $V2
run control_lr1e4 2 --lr 1e-4
run thr04_lr1e4 3 --lr 1e-4 $V2
wait
