#!/usr/bin/env bash
# Usage: eval_arm.sh <train_dir> <arm_name> <final_iter> <gpu> <seed> <games> [checkpoints]
# Symlinks it4000 as phase1 (reference) and <train_dir>/final_itN.pt as phase2_itN.pt so
# main's scripts/eval_transformer_checkpoint_sweep.py runs unchanged.
set -euo pipefail
PY=/mnt/jpfs/afs/wangyaqi/code_room/qqt-gpu-sim/.venv/bin/python
cd /mnt/jpfs/afs/wangyaqi/code_room/qqt-reward-v2
TRAIN=$1; ARM=$2; FINAL=$3; GPU=$4; SEED=$5; GAMES=$6; SEL=${7:-all}
IT4000=/mnt/jpfs/afs/wangyaqi/code_room/qqt-RL/runs/overnight_tf_v2/phase2_it4000.pt
LAYOUT=runs/reward_v2_20260930/eval_layout/$ARM
mkdir -p $LAYOUT
ln -sfn $IT4000 $LAYOUT/phase1.pt
for f in $TRAIN/final_it*.pt; do
  [ -e "$f" ] || continue
  n=$(basename $f .pt); n=${n#final_it}
  ln -sfn $(realpath $f) $LAYOUT/phase2_it$n.pt
done
[ -e $TRAIN/final.pt ] && ln -sfn $(realpath $TRAIN/final.pt) $LAYOUT/phase2.pt
OUT=runs/reward_v2_20260930/eval/seed${SEED}_g${GAMES}/$ARM
start=$(date +%s)
JAXBOMB_RULE=bun $PY scripts/eval_transformer_checkpoint_sweep.py \
  --run-dir $LAYOUT --out-dir $OUT --checkpoints "$SEL" --final-iter $FINAL \
  --seed $SEED --games $GAMES --device $GPU --host-workers 40 \
  --jax-cache-dir runs/reward_v2_20260930/.jax_cache \
  --reuse-dir runs/reward_v2_20260930/eval/seed${SEED}_g${GAMES}/_it4000 \
  --teacher-json /mnt/jpfs/afs/wangyaqi/code_room/qqt-RL/runs/eval_v2/rulebot_baseline.json
echo "eval wall_seconds=$(( $(date +%s) - start ))"
