#!/usr/bin/env bash
# 通宵 transformer 两阶段训练（4×H200 DP）：
#   阶段1：对 jax 规则 bot 对弈（flee-bot-ratio 0.3）+ legacy 奖励 → 学基本能力。
#   阶段2：大规模自对弈（flee-bot-ratio 0.0）+ combat_evolution 奖励 + 混合出生课程 → 强化安全击杀/对抗。
# 阶段2 --load 阶段1 权重续跑；--save-every 定期存中间 ckpt（供 10am 取用）。
# 看门狗在 10:00 硬停：即便仍在跑，最近的中间 ckpt 也是新鲜可用的（含 .json 元数据）。
set -uo pipefail

PY=/mnt/jpfs/afs/wangyaqi/code_room/qqt-gpu-sim/.venv/bin/python
cd /mnt/jpfs/afs/wangyaqi/code_room/qqt-RL || exit 1
OUT=runs/overnight_tf_v2
mkdir -p "$OUT"

COMMON="--arch transformer --embed 192 --depth 4 --devices 4 \
  --num-envs 4096 --num-steps 256 --minibatch 2048 --epochs 2 \
  --seed 0 --save-every 500"

# ---- 可选硬截止看门狗（默认关闭，训练完整跑完）----
# 如需恢复截止，显式设置 STOP_AT_10=1 后重启新run。
if [ "${STOP_AT_10:-0}" = "1" ]; then
  DEADLINE=$(date -d "today 10:00" +%s)
  NOW=$(date +%s)
  if [ "$NOW" -ge "$DEADLINE" ]; then DEADLINE=$(date -d "tomorrow 10:00" +%s); fi
  (
    while [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 60; done
    echo "=== WATCHDOG: 10:00 reached, stopping training $(date) ==="
    pkill -f "jax_bomb.bun_train"
  ) &
  WATCHDOG=$!
  trap 'kill $WATCHDOG 2>/dev/null' EXIT
else
  DEADLINE=32503680000
fi

echo "=== PHASE 1 (vs jax bot, legacy) start $(date) ==="
$PY -m jax_bomb.bun_train $COMMON \
  --iters 1800 \
  --flee-bot-ratio 0.3 \
  --bun-reward-profile legacy \
  --bun-curriculum "full=0.3,carry_home=0.25,near_steal=0.2,route_break=0.15,full_ambush=0.1" \
  --save "$OUT/phase1.pt"
echo "=== PHASE 1 done rc=$? $(date) ==="

if [ "$(date +%s)" -ge "$DEADLINE" ]; then
  echo "=== deadline hit during phase1; skip phase2 ==="
  exit 0
fi

echo "=== PHASE 2 (self-play, combat_evolution) start $(date) ==="
$PY -m jax_bomb.bun_train $COMMON \
  --iters 8000 \
  --flee-bot-ratio 0.0 \
  --bun-reward-profile combat_evolution \
  --bun-curriculum "danger_arena=0.35,full_ambush=0.25,full=0.2,carry_home=0.2" \
  --load "$OUT/phase1.pt" \
  --save "$OUT/phase2.pt"
echo "=== PHASE 2 done rc=$? $(date) ==="
echo "=== ALL DONE $(date) ==="
