#!/usr/bin/env bash
# Reward V2 + 分级 JAX Bot 池 + 自对弈，8 卡 pmap DP 正式训练（约一天预算）。
# 与 2026-10-01 已验证的 4 卡 DP smoke 命令相同，只改 --devices 8 与按卡整除的
# --num-envs / --minibatch（每卡 256 env、minibatch 512，与 smoke 每卡形状一致），
# 并默认启用设备端出生位置分桶（--bun-spawn-buckets，SPAWN_BUCKETS 可覆盖，
# 设为空串则回到原始出生分布）。
# 预检全部通过才启动；任何一项失败立即非 0 退出。8 卡机器不做 smoke。
#
# 用法：bash scripts/train_v2_jaxbot_dp8.sh
#   续跑：RESUME_CKPT=<旧run>/ckpt/final_itN.pt bash scripts/train_v2_jaxbot_dp8.sh
#   只打印训练参数：PRINT_ARGS=1 bash scripts/train_v2_jaxbot_dp8.sh
# 续跑语义：jax_train 只存 params，不存 Adam 状态 / bot 自适应 EMA / env 状态。
#   续跑 = 从 final_itN.pt 权重热启动、Adam 与 bot 权重重新初始化、--iter-offset N
#   让 ckpt 编号接着全局 iter 走；总 iter 仍以 ITERS 为准，seed 偏移 N 避免重复轨迹。
#   续跑总是写入新的唯一 run 目录，旧 run 不会被改动。
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# 不使用容易被shell/调度系统预设的通用PY变量；需要覆盖解释器时显式设置TRAIN_PY。
TRAIN_PY=${TRAIN_PY:-python3}
INIT_CKPT=${INIT_CKPT:-$REPO/runs/overnight_tf_v2/phase2_it4000.pt}
INIT_SHA256=6e190b009f180f2e449fd19941c930c0d1a86d680071eeb02262aadb6808ebd0
# 训练代码必须与已验证提交逐字节一致（按内容比对，rebase/cherry-pick 到 main 后仍可用）。
VERIFIED_CODE_SHA=${VERIFIED_CODE_SHA:-870b7dd}
CODE_PATHS=(jax_bomb qqt_rl web/assets/maps)
RUN_ROOT=${RUN_ROOT:-$REPO/runs}
N_GPU=8
MIN_FREE_GB=${MIN_FREE_GB:-50}
SEED=${SEED:-20261001}
# 预算：4 卡 smoke（含出生分桶）稳态 1.0–1.05 s/iter；8 卡每卡负载相同，按约 1.3 s/iter 保守估计，
# 60000 iter ≈ 22 h。MAX_HOURS 为硬上限，到点 SIGTERM（已存 ckpt 保留）。
ITERS=${ITERS:-60000}
SAVE_EVERY=${SAVE_EVERY:-500}
MAX_HOURS=${MAX_HOURS:-24}
SPAWN_BUCKETS=${SPAWN_BUCKETS-native=0.3,near=0.1,mid=0.15,far=0.1,below=0.15,upper_left=0.1,upper_right=0.1}

die() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }

ITER_OFFSET=0
LOAD=$INIT_CKPT
if [ -n "${RESUME_CKPT:-}" ]; then
  [[ $(basename "$RESUME_CKPT") =~ _it([0-9]+)\.pt$ ]] \
    || die "RESUME_CKPT 必须是 *_itN.pt：$RESUME_CKPT"
  ITER_OFFSET=${BASH_REMATCH[1]}
  LOAD=$RESUME_CKPT
fi
REMAINING=$((ITERS - ITER_OFFSET))

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${RUN_NAME:-v2_jaxbot_dp8_${STAMP}_$(hostname -s)}
RUN_DIR=$RUN_ROOT/$RUN_NAME

TRAIN_ARGS=(
  --devices "$N_GPU"
  --arch transformer --embed 192 --depth 4
  --num-envs 2048 --num-steps 64 --minibatch 4096 --epochs 1
  --iters "$REMAINING" --iter-offset "$ITER_OFFSET" --save-every "$SAVE_EVERY"
  --seed "$((SEED + ITER_OFFSET))"
  --load "$LOAD"
  --flee-bot-ratio 0.3
  --jax-bot-pool dodge=1,bomber_easy=2,hunter=2,hunter_hard=1,legacy_flee=1
  --jax-bot-adaptive
  --bun-curriculum danger_arena=1 --bun-reward-profile danger_arena
  --bun-base-bomb-reward 0.06 --bun-tactical-bomb-placement-reward 0.6
  --bun-forced-kill-reward 2.0 --bun-tactical-bomb-resolution-reward 1.0
  --bun-enemy-threat-reward 0.4
  --bun-spawn-buckets "$SPAWN_BUCKETS"
  --save "$RUN_DIR/ckpt/final.pt"
)

if [ "${PRINT_ARGS:-0}" = "1" ]; then
  printf '%s\n' "${TRAIN_ARGS[@]}"
  exit 0
fi

# ---------------- 预检 ----------------
[ "$REMAINING" -gt 0 ] || die "ITERS=$ITERS 不大于续跑起点 $ITER_OFFSET"
(( SAVE_EVERY > 0 )) || die "SAVE_EVERY 必须 > 0"
cd "$REPO"
command -v git >/dev/null || die "缺少 git"
command -v nvidia-smi >/dev/null || die "缺少 nvidia-smi"
[ -x "$TRAIN_PY" ] || die "Python 不可执行：$TRAIN_PY"
"$TRAIN_PY" -c 'import jax' 2>/dev/null \
  || die "训练Python无法import jax：$TRAIN_PY（如需覆盖请设置TRAIN_PY）"

git cat-file -e "${VERIFIED_CODE_SHA}^{commit}" 2>/dev/null \
  || die "仓库中找不到已验证提交 $VERIFIED_CODE_SHA"
git diff --quiet "$VERIFIED_CODE_SHA" HEAD -- "${CODE_PATHS[@]}" \
  || die "HEAD 的训练代码与已验证提交 $VERIFIED_CODE_SHA 不一致"
git diff --quiet HEAD -- "${CODE_PATHS[@]}" && git diff --cached --quiet HEAD -- "${CODE_PATHS[@]}" \
  || die "训练代码有未提交改动"
[ -z "$(git ls-files --others --exclude-standard -- "${CODE_PATHS[@]}")" ] \
  || die "训练代码目录有未跟踪文件"
CODE_SHA=$(git rev-parse HEAD)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | head -n "$N_GPU")
fi
[ "${#GPU_IDS[@]}" -eq "$N_GPU" ] || die "需要 $N_GPU 张 GPU，实际可用 ${#GPU_IDS[@]}"
for gpu in "${GPU_IDS[@]}"; do
  row=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits) || die "GPU $gpu 查询失败"
  mem=${row%%,*}; util=${row##*, }
  [ "${mem// /}" -lt 2000 ] || die "GPU $gpu 显存已占用 ${mem} MiB"
  [ "${util// /}" -lt 10 ] || die "GPU $gpu 利用率 ${util}%"
  apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
  [ -z "$apps" ] || die "GPU $gpu 上有进程：$apps"
done
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_IDS[*]}")

[ -f "$LOAD" ] || die "checkpoint 不存在：$LOAD"
if [ -z "${RESUME_CKPT:-}" ]; then
  got=$(sha256sum "$LOAD" | cut -d' ' -f1)
  [ "$got" = "$INIT_SHA256" ] || die "it4000 sha256 不符：$got"
fi

mkdir -p "$RUN_ROOT"
free_gb=$(df -BG --output=avail "$RUN_ROOT" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "$MIN_FREE_GB" ] || die "$RUN_ROOT 剩余 ${free_gb}G < ${MIN_FREE_GB}G"

JAXBOMB_RULE=bun XLA_PYTHON_CLIENT_PREALLOCATE=false "$TRAIN_PY" -c \
  "import jax, sys; n=jax.device_count(); p=jax.devices()[0].platform; \
print('jax', jax.__version__, p, n); sys.exit(0 if (p=='gpu' and n==$N_GPU) else 1)" \
  || die "JAX 未看到 $N_GPU 张 GPU"

[ ! -e "$RUN_DIR" ] || die "run 目录已存在：$RUN_DIR"
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/ckpt"

# ---------------- 记录 ----------------
{
  echo "code_sha=$CODE_SHA"
  echo "verified_code_sha=$VERIFIED_CODE_SHA"
  echo "host=$(hostname)"
  echo "start=$(date -Is)"
  echo "gpus=$N_GPU cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "seed=$((SEED + ITER_OFFSET)) iters_total=$ITERS iter_offset=$ITER_OFFSET iters_this_run=$REMAINING"
  echo "save_every=$SAVE_EVERY max_hours=$MAX_HOURS"
  echo "spawn_buckets=${SPAWN_BUCKETS:-native(disabled)}"
  echo "load=$LOAD"
  echo "load_sha256=$(sha256sum "$LOAD" | cut -d' ' -f1)"
  echo "resume_from=${RESUME_CKPT:-none}"
} > "$RUN_DIR/run_info.txt"
git status --short > "$RUN_DIR/git_status.txt"
git log -5 --oneline > "$RUN_DIR/git_log.txt"
{
  printf 'cd %q && JAXBOMB_RULE=bun CUDA_VISIBLE_DEVICES=%q timeout --signal=TERM %qh %q -m jax_bomb.jax_train' \
    "$REPO" "$CUDA_VISIBLE_DEVICES" "$MAX_HOURS" "$TRAIN_PY"
  printf ' %q' "${TRAIN_ARGS[@]}"
  echo
} > "$RUN_DIR/command.txt"
{
  env | grep -E '^(CUDA|XLA|JAX|NCCL|TF_|PYTHON|PATH=|HOSTNAME|USER=)' | sort
  "$TRAIN_PY" -m pip freeze 2>/dev/null | grep -iE '^(jax|jaxlib|jax-cuda|flax|optax|numpy)' || true
  nvidia-smi
} > "$RUN_DIR/env.txt" 2>&1

# ---------------- 启动（前台运行，保持训练机器任务存活）----------------
export JAXBOMB_RULE=bun
{
  echo $$ > "$RUN_DIR/train.pid"
  set +e
  timeout --signal=TERM "${MAX_HOURS}h" "$TRAIN_PY" -m jax_bomb.jax_train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "$RUN_DIR/train.log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "rc=$rc end=$(date -Is)" > "$RUN_DIR/exit_status"
  exit "$rc"
}
