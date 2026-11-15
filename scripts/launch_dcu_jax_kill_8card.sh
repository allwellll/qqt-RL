#!/usr/bin/env bash
# Foreground job; retain it using the platform scheduler or tmux.
set -euo pipefail
qqt_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$qqt_project"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/launch_dcu_jax_kill_8card.sh NEW_OUTPUT_DIRECTORY [trainer overrides...]' >&2
  exit 2
fi
qqt_output="$1"
shift
source scripts/dcu_env.sh
ulimit -c 0
exec timeout --signal=TERM --kill-after=90s 5h \
  python -u -m scripts.train_dcu_jax_kill train \
  --devices 0,1,2,3,4,5,6,7 --envs 512 --ticks 64 --minibatch 512 \
  --hours 4.95 --critic-warmup 10 \
  --out "$qqt_output" --acknowledge-training "$@"
