#!/usr/bin/env bash
# Explicit single-stage launch; NEVER starts the next stage automatically.
set -euo pipefail
qqt_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$qqt_project"
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/launch_dcu_jax_carry_8card.sh interceptor|courier NEW_OUTPUT [overrides...]' >&2
  exit 2
fi
qqt_stage="$1"
qqt_output="$2"
shift 2
case "$qqt_stage" in
  interceptor) qqt_hours=0.75 ;;
  courier) qqt_hours=4 ;;
  *) echo 'Expected interceptor or courier' >&2; exit 2 ;;
esac
source scripts/dcu_env.sh
ulimit -c 0
exec timeout --signal=TERM --kill-after=90s 5h \
  python -u -m scripts.train_dcu_jax_carry train \
  --stage "$qqt_stage" --devices 0,1,2,3,4,5,6,7 \
  --envs 2048 --ticks 64 --minibatch 2048 --collector fused \
  --policy-chunk 2048 --value-chunk 2048 --compiler-profile throughput --hours "$qqt_hours" \
  --out "$qqt_output" --acknowledge-training "$@"
