#!/usr/bin/env bash
# Run in the foreground; use tmux/platform job scheduling to retain the terminal.
set -euo pipefail
qqt_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$qqt_project"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/launch_dcu_kill_8card.sh NEW_OUTPUT_DIRECTORY [trainer overrides...]' >&2
  exit 2
fi
qqt_output="$1"
shift
source scripts/dcu_env.sh
export PATH="$qqt_project/.tools/node/bin:$PATH"
ulimit -c 0
# A 4h57m internal budget leaves time for checkpoints. External watchdog caps
# the job at about 5h even if a compiler/driver call stops responding to Python.
exec timeout --signal=TERM --kill-after=90s 5h \
  python -u -m scripts.train_dcu_kill train \
  --devices 0,1,2,3,4,5,6,7 --envs 8 --ticks 64 --minibatch 64 \
  --hours 4.95 --critic-warmup 10 --save-every 20 --eval-every 1800 \
  --out "$qqt_output" --acknowledge-training "$@"
