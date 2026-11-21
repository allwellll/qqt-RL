#!/usr/bin/env bash
# Alternating co-training: update courier, then interceptor against that courier.
# Each segment freezes only its current opponent; the next segment refreshes it.
set -euo pipefail
qqt_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$qqt_root"
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash scripts/launch_dcu_jax_carry_joint_8card.sh NEW_OUTPUT_ROOT' >&2
  exit 2
fi
qqt_root_out="$1"
if [[ -e "$qqt_root_out" || -e "$qqt_root_out.log" ]]; then
  echo 'Output root or log already exists; choose a new directory' >&2
  exit 2
fi
mkdir -p "$qqt_root_out"
source scripts/dcu_env.sh

qqt_courier="runs/carry-courier-u244-defender-u104-20261121/u000587"
qqt_defender="runs/carry-defender-u244-20261121/u000104"
qqt_start="u244"
qqt_segments=6
qqt_updates=20

for qqt_segment in $(seq 1 "$qqt_segments"); do
  qqt_courier_out="$qqt_root_out/courier-s$(printf '%02d' "$qqt_segment")"
  bash scripts/launch_dcu_jax_carry_8card.sh courier "$qqt_courier_out" \
    --interceptor "$qqt_defender" --checkpoint "$qqt_courier" \
    --max-updates "$qqt_updates" --hours 0.75
  qqt_courier="$qqt_courier_out/$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint"])' "$qqt_courier_out/LATEST.json")"

  qqt_defender_out="$qqt_root_out/interceptor-s$(printf '%02d' "$qqt_segment")"
  bash scripts/launch_dcu_jax_carry_8card.sh interceptor "$qqt_defender_out" \
    --interceptor-start "$qqt_start" --checkpoint "$qqt_defender" \
    --courier-opponent "$qqt_courier" --max-updates "$qqt_updates" --hours 0.75
  qqt_defender="$qqt_defender_out/$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint"])' "$qqt_defender_out/LATEST.json")"
done
