#!/usr/bin/env bash
# Long alternating continuation from the latest completed joint run.
set -euo pipefail
qqt_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$qqt_root"
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash scripts/launch_dcu_jax_carry_joint_long_8card.sh NEW_OUTPUT_ROOT' >&2
  exit 2
fi
qqt_root_out="$1"
if [[ -e "$qqt_root_out" || -e "$qqt_root_out.log" ]]; then
  echo 'Output root or log already exists; choose a new directory' >&2
  exit 2
fi
mkdir -p "$qqt_root_out"
source scripts/dcu_env.sh

qqt_courier="runs/carry-joint-u587-u104-20261122d/courier-s02/u000170"
qqt_defender="runs/carry-joint-u587-u104-20261122d/interceptor-s02/u000170"
qqt_start="u244"
qqt_segments=4
qqt_hours=1.1

for qqt_segment in $(seq 1 "$qqt_segments"); do
  qqt_courier_out="$qqt_root_out/courier-s$(printf '%02d' "$qqt_segment")"
  bash scripts/launch_dcu_jax_carry_8card.sh courier "$qqt_courier_out" \
    --interceptor "$qqt_defender" --checkpoint "$qqt_courier" \
    --courier-self-opponent "$qqt_courier" --interceptor-fraction 0.60 \
    --max-updates 100000 --hours "$qqt_hours"
  qqt_courier="$qqt_courier_out/$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint"])' "$qqt_courier_out/LATEST.json")"

  qqt_defender_out="$qqt_root_out/interceptor-s$(printf '%02d' "$qqt_segment")"
  bash scripts/launch_dcu_jax_carry_8card.sh interceptor "$qqt_defender_out" \
    --interceptor-start "$qqt_start" --checkpoint "$qqt_defender" \
    --courier-opponent "$qqt_courier" --max-updates 100000 --hours "$qqt_hours"
  qqt_defender="$qqt_defender_out/$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint"])' "$qqt_defender_out/LATEST.json")"
done
