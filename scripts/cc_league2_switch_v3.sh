#!/bin/bash
if [ "${1:-}" = '--help' ]; then
  printf '%s\n' 'usage: scripts/cc_league2_switch_v3.sh OLD_ARM GPU CPUS NEW_ARM NEW_SUP_MODULE FB_ARM FB_SUP_MODULE PREP(0|1)'
  exit 0
fi
# Stop OLD_ARM at its next save boundary, (optionally write NEW_ARM's next protocol), launch NEW_ARM's supervisor on the
# same GPU, require a reward receipt after smoke (variant actually applied), else fall back to FB_ARM (GPU never idles).
# usage: scripts/cc_league2_switch_v3.sh OLD_ARM GPU CPUS NEW_ARM NEW_SUP_MODULE FB_ARM FB_SUP_MODULE PREP(0|1)
#   PREP=1: python -m scripts.cc_league2_variant_next NEW_ARM after the stop (same-arm hook fix: OLD_ARM == NEW_ARM)
# Log markers: "SWITCH_OK" (formal started with receipt), "FALLBACK <cfg>" (FB_ARM resumed).
set -u
OLD=$1; GPU=$2; CPUS=$3; NEW=$4; NMOD=$5; FB=$6; FMOD=$7; PREP=$8
F=runs/cc_train_20261105/training/formal; O=$F/$OLD; N=$F/$NEW; PY=python_environment/bin/python
log() { echo "$(date +%T) $*"; }
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU; }
stage() { python3 -c "import json;d=json.load(open('$O/CYCLE_STAGE.json'));print(d['update'],d['stage'])"; }
mancfg() { $PY -c "from scripts.model_launch import manifest;print(manifest(dict(module='$1',args=['$2','$GPU','$CPUS','CC_LEAGUE2_$2'])))"; }
stopped_note() { $PY -c "import json,time;from scripts.population_run_state import atomic_json;atomic_json('$F/$1/SUPERVISOR.json',dict(status='STOPPED',owner='cc-league-v2',reason='$2',unix=time.time()))"; }
fallback() {
  log "$NEW failed: $1"; tail -n 4 $N/supervisor.jsonl 2>/dev/null | cut -c1-200
  if [ -n "${SUP:-}" ]; then for c in $(ps -o pid= --ppid $SUP); do kill $c; done; kill $SUP 2>/dev/null; fi
  for k in $(seq 1 120); do [ "$(mem)" -lt 1000 ] && break; sleep 5; done
  touch $N/USER_STOP.json; stopped_note $NEW "switch failed: $1"
  rm -f $F/$FB/USER_STOP.json; FC=$(mancfg $FMOD $FB)
  nohup $PY -u model_diagnostics.py $FC > runs/cc_train_20261105/training/logs/supervisor_$FB.log 2>&1 &
  log "FALLBACK $FC pid $! ($FB)"; exit 1
}
read U S <<<"$(stage)"; B=$(( ( (U-18262)/40 + 1 )*40 + 18262 )); log "$OLD at $U $S boundary $B"
for i in $(seq 1 3600); do
  read u s <<<"$(stage)"
  if { [ "$u" -eq "$B" ] && [ "$s" != ATOMIC_SAVE ]; } || [ "$u" -gt "$B" ]; then touch $O/USER_STOP.json; log "stop at $u $s"; break; fi
  sleep 1
done
for i in $(seq 1 360); do
  tail -n 3 $O/supervisor.jsonl | grep -q '"exit", "phase": "formal' && [ "$(mem)" -lt 1000 ] && break; sleep 5
done
log "GPU$GPU $(mem)MiB, $OLD latest $(ls -d $O/u*/COMMITTED.json | sort | tail -1)"
if [ "$PREP" = 1 ]; then $PY -m scripts.cc_league2_variant_next $NEW || fallback "variant_next failed"; fi
PSTEM=$(ls runs/cc_train_20261105/training/protocols | grep "^CC_LEAGUE2_${NEW}_V" | sort -V | tail -1 | sed "s/.json$//"); log "protocol $PSTEM"
rm -f $N/USER_STOP.json; T0=$(date +%s)
CFG=$(mancfg $NMOD $NEW); nohup $PY -u model_diagnostics.py $CFG > runs/cc_train_20261105/training/logs/supervisor_$NEW.log 2>&1 &
SUP=$!; log "$NEW supervisor $SUP $CFG"
for i in $(seq 1 1800); do
  st=$(ps -o stat= -p $SUP 2>/dev/null | tr -d ' ')
  { [ -z "$st" ] || [ "${st:0:1}" = Z ] || grep -q "_${PSTEM}\", \"code\": [1-9]" $N/supervisor.jsonl; } && fallback "supervisor exit or nonzero phase"
  if grep -q "\"start\", \"phase\": \"formal_${PSTEM}\"" $N/supervisor.jsonl; then
    R=$(find $N -maxdepth 1 -name 'REWARD_RECEIPT_*.json' -newermt "@$T0" | head -1)
    [ -z "$R" ] && fallback "no reward receipt after smoke/restore"
    log "receipt $R $(cut -c1-160 $R)"; log "SWITCH_OK formal started"; exit 0
  fi
  sleep 2
done
fallback "timeout waiting for formal"
