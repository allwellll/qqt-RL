#!/bin/bash
if [ "${1:-}" = '--help' ]; then
  printf '%s\n' 'usage: scripts/cc_league2_switch_v2.sh OLD_ARM GPU CPUS NEW_ARM NEW_SUPERVISOR_MODULE OLD_SUPERVISOR_MODULE'
  exit 0
fi
# Stop OLD_ARM at its next save boundary, launch an already-prepared NEW_ARM supervisor on the same GPU, and fall back
# to OLD_ARM if the new smoke/restore fails (GPU never idles). Protocol of NEW_ARM must exist (prepare first).
# usage: scripts/cc_league2_switch_v2.sh OLD_ARM GPU CPUS NEW_ARM NEW_SUPERVISOR_MODULE OLD_SUPERVISOR_MODULE
# Completion markers in the log: "SWITCH_OK" (new arm reached formal), "FALLBACK <cfg>" (old arm resumed).
set -u
OLD=$1; GPU=$2; CPUS=$3; NEW=$4; NMOD=$5; OMOD=$6
F=runs/cc_train_20261105/training/formal; O=$F/$OLD; N=$F/$NEW; PY=python_environment/bin/python
T=$(date +%T); log() { echo "$(date +%T) $*"; }
stage() { python3 -c "import json;d=json.load(open('$O/CYCLE_STAGE.json'));print(d['update'],d['stage'])"; }
mancfg() { $PY -c "from scripts.model_launch import manifest;print(manifest(dict(module='$1',args=['$2','$GPU','$CPUS','CC_LEAGUE2_$2'])))"; }
read U S <<<"$(stage)"; B=$(( ( (U-18262)/40 + 1 )*40 + 18262 )); log "$OLD at $U $S boundary $B"
for i in $(seq 1 3600); do
  read u s <<<"$(stage)"
  if { [ "$u" -eq "$B" ] && [ "$s" != ATOMIC_SAVE ]; } || [ "$u" -gt "$B" ]; then touch $O/USER_STOP.json; log "stop at $u $s"; break; fi
  sleep 1
done
for i in $(seq 1 360); do
  tail -n 3 $O/supervisor.jsonl | grep -q '"exit", "phase": "formal' && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)" -lt 1000 ] && break; sleep 5
done
log "GPU$GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)MiB, $OLD latest $(ls -d $O/u*/COMMITTED.json | sort | tail -1)"
CFG=$(mancfg $NMOD $NEW); nohup $PY -u model_diagnostics.py $CFG > runs/cc_train_20261105/training/logs/supervisor_$NEW.log 2>&1 &
SUP=$!; log "$NEW supervisor $SUP $CFG"
for i in $(seq 1 1800); do
  if grep -q '"phase": "formal' $N/supervisor.jsonl 2>/dev/null; then log "SWITCH_OK formal started"; exit 0; fi
  st=$(ps -o stat= -p $SUP 2>/dev/null | tr -d ' ')
  if [ -z "$st" ] || [ "${st:0:1}" = Z ] || grep -q '"code": [1-9]' $N/supervisor.jsonl 2>/dev/null; then
    log "$NEW failed before formal:"; tail -n 5 $N/supervisor.jsonl; tail -n 5 runs/cc_train_20261105/training/logs/supervisor_$NEW.log
    kill $SUP 2>/dev/null; sleep 5
    for k in $(seq 1 120); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)" -lt 1000 ] && break; sleep 5; done
    rm -f $O/USER_STOP.json; OC=$(mancfg $OMOD $OLD)
    nohup $PY -u model_diagnostics.py $OC > runs/cc_train_20261105/training/logs/supervisor_$OLD.log 2>&1 &
    log "FALLBACK $OC pid $!"; exit 1
  fi
  sleep 2
done
log "timeout waiting for $NEW formal"; exit 2
