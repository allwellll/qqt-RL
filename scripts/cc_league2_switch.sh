#!/bin/bash
if [ "${1:-}" = '--help' ]; then
  printf '%s\n' 'usage: scripts/cc_league2_switch.sh OLD_ARM OLD_SUPERVISOR_PID BOUNDARY NEW_ARM SOURCE_ARM SIL_COEF GPU CPUS OLD_TRAINER_PID'
  exit 0
fi
# Stop OLD_ARM at the save boundary BOUNDARY, then prepare and launch NEW_ARM (CC_LEAGUE_V2) on the same GPU.
# usage: scripts/cc_league2_switch.sh OLD_ARM OLD_SUPERVISOR_PID BOUNDARY NEW_ARM SOURCE_ARM SIL_COEF GPU CPUS OLD_TRAINER_PID
# The stop file is dropped while OLD_ARM is inside update BOUNDARY, so the trainer commits BOUNDARY and
# exits at the next loop check without running BOUNDARY+1.
set -u
OLD=$1; SUP=$2; B=$3; NEW=$4; SRC=$5; COEF=$6; GPU=$7; CPUS=$8; TR=$9
F=runs/cc_train_20261105/training/formal; OUT=$F/$OLD
log() { echo "$(date +%FT%T) $*"; }
stage() { python3 -c "import json;d=json.load(open('$OUT/CYCLE_STAGE.json'));print(d['update'],d['stage'])"; }
[ "$SUP" != 0 ] && kill "$SUP" && log "killed supervisor $SUP"
log "trainer pid $TR"
for i in $(seq 1 3600); do
  read u s <<<"$(stage)"
  if [ "$u" -eq "$B" ] && [ "$s" != ATOMIC_SAVE ]; then touch "$OUT/USER_STOP.json"; log "stop file at $u $s"; break; fi
  if [ "$u" -gt "$B" ]; then log "missed boundary: at $u"; touch "$OUT/USER_STOP.json"; break; fi
  sleep 1
done
for i in $(seq 1 900); do
  st=$(ps -o stat= -p "$TR" 2>/dev/null | tr -d ' ')
  [ -z "$st" ] || [ "${st:0:1}" = Z ] && break; sleep 2
done
LATEST=$(ls -d $OUT/u*/COMMITTED.json | sort | tail -1 | xargs dirname | xargs basename)
log "trainer exited; latest $LATEST; stage $(stage)"
grep -E "Traceback|Error" $OUT/formal_*.log 2>/dev/null | tail -3
for i in $(seq 1 120); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU); [ "$m" -lt 1000 ] && break; sleep 5
done
log "GPU$GPU memory ${m}MiB"
python_environment/bin/python - <<EOF
import json,time
from scripts.population_run_state import atomic_json
atomic_json('$OUT/SUPERVISOR.json', dict(status='STOPPED', latest=int('$LATEST'[1:]), owner='cc-league-v2',
    reason='switched to CC_LEAGUE_V2 as $NEW at save boundary', unix=time.time()))
EOF
# Same arm: start exactly at the stop point. Other source arm: its latest COMMITTED (read only).
U=""; [ "$SRC" = "$OLD" ] && U=$((10#${LATEST:1}))
python_environment/bin/python -m scripts.cc_league2_prepare "$NEW" "$SRC" "$COEF" $U || { log "prepare failed"; exit 1; }
CFG=$(python_environment/bin/python -c "from scripts.model_launch import manifest;print(manifest(dict(module='scripts.cc_league2_supervise',args=['$NEW','$GPU','$CPUS','CC_LEAGUE2_$NEW'])))")
mkdir -p runs/cc_train_20261105/training/logs
nohup python_environment/bin/python -u model_diagnostics.py "$CFG" > runs/cc_train_20261105/training/logs/supervisor_$NEW.log 2>&1 &
log "launched $NEW supervisor pid $! config $CFG"
