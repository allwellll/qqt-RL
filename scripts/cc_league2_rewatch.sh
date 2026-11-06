#!/bin/bash
if [ "${1:-}" = '--help' ]; then
  printf '%s\n' 'usage: scripts/cc_league2_rewatch.sh OLD_WATCHER_PID TASKSET_CPUS ARM [ARM...]'
  exit 0
fi
# Replace a strong watcher with the current code without killing or duplicating its in-flight eval.
# usage: scripts/cc_league2_rewatch.sh OLD_WATCHER_PID TASKSET_CPUS ARM [ARM...]
OLD=$1; CPUS=$2; shift 2
CHILD=$(ps -o pid= --ppid $OLD | head -1 | tr -d ' ')
kill $OLD
[ -n "$CHILD" ] && while kill -0 $CHILD 2>/dev/null && [ "$(ps -o stat= -p $CHILD | cut -c1)" != Z ]; do sleep 10; done
ARGS=$(printf "'%s'," "$@")
CFG=$(python_environment/bin/python -c "from scripts.model_launch import manifest;print(manifest(dict(module='scripts.cc_league2_strong_watch',args=[${ARGS%,}])))")
nohup taskset -c $CPUS python_environment/bin/python -u model_diagnostics.py $CFG > runs/cc_train_20261105/training/logs/strong_watch_$1.log 2>&1 &
echo "$(date +%T) old $OLD child $CHILD -> new $! $CFG $*"
