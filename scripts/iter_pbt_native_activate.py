"""One-shot authorized evaluation-only migration after persisted parity PASS."""
import json,os,signal,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.iter_pbt_owner import live_workers
from scripts.iter_pbt_eval_scheduling import pending_latest
B=Path('runs/iter_pbt_v1')
def main():
 gate=json.loads((B/'native_gpu/PARITY_GATE.json').read_text());assert gate['status']=='PASS'
 atomic_json(B/'native_gpu/ACTIVATION_STARTED.json',dict(authorization='.hermes/pbt_eval_gpu_native_0920.md',training_runs_before=sorted(live_workers('scripts.iter_pbt_train')),unix=time.time()))
 # Stop only CPU evaluator/controller groups; all completed atomic shards remain.
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:args=(d/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if b'scripts.iter_pbt_eval_owner' in args:os.kill(int(d.name),signal.SIGTERM)
 requests={p.stem:json.loads(p.read_text()) for p in (B/'eval_requests').glob('*.json')};keep=pending_latest(list(requests.values()))
 retained=[];deferred=[]
 for n,identity in live_workers('scripts.iter_pbt_eval').items():
  out=B/'eval'/n
  # Dedicated process sessions established by the original owner. Protect
  # against accidental parent-group signaling if process ownership changes.
  assert os.getpgid(identity)==identity
  if (out/'COMPLETE.json').exists():continue
  if n not in keep:
   atomic_json(out/'SKIPPED_SUPERSEDED.json',dict(reason='0920 evaluation-only migration; stale request, completed shards retained',unix=time.time()));deferred.append(n)
  else:retained.append(n)
  os.killpg(identity,signal.SIGTERM)
  atomic_json(out/'CPU_TO_GPU_MIGRATION.json',dict(completed_shards_retained=True,training_stopped=False,unix=time.time()))
 # Native owner will consume requests after resumed CPU JS merger publishes them.
 config=json.loads((B/'native_gpu/CONFIG.json').read_text());config.update(enabled=True,wait_reason=None,activation_unix=time.time());atomic_json(B/'native_gpu/CONFIG.json',config)
 atomic_json(B/'native_gpu/ACTIVATION.json',dict(status='ENABLED',parity_gate_sha256=sha(B/'native_gpu/PARITY_GATE.json'),retained_jobs=retained,deferred_stale_jobs=deferred,no_training_signal=True,unix=time.time()))
if __name__=='__main__':main()
