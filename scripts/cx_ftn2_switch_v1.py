"""Owned GPU4 committed-boundary switch, bounded smoke/restore/formal, safe old-arm fallback."""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.model_launch import manifest,command,public_project
from scripts.cx_ftn2_prepare_v1 import BASE,ROOT,PROTOCOL
OLD=BASE/'formal/L2_FTN';OUT=ROOT/'L2_FTN2';TOKEN='CX-FTN2-20261106-V1'
def read(p):return json.loads(Path(p).read_text())
def latest(d):return sorted(d.glob('u*/COMMITTED.json'))[-1]
def held(p):
 with p.open('a+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  return False
def state(stage,**kw):
 atomic_json(ROOT/'SWITCH_STATUS.json',dict(stage=stage,pid=os.getpid(),unix=time.time(),**kw))
 with (ROOT/'switch.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,unix=time.time(),**kw))+'\n')
def audit():
 rows=[json.loads(l) for l in (OUT/'updates.jsonl').read_text().splitlines() if l.strip()]
 receipts=list(OUT.glob('REWARD_RECEIPT_*.json'));assert receipts
 for f in receipts:assert read(f)['version']=='cx-ftn2-continuation-frontier-v1'
 for r in rows:
  e=r['exposure']['bun'];n=r['actual_environment_steps_this_update'];assert r['finite']
  assert e['r3_f2_variant']==2*n and e['r3_f2_no_sync_chain']==n
  assert abs(e['r3_candidate'][5]+e['r3_f2_clawback']-e['r3_f2_frontier_candidate'])<.01
  assert abs(.03*e['r3_f2_frontier_event']-e['r3_f2_frontier_candidate'])<.01
  assert e['r3_f2_frontier_event']<=min(e['r3_f2_selected_safe'],e['r3_f2_post_safe'],e['r3_f2_c_eligible'],e['r3_f2_strict_crossing'])
 assert sum(r['exposure']['bun']['r3_f2_frontier_event'] for r in rows)>0,'no native-safe positive event'
 atomic_json(ROOT/'REWARD_ROLLOUT_ACCEPTANCE.json',dict(status='PASS',rows=len(rows),worker_pids=[read(f)['pid'] for f in receipts],receipts={str(f):sha(f) for f in receipts},
  first_update=rows[0],last_update=rows[-1],positive_frontier=sum(r['exposure']['bun']['r3_f2_frontier_event'] for r in rows),no_sync_chain=True,fields_crosschecked=True,unix=time.time()))
 return rows[-1]
def run(phase,*flags,timeout=1200):
 p=read(PROTOCOL);c=manifest(dict(module='scripts.cx_ftn2_train_v1',gpu=4,args=[str(PROTOCOL),'L2_FTN2',str(p['seed'])]+list(flags)))
 with (OUT/f'{phase}.log').open('a') as log:
  child=subprocess.Popen(['taskset','-c','48-55']+command('training',c),cwd=public_project(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  atomic_json(OUT/'SUPERVISOR.json',dict(status='RUNNING',owner='CX',pid=os.getpid(),worker_pid=child.pid,gpu=4,cpus='48-55',protocol=str(PROTOCOL),phase=phase,unix=time.time()))
  state(phase,worker_pid=child.pid,config=c,deadline_seconds=timeout)
  end=time.monotonic()+timeout
  try:
   while child.poll() is None:
    if time.monotonic()>end:raise TimeoutError(f'{phase} exceeded {timeout}s')
    time.sleep(5)
  except BaseException:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise
  assert child.returncode==0,(phase,child.returncode)
def tests():
 for name in ('UNIT_ANTIHACK','NATIVE_INTEGRATION','COLLECTOR_ABI'):
  r=read(ROOT/'tests'/f'{name}.json');assert r['status']=='PASS',name
  for f,h in r.get('sources',r.get('reward_sources',{})).items():assert sha(f)==h,(name,f)
 for f,h in read(PROTOCOL)['source_sha256'].items():assert sha(f)==h,f
def main():
 lock=(ROOT/'switch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tests();oldsuper=read(OLD/'SUPERVISOR.json')['pid'];oldpid=int(oldsuper)
 cmd=(Path('/proc')/str(oldpid)/'cmdline').read_bytes().decode();assert 'model_diagnostics.py' in cmd
 starttime=(Path('/proc')/str(oldpid)/'stat').read_text().split()[21]
 oldcp=latest(OLD);old_u=read(oldcp)['update'];marker=OLD/'USER_STOP.json'
 assert not marker.exists()
 stopped=False;marked=False;success=False
 try:
  # Pause only the owner supervisor, never its healthy worker or CPU evaluations.
  os.kill(oldpid,signal.SIGSTOP);stopped=True
  state('WAIT_NEW_COMMITTED',previous_update=old_u,old_supervisor=oldpid,old_supervisor_starttime=starttime)
  deadline=time.monotonic()+1200
  while read(latest(OLD))['update']<=old_u:
   assert time.monotonic()<deadline,'old FTN no new COMMITTED within 1200s'
   time.sleep(5)
  boundary=latest(OLD);cm=read(boundary)
  assert sha(boundary.parent/'state.pkl')==cm['state_sha256'] and sha(boundary.parent/'policy.pt')==cm['policy_sha256']
  atomic_json(marker,dict(owner=TOKEN,reason='authorized committed-boundary FTN2 switch; old state retained',checkpoint=str(boundary.parent),unix=time.time()));marked=True
  state('OLD_BOUNDARY_STOP',committed=str(boundary.parent),update=cm['update'])
  deadline=time.monotonic()+180
  while held(BASE/'GPU4.lock') or held(OLD/'worker.lock'):
   assert time.monotonic()<deadline,'old GPU4 worker did not release locks'
   time.sleep(3)
  run('smoke2','--smoke',timeout=900);audit()
  run('fresh2','--verify-restore',timeout=900)
  assert read(OUT/'RESTORE.json')['full_Actor_Critic_Adam_env_aux_RNG_league_exact']
  state('SMOKE_RESTORE_PASS',latest=audit()['new_update'])
  # Keep the old supervisor parked until the bounded trial ends, for a reliable same-owner fallback.
  success=True
  for target in (19382,19462):
   run(f'formal_to{target}','--until',str(target),timeout=4000);row=audit()
   run(f'fresh_u{target}','--verify-restore',timeout=900)
   assert read(OUT/f'RESTORE_u{target:06d}.json')['full_Actor_Critic_Adam_env_aux_RNG_league_exact']
   state('SEGMENT_ACCEPTED',update=target,finite=row['finite'],envsteps_s=row['env_per_second'])
   cfg=manifest(dict(module='scripts.cc_strong_eval_v1',args=['point',f'L2_FTN2@u{target:06d}','--seats','0,1','--opponents','G_frozen'],env=dict(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',CC_EVAL_WORKERS='4')))
   with (ROOT/'evaluation.log').open('a') as log:
    ep=subprocess.Popen(command('evaluation',cfg),cwd=public_project(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   atomic_json(ROOT/f'EVAL_LAUNCH_u{target:06d}.json',dict(pid=ep.pid,config=cfg,unix=time.time()))
  os.kill(oldpid,signal.SIGCONT);stopped=False
  atomic_json(OUT/'SUPERVISOR.json',dict(status='COMPLETED_BOUNDED',owner='CX',pid=os.getpid(),gpu=4,protocol=str(PROTOCOL),latest=19462,unix=time.time()))
  state('COMPLETED_BOUNDED',latest=19462)
 except BaseException as e:
  state('FAILED',error=f'{type(e).__name__}: {e}',switch_accepted=success)
  # On ANY trial failure retain all trial artifacts, return card to original healthy FTN.
  if marked and marker.exists() and read(marker).get('owner')==TOKEN:marker.unlink()
  if stopped:os.kill(oldpid,signal.SIGCONT);stopped=False
  # Original supervisor may have exited after acceptance. Relaunch its same generic config only if gone.
  if not (Path('/proc')/str(oldpid)).exists():
   cfg='local_config/20261106/parameters_01105.json'
   with (ROOT/'fallback.log').open('a') as log:
    p=subprocess.Popen(command('diagnostics',cfg),cwd=public_project(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   state('FALLBACK_RELAUNCHED',pid=p.pid,config=cfg)
  raise
 finally:
  if stopped:os.kill(oldpid,signal.SIGCONT)
if __name__=='__main__':main()
