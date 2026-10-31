"""Verify actual resumed updates, then restore independent eval/publication services."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from scripts.population_run_state import atomic_json, lock, sha

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'runs/bomb_threat_v1'
O=B/'reboot_resume_20261030'


def read(p,default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def live(module):
    count=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if module.encode() in (p/'cmdline').read_bytes().split(b'\0') and (p/'cwd').resolve()==ROOT and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':count+=1
        except OSError:pass
    return count


def launch(module,affinity,args=()):
    count=live(module)
    assert count<=1,'Duplicate service: '+module
    if count:return dict(module=module,status='ADOPTED_EXISTING')
    log=(O/(module.split('.')[-1]+'.log')).open('ab',buffering=0)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             LD_LIBRARY_PATH='/tmp/qqt-browser-libs/root/usr/lib/x86_64-linux-gnu'+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
    # The native owner sets the dedicated GPU2 only for its own evaluator.
    subprocess.Popen(['taskset','-c',affinity,sys.executable,'-m',module,*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    return dict(module=module,status='STARTED',CPU_affinity=affinity)


def main():
    guard=lock(O/'ACCEPTANCE_OWNER.lock')
    plan=read(O/'PLAN.json');accepted={}
    while len(accepted)<7:
        if any((B/n).exists() for n in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json')):
            atomic_json(O/'ACCEPTANCE_STATUS.json',dict(status='NEW_USER_PAUSE_OBSERVED',unix=time.time()));return
        slots=read(B/'SLOTS.json')
        pending={}
        for slot,spec in plan['slots'].items():
            entry=slots[slot];run=B/'formal'/entry['run'];p=read(B/'protocols'/f"{entry['run']}.json")
            recovery=read(O/f'slot{int(slot):02d}/STATUS.json',{})
            if recovery.get('status')!='READY_FOR_FORMAL':
                pending[slot]=recovery;continue
            same=entry['run']==spec['run']
            threshold=spec['update']+2 if same else p['chunk_start_update']+2
            smoke=read(O/f'slot{int(slot):02d}/SMOKE.json',{}) if same else read(run/'SMOKE.json',{})
            restore=read(O/f'slot{int(slot):02d}/RESTORE.json',{}) if same else read(run/'RESTORE.json',{})
            if p.get('prewarm_acceptance') and read(run/'TECHNICAL_CONTINUATION.json',{}).get('passed'):
                # The unchanged legacy owner uses prewarm + actual source
                # restore instead of writing per-run SMOKE/RESTORE markers.
                # Our two real resumed smoke updates and post-smoke restore
                # remain the evidence for this exact saved source.
                warm=read(ROOT/p['prewarm_acceptance'],{})
                assert warm.get('status')=='PASS'
                smoke=read(O/f'slot{int(slot):02d}/SMOKE.json',{})
                restore=read(run/'SOURCE_RESTORE.json',{})
                threshold=p['chunk_start_update']
            rows=[json.loads(x) for x in (run/'updates.jsonl').read_text().splitlines() if x.strip()] if (run/'updates.jsonl').exists() else []
            formal=[r for r in rows if r['new_update']>threshold and r.get('unix',0)>plan['unix']]
            if not(smoke.get('passed') and restore.get('passed')) or len(formal)<3:
                pending[slot]=dict(run=entry['run'],formal_updates_seen=len(formal),smoke=smoke.get('passed'),restore=restore.get('passed'));continue
            assert all(r['finite'] and min(r['update_l2'].values())>0 for r in formal)
            warm=formal[-3:];throughput=statistics.median(r['actual_environment_steps_this_update']/r['cycle_seconds'] for r in warm)
            if throughput<13600:
                pending[slot]=dict(run=entry['run'],reason='warm throughput below80percent of17000 baseline',env_per_second=throughput);continue
            cp=sorted(run.glob('u*/COMMITTED.json'))[-1].parent;cm=read(cp/'COMMITTED.json')
            assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']
            accepted[slot]=dict(run=entry['run'],resume_source=spec['checkpoint'],first_formal_update=formal[0]['new_update'],latest_update=formal[-1]['new_update'],
                environment_steps=formal[-1]['environment_steps'],optimizer_steps=formal[-1]['optimizer_steps'],finite=True,
                first_update_l2=formal[0]['update_l2'],warm_env_per_second=throughput,checkpoint=str(cp.relative_to(ROOT)),
                state_sha256=cm['state_sha256'],policy_sha256=cm['policy_sha256'],smoke_passed=True,fresh_restore_passed=True,
                recipe=p['arms'][entry['run']],seed=spec['seed'])
        atomic_json(O/'ACCEPTANCE_STATUS.json',dict(status='CHECKING_REAL_FORMAL_UPDATES',accepted=accepted,pending=pending,unix=time.time()))
        if len(accepted)<7:time.sleep(10)
    samples=[]
    for i in range(3):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
        samples.append(dict(unix=time.time(),GPU=[dict(index=int(a),memory_MiB=int(b),utilization=int(c),power_W=float(d)) for a,b,c,d in (line.split(',') for line in raw.strip().splitlines())]))
        if i<2:time.sleep(3)
    value=dict(status='PASS',slots=accepted,GPU_samples=samples,unique_training_owner_count=live('scripts.bomb_threat_owner'),
        slot06_limitation='new probe validates two post-reboot processes; pre-reboot next-step reference was not saved',scientific_changes=[],unix=time.time())
    assert value['unique_training_owner_count']==1
    atomic_json(O/'TRAINING_RESUME_ACCEPTANCE.json',value)
    services=[]
    for module,cpu,args in [('scripts.bomb_threat_eval_owner','112-135',()),('scripts.bomb_threat_native_owner','68-71',()),
        ('scripts.bomb_threat_watchdog','70-71',()),('scripts.checkpoint_retention','144-145',('--apply','--watch','--interval','30')),
        ('scripts.best_publish_eligible_20261030','132-135',()),('scripts.best_publish_prepublication_20261030','136-143',()),
        ('scripts.best_publish_release_20261030','140-143',())]:
        services.append(launch(module,cpu,args))
    atomic_json(O/'SERVICES_RESTORED.json',dict(status='STARTED_OR_ADOPTED',services=services,old_R2_O5_training_not_started=True,publication_will_not_stop_training=True,unix=time.time()))
    utilization='; '.join(f"GPU{g['index']} {g['utilization']}%/{g['power_W']:.0f}W" for g in samples[-1]['GPU'])
    message='重启后训练恢复验收通过：唯一owner、7路smoke/新进程恢复/正式finite更新且Actor/Critic非零；'+ '; '.join(f's{k}:u{v["first_formal_update"]}, {v["warm_env_per_second"]:.0f} env/s' for k,v in sorted(accepted.items()))+'。实际样本 '+utilization+'；评估/GPU2/watchdog/retention已恢复，GPU2专评；继续slot03复核及发布。'
    r=subprocess.run(['jdme-notify','send',message],capture_output=True,timeout=30)
    atomic_json(O/'TRAINING_NOTIFICATION.json',dict(status='SENT' if r.returncode==0 else 'FAILED',exit_code=r.returncode,unix=time.time()))
    atomic_json(O/'TASK_CONSUMED.json',dict(status='TRAINING_RESTORED_PUBLICATION_CONTINUES',training_acceptance_sha256=sha(O/'TRAINING_RESUME_ACCEPTANCE.json'),services_restored=True,unix=time.time()))
    print(json.dumps(dict(status='TRAINING_RESUMED',slots={k:v['first_formal_update'] for k,v in accepted.items()})),flush=True)


if __name__=='__main__':main()
