"""Passive receipt/notification watcher; never starts a GPU worker."""
import json
import os
import subprocess
import time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.r3_selfplay_owner_20261031 import tail_rows

B=Path('runs/r3_resume_publish_2110_20261031')
def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def notify(key,message):
    f=B/'notifications'/f'{key}.json'
    if f.exists():return
    result=subprocess.run(['jdme-notify','send',message],capture_output=True,check=False)
    atomic_json(f,dict(message=message,returncode=result.returncode,unix=time.time()))

def main():
    guard=lock(B/'WATCH.lock')
    roots=[Path('runs/r3_selfplay_ramp_20261031'),Path('runs/r3_selfplay_ramp_20261031/seed02')]
    while True:
        lines={}
        for root in roots:
            reg=read(root/'REGISTRATION.json');entry=reg['entry'];folder=root/'formal'/entry['run']
            request=read(root/'RESUME_2110.json');rows=tail_rows(folder/'updates.jsonl',16)
            fresh=[r for r in rows if r['new_update']>request['start_update']]
            warm=fresh[1:][-5:]
            seconds=sum(r['cycle_seconds'] for r in warm)
            warm_rate=sum(r['actual_environment_steps_this_update'] for r in warm)/seconds if seconds else None
            commits=sorted(p for p in folder.glob('u*/COMMITTED.json') if int(p.parent.name[1:])>request['start_update'] and (p.parent/'state.pkl').exists() and (folder/f'NEXT_STEP_{p.parent.name}.json').exists())
            if commits:
                cp=commits[-1].parent;cm=read(cp/'COMMITTED.json')
                assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']
                atomic_json(root/'RESUME_2110_ATOMIC_ACCEPTANCE.json',dict(status='PASS',source=cm,next_step_sha256=sha(folder/f'NEXT_STEP_{cp.name}.json'),unix=time.time()))
            stage=read(folder/'CYCLE_STAGE.json',{})
            lines[str(root)]=dict(run=entry['run'],gpu=entry['gpu'],last_update=rows[-1]['new_update'],
                restore_pass=read(root/'RESUME_2110_RESTORE_ACCEPTANCE.json',{}).get('status')=='PASS',
                new_update_pass=read(root/'RESUME_2110_FIRST_UPDATE.json',{}).get('status')=='PASS',
                warm_env_per_second=warm_rate,warm_updates=[r['new_update'] for r in warm],
                stage=stage,stage_age_seconds=time.time()-stage.get('unix',time.time()),
                atomic_update=read(commits[-1])['update'] if commits else None)
            blocked=read(root/'OWNER_BLOCKED.json')
            if blocked:notify(root.name+'_BLOCKED','R3恢复owner技术阻塞：'+str(root/'OWNER_BLOCKED.json')+'；未对健康训练发信号。')
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,power.draw,memory.used','--format=csv,noheader'],text=True).splitlines()
        with (B/'GPU_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),rows=gpu))+'\n')
        pub=B/'selection/publication';done=read(pub/'DONE.json');failure=read(B/'selection/BLOCKED.json')
        if failure:notify('PUBLICATION_BLOCKED','u4002发布链技术阻塞：'+str(B/'selection/BLOCKED.json')+'；训练继续，未强推或改规则。')
        queue=read(B/'R3_ALLOCATION_QUEUE.json',{})
        controls={}
        for arm,entry in read('runs/r3_items_route_20261030/R3_SLOTS.json',{}).items():
            folder=Path('runs/r3_items_route_20261030/formal')/entry['run'];rows=tail_rows(folder/'updates.jsonl',6) if (folder/'updates.jsonl').exists() else []
            controls[arm]=dict(run=entry['run'],gpu=entry['gpu'],allocation=queue.get(arm,{}),last_update=rows[-1]['new_update'] if rows else None,
                first_formal_pass=read(folder/'FIRST_FORMAL_UPDATE.json',{}).get('status')=='PASS')
        atomic_json(B/'EXECUTION_STATUS.json',dict(status='RUNNING',selfplay=lines,controls=controls,publication=done or read(pub/'STATUS.json',{}),publication_failure=failure,gpu_samples=gpu,unix=time.time()))
        time.sleep(30)

if __name__=='__main__':main()
