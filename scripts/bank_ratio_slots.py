"""Single campaign-owned GPU slot controller, bounded Bank-ratio queue only."""
from __future__ import annotations
import json,os,subprocess,sys,time,statistics
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.bank_ratio_slot_train import B,CP,read,notify
RUNS=[(a,s) for s in (2026103151,2026103152,2026103153) for a in ('control','reduced')]


def rows(out):
    f=out/'updates.jsonl'
    return [json.loads(v) for v in f.read_text().splitlines()] if f.exists() else []


def rate(out,start,count=8):
    selected=[r for r in rows(out) if start<=r['new_update']<start+count]
    return statistics.median(r['env_per_second'] for r in selected) if len(selected)==count else None


def main():
    guard=lock(B/'slot_controller.lock');children={};streams=[];last_sample=0;idle_since=None;last_idle=0
    state=read(B/'SLOT_STATE.json') if (B/'SLOT_STATE.json').exists() else dict(stage='BASELINE',concurrency=6,baseline=None)
    def out(i):a,s=RUNS[i];return B/'formal'/f'{a}_s{s}'
    def command(i,target,**extra):atomic_json(out(i)/'COMMAND.json',dict(target_update=target,**extra))
    def alive_worker(i):
        a,s=RUNS[i];wanted=['-m','scripts.bank_ratio_slot_train',a,str(s),'--devices',str(state.get('devices',1))]
        found=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:argv=(p/'cmdline').read_bytes().decode().split('\0')
            except OSError:continue
            if len(argv)>len(wanted) and Path(argv[0]).name.startswith('python') and argv[1:1+len(wanted)]==wanted and '--verify' not in argv:found.append(int(p.name))
        if len(found)>1:raise RuntimeError('Duplicate Bank slot worker')
        return found[0] if found else None
    def launch(i,slot=None):
        if alive_worker(i):return
        if (out(i)/'COMPLETE.json').exists():return
        devices=state.get('devices',1);slot=i if slot is None else slot
        group=list(range(16+3*slot,19+3*slot)) if devices==1 else list(range(16,34))
        gpu=str(slot) if devices==1 else '0,1,2,3,4,5,6,7'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        stream=(B/f'{out(i).name}_d{devices}.log').open('a');streams.append(stream)
        proc=subprocess.Popen(['taskset','-c',','.join(map(str,group)),sys.executable,'-m','scripts.bank_ratio_slot_train',*map(str,RUNS[i]),'--devices',str(devices)],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children[i]=proc
        atomic_json(out(i)/'ASSIGNMENT.json',dict(GPU_slot=slot,devices=devices,cpu_cores=group,owned_by='existing campaign Bank-slot child',unix=time.time()))
        notify('SLOT_STARTED',dict(run=out(i).name,slot=slot,devices=devices,awaiting_real_update=True))
    def wait_tick(indices,targets):
        nonlocal last_sample,idle_since,last_idle
        if any((CP/p).exists() for p in ('USER_STOP.json','USER_PAUSE.json')) or (B/'USER_STOP.json').exists():
            for i in indices:command(i,0,exit_at_boundary=True)
            state['stage']='USER_PAUSED';atomic_json(B/'SLOT_STATE.json',state);raise SystemExit(0)
        for i in indices:
            if i in children and children[i].poll() is not None and children[i].returncode:
                atomic_json(B/'TECHNICAL_BLOCK.json',dict(run=out(i).name,code=children[i].returncode,stage=state['stage'],unix=time.time()))
                notify('TECHNICAL_BLOCK',dict(run=out(i).name,log=str(B/f'{out(i).name}_d{state.get("devices",1)}.log')))
                raise RuntimeError('Worker failed; atomic checkpoints preserved')
        if time.time()-last_sample>10:
            last_sample=time.time()
            sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
            with (B/'GPU_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),stage=state['stage'],samples=sample.splitlines()))+'\n')
            free=all(float(r.split(',')[1])==0 for r in sample.splitlines())
            idle_since=(idle_since or time.time()) if free else None
            if idle_since and time.time()-idle_since>600 and time.time()-last_idle>600:
                notify('GPU_IDLE_GT_10_MIN',dict(stage=state['stage'],reason='inspect worker progress and compile; no synthetic load'));last_idle=time.time()
        progress={out(i).name:(read(out(i)/'PROGRESS.json') if (out(i)/'PROGRESS.json').exists() else {'iterations':0}) for i in range(6)}
        atomic_json(B/'SLOT_PROGRESS.json',dict(stage=state['stage'],iterations=sum(v['iterations'] for v in progress.values()),runs=progress,unix=time.time()))
        atomic_json(B/'SLOT_STATE.json',state)
        return all((out(i)/f'u{targets[i]:06d}'/'COMMITTED.json').exists()
            and (out(i)/f'NEXT_STEP_u{targets[i]:06d}.json').exists() for i in indices)
    def wait(indices,targets):
        while not wait_tick(indices,targets):time.sleep(3)
    # Baseline has no other training worker. Pre-existing u1..10 is never rerun.
    if state['stage']=='BASELINE':
        command(0,10);launch(0);wait([0],{0:10})
        baseline=rate(out(0),3);assert baseline and baseline>0
        state.update(baseline=baseline,stage='RESTORE')
        atomic_json(B/'BASELINE.json',dict(passed=True,B=baseline,warm_updates=list(range(3,11)),run=out(0).name))
        notify('BASELINE_COMPLETE',dict(B=baseline,run=out(0).name,warm_updates=8))
    if state['stage']=='RESTORE':
        if not (out(0)/'RESTORE.json').exists():
            stream=(B/'FRESH_RESTORE.log').open('a')
            result=subprocess.run(['taskset','-c','16,17,18',sys.executable,'-m','scripts.bank_ratio_slot_train',*map(str,RUNS[0]),'--devices','1','--verify'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=stream,stderr=subprocess.STDOUT)
            stream.close();assert result.returncode==0,'Fresh-process restore failed'
        state['stage']='CONCURRENT6';atomic_json(B/'SLOT_STATE.json',state)
    if state['stage']=='CONCURRENT6':
        targets={i:18 if i==0 else 10 for i in range(6)}
        for i in range(6):command(i,targets[i]);launch(i)
        wait(list(range(6)),targets)
        state.update(stage='CONCURRENT6_SYNC',sync_starts={str(i):targets[i]+1 for i in range(6)})
        atomic_json(B/'SLOT_STATE.json',state)
    if state['stage']=='CONCURRENT6_SYNC':
        starts={int(i):u for i,u in state['sync_starts'].items()};targets={i:starts[i]+7 for i in range(6)}
        # All six have already compiled collector, updater and next-step token.
        # Release one common window only after the last cold slot is ready.
        for i in range(6):command(i,targets[i]);launch(i)
        wait(list(range(6)),targets)
        speeds={out(i).name:rate(out(i),starts[i]) for i in range(6)}
        passed=all(v and v>=.85*state['baseline'] for v in speeds.values())
        atomic_json(B/'CONCURRENT6_ACCEPTANCE.json',dict(passed=passed,B=state['baseline'],threshold=.85,speeds=speeds))
        notify('CONCURRENT6_SMOKE',dict(passed=passed,speeds=speeds,B=state['baseline']))
        state['stage']='FORMAL6' if passed else 'CONCURRENT3'
        atomic_json(B/'SLOT_STATE.json',state)
    if state['stage']=='CONCURRENT3':
        speeds={}
        for group in (list(range(3)),list(range(3,6))):
            targets={i:rows(out(i))[-1]['new_update']+8 for i in group}
            starts={i:targets[i]-7 for i in group}
            for i in group:command(i,targets[i]);launch(i)
            wait(group,targets)
            speeds.update({out(i).name:rate(out(i),starts[i]) for i in group})
        passed=all(v>=.85*state['baseline'] for v in speeds.values())
        atomic_json(B/'CONCURRENT3_ACCEPTANCE.json',dict(passed=passed,B=state['baseline'],speeds=speeds))
        notify('CONCURRENT3_SMOKE',dict(passed=passed,speeds=speeds))
        state['stage']='FORMAL3' if passed else 'FALLBACK8'
    if state['stage']=='FALLBACK8':
        for i in range(6):command(i,0,exit_at_boundary=True)
        while any(alive_worker(i) for i in range(6)):time.sleep(2)
        state.update(stage='FORMAL8_SERIAL',devices=8);atomic_json(B/'SLOT_STATE.json',state)
        notify('FALLBACK_8_SERIAL',dict(reason='6 and3 smoke below .85 baseline; retain all checkpoints'))
    if state['stage'] in ('FORMAL6','FORMAL3','FORMAL8_SERIAL'):
        width={'FORMAL6':6,'FORMAL3':3,'FORMAL8_SERIAL':1}[state['stage']]
        pending=[i for i in range(6) if not (out(i)/'COMPLETE.json').exists()]
        while pending:
            group=pending[:width]
            for i in group:command(i,320);launch(i)
            while not all((out(i)/'COMPLETE.json').exists() for i in group):
                wait_tick(group,{i:320 for i in group})
                for i in group:
                    r=rows(out(i))
                    if len(r)>=20 and state['stage']!='FORMAL8_SERIAL':
                        speed=statistics.median(x['env_per_second'] for x in r[-20:])
                        if speed<.85*state['baseline']:
                            atomic_json(B/'THROUGHPUT_GUARD.json',dict(pause_new_entries=True,run=out(i).name,speed=speed,B=state['baseline'],unix=time.time()))
                time.sleep(3)
            for i in group:
                complete=read(out(i)/'COMPLETE.json');cp=Path(complete['checkpoint'])
                assert sha(cp/'state.pkl')==complete['state_sha256'] and sha(cp/'policy.pt')==complete['policy_sha256']
            pending=[i for i in pending if i not in group]
            if pending and (B/'THROUGHPUT_GUARD.json').exists():
                notify('NEW_ENTRIES_PAUSED_THROUGHPUT',dict(remaining=[out(i).name for i in pending]));raise RuntimeError('Throughput guard requires diagnosis; current runs completed')
        state['stage']='TRAINING_COMPLETE';atomic_json(B/'SLOT_STATE.json',state)
        atomic_json(B/'TRAINING_COMPLETE.json',dict(passed=True,runs=[read(out(i)/'COMPLETE.json') for i in range(6)],evaluation_pending=True))
        notify('ALL_SIX_TRAINING_COMPLETE',dict(environment_steps=6*5242880,CPU_evaluation_async=True))

if __name__=='__main__':main()
