"""Own-side paired bot holdout; CPU only and independent from GPU progress."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import json
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json, lock, sha
from scripts.r3_items_eval_owner import summarize

OUT = Path(os.environ.get('R3_SP_NAMESPACE','runs/r3_selfplay_ramp_20261031'))


def cpu_cores():
    return (148,149) if OUT.name=='seed02' else (144,145)


def read(p):
    return json.loads(Path(p).read_text())


def requests():
    registration = OUT/'REGISTRATION.json'
    if not registration.exists():
        return []
    p = read(registration)
    result = [dict(key='parent', source=p['parent_source'], priority=0)]
    for folder in (OUT/'formal').glob('r3_sp*'):
        points = sorted((folder/'eval_queue').glob('*.json'), reverse=True)
        for i, f in enumerate(points):
            r=read(f);result.append(dict(key=folder.name+'_'+f.stem, source=r, priority=1 if i==0 else 3))
        if (folder/'COMPLETE.json').exists():
            r=read(folder/'COMPLETE.json');result.append(dict(key=folder.name+'_final', source=r, priority=1))
    seen=set();unique=[]
    for r in sorted(result,key=lambda r:(r['priority'],-r['source']['update'],r['key'])):
        identity=(r['source']['checkpoint'],r['source']['policy_sha256'])
        if identity in seen:continue
        seen.add(identity);unique.append(r)
    return unique


def export(req):
    folder=OUT/'evaluation'/req['key']/'model';folder.mkdir(parents=True,exist_ok=True)
    proof=folder/'IDENTITY.json'
    if proof.exists():
        x=read(proof);assert sha(Path(x['model_json']))==x['model_sha256'];return x
    cp=Path(req['source']['checkpoint']);assert sha(cp/'policy.pt')==req['source']['policy_sha256']
    with (folder/'EXPORT.log').open('ab') as log:
        subprocess.run(['taskset','-c',','.join(map(str,cpu_cores())),sys.executable,'-c',
            'from scripts.r3_items_export import export;import sys;export(*sys.argv[1:])',str(cp),str(folder)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    x=dict(model_json=str(folder/'model.json'),model_sha256=sha(folder/'model.json'),source=req['source'])
    atomic_json(proof,x);return x


def collect(req, condition, start, model, core):
    p=read(OUT/'PROTOCOL.json');f=OUT/'evaluation'/req['key']/condition/f'shard_{start:03d}.json'
    f.parent.mkdir(parents=True,exist_ok=True);guard=lock(Path(str(f)+'.lock'))
    proof=Path(str(f)+'.ACCEPTED.json');side=Path(str(f)+'.outcome.json')
    if proof.exists():
        x=read(proof);assert all(sha(Path(k))==v for k,v in x['files'].items());return
    source=p['evaluation']['holdout'];assert sha(Path(source['spawn_manifest']))==source['spawn_sha256']
    from scripts.bun_native_v2_owner import processes
    while list(processes('scripts/r3_items_candidate_eval.js',str(f))):time.sleep(3)
    if not (f.exists() and side.exists()):
        env=dict(os.environ,QQT_STOCK_RULE='stock-window-v1',BUN_V2_RULE='bun-native-v2', BUN_V2_TIER='hard',
            BUN_V2_OPPONENT='bun.coop_hunter' if condition=='coop_hard' else 'bun.hunter',
            JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='128',JE10_ROUTE_START=str(start),
            JE10_ROUTE_END=str(start+8),JE10_ROUTE_SEED_START=str(source['seed']))
        with f.with_suffix('.log').open('ab') as log:
            subprocess.run(['taskset','-c',str(core),'node','scripts/r3_items_candidate_eval.js','greedy',model['model_json'],str(f),
                'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',source['spawn_manifest']],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    doc=read(side);episodes=doc['episodes']
    assert read(f)['model_json_sha256']==model['model_sha256']
    assert doc['mode']=='greedy' and doc['rule_version']=='bun-native-v2' and not doc['teacher']
    assert len(episodes)==8 and [e['seed'] for e in episodes]==list(range(source['seed']+start,source['seed']+start+8))
    from scripts.bun_spawn_side import validate
    for i,e in enumerate(episodes):
        assert e['done'] and e['seat']==(start+i)%2 and e['spawn_side_version']=='own-side-native-v1'
        validate(e['spawn_cells']);validate(e['respawn_cells'])
    atomic_json(proof,dict(status='PASS',files={str(x):sha(x) for x in (f,side)},unix=time.time()))


def finish(req):
    folder=OUT/'evaluation'/req['key'];metrics={};files={}
    for condition in ('hunter_hard','coop_hard'):
        episodes=[]
        for start in range(0,128,8):
            f=folder/condition/f'shard_{start:03d}.json';proof=Path(str(f)+'.ACCEPTED.json')
            if not proof.exists():return None
            x=read(proof);assert all(sha(Path(k))==v for k,v in x['files'].items());files.update(x['files'])
            episodes.extend(read(str(f)+'.outcome.json')['episodes'])
        assert len(episodes)==128==len({(e['seed'],e['seat']) for e in episodes})
        metrics[condition]={f'P{s}':summarize(episodes,s) for s in (0,1)}
    result=dict(status='COMPLETE',source=req['source'],metrics=metrics,files=files,mode='greedy',games=256,unix=time.time())
    atomic_json(folder/'COMPLETE.json',result)
    if req['key']!='parent':
        parent=read(OUT/'evaluation/parent/COMPLETE.json');comparisons={}
        for c,seats in metrics.items():
            for seat,value in seats.items():
                base=parent['metrics'][c][seat];assert value['games']==base['games']==64
                delta=value['win_minus_loss']-base['win_minus_loss']
                comparisons[c+'_'+seat]=dict(candidate=value['win_minus_loss'],parent=base['win_minus_loss'],delta=delta,rollback=delta < -6)
        atomic_json(folder/'BOT_GATE.json',dict(status='ROLLBACK' if any(x['rollback'] for x in comparisons.values()) else 'PASS',comparisons=comparisons,unix=time.time()))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'EVAL_OWNER.lock')
    while True:
        if any((root/f).exists() for root in (OUT,Path('runs/bomb_threat_v1'),Path('runs/bun_native_v2_20261030'),Path('runs/iter_pbt_v1'))
               for f in ('USER_STOP.json','USER_PAUSE.json','STOP_FOR_REBOOT.json')):
            atomic_json(OUT/'EVAL_STATUS.json',dict(status='USER_PAUSED',unix=time.time()));return
        if (OUT/'FIXED_HISTORY_EVALUATION_V1.json').exists() and not (OUT/'FIXED_HISTORY_BLOCKED.json').exists():
            from scripts.bun_native_v2_owner import processes
            module='scripts.r3_sp_fixed_history_owner_20261031'
            namespace=str(OUT).encode()
            live=[]
            for process in processes(module):
                try:
                    values=(process/'environ').read_bytes().split(b'\0')
                    identity=next((x.split(b'=',1)[1] for x in values if x.startswith(b'R3_SP_NAMESPACE=')),b'runs/r3_selfplay_ramp_20261031')
                    if identity==namespace:live.append(process)
                except OSError:continue
            if not live:
                with (OUT/'FIXED_HISTORY_OWNER.log').open('ab') as log:
                    subprocess.Popen([sys.executable,'-m',module],stdout=log,stderr=subprocess.STDOUT,
                                     env=dict(os.environ,R3_SP_NAMESPACE=str(OUT)),stdin=subprocess.DEVNULL,start_new_session=True)
        req=next((r for r in requests() if not (OUT/'evaluation'/r['key']/'COMPLETE.json').exists()),None)
        if req is None:
            atomic_json(OUT/'EVAL_STATUS.json',dict(status='CAUGHT_UP_OR_WAIT_PARENT_GATE',unix=time.time()));time.sleep(20);continue
        model=export(req)
        accepted=0
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(collect,req,c,start,model,cpu_cores()[i%2]) for i,(c,start) in enumerate(
                (c,start) for c in ('hunter_hard','coop_hard') for start in range(0,128,8))]
            for future in futures:
                future.result();accepted+=8
                atomic_json(OUT/'EVAL_STATUS.json',dict(status='ACCEPTING_FULL_GAMES',key=req['key'],
                    games_confirmed_in_this_pass=accepted,budget=256,unix=time.time()))
        finish(req)
        atomic_json(OUT/'EVAL_STATUS.json',dict(status='REQUEST_COMPLETE',key=req['key'],unix=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(OUT/'EVAL_BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error),unix=time.time()));raise
