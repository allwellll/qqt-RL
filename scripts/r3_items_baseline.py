"""Resumable CPU-only R3 full-game baseline, independent from R1/R2 owners."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha

OUT=Path('runs/r3_items_route_20261030')
SOURCES=['scripts/r3_items_metrics.js','scripts/r3_items_eval.js','scripts/bun_native_v2_eval.js',
         'scripts/bun_native_v2_metrics.js','scripts/bomb_threat_eval.js','web/sim-bun-native-v2.js']


def read(p):return json.loads(Path(p).read_text())


def freeze():
    dest=OUT/'BASELINE_PROTOCOL.json'
    if dest.exists():return read(dest)
    old=read('runs/bun_native_v2_20261030/PROTOCOL.json')
    cps=sorted(Path('runs/bun_native_v2_20261030/formal/r2_a_c000000').glob('u*/COMMITTED.json'))
    assert cps
    source=read(cps[-1]);cp=Path(source['checkpoint'])
    assert sha(cp/'state.pkl')==source['state_sha256'] and sha(cp/'policy.pt')==source['policy_sha256']
    models={'historical_slot00_u2400':old['model'],'r2_a_latest_at_freeze':source}
    conditions={}
    for i,c in enumerate(('hard','normal','coop_hard')):
        v=old['conditions'][c];spawn=read(v['spawn_manifest']);seed=2026208100+i*1000
        spawn.update(seed=seed,spawn_cells=spawn['spawn_cells'][:128])
        f=OUT/'spawns'/f'{c}.json';atomic_json(f,spawn)
        conditions[c]=dict(v,seed=seed,spawn_manifest=str(f),spawn_sha256=sha(f))
    amendment=dict(status='FROZEN',model_identity='task explicitly names historical slot00 c34 u2400; retained as named, not represented as current online model',
        current_online='R1 slot03 c36 u2760; baseline not silently substituted',
        sampling='production greedy, teacher off, map806,128 games/condition/model=64 per seat',
        geometry='prior natural frozen birth list,128 entries may repeat geometries; games not independent maps',
        critical_path='carrier OR geodesic distance to home/loose decreasing OR observed chase distance decreasing at <=4; geometric proxy, intention unavailable',
        glue='5-second subsequent kill/pickup windows observational; causal effect unavailable without matched branch',
        banana='completed slide path/native carry speed minus actual elapsed, positive home progress required; estimate, not empirical Q or counterfactual',
        timing='8-tick adjacent placement gap starts batch; 5sec=50tick,3sec=30tick at10Hz',
        R2_precedence='R3 glue event reward never restores legacy defensive kill shaping forbidden by custody amendment',
        R3_parent='first complete atomic checkpoint from revised R2-b, including counted smoke u2722; freeze by chronological first update, no peak selection',
        no_rule_or_training_change=True,unix=time.time())
    atomic_json(OUT/'AMENDMENT01_BASELINE_IDENTITY_AND_MEASUREMENT.json',amendment)
    result=dict(status='FROZEN',task='.hermes/TASK_r3_items_route_chain_20261030.md',task_sha256=sha(Path('.hermes/TASK_r3_items_route_chain_20261030.md')),
        models=models,conditions=conditions,games_per_condition=128,total_games=768,mode='greedy',teacher=False,
        workers=8,cpu_cores=list(range(176,184)),CUDA_VISIBLE_DEVICES='',source_sha256={f:sha(Path(f)) for f in SOURCES},
        amendment=str(OUT/'AMENDMENT01_BASELINE_IDENTITY_AND_MEASUREMENT.json'),unix=time.time())
    atomic_json(dest,result)
    with Path('docs/training_experiment_ledger.md').open('a') as f:
        f.write('\n## R3-ITEMS-ROUTE-CHAIN-20261030\n\n用户任务已消费。先768完整CPU生产JS局：指定历史slot00 c34 u2400与冻结时最新R2-a，3类对手、每席位每档64局、greedy、老师关闭。'
          '旧网页称谓已更正，不择点；独立seed/来源/SHA/分母见 `runs/r3_items_route_20261030/BASELINE_PROTOCOL.json`。'
          '随后独立R3两seed2026208001/2026208002、640更新/chunk，父固定为custodyv2 b首原子点u2722；实现、事件防hack、零输入扩展、恢复与资源验收通过前不启动。'
          'R1/R2不修改，CPU176–183，GPU优先2且既有线吞吐下降不得>15%；旧O5/储备继续暂停，无发布。\n')
    return result


def model_for(key,model):
    dest=OUT/'exports'/key;receipt=dest/'EXPORT.json'
    if receipt.exists():
        d=read(receipt);assert sha(Path(d['model_json']))==d['model_sha256'];return d
    if 'model_json' in model:
        assert sha(Path(model['model_json']))==model['model_sha256'];return model
    cp=Path(model['checkpoint']);assert sha(cp/'policy.pt')==model['policy_sha256']
    dest.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    code='from pathlib import Path;from scripts.je08_bun_authority import export;import sys;export(Path(sys.argv[1]),Path(sys.argv[2]).resolve())'
    with (dest/'EXPORT.log').open('ab') as log:
        subprocess.run(['taskset','-c','176-179',sys.executable,'-c',code,str(cp/'policy.pt'),str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    path=dest/'model.json';assert read(path)['meta']['checkpoint_sha256']==model['policy_sha256']
    result=dict(model,model_json=str(path),model_sha256=sha(path));atomic_json(receipt,result);return result


def collect(job,protocol,core):
    key,c,start=job;v=protocol['conditions'][c];model=protocol['models'][key]
    target=OUT/'baseline'/key/c/f'shard_{start:03d}.json';target.parent.mkdir(parents=True,exist_ok=True)
    guard=lock(Path(str(target)+'.lock'));accept=Path(str(target)+'.ACCEPTED.json')
    if accept.exists():
        assert all(sha(Path(f))==h for f,h in read(accept)['files'].items());return
    side=Path(str(target)+'.outcome.json')
    if not side.exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            QQT_STOCK_RULE='stock-window-v1',BUN_V2_RULE='bun-native-v2',BUN_V2_TIER=v['tier'],BUN_V2_OPPONENT=v['opponent'],
            JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='128',JE10_ROUTE_START=str(start),JE10_ROUTE_END=str(start+8),JE10_ROUTE_SEED_START=str(v['seed']))
        with target.with_suffix('.log').open('ab') as log:
            subprocess.run(['taskset','-c',str(core),'node','scripts/r3_items_eval.js','greedy',model['model_json'],str(target),
                'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',v['spawn_manifest']],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    doc=read(side);es=doc['episodes']
    assert doc['rule_version']=='bun-native-v2' and not doc['teacher'] and doc['mode']=='greedy'
    assert len(es)==8 and all(e['done'] and 'r3' in e for e in es)
    assert [e['seed'] for e in es]==list(range(v['seed']+start,v['seed']+start+8))
    assert [e['seat'] for e in es]==[(start+i)%2 for i in range(8)]
    assert read(target)['model_json_sha256']==model['model_sha256']
    atomic_json(accept,dict(status='PASS',games=8,files={str(f):sha(f) for f in (target,side)},unix=time.time()))


def main():
    guard=lock(OUT/'BASELINE_OWNER.lock');p=freeze()
    assert all(sha(Path(f))==h for f,h in p['source_sha256'].items())
    p=dict(p,models={k:model_for(k,v) for k,v in p['models'].items()});atomic_json(OUT/'BASELINE_MODELS.json',p['models'])
    jobs=[(k,c,i) for i in range(0,128,8) for k in p['models'] for c in p['conditions']]
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending={};done=set()
        while len(done)<len(jobs):
            for f in list(pending):
                if f.done():job,_=pending.pop(f);f.result();done.add(job)
            for job in jobs:
                if len(pending)>=8:break
                if job in done or any(v[0]==job for v in pending.values()):continue
                core=next(c for c in p['cpu_cores'] if c not in [v[1] for v in pending.values()])
                pending[pool.submit(collect,job,p,core)]=(job,core)
            atomic_json(OUT/'EXECUTION_STATUS.json',dict(status='BASELINE_RUNNING',completed_games=len(done)*8,budget=768,active=[v[0] for v in pending.values()],unix=time.time()))
            time.sleep(3)
    from scripts.r3_items_report import report
    report(p)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(OUT/'BASELINE_BLOCKED.json',dict(status='BLOCKED',error=type(exc).__name__,reason='inspect BASELINE_OWNER.log and shard logs',unix=time.time()))
        raise
