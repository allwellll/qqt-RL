"""Full-game carrier denominators on the already frozen parent seed list."""
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha

B=Path('runs/r4_combat_20261031/carrier_diagnostic_v2')
ROOT=B.parent
def read(p):return json.loads(Path(p).read_text())


def freeze():
    f=B/'PROTOCOL.json'
    if f.exists():return read(f)
    old=read(ROOT/'protocols/diagnostic_v1.json')
    d=dict(old,id='R4-CARRIER-DIAGNOSTIC-V2',status='FROZEN',
        supersedes_metrics_only='new carrier fields; original diagnostic and results remain immutable',
        reason='prior256 full games did not record the required carrier path exposure; recollect with passive instrumentation and verify exact prior joint actions',
        source_sha256={p:sha(Path(p)) for p in ['scripts/r4_carrier_metrics.js','scripts/r4_carrier_diagnostic_eval.js',
            'scripts/r4_combat_metrics.js','scripts/r3_items_candidate_eval.js','scripts/r3_items_metrics.js','web/sim-bun-native-v2.js']},
        amended_authority='.hermes/STEER_2225_carrier_kill.md',unix=time.time())
    atomic_json(f,d);return d


def collect(pair,p):
    start,core=pair;v=p['condition'];target=B/f'shard_{start:03d}.json'
    guard=lock(Path(str(target)+'.lock'));receipt=Path(str(target)+'.ACCEPTED.json')
    if receipt.exists():
        assert all(sha(Path(f))==h for f,h in read(receipt)['files'].items());return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        QQT_STOCK_RULE='stock-window-v1',BUN_V2_RULE='bun-native-v2',BUN_V2_TIER='hard',BUN_V2_OPPONENT='bun.hunter',
        JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='256',JE10_ROUTE_START=str(start),JE10_ROUTE_END=str(start+8),
        JE10_ROUTE_SEED_START=str(v['seed']))
    if not Path(str(target)+'.outcome.json').exists():
        with target.with_suffix('.log').open('ab') as log:
            subprocess.run(['taskset','-c',str(core),'node','scripts/r4_carrier_diagnostic_eval.js',p['mode'],p['model']['model_json'],
                str(target),'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',v['spawn_manifest']],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    es=read(str(target)+'.outcome.json')['episodes']
    assert len(es)==8 and all(e['done'] and 'r4_carrier' in e for e in es)
    assert [e['seed'] for e in es]==list(range(v['seed']+start,v['seed']+start+8))
    assert [e['seat'] for e in es]==[(start+i)%2 for i in range(8)]
    assert read(target)['model_json_sha256']==p['model']['model_sha256']
    old=read(ROOT/'diagnostic'/f'shard_{start:03d}.json.outcome.json')['episodes']
    assert [e['r4']['joint_actions_sha256'] for e in es]==[e['r4']['joint_actions_sha256'] for e in old]
    atomic_json(receipt,dict(status='PASS',games=8,passive_joint_actions_exact=True,
        files={str(f):sha(f) for f in [target,Path(str(target)+'.outcome.json')]},unix=time.time()))


def finish(p):
    es=[];files={}
    for start in range(0,256,8):
        f=B/f'shard_{start:03d}.json';proof=read(str(f)+'.ACCEPTED.json')
        for path,h in proof['files'].items():assert sha(Path(path))==h;files[path]=h
        es+=read(str(f)+'.outcome.json')['episodes']
    assert len(es)==len({(e['seed'],e['seat']) for e in es})==256
    results={}
    keys=['carrier_ticks','own_actionable_ticks','enemy_free_ticks','reachable_path_ticks','interception_ticks',
        'free_interception_ticks','exposed_carrier_lives','carrier_deaths','carrier_causal_kills','enemy_deliveries',
        'own_deliveries','stock_pickups','completed_antifarm_cycles']
    for seat in (0,1):
        group=[e for e in es if e['seat']==seat]
        v={k:sum(e['r4_carrier'][k][seat] for e in group) for k in keys};v['games']=len(group)
        v['interception_rate']=v['interception_ticks']/v['own_actionable_ticks'] if v['own_actionable_ticks'] else None
        v['carrier_killed_rate']=v['carrier_causal_kills']/v['exposed_carrier_lives'] if v['exposed_carrier_lives'] else None
        for key in ['enemy_deliveries','own_deliveries','completed_antifarm_cycles']:v[key+'_per_game']=v[key]/len(group)
        results[f'P{seat}']=v
    d=dict(status='COMPLETE',games=256,results=results,files=files,
        definitions=es[0]['r4_carrier']['definitions'],limitations=es[0]['r4_carrier']['limitations'],
        passive_joint_actions_exact=True,protocol_sha256=sha(B/'PROTOCOL.json'),unix=time.time())
    atomic_json(B/'COMPLETE.json',d)
    with Path('reports/r4_combat_20261031.md').open('a') as f:
        f.write('\n## u4002 持包者拦截基线（22:25修订）\n\n生产JS Hunter hard、greedy、老师关，沿已冻结256自然整局重采缺失被动字段；全部逐局动作SHA与原诊断一致。\n\n|席位|局数|可行动持包tick|拦截tick/比例|暴露持包生命|因果杀持包者/比例|敌送达/局|己送达/局|完整取杀夺回循环/局|\n|---|---|---|---|---|---|---|---|---|\n')
        for seat,v in results.items():
            f.write(f"|{seat}|{v['games']}|{v['own_actionable_ticks']}|{v['interception_ticks']}/{v['interception_rate']}|{v['exposed_carrier_lives']}|{v['carrier_causal_kills']}/{v['carrier_killed_rate']}|{v['enemy_deliveries_per_game']}|{v['own_deliveries_per_game']}|{v['completed_antifarm_cycles_per_game']}|\n")
        f.write('\n拦截是确定性最短返程路3格内的几何暴露，不证明有效封路或策略意图；不可达仍计可行动分母。生命去重，敌持包死亡按pre-action包归属，不猜同tick拾取后死亡；无原生JAX奖励账本。逐局、事件、分母、SHA与定义见 `runs/r4_combat_20261031/carrier_diagnostic_v2/COMPLETE.json`。\n')
    return d


def main():
    B.mkdir(parents=True,exist_ok=True);guard=lock(B/'OWNER.lock')
    if (B/'COMPLETE.json').exists():return
    p=freeze();assert all(sha(Path(f))==h for f,h in p['source_sha256'].items())
    assert sha(Path(p['condition']['spawn_manifest']))==p['condition']['spawn_sha256']
    began=time.monotonic()
    for offset in range(0,32,8):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda pair:collect(pair,p),[(start,p['cores'][i]) for i,start in enumerate(range(offset*8,(offset+8)*8,8))]))
        atomic_json(B/'STATUS.json',dict(status='RUNNING',completed_games=(offset+8)*8,budget=256,CPU_only=True,wall_seconds=time.monotonic()-began,unix=time.time()))
    finish(p)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(B/'BLOCKED.json',dict(status='BLOCKED',error_type=type(exc).__name__,reason=str(exc),training_signals=0,unix=time.time()));raise
