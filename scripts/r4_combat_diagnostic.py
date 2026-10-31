"""Bounded CPU production full games. Native rewards are never fabricated."""
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

B=Path('runs/r4_combat_20261031')
def read(p):return json.loads(Path(p).read_text())

def freeze():
    path=B/'protocols/diagnostic_v1.json'
    if path.exists():return read(path)
    old=read('runs/r3_resume_publish_2110_20261031/selection/PROTOCOL.json')
    condition=old['conditions']['hunter_hard']
    doc=dict(id='R4-COMBAT-FULLGAME-DIAGNOSTIC-V1',status='FROZEN',model=old['models']['r3_a_u4002'],
        condition=condition,mode='greedy',teacher=False,games=256,per_seat=128,horizon=2400,
        seed_list=list(range(condition['seed'],condition['seed']+256)),workers=8,cores=list(range(80,88)),
        new_passive_metrics=True,source_sha256={p:sha(Path(p)) for p in
            ['scripts/r4_combat_metrics.js','scripts/r4_combat_diagnostic_eval.js','scripts/r3_items_candidate_eval.js',
             'scripts/r3_items_metrics.js','scripts/bun_native_v2_eval.js','web/sim-bun-native-v2.js']},
        limitation='JS has no native JAX reward ledger; total reward shares/pending/clawback reported unavailable, not zero',unix=time.time())
    atomic_json(path,doc);return doc

def collect(pair,p):
    start,core=pair;v=p['condition'];target=B/'diagnostic'/f'shard_{start:03d}.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    guard=lock(Path(str(target)+'.lock'));done=Path(str(target)+'.ACCEPTED.json')
    if done.exists():
        assert all(sha(Path(f))==h for f,h in read(done)['files'].items());return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',QQT_STOCK_RULE='stock-window-v1',BUN_V2_RULE='bun-native-v2',
        BUN_V2_TIER='hard',BUN_V2_OPPONENT='bun.hunter',JE10_ROUTE_TASK='natural_full2400',
        JE10_ROUTE_GAMES='256',JE10_ROUTE_START=str(start),JE10_ROUTE_END=str(start+8),JE10_ROUTE_SEED_START=str(v['seed']))
    if not Path(str(target)+'.outcome.json').exists():
        with target.with_suffix('.log').open('ab') as log:
            subprocess.run(['taskset','-c',str(core),'node','scripts/r4_combat_diagnostic_eval.js',p['mode'],
                p['model']['model_json'],str(target),'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',v['spawn_manifest']],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    side=read(str(target)+'.outcome.json');episodes=side['episodes']
    assert len(episodes)==8 and all(e['done'] and 'r4' in e for e in episodes)
    assert [e['seed'] for e in episodes]==list(range(v['seed']+start,v['seed']+start+8))
    assert [e['seat'] for e in episodes]==[(start+i)%2 for i in range(8)]
    assert read(target)['model_json_sha256']==p['model']['model_sha256']
    # New instrumentation must reproduce already accepted full joint actions.
    previous=Path('runs/r3_resume_publish_2110_20261031/selection/shards/hunter_hard/r3_a_u4002')/target.name
    old=read(str(previous)+'.outcome.json')['episodes']
    assert [e['r3']['joint_actions_sha256'] for e in episodes]==[e['r3']['joint_actions_sha256'] for e in old]
    atomic_json(done,dict(status='PASS',games=8,passive_joint_actions_exact=True,
        files={str(f):sha(f) for f in [target,Path(str(target)+'.outcome.json')]},unix=time.time()))

def finish(p):
    es=[];files={}
    for start in range(0,256,8):
        f=B/'diagnostic'/f'shard_{start:03d}.json';a=read(str(f)+'.ACCEPTED.json')
        for path,h in a['files'].items():assert sha(Path(path))==h;files[path]=h
        es+=read(str(f)+'.outcome.json')['episodes']
    assert len({(e['seed'],e['seat']) for e in es})==256
    from scripts.r3_items_eval_owner import summarize
    results={}
    for seat in (0,1):
        group=[e for e in es if e['seat']==seat];m=summarize(es,seat)
        opportunities=sum(e['r4']['opportunity'][seat] for e in group)
        placed=sum(e['r4']['placements'][seat] for e in group)
        requested=sum(e['r4']['requests'][seat] for e in group)
        probability_n=sum(e['r4']['probability_count'][seat] for e in group)
        probability_sum=sum(e['r4']['probability_sum'][seat] for e in group)
        m.update(opportunity_ticks=opportunities,opportunity_placements=placed,opportunity_requests=requested,
            placement_rate=placed/opportunities if opportunities else None,
            probability_recorded_ticks=probability_n,mean_bomb_probability=probability_sum/probability_n if probability_n else None)
        results[f'P{seat}']=m
    d=dict(status='COMPLETE',games=256,results=results,files=files,protocol_sha256=sha(B/'protocols/diagnostic_v1.json'),
        metrics_status=dict(implemented=['opportunity','true placement','learner probability','causal kills/death','R3 item/chain/road events'],
        recomputable=['wins/draws/losses','deliveries','per-phase exposure'],missing_data=['native per-tick reward components','pending/clawback histories'],
        unavailable=['native JAX training reward fractions on JS Hunter episodes']),unix=time.time())
    atomic_json(B/'DIAGNOSTIC_COMPLETE.json',d)
    with Path('reports/r4_combat_20261031.md').open('a') as f:
        f.write('\n## 生产JS完整局机会诊断\n\n父R3-a u4002，Hunter hard，greedy/老师关闭，本侧出生，P0/P1各128局；新被动记录与已有逐局joint-actions SHA完全一致。\n\n|席位|胜/平/负|因果击杀/局|死亡/局|放泡机会tick|机会中真实放泡|比例|平均放泡概率|\n|---|---|---|---|---|---|---|---|\n')
        for seat,v in results.items():f.write(f"|{seat}|{v['wins']}/{v['draws']}/{v['losses']}|{v['causal_kills']/128:.4f}|{v['deaths']/128:.4f}|{v['opportunity_ticks']}|{v['opportunity_placements']}|{v['placement_rate']}|{v['mean_bomb_probability']}|\n")
        f.write('\n原生训练奖励占比、pending/clawback在这些JS轨迹里不可用，不记0，不把事件数量当奖励。原始逐局与SHA见 `runs/r4_combat_20261031/DIAGNOSTIC_COMPLETE.json`。\n')
    return d

def main():
    B.mkdir(parents=True,exist_ok=True);guard=lock(B/'DIAGNOSTIC_OWNER.lock');p=freeze()
    if (B/'DIAGNOSTIC_COMPLETE.json').exists():return
    assert all(sha(Path(f))==h for f,h in p['source_sha256'].items())
    assert sha(Path(p['condition']['spawn_manifest']))==p['condition']['spawn_sha256']
    jobs=list(range(0,256,8));began=time.monotonic()
    # Fixed chunks of eight ensure exclusive single-core workers, no shared lane oversubscription.
    for offset in range(0,len(jobs),8):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda pair:collect(pair,p),[(start,p['cores'][i]) for i,start in enumerate(jobs[offset:offset+8])]))
        atomic_json(B/'DIAGNOSTIC_STATUS.json',dict(status='RUNNING',completed_games=min(256,(offset+8)*8),budget=256,
            elapsed_seconds=time.monotonic()-began,CPU_only=True,unix=time.time()))
    finish(p)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(B/'DIAGNOSTIC_BLOCKED.json',dict(status='BLOCKED',error_type=type(exc).__name__,
            reason=str(exc),training_signals=0,unix=time.time()));raise
