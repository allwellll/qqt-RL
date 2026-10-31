"""Resumable CPU-only full-game v1/v2 diagnostic, fixed checkpoint and seeds."""
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha

OUT = Path('runs/bun_native_v2_20261030')


def read(p):
    return json.loads(Path(p).read_text())


def jobs(protocol):
    return [dict(rule=r, condition=c, start=i, **v) for c,v in protocol['conditions'].items()
            for i in range(0,256,8) for r in protocol['rules']]


def path(job):
    if 'eval_key' in job:
        return OUT/job.get('evaluation_directory','evaluation')/job['eval_key']/job['condition']/job.get('mode','greedy')/f"shard_{job['start']:03d}.json"
    return OUT / 'diagnostic' / job['rule'] / job['condition'] / f"shard_{job['start']:03d}.json"


def running(output):
    target = str(output).encode()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if target in args and b'scripts/bun_native_v2_eval.js' in args and (p / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                return True
        except OSError:
            pass
    return False


def collect(job, protocol, core):
    assert sha(Path(job['spawn_manifest']))==job['spawn_sha256']
    output = path(job)
    output.parent.mkdir(parents=True, exist_ok=True)
    guard = lock(Path(str(output) + '.lock'))
    accepted = Path(str(output) + '.ACCEPTED.json')
    if accepted.exists():
        for f, h in read(accepted)['files'].items():
            assert sha(Path(f)) == h
        return
    while running(output):
        time.sleep(3)
    side = Path(str(output) + '.outcome.json')
    if not (output.exists() and side.exists()):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', QQT_STOCK_RULE='stock-window-v1',
                   BUN_V2_RULE=job['rule'], BUN_V2_TIER=job['tier'], BUN_V2_OPPONENT=job['opponent'],
                   JE10_ROUTE_TASK='natural_full2400', JE10_ROUTE_GAMES=str(job.get('games',256)),
                   JE10_ROUTE_START=str(job['start']), JE10_ROUTE_END=str(job['start']+8),
                   JE10_ROUTE_SEED_START=str(job['seed']))
        with output.with_suffix('.log').open('ab') as log:
            subprocess.run(['taskset', '-c', str(core), 'node', 'scripts/bun_native_v2_eval.js', job.get('mode','greedy'),
                            protocol['model']['model_json'], str(output),
                            'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json', job['spawn_manifest']],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    doc = read(side)
    assert doc['rule_version'] == job['rule'] and doc['teacher'] is False and doc['mode']==job.get('mode','greedy')
    simulator='web/sim-bun-native-v2.js' if job['rule']=='bun-native-v2' else 'web/sim-stock-window-v1.js'
    assert doc['simulator_sha256']==sha(Path(simulator))
    assert read(output)['model_json_sha256']==protocol['model']['model_sha256']
    eps = doc['episodes']
    assert len(eps) == 8 and all(e['done'] for e in eps)
    assert [e['seed'] for e in eps] == list(range(job['seed']+job['start'], job['seed']+job['start']+8))
    assert [e['seat'] for e in eps] == [(job['start']+i)%2 for i in range(8)]
    assert all('antifarm' in e for e in eps)
    atomic_json(accepted, dict(status='PASS', job=job, files={str(f):sha(f) for f in (output, side)},
                              model_sha256=sha(Path(protocol['model']['model_json'])), unix=time.time()))


def summarize(protocol):
    result = {}
    for rule in protocol['rules']:
        result[rule] = {}
        for condition in protocol['conditions']:
            es=[]
            for job in jobs(protocol):
                if job['rule']==rule and job['condition']==condition and Path(str(path(job))+'.ACCEPTED.json').exists():
                    es.extend(read(str(path(job))+'.outcome.json')['episodes'])
            result[rule][condition]={}
            for seat in (0,1):
                rows=[e for e in es if e['seat']==seat]
                kills=[k for e in rows for k in e['kills'] if k['attacker']==seat]
                def flag(k): return (k['victim_carried']>=0,k['in_attacker_cage'],k['within3_attacker_home'])
                counts=[sum(flag(k)[i] for k in kills) for i in range(3)]
                union=sum(any(flag(k)) for k in kills)
                total=lambda name:sum(e['antifarm'][name][seat] for e in rows)
                deliveries=[v for e in rows for v in e['antifarm']['events'] if v['type']=='delivery' and v['player']==1-seat and v['origin']==seat]
                result[rule][condition][f'P{seat}']=dict(games=len(rows),wins=sum(e['winner']==seat for e in rows),
                    draws=sum(e['winner'] is None or e['winner']==-1 for e in rows),losses=sum(e['winner']==1-seat for e in rows),
                    causal_kills=len(kills),farm_carry=counts[0],farm_in_home=counts[1],farm_near_home=counts[2],
                    farm_union=union,farm_fraction=union/len(kills) if kills else None,
                    farm_overlap_all=sum(all(flag(k)) for k in kills),farm_no_kill_games=sum(not any(k['attacker']==seat for k in e['kills']) for e in rows),
                    farm_overlap_carry_home=sum(flag(k)[0] and flag(k)[1] for k in kills),
                    farm_overlap_carry_near=sum(flag(k)[0] and flag(k)[2] for k in kills),
                    farm_overlap_home_near=sum(flag(k)[1] and flag(k)[2] for k in kills),
                    stock_pickups=total('stock_pickups'),deliveries=total('deliveries'),
                    enemy_origin_deliveries=sum(v['type']=='delivery' and v['player']==seat and v['origin']==1-seat for e in rows for v in e['antifarm']['events']),
                    enemy_cage_entries=total('enemy_cage_entries'),
                    alive_ticks=total('alive_ticks'),actionable_ticks=total('actionable_ticks'),near_home_ticks=total('near_home_ticks'),
                    near_home_fraction=total('near_home_ticks')/total('alive_ticks') if total('alive_ticks') else None,
                    opponent_delivery_losses=sum(e['winner']==1-seat and any(v['type']=='delivery' and v['player']==1-seat and v['origin']==seat and v['tick']==e['ticks'] for v in e['antifarm']['events']) for e in rows),
                    opponent_delivery_ticks=[x['pickup_to_delivery_ticks'] for x in deliveries],
                    zero_bomb_games=sum(e['placed'][seat]==0 for e in rows),bombs=sum(e['placed'][seat] for e in rows),
                    self_deaths=sum(e['self_deaths'][seat] for e in rows))
    return result


def main():
    guard=lock(OUT/'DIAGNOSTIC_OWNER.lock')
    protocol=read(OUT/'PROTOCOL.json')
    assert sha(Path(protocol['model']['model_json']))==protocol['model']['model_sha256']
    work=jobs(protocol)
    schedule=read(OUT/'SCHEDULING.json') if (OUT/'SCHEDULING.json').exists() else {}
    workers=schedule.get('workers',8)
    cores=schedule.get('cores',list(range(144,144+workers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active={}
        while True:
            for future in list(active):
                if future.done():
                    future.result();del active[future]
            for job in work:
                if len(active)>=workers:break
                if any(job==x[0] for x in active.values()) or Path(str(path(job))+'.ACCEPTED.json').exists():continue
                free=next(i for i in range(workers) if i not in [x[1] for x in active.values()])
                active[pool.submit(collect,job,protocol,cores[free])]=(job,free)
            n=sum(8 for job in work if Path(str(path(job))+'.ACCEPTED.json').exists())
            atomic_json(OUT/'DIAGNOSTIC_STATUS.json',dict(status='RUNNING',games=n,budget=2048,active=[x[0] for x in active.values()],unix=time.time()))
            if n==2048:
                results=summarize(protocol)
                atomic_json(OUT/'DIAGNOSTIC_COMPLETE.json',dict(status='COMPLETE',games=n,results=results,
                    protocol_sha256=sha(OUT/'PROTOCOL.json'),unix=time.time()))
                report=Path('reports/bun_native_v2_antifarm_20261030.md')
                with report.open('a') as f:
                    f.write('\n## 完整自然局诊断\n\n固定旧网页 c34 u2400，2048完整局，argmax、老师关闭。\n\n')
                    f.write('|规则|对手|席位|局数|胜/平/负|刷分并集/因果击杀|主动取包/交付|敌交付负局|零泡|\n|---|---|---|---|---|---|---|---|---|\n')
                    for r,cs in results.items():
                        for c,seats in cs.items():
                            for s,v in seats.items():f.write(f"|{r}|{c}|{s}|{v['games']}|{v['wins']}/{v['draws']}/{v['losses']}|{v['farm_union']}/{v['causal_kills']}|{v['stock_pickups']}/{v['enemy_origin_deliveries']}|{v['opponent_delivery_losses']}|{v['zero_bomb_games']}|\n")
                h1=results['stock-window-v1']['hard']['P1'];h2=results['bun-native-v2']['hard']['P1']
                subprocess.run(['jdme-notify','send',f"bun-native-v2诊断2048完整局完成；hard P1 v1 {h1['wins']}/{h1['draws']}/{h1['losses']}→v2 {h2['wins']}/{h2['draws']}/{h2['losses']}，敌交付负局{h1['opponent_delivery_losses']}→{h2['opponent_delivery_losses']}。其他三类与P0分列见reports/bun_native_v2_antifarm_20261030.md；R1继续，网页规则未切换。"],check=False)
                return
            time.sleep(5)


if __name__=='__main__':
    try: main()
    except Exception as e:
        atomic_json(OUT/'DIAGNOSTIC_BLOCKED.json',dict(status='BLOCKED',error_type=type(e).__name__,reason=str(e) if '<' in str(e) else 'inspect diagnostic shard log; training untouched',unix=time.time()))
        raise
