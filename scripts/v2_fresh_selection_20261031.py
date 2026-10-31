"""Fixed production-JS v2 matrix; independent from all training owners."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha
from scripts.r3_items_eval_owner import summarize
from scripts.r1_fresh_tau_completion import comparison, context
from scripts.r1_fresh_tau_0735 import training_snapshot, paired

OUT=Path(os.environ.get('V2_SELECTION_OUTPUT','runs/v2_fresh_selection_20261031'))
REPORT=Path(os.environ.get('V2_SELECTION_REPORT','reports/v2_fresh_selection_20261031.md'))
def read(p):return json.loads(Path(p).read_text())


def models(p):
    result={}
    for label,source in p['models'].items():
        receipt=OUT/'models'/f'{label}.json'
        if receipt.exists():
            m=read(receipt)
        else:
            cp=Path(source['checkpoint']);assert sha(cp/'policy.pt')==source['policy_sha256']
            # Reuse exact existing exports before creating anything new.
            root=Path('runs/r3_items_route_20261030/evaluation') if source['r3'] else Path('runs/bomb_threat_v1/eval')
            matches=list(root.glob(cp.parent.name+'_'+cp.name+'*/model/model.json'))
            if not source['r3']:
                matches+=list(Path('runs/bun_native_v2_20261030/eval_exports').glob(cp.parent.name+'_'+cp.name+'*/model.json'))
            f=next((x for x in matches if read(x)['meta']['checkpoint_sha256']==source['policy_sha256']),None)
            if f is None:
                dest=OUT/'exports'/label;dest.mkdir(parents=True,exist_ok=True)
                code=('from scripts.r3_items_export import export;import sys;export(*sys.argv[1:])' if source['r3'] else
                      'from scripts.je08_bun_authority import export;from pathlib import Path;import sys;export(Path(sys.argv[1])/"policy.pt",Path(sys.argv[2]).resolve())')
                with (dest/'EXPORT.log').open('ab') as log:
                    subprocess.run([sys.executable,'-c',code,str(cp),str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
                f=dest/'model.json'
            meta=read(f)['meta'];graph=Path(meta.get('onnx_path',f.with_suffix('.onnx')))
            assert sha(graph)==meta['onnx_sha256']
            m=dict(source,model_json=str(f),model_sha256=sha(f),onnx_path=str(graph),onnx_sha256=sha(graph))
            atomic_json(receipt,m)
        assert sha(Path(m['model_json']))==m['model_sha256'] and sha(Path(m['onnx_path']))==m['onnx_sha256']
        result[label]=m
    atomic_json(OUT/'MODELS.json',result)
    return result


def jobs(p):
    return [dict(label=label,condition=c,start=start,**v) for c,v in p['conditions'].items()
            for start in range(0,256,8) for label in p['models']]


def path(j):return OUT/'shards'/j['condition']/j['label']/f"shard_{j['start']:03d}.json"


def validate(j,m):
    f=path(j);a=read(f);b=read(str(f)+'.outcome.json');rows=b['episodes']
    assert a['model_json_sha256']==m['model_sha256']
    assert b['rule_version']=='bun-native-v2' and b['mode']=='greedy' and b['engine']=='production_JS' and not b['teacher']
    assert b['simulator_sha256']==sha(Path('web/sim-bun-native-v2.js'))
    assert sha(Path(j['spawn_manifest']))==j['spawn_sha256']
    assert len(rows)==8 and [r['seed'] for r in rows]==list(range(j['seed']+j['start'],j['seed']+j['start']+8))
    for i,(r,e) in enumerate(zip(rows,a['results']['natural_full2400']['episodes'])):
        assert r['seat']==(j['start']+i)%2 and r['done'] and 0<r['ticks']<=2400
        assert (r['seed'],r['seat'],r['ticks'])==(e['seed'],e['team'],e['ticks'])
        assert 'r3' in r and 'antifarm' in r and 'resolved_bombs' in r['r3']
        if read(j['spawn_manifest']).get('spawn_side_version'):
            from scripts.bun_spawn_side import validate as side_validate
            assert r['spawn_side_version']=='own-side-native-v1'
            side_validate(r['spawn_cells']);side_validate(r['respawn_cells'])
    return rows


def collect(j,m):
    f=path(j);f.parent.mkdir(parents=True,exist_ok=True);guard=lock(Path(str(f)+'.lock'))
    proof=Path(str(f)+'.ACCEPTED.json')
    if proof.exists():
        assert all(sha(Path(k))==v for k,v in read(proof)['files'].items());return validate(j,m)
    if not (f.exists() and Path(str(f)+'.outcome.json').exists()):
        env=dict(os.environ,QQT_STOCK_RULE='stock-window-v1',BUN_V2_RULE='bun-native-v2',
                 BUN_V2_TIER=j['tier'],BUN_V2_OPPONENT=j['opponent'],JE10_ROUTE_TASK='natural_full2400',
                 JE10_ROUTE_GAMES='256',JE10_ROUTE_START=str(j['start']),JE10_ROUTE_END=str(j['start']+8),
                 JE10_ROUTE_SEED_START=str(j['seed']))
        with f.with_suffix('.log').open('ab') as log:
            subprocess.run(['node','scripts/r3_items_candidate_eval.js','greedy',m['model_json'],str(f),
                            'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',j['spawn_manifest']],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=validate(j,m)
    atomic_json(proof,dict(status='PASS',job=j,model=m,protocol_sha256=sha(OUT/'PROTOCOL.json'),
                          files={str(z):sha(z) for z in (f,Path(str(f)+'.outcome.json'))},unix=time.time()))
    return rows


def select(results,p):
    tiers=['hunter_hard','hunter_normal','hunter_easy'];base=results['baseline'];detail={}
    for label,cs in results.items():
        p1={t:cs[t]['P1']['win_minus_loss'] for t in tiers};p0=sum(cs[t]['P0']['win_minus_loss'] for t in tiers)
        selfd=sum(cs[t][s]['self_deaths'] for t in tiers for s in ('P0','P1'))
        guards=dict(each_P1=all(p1[t]>=base[t]['P1']['win_minus_loss']-5 for t in tiers),
                    P0=p0>=sum(base[t]['P0']['win_minus_loss'] for t in tiers)-10,
                    self_deaths=selfd<=sum(base[t][s]['self_deaths'] for t in tiers for s in ('P0','P1')))
        detail[label]=dict(P1=p1,P1_sum=sum(p1.values()),P0_sum=p0,self_deaths=selfd,guards=guards,eligible=all(guards.values()))
    winner='baseline'
    for label in p['candidate_order']:
        if detail[label]['eligible'] and detail[label]['P1_sum']>detail[winner]['P1_sum']:winner=label
    return dict(winner=winner,eligible_for_conditional_publish=winner!='baseline',comparison=detail,
                coop_hard_separate=True,selection='fixed three-tier primary and guards; strict improvement, ties keep baseline')


def aggregate(p,ms,final=False):
    raw={label:{} for label in ms};files={};games=0
    for j in jobs(p):
        proof=Path(str(path(j))+'.ACCEPTED.json')
        if not proof.exists():continue
        a=read(proof);assert all(sha(Path(k))==v for k,v in a['files'].items())
        rows=validate(j,ms[j['label']]);files.update(a['files']);games+=len(rows)
        raw[j['label']].setdefault(j['condition'],[]).extend(rows)
    metrics={label:{c:{f'P{s}':summarize(es,s) for s in (0,1)} for c,es in cs.items()} for label,cs in raw.items()}
    result=dict(status='COMPLETE' if games==p['budget'] else 'RUNNING',games=games,budget=p['budget'],metrics=metrics,
                files=files,models=ms,protocol_sha256=sha(OUT/'PROTOCOL.json'),unix=time.time())
    if final:
        assert games==p['budget']
        for cs in raw.values():
            for es in cs.values():
                assert len(es)==256==len({(r['seed'],r['seat']) for r in es})
                assert [sum(e['seat']==s for e in es) for s in (0,1)]==[128,128]
        result['paired_per_game']={label:{c:paired(es,raw['baseline'][c]) for c,es in cs.items()}
                                   for label,cs in raw.items() if label!='baseline'}
        result['selection']=select(metrics,p)
    atomic_json(OUT/'RESULTS.json',result)
    return result


def report(r):
    p=read(OUT/'PROTOCOL.json');d=r['selection']
    lines=['# bun-native-v2 fresh256固定候选复核（20261031）','',
           f"6144完整局，6模型×Hunter三档及协作Hunter hard×256；每条件P0/P1各128。候选冻结后不随训练追新，最新点R3-a u3240、R3-b u3200。winner={d['winner']}，具备条件发布资格={d['eligible_for_conditional_publish']}。",'',
           '规则bun-native-v2、生产JS Sim.step、argmax、带砖、nativeItems/nativeTrap=true、2400tick、老师关闭。hard/normal/easy新seed2026314000/2026315000/2026316000，协作hard复用hard同seed；64出生几何循环，不是256独立地图。协作Hunter为1v1压力项，无同队搭档，单列不混主排名。', '',
           '|模型|P1 hard/normal/easy|P1合计|P0合计|自伤|护栏|','|---|---|---:|---:|---:|---|']
    for label,x in d['comparison'].items():lines.append(f"|{label}|{list(x['P1'].values())}|{x['P1_sum']:+d}|{x['P0_sum']:+d}|{x['self_deaths']}|{x['guards']}|")
    lines+=['','|模型/对手/席位|局数|W/D/L|WL|因果击杀|死亡|泡/局|原始/单方零泡|刷分代理份额|主动送达|被送达负局|自伤|',
            '|---|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|']
    for label,cs in r['metrics'].items():
        for c,ss in cs.items():
            for seat,v in ss.items():
                lines.append(f"|{label}/{c}/{seat}|{v['games']}|{v['wins']}/{v['draws']}/{v['losses']}|{v['win_minus_loss']:+d}|{v['causal_kills']}|{v['deaths']}|{v['bombs']/v['games']:.3f}|{v['zero_bomb']}/{v['unilateral_zero_bomb']}|{v['farm_union']}/{v['causal_kills']}|{v['active_deliveries']}|{v['opponent_delivery_losses']}|{v['self_deaths']}|")
    lines+=['','刷分份额仅union(击杀持包者、己笼内、己家3格内)/因果击杀，行为代理而非意图证据；零击杀为null，不自动当安全。被偷家列严格为对手送达己方来源包导致的终局负局，不推断追击。', '',
            '|模型/对手/席位|拾取泡/威/速/蕉/胶|蕉/胶释放|持包自踩蕉|胶敌/自触发|完整滑行/未完成|估算省tick|真连锁|泡mean/P90/max|批次mean/max|',
            '|---|---|---|---:|---|---|---:|---:|---|---|']
    for label,cs in r['metrics'].items():
        for c,ss in cs.items():
            for seat,v in ss.items():
                lines.append(f"|{label}/{c}/{seat}|{v['pickup_events']}|{v['release_counts']}|{v['carrying_own_banana_contacts']}|{v['enemy_glue_contacts']}/{v['glue_self_contacts']}|{v['banana_completed']}/{v['banana_unfinished']}|{v['banana_estimated_saved_ticks']:.2f}|{v['chain_events']}|{v['live_bomb_mean']:.2f}/{v['live_bomb_p90']:.0f}/{v['live_bomb_max']}|{v['live_batch_mean']:.2f}/{v['live_batch_max']}|")
    lines+=['','完整终局三属性、逐局配对差、道具机会/来源/滑行危险、连锁批次、末端未爆与所有SHA在 `runs/v2_fresh_selection_20261031/RESULTS.json`。',
            '香蕉主语义为持包自己踩加速回笼；省tick是同路径原生速度估计，不是无香蕉分支因果收益。胶后击杀/取包是观察相关，不是因果效用。未记录原生训练奖励，不能以JS指标冒充奖励；本比较不代表真人frameStep或其他地图泛化。', '',
            '训练GPU未被停止或挤占；生产JS/ONNX入口仅支持CPU，本次使用独立限额CPU而不把JAX替代成JS。资源与吞吐日志 `SCHEDULER_STATUS.json`；可复现命令 `.venv/bin/python -m scripts.v2_fresh_selection_20261031`，已验收分片不会重跑。', '',
            '后继：只有同三档P1主排序严格超过基线且每档≥基线−5、P0合计≥基线−10、自伤不增才进入用户授权的模型+规则同步发布；否则保留网页。协作hard单列，未临时增加未授权选型门槛。']
    if p.get('spawn_side_version'):
        lines=[line.replace('64出生几何循环，不是256独立地图。','16个原生本侧出生组合循环，不是256独立地图；P0列≤6、P1列≥8，中线列7排除。').replace('64 reused geometries','16 own-side native geometries') for line in lines]
        lines.insert(2,'旧错侧出生复核不再用于选型；本报告使用独立新seed和本侧出生/复活清单。')
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models-only',action='store_true');args=parser.parse_args()
    guard=lock(OUT/'OWNER.lock');p=read(OUT/'PROTOCOL.json');ms=models(p)
    if args.models_only:return
    files=['scripts/v2_fresh_selection_20261031.py','scripts/r3_items_candidate_eval.js','scripts/r3_items_candidate_metrics.js',
           'scripts/r3_items_metrics.js','scripts/r3_items_inference.js','scripts/bun_native_v2_eval.js',
           'scripts/bun_native_v2_metrics.js','scripts/bomb_threat_eval.js','scripts/expert_validation_js_eval.js',
           'scripts/dagger_rl_route.js','web/sim-bun-native-v2.js','web/bun_hunter_bot.js','web/bun_coop_hunter_bot.js']
    seal=OUT/'SOURCE_SEAL.json'
    if not seal.exists():atomic_json(seal,{f:sha(Path(f)) for f in files})
    assert all(sha(Path(f))==h for f,h in read(seal).items())
    pending=[j for j in jobs(p) if not Path(str(path(j))+'.ACCEPTED.json').exists()]
    refs=training_snapshot();cohort=context();workers=12;history=[]
    while pending:
        batch,pending=pending[:workers],pending[workers:]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(collect,j,ms[j['label']]) for j in batch]):f.result()
        # Full metrics are expensive: only aggregate once all shards complete.
        done=len(list((OUT/'shards').glob('*/*/*.ACCEPTED.json')))*8
        now,current=training_snapshot(),context();ratios=comparison(now,refs) if current==cohort else {}
        if current!=cohort:refs,cohort=now,current
        if ratios and min(ratios.values())<.85:workers=max(2,workers//2)
        record=dict(games=done,budget=p['budget'],workers=workers,ratios=ratios,training_current=now,gpu_cohort=current,unix=time.time())
        history.append(record);atomic_json(OUT/'SCHEDULER_STATUS.json',dict(status='RUNNING',history=history));atomic_json(OUT/'EXECUTION_STATUS.json',record)
        print(json.dumps(dict(games=done,workers=workers,ratios=ratios)),flush=True)
    r=aggregate(p,ms,True);report(r)
    atomic_json(OUT/'COMPLETE.json',dict(status='COMPLETE',games=r['games'],results_sha256=sha(OUT/'RESULTS.json'),
                                       report_sha256=sha(Path('reports/v2_fresh_selection_20261031.md')),selection=r['selection'],unix=time.time()))
    subprocess.run(['jdme-notify','send',f"v2 fresh256复核6144局完成，固定规则候选={r['selection']['winner']}；报告reports/v2_fresh_selection_20261031.md；进入录像/条件发布验收，训练不停。"],check=False)


if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(OUT/'BLOCKED.json',dict(status='BLOCKED',error_type=type(e).__name__,reason=str(e),training_untouched=True,unix=time.time()));raise
