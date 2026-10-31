"""Round14 CPU fresh256 JS recheck; fixed identities, atomic resumable shards."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',QQT_STOCK_RULE='stock-window-v1')
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from scripts.population_run_state import atomic_json, sha, lock
from scripts import bomb_threat_fresh256 as F

BASE=Path('runs/bomb_threat_v1')
OUT=BASE/'fresh256_a3sr_u640'


def read(p):return json.loads(Path(p).read_text())


def freeze():
    target=OUT/'PROTOCOL.json'
    if target.exists():return read(target)
    audit=read(BASE/'iter2/a3sr/round14/HOLDOUT_AUDIT.json')
    assert audit['gates']['absolute']['status']=='PASS'
    old=read(BASE/'fresh256_slot01_u800/PROTOCOL.json')
    models=dict(old['models'])
    cp=BASE/'formal/bt_slot05_c000026/u000640'
    cm=read(cp/'COMMITTED.json');model=BASE/'eval/bt_slot05_c000026_u000640_holdout/model/model.json'
    models['candidate']=dict(checkpoint=str(cp),policy_sha256=cm['policy_sha256'],
        model_json=str(model),model_sha256=sha(model))
    tiers={}
    for i,tier in enumerate(F.TIERS):
        seed=2026153900+i*1000
        spawn=read(BASE/'holdout_spawns.json')
        # Match the holdout geometry exactly, independently new environment/action seeds.
        spawn=dict(spawn,seed=seed,source='holdout64 birth geometry cyclic reuse; not fresh maps')
        p=OUT/f'{tier}_spawns.json';atomic_json(p,spawn)
        tiers[tier]=dict(seed=seed,seed_end=seed+255,spawn_manifest=str(p),spawn_sha256=sha(p))
    paths=[p for p in old['source_sha256'] if p!='scripts/bomb_threat_fresh256.py']
    paths+=['scripts/bomb_threat_a3sr_fresh.py','scripts/bomb_threat_fresh256.py']
    proto=dict(id='A3SR-U640-FRESH256-20261030',status='FROZEN',
        instruction='.hermes/STEER_a3sr_fresh256_20261030_0405.md',
        models=models,tiers=tiers,source_sha256={p:sha(p) for p in paths},
        native_shaping_share=audit['reward_monitor']['threat_reward_fraction_total'],
        modes=['greedy','categorical'],games_per_mode_per_tier=256,seats={'P0':128,'P1':128},
        gate='existing fresh256 proportional B thresholds; argmax primary, categorical report only',
        baselines='same four identities and matched seeds as previous fresh protocol; no peak selection',
        teacher=False,total_games=3840,cpu_workers=4,no_publication=True,no_promotion=True,
        limitations=['holdout64 geometries cyclic reuse, not256 independent geometries',
                     'JAX auxiliary is distinct engine/opponent and is not JS gate evidence'],unix=time.time())
    atomic_json(target,proto)
    # A named split lets the unmodified persistent GPU2 engine consume fresh seeds.
    native_spawn=BASE/'fresh_a3sr_u640_spawns.json'
    atomic_json(native_spawn,read(OUT/'hard_spawns.json'))
    req=dict(name='fresh256_a3sr_u640',checkpoint=str(cp),split='fresh_a3sr_u640',
        iter2_arm='A3SR',rule_version='stock-window-v1',reward_version='bomb-threat-iter2-A3',
        reason='round14 priority fresh256; GPU2 auxiliary only, JS separately collected',
        native_only=True,protocol=str(target),protocol_sha256=sha(target))
    atomic_json(BASE/'eval_requests'/f"{req['name']}.json",req)
    p=BASE/'eval_requests/bt_slot05_c000026_u000600_holdout.json'
    if not p.exists():atomic_json(p,dict(name=p.stem,checkpoint=str(cp.parent/'u000600'),split='holdout',
        iter2_arm='A3SR',rule_version='stock-window-v1',reward_version='bomb-threat-iter2-A3',reason='round14 frozen neighbor, not peak selection'))
    atomic_json(OUT/'TASK_CONSUMED.json',dict(status='FROZEN_ENQUEUED',protocol_sha256=sha(target),unix=time.time()))
    return proto


def configure():F.OUT=OUT


def publish(protocol):
    from scripts.bomb_threat_eval import summarize
    summaries={};files={};count=0
    for tier in F.TIERS:
        summaries[tier]={}
        for role in F.ROLES:
            summaries[tier][role]={}
            for mode in (F.MODES if role=='candidate' else ('greedy',)):
                rows=[]
                for start in range(0,256,8):
                    job=dict(tier=tier,role=role,mode=mode,start=start);p=F.job_path(job)
                    proof=Path(str(p)+'.ACCEPTED.json')
                    if not proof.exists():continue
                    for f,h in read(proof)['files'].items():assert sha(f)==h;files[f]=h
                    rows+=F.validate_shard(p,job,protocol);count+=8
                if len(rows)==256:
                    assert len({r['seed'] for r in rows})==256
                    summaries[tier][role][mode]=summarize(rows)
    gates={t:F.gate(summaries[t]['candidate'].get('greedy',{}),summaries[t]['parent'].get('greedy',{}),
        {r:summaries[t][r].get('greedy',{}) for r in F.ROLES[2:]},protocol['native_shaping_share']) for t in F.TIERS}
    done=count==protocol['total_games']
    result=dict(status=F.verdict_status(gates.values()),collection_status='COMPLETE' if done else 'RUNNING',
        completed_games=count,total_games=protocol['total_games'],metrics=summaries,per_tier=gates,files=files,
        protocol_sha256=sha(OUT/'PROTOCOL.json'),best_changed=False,teacher=False,unix=time.time())
    atomic_json(OUT/'RESULTS.json',result)
    if done:
        atomic_json(OUT/'COMPLETE.json',dict(status=result['status'],result_sha256=sha(OUT/'RESULTS.json'),unix=time.time()))
        from scripts.iter_pbt_owner import notify
        notify('A3SR_FRESH256_COMPLETE',f"A3SR u640 fresh256三个档与匹配基线完整采集完成，B={result['status']}；不晋升不发布。")
    return result


def main():
    protocol=freeze();configure();guard=lock(OUT/'OWNER.lock')
    for f,h in protocol['source_sha256'].items():assert sha(f)==h,f
    for m in protocol['models'].values():assert sha(m['model_json'])==m['model_sha256']
    if (OUT/'COMPLETE.json').exists():return
    atomic_json(OUT/'EXECUTION.json',dict(status='RUNNING',workers=4,cpu_only=True,unix=time.time()))
    def run(job):
        p=F.job_path(job);p.parent.mkdir(parents=True,exist_ok=True);proof=Path(str(p)+'.ACCEPTED.json')
        if proof.exists():
            for f,h in read(proof)['files'].items():assert sha(f)==h
            F.validate_shard(p,job,protocol);return
        if not (p.exists() and Path(str(p)+'.outcome.json').exists()):
            env=dict(os.environ,BT_FRESH_TIER=job['tier'],JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='256',
                JE10_ROUTE_START=str(job['start']),JE10_ROUTE_END=str(job['start']+8),JE10_ROUTE_SEED_START=str(protocol['tiers'][job['tier']]['seed']))
            with p.with_suffix('.log').open('ab') as log:
                subprocess.run(['node','scripts/bomb_threat_fresh256_eval.js',job['mode'],protocol['models'][job['role']]['model_json'],str(p),
                    'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',protocol['tiers'][job['tier']]['spawn_manifest']],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        F.validate_shard(p,job,protocol)
        atomic_json(proof,dict(job=job,protocol_sha256=sha(OUT/'PROTOCOL.json'),
            files={str(f):sha(f) for f in (p,Path(str(p)+'.outcome.json'))},unix=time.time()))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(run,j):j for j in F.jobs()}
            for i,f in enumerate(as_completed(futures)):
                f.result();atomic_json(OUT/'PROGRESS.json',dict(completed_shards=i+1,total_shards=len(futures),last_job=futures[f],unix=time.time()))
                if (i+1)%8==0:publish(protocol)
        result=publish(protocol)
        atomic_json(OUT/'EXECUTION.json',dict(status='COMPLETED',B=result['status'],unix=time.time()))
    except Exception as error:
        atomic_json(OUT/'EXECUTION.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error),unix=time.time()))
        raise


if __name__=='__main__':main()
