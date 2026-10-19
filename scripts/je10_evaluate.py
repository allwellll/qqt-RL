"""Unique asynchronous CPU owner for fixed tactical, JS Hunter and route funnels."""
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.je10_campaign import NAMESPACE,BASE,ARM,POINTS,read,notify
from scripts.population_run_state import atomic_json,sha,lock
OUT=BASE/'authority_je10'


def unresolved_training_failure(base):
    failure=Path(base)/'TECHNICAL_FAILURE.json'
    if not failure.exists():return False
    recovery=Path(base)/'FORMAL_NONZERO_CONFIRMED.json'
    if not recovery.exists():return True
    try:
        receipt=read(recovery)
        assert receipt['formal_nonzero_verified'] and receipt['finite'] and receipt['replicas_exact']
        assert receipt['devices']==8 and receipt['generation']==1 and receipt['fresh_ticks']==2097152
        for name in ['actor_update_l2','critic_update_l2']:
            values=receipt[name]
            assert len(values)==8 and all(math.isfinite(x) and x>0 for x in values)
    except (OSError,ValueError,TypeError,KeyError,AssertionError):return True
    return failure.stat().st_mtime_ns>=recovery.stat().st_mtime_ns


def tactical(policy,target):
    from scripts import je08_bun_authority as authority
    authority.OUT=OUT
    authority.tactical(Path(policy),Path(target))


def evaluate(g):
    from scripts import je08_bun_authority as authority
    from scripts.je03_admission_hunter_outcomes import episode_outcome
    authority.OUT=OUT
    cp=BASE/'formal'/ARM/f'gen{g:06d}'
    out=OUT/f'g{g:06d}';out.mkdir(parents=True,exist_ok=True)
    policy=cp/'policy.pt';policy_hash=sha(policy)
    model=authority.export(policy,out/'model')
    def hunter(job):
        tier,start=job;target=out/f'hunter_{tier}_{start:03d}.json'
        if not target.exists():
            subprocess.run(['node','scripts/je10_web_hunter.js','--model',str(model.relative_to(ROOT)),
                '--difficulty',tier,'--spawns',str((OUT/'spawns.json').relative_to(ROOT)),
                '--start',str(start),'--end',str(start+8),'--seed','20261001',
                '--checkpoint-sha',policy_hash,'--out',str(target.relative_to(ROOT))],check=True)
        authority.validate_hunter(read(target),policy_hash,tier,start)
        return target
    target=out/'tactical_v2.json'
    tproc=None
    if not target.exists():
        tproc=subprocess.Popen([sys.executable,'scripts/je10_evaluate.py','tactical',str(policy.relative_to(ROOT)),str(target.relative_to(ROOT))],
            stdout=(out/'tactical.log').open('a'),stderr=subprocess.STDOUT)
    # JS route work runs alongside Hunter shards, on CPU only.
    route=out/'ROUTE_FUNNEL.json'
    rproc=None
    if not route.exists():
        rproc=subprocess.Popen(['node','scripts/je10_route_funnel.js',str(model.relative_to(ROOT)),str(route.relative_to(ROOT)),str((BASE/'regions.json').relative_to(ROOT))],
            stdout=(out/'route.log').open('a'),stderr=subprocess.STDOUT)
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        shards=list(pool.map(hunter,[(tier,s) for tier in ['hard','normal','easy'] for s in range(0,64,8)]))
    if tproc and tproc.wait():raise RuntimeError('CPU Tactical failed')
    if rproc and rproc.wait():raise RuntimeError('JS route funnel failed')
    hunter_summary={tier:authority.summarize_hunter([e for s in range(0,64,8) for e in read(out/f'hunter_{tier}_{s:03d}.json')['episodes']])
        for tier in ['hard','normal','easy']}
    control={tier:{key:sum(e.get('je10_control',{}).get(key,0) for s in range(0,64,8) for e in read(out/f'hunter_{tier}_{s:03d}.json')['episodes'])
        for key in ['first_control','attributed_first_control','final_pop','active_contact_pop','causal_kill','surviving_kill','native_causal_kill']}
        for tier in ['hard','normal','easy']}
    exposure=dict(current=0.,ring=0.,fixed_league=0.,fixed_g0=0.,fixed_stage1=0.,responsive_rule=0.,weak_course_ticks=0.,valid_decision_ticks=0.)
    executed={};executed_generations=0
    events=BASE/'formal'/ARM/'events.jsonl'
    for line in events.read_text().splitlines() if events.exists() else []:
        row=json.loads(line)
        if row['generation']>g:continue
        final_source=row.get('executed_opponent_exposure',{})
        if final_source:
            executed_generations+=1
            for name in ['current','ring','fixed_league','fixed_g0','fixed_stage1','responsive_rule','weak_courier','weak_guard','carry_assist','valid_decision_ticks']:
                executed[name]=executed.get(name,0)+final_source[name]
        league=row.get('actual_league_exposure',{})
        exposure['current']+=league.get('current',0.);exposure['ring']+=league.get('ring',0.);exposure['fixed_league']+=league.get('fixed',0.)
        family=row['actual_opponent_valid_ticks']
        exposure['fixed_g0']+=family[1];exposure['fixed_stage1']+=family[2];exposure['responsive_rule']+=family[0]
        exposure['valid_decision_ticks']+=sum(family)
        exposure['weak_course_ticks']+=sum(row['dagger']['opponent_course'][str(i)]['visited'] for i in [1,2])
    denominator=max(1.,exposure['valid_decision_ticks'])
    exposure['league_fraction_of_valid']={k:exposure[k]/denominator for k in ['current','ring','fixed_league']}
    exposure['executed_after_overrides']=dict(**executed,generations=executed_generations,
        fraction_of_valid={k:executed[k]/max(1,executed['valid_decision_ticks']) for k in ['current','ring','fixed_league']},
        scope='Final action-producing source after course/assist overrides; actual alive-learner tick denominator') if executed else None
    exposure['limitation']='Underlying sampled family ticks include natural weak-course overrides; actual weak-role visited ticks are reported separately, not mislabelled as unmodified policy exposure.'
    result=dict(generation=g,policy_sha256=policy_hash,hunter_js=hunter_summary,control_funnel=control,control_protocol=dict(hunter_native_trap=False,control_available_in_hunter=False,native_control_measured_in='production JS route/native_full tasks'),
        tactical_python=read(target)['settled_style_summary'],route=read(route)['results'],
        wall_seconds=time.monotonic()-start,source_simulators=dict(tactical='Python Tactical v2 via JAX simulator, CPU',hunter='production JS sim.js and Hunter',routes='production JS sim.js and Hunter normal',training_bots='JAX training only; never merged with evaluation'),
        actual_training_exposure=exposure)
    def numeric_delta(left,right):
        if isinstance(left,dict) and isinstance(right,dict):return {k:numeric_delta(left[k],right[k]) for k in left.keys()&right.keys() if isinstance(left[k],(int,float,dict))}
        return left-right if isinstance(left,(int,float)) and isinstance(right,(int,float)) else None
    previous=[p for p in POINTS if p<g and (OUT/f'g{p:06d}'/'SUMMARY.json').exists()]
    result['delta_parent']=numeric_delta(result,read(OUT/'g000000'/'SUMMARY.json')) if g else None
    result['delta_previous']=numeric_delta(result,read(OUT/f'g{max(previous):06d}'/'SUMMARY.json')) if previous else None
    atomic_json(out/'SUMMARY.json',result)
    from scripts.je10_diagnostics import diagnose
    diagnose(BASE,result)
    files=shards+[target,route,out/'SUMMARY.json',out/'DIAGNOSIS.json']
    atomic_json(out/'COMPLETE.json',dict(passed=True,generation=g,checkpoint_sha256=policy_hash,
        shards={str(p.relative_to(out)):sha(p) for p in files},capability_gate=False))
    notify(f'JE10异步评估g{g}完成：Hunter难/中/易安全击杀'+str([hunter_summary[d].get('safe_win') for d in ['hard','normal','easy']])+'；固定训练继续，详见authority_je10。')
    # Native joint fit under the same CPU authority.
    # Fixed quick-screen COMPLETE is already durable; audit is diagnostic only.
    if read(BASE/'plan.json').get('prestock_preparation'):
        try:
            from scripts.audit_joint_dagger import audit as joint_fifo_audit
            joint_fifo_audit(BASE,g)
        except Exception as error:
            atomic_json(out/'DAGGER_JOINT_AUDIT_FAILURE.json',dict(error=str(error),controls_gpu=False,
                fixed_evaluation_complete=True,capability_gate=False))
    return result


def owner():
    OUT.mkdir(parents=True,exist_ok=True);unique=lock(OUT/'evaluator.lock')
    # One CPU/JS authority across overlapping fixed-run relays. Queueing is
    # strictly CPU-side and never delays a GPU successor.
    import fcntl
    global_lock=(ROOT/'runs/je10_bun_20261018/evaluation.lock').open('a')
    atomic_json(OUT/'OWNER_WAIT.json',dict(reason='global CPU authority serialization',controls_gpu=False))
    fcntl.flock(global_lock.fileno(),fcntl.LOCK_EX)
    manifest=read(ROOT/'data/evaluation/bun_hunter_spawns_v1.json')
    manifest.update(seed=20261001,source_spawn_seed=manifest['seed'],seed_scope='evaluation RNG fixed20261001; identical frozen64 spawn cells across every checkpoint')
    atomic_json(OUT/'spawns.json',manifest)
    atomic_json(OUT/'PROTOCOL.json',read(BASE/'ROUND_DECLARATION.json')['evaluation'])
    atomic_json(OUT/'OWNER.json',dict(unique=True,global_unique=True,cpu_only=True,controls_gpu=False,namespace=NAMESPACE))
    if (OUT/'OWNER_BLOCKED.json').exists() and not unresolved_training_failure(BASE):
        atomic_json(OUT/'OWNER_RESUMED.json',dict(reason='historical technical failure predates verified formal nonzero recovery',
            failure_marker_preserved=True,queue_resumes_from='first incomplete fixed checkpoint',controls_gpu=False))
    completed=[];failed={}
    for g in POINTS:
        cp=BASE/'formal'/ARM/f'gen{g:06d}'/'COMMITTED.json'
        while not cp.exists():
            atomic_json(OUT/'QUEUE.json',dict(completed=completed,pending=[i for i in POINTS if i not in completed],running=None,failed=failed))
            if unresolved_training_failure(BASE):
                atomic_json(OUT/'OWNER_BLOCKED.json',dict(reason='training technical failure; no GPU action taken'));return
            time.sleep(5)
        atomic_json(OUT/'QUEUE.json',dict(completed=completed,pending=[i for i in POINTS if i not in completed],running=g,failed=failed))
        if not (OUT/f'g{g:06d}'/'COMPLETE.json').exists():
            for attempt in range(3):
                try:evaluate(g);break
                except Exception as error:
                    failed[str(g)]=str(error)
                    atomic_json(OUT/f'FAILED_g{g:06d}_{attempt}.json',dict(error=str(error),retry=attempt,controls_gpu=False))
                    if attempt==2:
                        notify(f'JE10评估技术阻塞g{g}：{error}；GPU不受影响。');return
        completed.append(g)
    atomic_json(OUT/'QUEUE.json',dict(completed=completed,pending=[],running=None,failed=failed))
    atomic_json(OUT/'EVALUATION_COMPLETE.json',dict(passed=True,completed=completed,cpu_only=True,capability_gate=False))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='tactical':tactical(sys.argv[2],sys.argv[3])
    else:owner()
