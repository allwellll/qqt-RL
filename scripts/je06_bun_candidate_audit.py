"""Read-only candidate integrity and same-protocol native web task comparison."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,pickle,json,time,concurrent.futures,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.je06_bun_recipe import BASE,ROOT,PREVIOUS,PARENT_CP,read,sha,atomic_json,digest,verify_checkpoint,unrep,child_env
from scripts.je05_bun_authority import export
CANDIDATES=[('je04',768),('je04',2048),('je05',1536),('je05',3072),('je05',4096)]

def source(label,g):
    run=ROOT/('runs/je04_bun_20261012/retention3_pfsp8_v1' if label=='je04' else 'runs/je05_bun_20261013/combat_transport_pfsp8_v1')
    arm='bun392_retention3_pfsp8' if label=='je04' else 'bun392_combat_transport_pfsp8'
    ns='authority_je04_v1' if label=='je04' else 'authority_je05_v3'
    return run,run/'formal'/arm/f'gen{g:06d}',run/ns/f'{arm}_g{g:06d}'

def check(label,g):
    run,cp,out=source(label,g);m=verify_checkpoint(cp,sha(run/'plan.json'))
    assert sha(cp/'ema_policy.pt')==m['ema_policy_sha256']
    d=read(out/'COMPLETE.json');t=read(out/'COMPLETE_WITH_BUN_TASKS.json')
    assert d['policy_sha256']==m['policy_sha256']==t['policy_sha256']
    assert t['authority_sha256']==sha(out/'COMPLETE.json')
    assert all(sha(out/f)==h for f,h in d['shards'].items())
    assert all(sha(out/f)==h for f,h in t['additional_shards'].items())
    with (cp/'state.pkl').open('rb') as f:state=pickle.load(f)
    assert digest(state['live'])==m['full_state_digest']
    required={'params','adam','ema','states','aux','key','pool','funnel','pool_counts','course','pfsp'}
    assert set(state['live'])==required and state['shape']==[8,1024,128,4096]
    p=unrep(state['live']['params'])
    with (cp/'policy.pt').open('rb') as f:raw=pickle.load(f)
    assert digest(p['actor'])==digest(raw)
    critic=digest(p['critic']);del state,p,raw
    return dict(label=label,generation=g,checkpoint=str(cp.relative_to(ROOT)),checkpoint_proof=m,committed_sha256=sha(cp/'COMMITTED.json'),
        raw_authority_sha256=sha(out/'COMPLETE.json'),full_state_valid=True,critic_digest=critic,hunter=d['hunter'],tactical=d['tactical'],tasks=t['tasks'],
        task_protocol='legacy JAX idle' if label=='je04' else 'je05_native_transport_web_v2')

def native_tasks(label,g):
    _,cp,_=source(label,g);out=BASE/'parent_candidates'/f'{label}_g{g:06d}';model=export(cp/'policy.pt',out/'model')
    def task(name):
        target=out/f'web_task_{name}.json'
        if not target.exists():subprocess.run(['node','scripts/je05_web_tasks.js',str(model),str(target),name,str(PREVIOUS/'regions.json')],cwd=ROOT,env=child_env('',True),check=True)
        d=read(target);assert d['passed'] and d['model_json_sha256']==sha(model)
        return name,d['results'][name]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(task,['carry_home','near_steal','route_break','full']))
    atomic_json(out/'SAME_PROTOCOL_TASKS_COMPLETE.json',dict(passed=True,policy_sha256=sha(cp/'policy.pt'),protocol='je05_native_transport_web_v2',tasks=results))
    return results

def main():
    BASE.mkdir(parents=True,exist_ok=True);start=time.time()
    # Missing native task comparisons run in parallel with read-only state hashes.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        work={g:pool.submit(native_tasks,'je04',g) for g in [768,2048]}
        rows=[check(label,g) for label,g in CANDIDATES]
        for row in rows:
            if row['label']=='je04':row['tasks']=work[row['generation']].result();row['task_protocol']='je05_native_transport_web_v2 (fresh re-evaluation)'
    # Frozen reasoned holistic choice: hard/easy combat, no self episodes,
    # stronger Tactical than later JE05 points and native transport retention.
    selected=next(r for r in rows if r['label']=='je05' and r['generation']==1536)
    atomic_json(BASE/'PARENT_SELECTION.json',dict(passed=True,selected_checkpoint=str(PARENT_CP.relative_to(ROOT)),policy_kind='raw',candidates=rows,
        reason='JE05 g1536 retains hard3/easy32, Tactical22 without Tactical death/self/trade, Hunter self0 and fewer deaths than g3072; carry15/16, steal15/16. JE04 g768 has better normal/safety but Tactical19 and no hard wins; g2048 has Tactical26 but Hunter deaths6/self1/trade2 and hard0. No candidate solves route/full. Choose a Pareto compromise, not statistically significant dominance.',
        adjacency='JE05 g1024 and g1536 Hunter total36, g2048 regresses23 with safety loss; g1536 not stable all-tier superiority. Later3072/4096 Tactical17/18 and g3072 self3. All curves retained.',
        screening_only=True,wall_seconds=time.time()-start))
    print('PARENT_SELECTION COMPLETE JE05 raw g1536',flush=True)

if __name__=='__main__':main()
