"""Unique CPU/production-JS retention lane; never launches GPU training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
os.environ['NODE_OPTIONS']='--require=./scripts/dagger_rl_metrics_v2_preload.js'
import sys,json,time,pickle,shutil,subprocess,concurrent.futures
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.ppo_scale_longrun_run import B,read,notify
from scripts.population_run_state import atomic_json,sha,lock


def evaluate(cp):
    cp=Path(cp);out=cp/'retention';out.mkdir(exist_ok=True)
    receipt=read(cp/'COMMITTED.json');assert sha(cp/'state.pkl')==receipt['state_sha256'] and sha(cp/'policy.pt')==receipt['policy_sha256']
    if (out/'COMPLETE.json').exists():return
    from scripts.retention_phase_export import authority_export
    from scripts.je08_bun_authority import summarize_hunter
    model=authority_export(cp/'policy.pt',out/'model');ph=sha(cp/'policy.pt');protocol=read(B/'CPU_EVAL_PROTOCOL.json')
    spawn=B/'cpu_eval_inputs/spawns.json';regions=B/'cpu_eval_inputs/regions.json'
    def hunter(job):
        tier,start=job;target=out/f'hunter_{tier}_{start:03d}.json'
        if not target.exists():
            subprocess.run(['node','scripts/je10_web_hunter.js','--model',str(model.relative_to(R)),'--difficulty',tier,'--spawns',str(spawn.relative_to(R)),'--start',str(start),'--end',str(start+8),'--seed',str(protocol['hunter_seed']),'--checkpoint-sha',ph,'--out',str(target.relative_to(R))],check=True,stdout=subprocess.DEVNULL)
        d=read(target);assert d['checkpoint_sha256']==ph if 'checkpoint_sha256' in d else True
        assert len(d['episodes'])==8
        return d
    tactical=out/'tactical_v2.json';tp=None
    if not tactical.exists():tp=subprocess.Popen([sys.executable,'scripts/ppo_scale_longrun_cpu.py','tactical',str(cp.relative_to(R))],stdout=(out/'tactical.log').open('a'),stderr=subprocess.STDOUT)
    natural=out/'natural_full64.json';np_=None
    if not natural.exists():
        env=dict(os.environ,JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='64',JE10_ROUTE_SEED_START=str(protocol['natural_seed_start']))
        np_=subprocess.Popen([sys.executable,'scripts/ppo_scale_longrun_cpu.py','natural',str(cp.relative_to(R))],env=env,stdout=(out/'natural.log').open('a'),stderr=subprocess.STDOUT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:shards=list(pool.map(hunter,[(t,s) for t in ['hard','normal','easy'] for s in range(0,64,8)]))
    if tp:assert tp.wait()==0,'Python Tactical failure'
    if np_:assert np_.wait()==0,'natural full JS failure'
    hunter_summary={tier:summarize_hunter([e for s in range(0,64,8) for e in read(out/f'hunter_{tier}_{s:03d}.json')['episodes']]) for tier in ['hard','normal','easy']}
    atomic_json(out/'COMPLETE.json',dict(passed=True,policy_sha256=ph,protocol_sha256=sha(B/'CPU_EVAL_PROTOCOL.json'),hunter=hunter_summary,tactical=read(tactical)['settled_style_summary'],natural=read(natural),engines_separate=True,controls_gpu=False,unix=time.time()))
    notify(f'PPO_SCALE_R1 CPU/JS保留评估终态：{cp.parent.name}/{cp.name}，Hunter难中易各64、Python Tactical64、native自然完整64已保存。')


def natural(cp, seed_start=None):
    from scripts.retention_route_eval import merge_episodes
    cp=Path(cp);out=cp/'retention';model=out/'model/model.json';regions=B/'cpu_eval_inputs/regions.json'
    if seed_start is None:seed_start=read(B/'CPU_EVAL_PROTOCOL.json')['natural_seed_start']
    def job(start):
        folder=out/'natural_shards'/f'{start:03d}';folder.mkdir(parents=True,exist_ok=True);target=folder/'RESULT.json'
        if not target.exists():
            env=dict(os.environ,JE10_ROUTE_TASK='natural_full2400',JE10_ROUTE_GAMES='64',JE10_ROUTE_START=str(start),JE10_ROUTE_END=str(start+8),JE10_ROUTE_SEED_START=str(seed_start))
            subprocess.run(['node','scripts/dagger_rl_route.js',str(model.relative_to(R)),str(target.relative_to(R)),str(regions.relative_to(R))],env=env,check=True,stdout=(folder/'run.log').open('a'),stderr=subprocess.STDOUT)
        for pattern in ['REVIEW_FIXED_*.json','fixed-*.trace.json']:
            for source in folder.glob(pattern):shutil.copyfile(source,out/source.name)
        return read(target)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(job,range(0,64,8)))
    result=merge_episodes('natural_full2400',parts);result['execution']='8 independent episode CPU shards; unchanged global seeds, seat indices, engine and per-tick ONNX batch1'
    atomic_json(out/'natural_full64.json',result)


def consume(cp):
    code=subprocess.run([sys.executable,'scripts/ppo_scale_longrun_cpu.py','evaluate',str(cp.relative_to(R))],stdout=(B/'cpu_evaluator.log').open('a'),stderr=subprocess.STDOUT).returncode
    return cp,code


def dispatch_pending(pending):
    atomic_json(B/'CPU_STATUS.json',dict(stage='FIXED_RETENTION' if pending else 'WAITING_SAVED_POINTS',checkpoints=[str(cp.relative_to(R)) for cp in pending],controls_gpu=False,unix=time.time()))
    atomic_json(B/'CPU_ACTIVE_JOBS.json',dict(unique_owner=True,max_concurrent_models=2,pending=[str(cp.relative_to(R)) for cp in pending],controls_gpu=False,unix=time.time()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(consume,pending))
