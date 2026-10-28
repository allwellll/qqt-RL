"""Stage E bounded campaign, using immutable Stage C/D production components."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('JE09_NAMESPACE','runs/je09_bun_20261015/stage_e_long_return_coverage_pfsp8_v10')
os.environ.setdefault('JAXBOMB_RULE','bun')
BASE=ROOT/os.environ['JE09_NAMESPACE'];ARM='bun392_bridge_pfsp8';POINTS=[0,1,4,8,16]

def configure():
    from scripts import je09_bun_migration as m
    m.NAMESPACE=os.environ['JE09_NAMESPACE'];m.STAGE_B=m.STAGE_C=m.STAGE_D=True
    m.SOURCE_NAMESPACE='runs/je09_bun_20261015/stage_c_return_set_pfsp8_v8';m.SOURCE_CHECKPOINT=m.SOURCE_NAMESPACE+'/formal/'+ARM+'/gen000008'
    m.SCHEMA='je09-eight-device-stage-e-long-return-v10'
    m.SOURCE_EVALUATION='authority_je09_v1/EVALUATION_COMPLETE.json'
    original=m.continuation_plan
    def plan(source):
        p=original(source);d=json.loads((BASE/'ROUND_DECLARATION.json').read_text())
        from scripts.population_run_state import sha
        p['recovery_every']=4
        p['campaign']['stage']='E';p['return_coverage']=d['curriculum'];p['return_replay_sha256']=sha(BASE/'inputs/return_replay.pkl');return p
    m.continuation_plan=plan
    from scripts import je09_bun_continuation as ctl
    import scripts.je09_stage_e_worker as ew
    sys.modules['scripts.je09_bun_worker']=ew
    ctl.POINTS=POINTS;ctl.ENTRY='scripts/je09_stage_e.py';ctl.SMOKE_NAME='production_smoke_stage_e'
    return ctl

def prepare(ctl):
    ctl.prepare()
    from scripts.population_run_state import sha,atomic_json
    import shutil
    seal=json.loads((BASE/'SOURCE_SEAL.json').read_text())
    for name in ['scripts/je09_stage_e.py','scripts/je09_stage_e_worker.py','scripts/je09_stage_e_ops.py','scripts/je09_stage_e_coverage.py','scripts/je09_stage_e_scan.py','scripts/je09_web_stage_e.js','scripts/je09_stage_e_oracle.js','tests/test_je09_stage_e.py']:
        seal['sources'][name]=sha(ROOT/name);target=BASE/'inputs/sealed_sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    atomic_json(BASE/'SOURCE_SEAL.json',seal)
    prepared=json.loads((BASE/'PREPARED.json').read_text());prepared['source_seal_sha256']=sha(BASE/'SOURCE_SEAL.json');atomic_json(BASE/'PREPARED.json',prepared)

def evaluate(g):
    from scripts import je09_bun_evaluator as ev
    from scripts.je03_bun831_20261011 import child_env
    ev.BASE=BASE;ev.POINTS=POINTS;ev.STAGE_B=True
    cp=BASE/'formal'/ARM/f'gen{g:06d}';out=BASE/'authority_je09_v1'/f'{ARM}_g{g:06d}';out.mkdir(parents=True,exist_ok=True)
    from scripts.je08_bun_authority import export
    model=export(cp/'policy.pt',out/'model')
    for mode,name in [('main','FIXED32.json'),('holdout','FIXED32_HOLDOUT.json'),('stage_b_main','STAGE_B.json'),('stage_b_holdout','STAGE_B_HOLDOUT.json'),('stage_a_main','STAGE_A.json'),('stage_a_holdout','STAGE_A_HOLDOUT.json')]:
        if not (out/name).exists():subprocess.run(['node','scripts/je09_web_stage_e.js',str(model),str(out/name),mode],env=child_env('',True),check=True)
    ev.evaluate_point(out.name,cp/'policy.pt',json.loads((cp/'COMMITTED.json').read_text()))

def run(ctl):
    from scripts.population_run_state import atomic_json
    from scripts.je03_bun831_20261011 import child_env
    ctl.check_seal();assert json.loads((BASE/'validation/RETURN_PRODUCTION_GRADIENT.json').read_text())['passed']
    ctl.smoke();smoke=BASE/ctl.SMOKE_NAME
    assert ctl.launch('restore',smoke,smoke/'formal'/ARM/'gen000001',cpu=True,log_name='restore.log').wait()==0
    assert ctl.launch('replay',smoke,smoke/'formal'/ARM/'gen000000',log_name='replay.log').wait()==0
    assert json.loads((smoke/'state/FRESH_GPU_REPLAY.json').read_text())['next_update_exact']
    assert not (BASE/'formal'/ARM/'gen000000/COMMITTED.json').exists()
    training=ctl.launch('worker',BASE,log_name='formal.log');pending=POINTS.copy();jobs={};done=[]
    def job(g):
        with (BASE/f'evaluate_g{g:06d}.log').open('a') as log:
            subprocess.run(['taskset','-c','80-127',sys.executable,'scripts/je09_stage_e.py','evaluate','--generation',str(g)],env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,check=True)
        return g
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while pending or jobs:
            for g in pending[:]:
                if len(jobs)>=3:break
                if (BASE/'formal'/ARM/f'gen{g:06d}/COMMITTED.json').exists():jobs[pool.submit(job,g)]=g;pending.remove(g)
            for future in list(jobs):
                if future.done():done.append(future.result());del jobs[future]
            if training.poll() is not None and training.returncode:raise RuntimeError('training failed; inspect formal.log')
            if pending or jobs:time.sleep(5)
    assert training.wait()==0
    assert ctl.launch('restore',BASE,BASE/'formal'/ARM/'gen000016',cpu=True,log_name='final_restore.log').wait()==0
    atomic_json(BASE/'authority_je09_v1/EVALUATION_COMPLETE.json',dict(passed=True,completed_generations=sorted(done),assist_off=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run','worker','replay','restore','evaluate']);ap.add_argument('--run',default=os.environ['JE09_NAMESPACE']);ap.add_argument('--resume');ap.add_argument('--generation',type=int);a=ap.parse_args()
    ctl=configure()
    if a.command=='prepare':prepare(ctl)
    elif a.command=='run':run(ctl)
    elif a.command=='evaluate':evaluate(a.generation)
    elif a.command=='restore':ctl.restore(ROOT/a.run,ROOT/a.resume)
    else:
        from scripts.je09_stage_e_coverage import install,bind_ops
        bind_ops();install(json.loads((BASE/'ROUND_DECLARATION.json').read_text())['curriculum']['long_return_fraction'])
        ctl.run_worker(ROOT/a.run,ROOT/a.resume if a.resume else None,a.command=='replay')
