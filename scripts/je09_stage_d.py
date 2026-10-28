"""Stage D isolated declaration, curriculum gates and complete CPU evaluation."""
import argparse,json,os,subprocess,sys,time,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.population_run_state import atomic_json,sha
from scripts.je03_bun831_20261011 import child_env
BASE=ROOT/os.environ.get('JE09_NAMESPACE','runs/je09_bun_20261015/stage_d_distance_set_pfsp8_v9')
ARM='bun392_bridge_pfsp8'

def gate(generation,stage):
    from scripts.je08_bun_authority import export
    cp=BASE/'formal'/ARM/f'gen{generation:06d}';out=BASE/'curriculum_gate'/f'g{generation:06d}';out.mkdir(parents=True,exist_ok=True)
    model=export(cp/'policy.pt',out/'model');target=out/'GATE.json'
    subprocess.run(['node','scripts/je09_web_stage_d.js',str(model),str(target),'gate',str(stage)],env=child_env('',True),check=True)
    r=json.loads(target.read_text())['results']['bucket'+str(stage)]
    atomic_json(BASE/'state'/f'DISTANCE_GATE_{generation:06d}.json',dict(assist=False,games=16,first_pickup=r['first_pickup'],stage=stage,next_stage=min(3,stage+int(r['first_pickup']>=12)),policy_sha256=sha(cp/'policy.pt'),evidence_sha256=sha(target)))

def evaluate(generation):
    from scripts import je09_bun_evaluator as ev
    cp=BASE/'formal'/ARM/f'gen{generation:06d}'
    label=f'{ARM}_g{generation:06d}';ev.BASE=BASE
    ev.POINTS=[1,4,5,8,9,12,13,16]
    out=BASE/'authority_je09_v1'/label;out.mkdir(parents=True,exist_ok=True)
    from scripts.je08_bun_authority import export
    model=export(cp/'policy.pt',out/'model')
    for mode,name in [('main','FIXED32.json'),('holdout','FIXED32_HOLDOUT.json')]:
        if not (out/name).exists():subprocess.run(['node','scripts/je09_web_stage_d.js',str(model),str(out/name),mode],env=child_env('',True),check=True)
    ev.evaluate_point(label,cp/'policy.pt',json.loads((cp/'COMMITTED.json').read_text()))

def oracle_audit():
    r=json.loads((BASE/'validation/ORACLE.json').read_text());rows=r['results']
    assert all(x['first_pickup']==16 for x in rows.values()),{k:v['first_pickup'] for k,v in rows.items()}
    atomic_json(BASE/'validation/ORACLE_AUDIT.json',dict(passed=True,protocol_unchanged=True,native_respawn_ticks=100,results={k:dict(first_pickup=v['first_pickup'],pickup_ticks=[e['pickup_tick'] for e in v['episodes']],distances=[e['initial']['distance'] for e in v['episodes']]) for k,v in rows.items()},source_sha256=sha(BASE/'validation/ORACLE.json')))

def run():
    from scripts import je09_bun_continuation as ctl
    ctl.check_seal();ctl.smoke()
    smoke=BASE/ctl.SMOKE_NAME
    cpu=ctl.launch('restore',smoke,smoke/'formal'/ARM/'gen000001',cpu=True,log_name='restore.log');assert cpu.wait()==0
    replay=ctl.launch('replay',smoke,smoke/'formal'/ARM/'gen000000',log_name='replay.log');assert replay.wait()==0
    assert json.loads((smoke/'state/FRESH_GPU_REPLAY.json').read_text())['next_update_exact']
    assert not (BASE/'formal'/ARM/'gen000000/COMMITTED.json').exists()
    training=ctl.launch('worker',BASE,log_name='formal.log')
    pending=list(ctl.POINTS);jobs={};completed=[]
    def evaluation_job(g):
        with (BASE/f'evaluate_g{g:06d}.log').open('a') as log:
            subprocess.run(['taskset','-c','96-127',sys.executable,'scripts/je09_stage_d.py','evaluate','--generation',str(g)],env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,check=True)
        return g
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        while pending or jobs:
            for g in pending[:]:
                if len(jobs)>=2:break
                if (BASE/'formal'/ARM/f'gen{g:06d}/COMMITTED.json').exists():jobs[pool.submit(evaluation_job,g)]=g;pending.remove(g)
            for future in list(jobs):
                if future.done():completed.append(future.result());del jobs[future]
            if training.poll() is not None and training.returncode:raise RuntimeError('training failed; inspect formal.log')
            if pending or jobs:time.sleep(5)
    assert training.wait()==0
    final=BASE/'formal'/ARM/'gen000016'
    cpu=ctl.launch('restore',BASE,final,cpu=True,log_name='final_restore.log');assert cpu.wait()==0
    atomic_json(BASE/'authority_je09_v1/EVALUATION_COMPLETE.json',dict(passed=True,completed_generations=sorted(completed),assist_off=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['gate','evaluate','oracle-audit','run']);ap.add_argument('--generation',type=int);ap.add_argument('--stage',type=int,default=0);a=ap.parse_args()
    if a.command=='gate':gate(a.generation,a.stage)
    elif a.command=='evaluate':evaluate(a.generation)
    elif a.command=='oracle-audit':oracle_audit()
    else:run()
