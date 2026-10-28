"""Independent fixed-budget iterative stage-switch DAgger adapter."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
os.environ.setdefault('JE09_NAMESPACE','runs/je09_bun_20261016/stage_h5c_native_course_dagger_pfsp8_v26')
BASE=ROOT/os.environ['JE09_NAMESPACE']
ARM='bun392_bridge_pfsp8'


def configure():
    from scripts import je09_stage_h3 as parent
    ctl=parent.configure()
    d=json.loads((BASE/'ROUND_DECLARATION.json').read_text())
    checkpoint=d['parent'];source_ns=str(Path(checkpoint).parents[2])
    import scripts.je09_bun_migration as migration
    migration.SOURCE_NAMESPACE=source_ns;migration.SOURCE_CHECKPOINT=checkpoint
    migration.SOURCE_EVALUATION=str(Path(checkpoint).relative_to(source_ns)/'COMMITTED.json')
    ctl.SOURCE_NAMESPACE=source_ns;ctl.SOURCE=ROOT/source_ns;ctl.SOURCE_CHECKPOINT=checkpoint
    ctl.SOURCE_EVALUATION=migration.SOURCE_EVALUATION
    ctl.SCHEMA=migration.SCHEMA='je09-eight-device-stage-h5c-native-course-dagger-v26'
    def source_audit():
        from scripts.population_run_state import sha
        cp=ROOT/checkpoint;marker=json.loads((cp/'COMMITTED.json').read_text())
        assert marker['passed'] and marker['generation']==d['parent_generation']
        assert sha(cp/'state.pkl')==marker['state_sha256'] and sha(cp/'policy.pt')==marker['policy_sha256']
        return cp,marker
    ctl.source_audit=source_audit
    ctl.ENTRY='scripts/je09_stage_h5c.py';ctl.SMOKE_NAME='production_smoke_stage_h5'
    from scripts import je09_stage_h5c_worker as worker,je09_stage_h5c_ops as ops,je09_stage_h_ops as old
    # Retain the exact installed Stage E/H return teacher and H3 lesson mix.
    ops.return_expert=old.return_expert
    sys.modules['scripts.je09_bun_worker']=worker
    sys.modules['scripts.je09_stage_h5c_worker']=worker
    original_launch=ctl.launch
    def launch(command,run,*args,**kwargs):
        if command=='worker' and Path(run).name==ctl.SMOKE_NAME:
            from scripts.population_run_state import atomic_json
            plan=json.loads((Path(run)/'plan.json').read_text())
            plan.update(envs_per_device=256,steps=1024)
            atomic_json(Path(run)/'plan.json',plan)
        return original_launch(command,run,*args,**kwargs)
    ctl.launch=launch
    original_worker=ctl.configure_worker
    def configure_worker(run):
        worker=original_worker(run)
        plan=json.loads((Path(run)/'plan.json').read_text())
        worker.SHAPE=[8,plan['envs_per_device'],plan['steps'],plan['minibatch_per_device']]
        worker.verify=verify
        return worker
    def verify(cp,force=False):
        from scripts.population_run_state import sha
        cp=Path(cp);marker=json.loads((cp/'COMMITTED.json').read_text())
        plan=json.loads((cp.parents[2]/'plan.json').read_text())
        assert marker['passed'] and marker['devices']==8 and marker['arm']==ARM
        assert marker['shape']==[8,plan['envs_per_device'],plan['steps'],plan['minibatch_per_device']]
        assert marker['state_schema']==ctl.SCHEMA and sha(cp/'state.pkl')==marker['state_sha256']
        assert marker['fresh_ticks']==marker['generation']*2097152
        assert marker['adam_steps']==[marker['generation']*128]*8
        return marker
    ctl.configure_worker=configure_worker;ctl.verify=verify
    return ctl


def prepare(ctl):
    from scripts import je09_stage_h3 as parent
    from scripts.population_run_state import atomic_json,sha
    parent.prepare(ctl)
    d=json.loads((BASE/'ROUND_DECLARATION.json').read_text())
    p=json.loads((BASE/'plan.json').read_text())
    p['campaign'].update(stage='H5',change=d['change'])
    p['parent_selection'].update(generation=d['parent_generation'],reason=d['parent_preservation'])
    p['dagger']=d['dagger'];p['dagger_opponent_course']=d['dagger_opponent_course'];p['state_schema']=ctl.SCHEMA
    p['continuation']['source_generation']=d['parent_generation']
    atomic_json(BASE/'plan.json',p)
    seal=json.loads((BASE/'SOURCE_SEAL.json').read_text());seal['plan_sha256']=sha(BASE/'plan.json')
    for name in ['scripts/je09_stage_h5c.py','scripts/je09_stage_h5c_ops.py','scripts/je09_stage_h5c_worker.py',
                 'scripts/je09_stage_h5c_dagger.py','scripts/je09_stage_h5c_proof.py','scripts/je09_stage_h5c_declare.py',
                 'scripts/je09_stage_h5c_relay.py','scripts/je09_stage_h5_course.py','tests/test_je09_stage_h5.py','tests/test_je09_stage_h5_course.py']:
        seal['sources'][name]=sha(ROOT/name)
        target=BASE/'inputs/sealed_sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    atomic_json(BASE/'SOURCE_SEAL.json',seal)
    atomic_json(BASE/'PREPARED.json',dict(passed=True,parent_generation=d['parent_generation'],
        plan_sha256=sha(BASE/'plan.json'),source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))
    atomic_json(BASE/'FROZEN_BEFORE_SMOKE.json',dict(round_sha256=sha(BASE/'ROUND_DECLARATION.json'),
        plan_sha256=sha(BASE/'plan.json'),source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','worker','restore','replay'])
    p.add_argument('--run',default=os.environ['JE09_NAMESPACE']);p.add_argument('--resume');a=p.parse_args()
    ctl=configure()
    if a.command=='prepare':prepare(ctl)
    elif a.command=='run':
        from scripts.je09_stage_h3 import run
        run(ctl)
    elif a.command=='restore':ctl.restore(ROOT/a.run,ROOT/a.resume)
    else:
        from scripts.je09_stage_h2_coverage import install
        install(json.loads((BASE/'ROUND_DECLARATION.json').read_text())['item_state_coverage']['fraction'])
        ctl.run_worker(ROOT/a.run,ROOT/a.resume if a.resume else None,a.command=='replay')
