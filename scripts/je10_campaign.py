"""Single durable eight-GPU campaign owner; technical smoke and fixed relay.

No capability/admission/promotion decisions control training. Evaluators are
separate CPU children. All public metadata uses repository-relative paths.
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('JAXBOMB_RULE','bun')
NAMESPACE=os.environ.get('JE10_NAMESPACE','runs/je10_bun_20261018/control_route_v38_r01_attempt04')
BASE=ROOT/NAMESPACE
ARM='bun392_bridge_pfsp8'
PARENT='runs/je09_bun_20261015/stage_c_return_set_pfsp8_v8/formal/bun392_bridge_pfsp8/gen000008'
TEMPLATE='runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36_attempt02'
GENERATIONS=128
POINTS=[0,4,8,16,32,64,96,128]
SHAPE=[8,1024,256,4096]


def read(p):return json.loads(Path(p).read_text())


def notify(message):
    print(message,flush=True)
    try:subprocess.run(['jdme-notify','send',message],timeout=20,check=False,stdout=subprocess.DEVNULL)
    except (FileNotFoundError,subprocess.TimeoutExpired):pass


def prepare(parent=PARENT,round_number=1):
    from scripts.population_run_state import atomic_json,sha
    from scripts.je03_bun831_20261011 import save_pickle
    d=read(ROOT/TEMPLATE/'ROUND_DECLARATION.json')
    parent_plan=ROOT/parent
    for _ in range(3): parent_plan=parent_plan.parent
    inherited_clean=parent_plan.joinpath('plan.json').exists() and read(parent_plan/'plan.json').get('clean_causal_init',False)
    clean_causal_init=os.environ.get('JE10_CLEAN_CAUSAL_INIT')=='1' or inherited_clean
    identity=f'JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r{round_number:02d}'
    if 'attempt' in NAMESPACE:identity+='-'+NAMESPACE.rsplit('_',1)[1]
    if clean_causal_init:
        identity += '-clean-causal-init'
    seeds={role:int.from_bytes(hashlib.sha256((identity+':'+role).encode()).digest()[:4],'big')
        for role in ['train','technical']}
    d.update(experiment_id=identity,namespace=NAMESPACE,round_number=round_number,parent=parent,parent_generation=8 if round_number==1 else GENERATIONS,
        seeds=dict(d['seeds'],**seeds),seed=seeds['train'],change='Integrated first-control causal credit + continuous PBRS .5 + set-valued learner-state DAgger .2, all route lessons and full custody lifetime; native contact residual wired, controlled death legacy16 removed',
        budget=dict(formal_generations=GENERATIONS,fresh_ticks=GENERATIONS*2097152,save_generations=POINTS,technical_smoke_generations=1),
        gates=dict(capability=False,admission=False,promotion=False,evaluation_blocks_training=False),
        terminal_relay=dict(successor=f'runs/je10_bun_20261018/control_route_v38_r{round_number+1:02d}'+('_causal_clean' if clean_causal_init else ''),
            parent='immediately preceding final raw Actor/Critic/pool',generations=GENERATIONS,
            policy='start fixed successor immediately before final evaluation/report; diagnose and propose next recipe asynchronously'),
        parent_selection=dict(checkpoint=parent,published='web-bun392-je09-stage-c-g8-v1' if round_number==1 else 'research successor',
            reason='Published g8 controlled capture15/16 main13/16 holdout Tactical17/64 Hunter2/3/29; transport1536 Hunter3/1/32 but no equivalent return-chain evidence; v36 full0/16 hard1/64. Prefer demonstrated return retention; no dominance/significance claim.'),
        route_pbrs=dict(enabled=True,lambda_value=.5,gamma=.995,continuous=True),
        kill_credit=dict(entry=4.,active_residual=.25,episode_cap=8.,death_clawback=True,capture_terminal=24.,controlled_death_legacy_kill=0.,direct_uncontrolled_settled_kill=16.,native_contact_popper=True),
        evaluation=dict(seed=20261001,points=POINTS,tactical_python_v2=64,hunter_production_js=dict(hard=64,normal=64,easy=64),
            max_steps=300,residual_steps=64,greedy=True,assist=False,route_funnel_seeds=[2026101800+i for i in range(16)],
            route_funnel=['native_death_stock','native_sugar_stock','real_drop','carry_near','carry_mid','carry_far','recovery_threat','natural_full2400'],
            simulator_separation='Python Tactical / production JS Hunter / JAX training bots reported separately',
            review='sole production Review protocol; fixed first failing seed in declaration order, no seed selection'),
        geometry_scope='Only native Bun map806 exists in repository. Native/open/dynamic wall-opening geometries, both seats and distance buckets are in training; additional map generalization remains unproven.')
    d['dagger'].update(coefficient=.2,query_policy='learner actual route scenes1..9; first deviation per stage; real custody to terminal, no120 cutoff; no takeover',
        loss='negative log total legal oracle action probability',oracle='next safe continuous high-return direction to real drop/enemy stock/own cage')
    if clean_causal_init:
        d['change'] += '; removed synthetic learner pre-stock scenes 2/3/4/6; learner custody requires native pickup/transfer event'
        d['training_scene_exclusions'] = {
            'scene_ids': [2,3,4,6],
            'scene_names': ['carry_route_far','carry_route_mid','carry_route_near','escort_under_pressure'],
            'reason': 'reset used to write learner bun_carried without a real event',
            'weight': 0.0,
        }
        d['dagger']['custody_source'] = 'native stock steal, credited loose pickup, or native contact transfer only'
        d['return_dagger_weight'] = {
            'enabled': True,
            'return_multiplier': 2.0,
            'other_phase_contributions_unchanged': True,
            'denominator': 'all valid accepted legal joint labels',
            'source': 'r11 declared return_dagger_weight; inherited from clean parent proposal',
        }
    assert not (BASE/'plan.json').exists(),'new namespace required'
    BASE.mkdir(parents=True,exist_ok=True)
    atomic_json(BASE/'ROUND_DECLARATION.json',d)
    p=read(ROOT/TEMPLATE/'plan.json')
    p.update(experiment_id=identity,generations=GENERATIONS,checkpoints=POINTS,recovery_every=4,
        total_fresh_ticks=GENERATIONS*2097152,seed=seeds['train'],state_schema='je10-control-route-v38',
        parent_selection=d['parent_selection'],route_pbrs=d['route_pbrs'],dagger=d['dagger'],
        no_cutoff=True,ring_interval=16,ring_generation_offset=0)
    if clean_causal_init:
        p['clean_causal_init']=True
        p['state_schema']='je10-control-route-v38-clean-causal-init-v1'
        p['return_dagger_weight']=d['return_dagger_weight']
    p['campaign'].update(stage='JE10-v38',change=d['change'],stopping='technical failure, hard safety deterioration or user stop only; no capability gates')
    # Keep 40% combat,25% real-drop bridge,35% full native learner games.
    p.update(combat_fraction=.4,bridge_fraction=.25)
    p['carry_assist']['anneal']='disabled; fixed .1 carry phase assistance; evaluation assist-off'
    atomic_json(BASE/'plan.json',p)
    for folder in ['inputs','state','formal','cache','validation','authority_je09_v1','diagnostics']:(BASE/folder).mkdir(exist_ok=True)
    cp=ROOT/parent;marker=read(cp/'COMMITTED.json')
    assert marker['passed'] and sha(cp/'policy.pt')==marker['policy_sha256'] and sha(cp/'state.pkl')==marker['state_sha256']
    with (cp/'state.pkl').open('rb') as f:saved=pickle.load(f)
    import jax
    initial={k:jax.tree.map(lambda x:x[0],saved['live']['params'][k]) for k in ['actor','critic']}
    save_pickle(BASE/'inputs/initial.pkl',initial)
    save_pickle(BASE/'inputs/retention_parent.pkl',initial)
    save_pickle(BASE/'inputs/initial_pool.pkl',saved['live']['pool'])
    del saved
    for name in ['frozen.pkl','regions.pkl','return_replay.pkl']:
        shutil.copyfile(ROOT/TEMPLATE/'inputs'/name,BASE/'inputs'/name)
    shutil.copyfile(ROOT/TEMPLATE/'regions.json',BASE/'regions.json')
    proof=ROOT/'runs/je09_bun_20261015/stage_e_long_return_coverage_pfsp8_v10/validation/RETURN_REPLAY.json'
    from scripts.je10_causal_replay import load_verified
    load_verified(BASE/'inputs/return_replay.pkl',proof)
    shutil.copyfile(proof,BASE/'inputs/RETURN_REPLAY.json')
    p['fixed_return_replay_provenance']=dict(proof='inputs/RETURN_REPLAY.json',
        native_drop_pickup=True,short_rows=1062,long_rows=996,reject_unproved=True)
    p['return_replay_sha256']=sha(BASE/'inputs/return_replay.pkl')
    atomic_json(BASE/'plan.json',p)
    names=list(Path('scripts').glob('je0[4-9]*.py'))+[x for x in Path('scripts').glob('je10*.py') if x.name!='je10_evaluate.py']+list(Path('jax_bomb').glob('je0*.py'))
    names+=[Path('jax_bomb/bun_env.py')]
    names+=list(Path('tests').glob('test_je10*.py'))+[Path('scripts/je03_bun831_20261011.py'),Path('jax_bomb/jax_train.py'),Path('jax_bomb/bun_jax_bots.py')]
    sources={str(name):sha(ROOT/name) for name in names}
    for name in sources:
        target=BASE/'inputs/sealed_sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    atomic_json(BASE/'SOURCE_SEAL.json',dict(sources=sources,plan_sha256=sha(BASE/'plan.json')))
    atomic_json(BASE/'PREPARED.json',dict(passed=True,parent=parent,parent_policy_sha256=sha(cp/'policy.pt'),seed=seeds['train']))


def configure_worker(run):
    from scripts import je10_worker as worker,je10_ops as ops
    from scripts.population_run_state import sha
    ops.install()
    worker.BASE=BASE;worker.RUN=Path(run);worker.ARM=ARM;worker.DEVICE_COUNT=8
    worker.SHAPE=SHAPE;worker.STATE_SCHEMA=read(Path(run)/'plan.json')['state_schema'];worker.POINTS=read(Path(run)/'plan.json')['checkpoints']
    def verify(cp,force=False):
        cp=Path(cp);m=read(cp/'COMMITTED.json')
        assert m['passed'] and m['replicas_exact'] and m['devices']==8 and m['shape']==SHAPE
        assert m['state_schema']==worker.STATE_SCHEMA and sha(cp/'state.pkl')==m['state_sha256']
        assert sha(cp/'policy.pt')==m['policy_sha256']
        return m
    worker.verify=verify
    return worker


def launch(command,run,resume=None,cpu=False,log_name=None):
    from scripts.je03_bun831_20261011 import child_env
    env=child_env('' if cpu else '0,1,2,3,4,5,6,7',cpu)
    env.update(JE10_NAMESPACE=NAMESPACE,JE09_H6_VERIFIER_SLOTS='4',JAX_COMPILATION_CACHE_DIR=str(BASE/'cache'/('cpu' if cpu else 'gpu')))
    args=['taskset','-c','96-111' if cpu else '0-79',sys.executable,'scripts/je10_campaign.py',command,'--run',str(Path(run).relative_to(ROOT))]
    if resume:args+=['--resume',str(Path(resume).relative_to(ROOT))]
    return subprocess.Popen(args,cwd=ROOT,env=env,stdout=(BASE/(log_name or command+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True)


def check_seal():
    from scripts.population_run_state import sha
    seal=read(BASE/'SOURCE_SEAL.json')
    assert sha(BASE/'plan.json')==seal['plan_sha256']
    assert all(sha(ROOT/name)==value for name,value in seal['sources'].items()),'sealed source changed'


def run(skip_smoke=False):
    from scripts.population_run_state import atomic_json,lock
    owner=lock(BASE/'campaign.lock');check_seal()
    reuse=os.environ.get('JE10_REUSE_TECHNICAL_CHECKPOINT')
    if reuse:
        cp=ROOT/reuse
        from scripts.population_run_state import sha
        marker=read(cp/'COMMITTED.json');row=read(cp.parent/'STATE.json')
        assert marker['passed'] and marker['replicas_exact'] and marker['devices']==8
        assert marker['fresh_ticks']==2097152 and row['finite'] and row['nonzero_update']
        assert sha(cp/'policy.pt')==marker['policy_sha256'] and sha(cp/'state.pkl')==marker['state_sha256']
        assert launch('restore',cp.parents[2],cp,cpu=True,log_name='smoke_restore.log').wait()==0
        atomic_json(BASE/'TECHNICAL_SMOKE_COMPLETE.json',dict(passed=True,finite=True,nonzero_update=True,devices=8,save_restore=True,capability_gate=False,reused_checkpoint=reuse,reason='host-only budget variable and JSON persistence fix; numerical learner unchanged'))
    elif not skip_smoke:
        smoke=BASE/'technical_smoke01';(smoke/'inputs').mkdir(parents=True)
        for name in ['initial.pkl','initial_pool.pkl','frozen.pkl','regions.pkl','retention_parent.pkl']:
            shutil.copyfile(BASE/'inputs'/name,smoke/'inputs'/name)
        shutil.copyfile(BASE/'regions.json',smoke/'regions.json')
        p=read(BASE/'plan.json');p.update(generations=1,checkpoints=[0,1],seed=read(BASE/'ROUND_DECLARATION.json')['seeds']['technical'])
        atomic_json(smoke/'plan.json',p)
        proc=launch('worker',smoke,log_name='smoke.log')
        notify('JE10 v38：8卡技术smoke已启动，只检查有效更新、NaN/OOM和保存恢复；不设能力门禁。')
        if proc.wait()!=0:raise RuntimeError('technical smoke worker failed; see smoke.log')
        row=read(smoke/'formal'/ARM/'STATE.json')
        assert row['finite'] and row['nonzero_update'] and row['fresh_ticks']==2097152
        if launch('restore',smoke,smoke/'formal'/ARM/'gen000001',cpu=True,log_name='smoke_restore.log').wait()!=0:
            raise RuntimeError('fresh CPU checkpoint restore failed')
        atomic_json(BASE/'TECHNICAL_SMOKE_COMPLETE.json',dict(passed=True,finite=True,nonzero_update=True,devices=8,save_restore=True,capability_gate=False))
    train=launch('worker',BASE,log_name='formal.log')
    atomic_json(BASE/'TRAIN_LAUNCH.json',dict(started=True,devices=list(range(8)),fixed_generations=GENERATIONS,fresh_ticks=GENERATIONS*2097152,capability_gate=False,namespace=NAMESPACE))
    notify(f'JE10 v38正式长训启动：{NAMESPACE}，8卡，固定128代/268435456 ticks；无能力门禁，终态先接后继。')
    # Evaluator is a CPU process and cannot delay or stop the GPU worker.
    eval_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JE10_NAMESPACE=NAMESPACE)
    subprocess.Popen(['taskset','-c','80-95',sys.executable,'scripts/je10_evaluate.py'],cwd=ROOT,env=eval_env,
        stdout=(BASE/'async_eval.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
    while train.poll() is None:time.sleep(10)
    if train.returncode:raise RuntimeError(f'formal worker failed with exit {train.returncode}')
    result=read(BASE/'formal'/ARM/'TRAINER_EXIT.json')
    assert result['generation']==GENERATIONS,'fixed run interrupted; successor withheld for technical audit'
    d=read(BASE/'ROUND_DECLARATION.json');successor=d['terminal_relay']['successor']
    # Durable successor owner starts before final restore, reports or eval wait.
    child=subprocess.Popen([sys.executable,'scripts/je10_campaign.py','relay','--parent',f'{NAMESPACE}/formal/{ARM}/gen{GENERATIONS:06d}',
        '--round',str(d['round_number']+1)],cwd=ROOT,
        env={**{k:v for k,v in os.environ.items() if k!='JE10_REUSE_TECHNICAL_CHECKPOINT'},'JE10_NAMESPACE':successor,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','JAX_PLATFORM_NAME':'cpu'},
        stdout=(BASE/'successor.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(BASE/'RUN_TERMINAL.json',dict(completed=True,**result,successor=successor,successor_owner_started=child.poll() is None))
    notify(f'JE10固定run完成：{NAMESPACE}，128代/268435456 ticks；预声明后继{successor}已启动，评估异步继续。')
    launch('restore',BASE,BASE/'formal'/ARM/f'gen{GENERATIONS:06d}',cpu=True,log_name='final_restore.log').wait()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run','worker','restore','relay'])
    ap.add_argument('--run',default=NAMESPACE);ap.add_argument('--resume');ap.add_argument('--parent',default=PARENT);ap.add_argument('--round',type=int,default=1)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.parent,a.round)
    elif a.command=='run':run()
    elif a.command=='relay':prepare(a.parent,a.round);run(skip_smoke=True)
    else:
        check_seal();w=configure_worker(ROOT/a.run)
        if a.command=='restore':w.restore_monitor(ROOT/a.resume)
        else:w.worker(ROOT/a.resume if a.resume else None)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if BASE.exists():
            from scripts.population_run_state import atomic_json
            atomic_json(BASE/'TECHNICAL_FAILURE.json',dict(error=str(error),command=sys.argv[1] if len(sys.argv)>1 else None))
        notify(f'JE10技术阻塞：{NAMESPACE}；{type(error).__name__}: {error}')
        raise
