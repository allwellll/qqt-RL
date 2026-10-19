"""Sealed JE07 continuation: CPU preparation now, GPU only after JE06 finish."""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.population_run_state import atomic_json,sha,lock
from scripts.je03_bun_stage1_20261011 import digest,read,verify_checkpoint,gpu_owners
from scripts.je03_bun831_20261011 import save_pickle,child_env
from scripts.je05_bun_recipe import POOL,unrep,alive,Adopted

BASE=ROOT/'runs/je07_bun_20261013/drop_recovery_carry_assist_pfsp8_v1'
SMOKE_NAME='production_smoke_v1'
RUN=BASE/SMOKE_NAME if os.environ.get('JE07_PHASE')=='smoke' else BASE
ARM='bun392_drop_recovery_carry_assist_pfsp8'
PREDECESSOR=ROOT/'runs/je06_bun_20261013/holistic_long256_pfsp8_v1'
POINTS=[0,8,16,32,64,128,256,512,768,1024,1536,2048,3072,4096]
SOURCES=['jax_bomb/je07_bun_env.py','scripts/je07_bun_curriculum.py','scripts/je07_bun_ops.py',
    'scripts/je07_bun_scenes.py','scripts/je07_bun_recipe.py','scripts/je07_bun_worker.py',
    'scripts/je07_bun_supervisor.py','scripts/je07_bun_evaluator.py','scripts/je07_bun_authority.py',
    'scripts/je07_bun_cpu_smoke.py','scripts/je07_web_probe.js',
    *read(PREDECESSOR/'SOURCE_SEAL.json')['sources'].keys()]

def plan():
    from scripts.je06_bun_recipe import plan as previous
    from scripts.je07_bun_curriculum import PROBE_MANIFEST,PROBE_MANIFEST_SHA
    p=previous();p.update(experiment_id='JE07-BUN392-DROP-RECOVERY-CARRY-ASSIST-PFSP8-20261013-v1',
        arms=[ARM],generations=4096,total_fresh_ticks=8589934592,seed=2026101307,
        checkpoints=POINTS,drop_recovery_reward=8,capture_reward=24,combat_fraction=.45,
        parent_rule='best_completed_same_protocol_restorable_raw_at_handoff',
        initial_policy='selected completed authority raw Actor AND matching Critic; fresh Adam/RNG/env',
        extension_rule='complete4096 regardless of ability fluctuation; no capacity or A/B change',
        drop_identity=dict(origin='conserved map806 object',nonce='increments only on real loose drop',
            team='scoring learner team',cycle='episode-scoped per-origin cycle; never renewed by custody loss',
            reward_once=True,enemy_origin_only=True,attribution='unique enemy physical/causal death source and scoring team survived',
            contact_transfer='no loose event, no bonus',max_episode_drop_bonus=8,combined_positive_shaping_cap=12,
            clawback='unsettled bonus reclaimed on carrying death/drop/trade; capture settles after hazard chain'),
        carry_assist=dict(probabilities=[.7,.5,.3,.1],sample='once per actual enemy-carrier phase',
            duration_limit=600,bot='uniform legal move/wait only; no bomb/use_item/pursuit',
            end=['drop','death','capture','timeout','episode_end'],
            anneal='each new unassisted fixed-seed policy probe: near/mid/far>=13/16 and full capture>=1/16; monotone one level; no wait',
            probe_manifest=PROBE_MANIFEST,probe_manifest_sha256=PROBE_MANIFEST_SHA,
            strong_share='30/50/70/90% carrier phases plus all non-carrier combat',
            checkpointed_fields=['phase','selected','carrying','ticks','level','proofs','manifest_sha256']),
        chain_scene=dict(weight=.10,name='drop_chain_return',learner_empty=True,
            enemy_carries_own_recovered_bun=True,scripted_kill=False,scripted_drop=False,mid_far_route=True),
        evaluation=dict(namespace='authority_je07_v1',hunter_games_per_tier=64,tactical_games=64,
            bun_task_games=16,assist_enabled=False,protocol='frozen JE05 web native task v2 + Stage2 Hunter/Tactical; screening only',
            side_horizons=[600,900],assisted_diagnostics_namespace='assist_diagnostic_v1',
            assisted_scores_never_select_parent=True),
        handoff=dict(predecessor='runs/je06_bun_20261013/holistic_long256_pfsp8_v1',requires_training_complete=True,
            requires_predecessor_gpu_worker_exit=True,requires_gpu_idle=True,wait_for_cpu_evaluation=False,
            no_ability_gate=True,no_training_hotpatch=True),
        namespaces=dict(state='state',cache='cache',judge='judge',eval='authority_je07_v1'))
    return p

def candidate_score(hunter,tactical,tasks):
    """Frozen screening utility; assistance never participates."""
    score=6*tasks['full']['capture_episodes']+4*tasks['route_break']['capture_episodes']
    score+=tasks['carry_home']['capture_episodes']+tasks['near_steal']['steal_episodes']
    score+=3*hunter['hard']['safe_win']+2*hunter['normal']['safe_win']+hunter['easy']['safe_win']+tactical['safe_win']
    score-=sum(2*v['death_episode']+4*v['self_episodes']+3*v['trade_episodes'] for v in hunter.values())
    score-=2*tactical['self_episodes']
    score-=sum(2*v['self_episodes']+v['trade_episodes'] for v in tasks.values())
    return float(score)

def seal_cpu():
    BASE.mkdir(parents=True,exist_ok=True)
    for name in ['state','cache','judge','validation','inputs','authority_je07_v1','assist_diagnostic_v1']:(BASE/name).mkdir(exist_ok=True)
    p=plan()
    if (BASE/'plan.json').exists():assert read(BASE/'plan.json')==p
    else:atomic_json(BASE/'plan.json',p)
    seal=dict(sources={s:sha(ROOT/s) for s in dict.fromkeys(SOURCES)},plan_sha256=sha(BASE/'plan.json'))
    if (BASE/'SOURCE_SEAL.json').exists() and read(BASE/'SOURCE_SEAL.json')!=seal:
        atomic_json(BASE/'validation/SOURCE_SEAL_SUPERSEDED.json',dict(old=read(BASE/'SOURCE_SEAL.json'),reason='CPU-only implementation completed after initial seal',unix=time.time()))
    atomic_json(BASE/'SOURCE_SEAL.json',seal)
    atomic_json(BASE/'CPU_READY.json',dict(passed=True,sealed=True,gpu_started=False,
        plan_sha256=seal['plan_sha256'],parent_pending_at_handoff=True,unix=time.time()))

def check_seal():
    seal=read(BASE/'SOURCE_SEAL.json')
    assert seal['plan_sha256']==sha(BASE/'plan.json')
    assert seal['sources']=={s:sha(ROOT/s) for s in seal['sources']}

def choose_parent():
    """Rank completed same-protocol Pareto candidates, never a live policy.

    Full capture dominates diagnostic proxy progress; safety is explicit.
    Hunter tiers are balanced, Tactical/transport and adjacent stability count.
    All candidates must provide exact complete raw Actor+Critic state.
    """
    candidates=[]
    for namespace,arm in [('je04_bun_20261012/retention3_pfsp8_v1','bun392_retention3_pfsp8'),
            ('je05_bun_20261013/combat_transport_pfsp8_v1','bun392_combat_transport_pfsp8'),
            ('je06_bun_20261013/holistic_long256_pfsp8_v1','bun392_holistic_long256_pfsp8')]:
        base=ROOT/'runs'/namespace
        authority=base/({'je04':'authority_je04_v1','je05':'authority_je05_v3','je06':'authority_je06_recovery_v2'}[namespace[:4]])
        for result in sorted(authority.glob(arm+'_g*/COMPLETE_WITH_BUN_TASKS.json')):
            row=read(result);g=int(result.parent.name.rsplit('_g',1)[1]);cp=base/'formal'/arm/f'gen{g:06d}'
            try:
                proof=verify_checkpoint(cp,sha(base/'plan.json'))
                assert row['policy_sha256']==proof['policy_sha256']
                combat=read(result.parent/'COMPLETE.json')
                assert all(sha(result.parent/k)==v for k,v in combat['shards'].items())
                assert row['authority_sha256']==sha(result.parent/'COMPLETE.json')
                assert all(sha(result.parent/k)==v for k,v in row['additional_shards'].items())
                cache=BASE/'state'/('PARENT_STATE_'+sha(cp/'COMMITTED.json')+'.json')
                if not cache.exists():
                    with (cp/'state.pkl').open('rb') as f:s=pickle.load(f)
                    assert digest(s['live'])==proof['full_state_digest'];params=unrep(s['live']['params']);del s
                    with (cp/'policy.pt').open('rb') as f:raw=pickle.load(f)
                    assert digest(params['actor'])==digest(raw) and 'critic' in params
                    atomic_json(cache,dict(passed=True,committed_sha256=sha(cp/'COMMITTED.json'),state_sha256=proof['state_sha256'],actor_digest=digest(params['actor']),critic_digest=digest(params['critic'])))
                    del params,raw
                assert read(cache)['state_sha256']==proof['state_sha256']
                tasks=row['tasks'];hunter=combat['hunter'];tactical=combat['tactical']
                if namespace.startswith('je04'):
                    same=PREDECESSOR/'parent_candidates'/f'je04_g{g:06d}'
                    native=read(same/'SAME_PROTOCOL_TASKS_COMPLETE.json')
                    assert native['policy_sha256']==proof['policy_sha256'] and native['protocol']=='je05_native_transport_web_v2'
                    assert all(read(same/f'web_task_{t}.json')['results'][t]==v for t,v in native['tasks'].items())
                    tasks=native['tasks']
                else:assert read(result.parent/'web_task_full.json')['protocol']=='je05_native_transport_web_v2'
                score=candidate_score(hunter,tactical,tasks)
                candidates.append(dict(checkpoint=str(cp.relative_to(ROOT)),result=str(result.relative_to(ROOT)),
                    score=float(score),generation=g,policy_sha256=proof['policy_sha256'],
                    committed_sha256=sha(cp/'COMMITTED.json'),result_sha256=sha(result),hunter=hunter,tactical=tactical,tasks=tasks))
            except Exception as error:
                atomic_json(BASE/'state'/('PARENT_REJECTED_'+sha(result)+'.json'),dict(result=str(result.relative_to(ROOT)),reason=repr(error)))
    assert candidates,'no completed restorable same-protocol parent'
    # Adjacent completed point stability adds only a bounded tie-break, never
    # selects based on training score or assisted diagnostic.
    for c in candidates:
        neighbors=sorted([x for x in candidates if Path(x['checkpoint']).parent==Path(c['checkpoint']).parent and x is not c],key=lambda x:abs(x['generation']-c['generation']))[:2]
        c['adjacent_scores']=[x['score'] for x in neighbors]
        c['stable_score']=c['score']+.05*min([x['score'] for x in neighbors] or [c['score']])
    winner=max(candidates,key=lambda c:(c['stable_score'],-c['generation']))
    atomic_json(BASE/'PARENT_SELECTION.json',dict(selected=winner,candidates=candidates,screening_only=True,
        rule='unassisted same-protocol composite + bounded adjacent stability; earliest tie'))
    return ROOT/winner['checkpoint']

def prepare_inputs():
    check_seal()
    if (BASE/'PREPARED.json').exists():
        assert all(sha(BASE/'inputs'/f)==h for f,h in read(BASE/'INITIALIZATION.json')['input_hashes'].items())
        return
    cp=choose_parent();m=read(cp/'COMMITTED.json')
    with (cp/'state.pkl').open('rb') as f:s=pickle.load(f)
    initial=unrep(s['live']['params']);assert digest(s['live'])==m['full_state_digest'];del s
    save_pickle(BASE/'inputs/initial.pkl',initial)
    for name in ['frozen.pkl','regions.pkl']:
        with (PREDECESSOR/'inputs'/name).open('rb') as f:save_pickle(BASE/'inputs'/name,pickle.load(f))
    atomic_json(BASE/'regions.json',read(PREDECESSOR/'regions.json'))
    atomic_json(BASE/'INITIALIZATION.json',dict(parent=str(cp.relative_to(ROOT)),policy_kind='raw',
        parent_committed_sha256=sha(cp/'COMMITTED.json'),parent_policy_sha256=m['policy_sha256'],parent_state_sha256=m['state_sha256'],
        actor_digest=digest(initial['actor']),critic_digest=digest(initial['critic']),fresh_optimizer=True,fresh_rng=True,fresh_env=True,
        input_hashes={f.name:sha(f) for f in (BASE/'inputs').iterdir()}))
    atomic_json(BASE/'PREPARED.json',dict(passed=True,unix=time.time()))

def verify(cp,force=False):
    m=verify_checkpoint(Path(cp),sha(RUN/'plan.json'));assert m['arm']==ARM
    assert sha(Path(cp)/'ema_policy.pt')==m['ema_policy_sha256'];return m

def resolve_checkpoint(generation=None):
    for marker in sorted((RUN/'formal'/ARM).glob('gen*/COMMITTED.json'),reverse=True):
        if generation is not None and not marker.parent.name.startswith(f'gen{generation:06d}'):continue
        try:
            proof=verify(marker.parent)
            if generation is None or proof['generation']==generation:return marker.parent
        except Exception as error:atomic_json(RUN/'state'/('CORRUPT_'+sha(marker)+'.json'),dict(reason=repr(error)))
    return None

def notify(event,message):
    marker=BASE/'state'/('NOTIFIED_'+event+'.json')
    if marker.exists():return
    atomic_json(marker,dict(claimed=True,unix=time.time()))
    try:code=subprocess.run(['jdme-notify','send',message],capture_output=True,timeout=30).returncode
    except Exception:code=-1
    atomic_json(marker,dict(claimed=True,returncode=code))

def predecessor_finished():
    if not (PREDECESSOR/'TRAINING_COMPLETE.json').exists():return False
    complete=read(PREDECESSOR/'TRAINING_COMPLETE.json')
    expected=4096 if (PREDECESSOR/'EXTENSION_AUTHORIZED.json').exists() else 2048
    if complete.get('generation')!=expected:return False
    # The GPU owner has completed original extension decision; CPU evaluators
    # may finish asynchronously. Never compete with a live predecessor worker.
    for marker in ['TRAIN_LAUNCH.json','SMOKE_LAUNCH.json']:
        if (PREDECESSOR/marker).exists():
            pid=read(PREDECESSOR/marker).get('pid')
            if pid and alive(pid):
                try:
                    if b'je06_bun_worker.py' in Path(f'/proc/{pid}/cmdline').read_bytes():return False
                except FileNotFoundError:pass
    return not any(gpu_owners().values())

def process_matches(pid,script):
    try:return alive(pid) and script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False

def gpu_sample(pid):
    sample=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    with (BASE/'gpu_samples.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),worker_pid=pid,phase='formal',gpu=sample))+'\n')

def owner():
    check_seal();lease=lock(BASE/'owner.lock')
    atomic_json(BASE/'OWNER.json',dict(pid=os.getpid(),unix=time.time(),gpu_started=False))
    while not (BASE/'HANDOFF_RELEASED.json').exists() and not predecessor_finished():
        atomic_json(BASE/'WAIT_STATE.json',dict(phase='WAIT_JE06_ORIGINAL_TRAINING_COMPLETE_AND_GPU_RELEASE',
            gpu_started=False,no_takeover=True,unix=time.time()));time.sleep(15)
    prepare_inputs()
    if not (BASE/'HANDOFF_RELEASED.json').exists():
        final=read(PREDECESSOR/'TRAINING_COMPLETE.json');g=final['generation']
        assert final['passed'] and final['fresh_ticks']==g*2097152
        markers=list((PREDECESSOR/'formal'/'bun392_holistic_long256_pfsp8').glob(f'gen{g:06d}*/COMMITTED.json'))
        assert any(verify_checkpoint(m.parent,sha(PREDECESSOR/'plan.json'))['generation']==g for m in markers)
        atomic_json(BASE/'HANDOFF_RELEASED.json',dict(predecessor_complete_sha256=sha(PREDECESSOR/'TRAINING_COMPLETE.json'),unix=time.time()))
    for attempt in range(1,33):
        cp=resolve_checkpoint();env=child_env('0,1,2,3,4,5,6,7',False)
        if cp and verify(cp)['generation']==4096:
            final=read(BASE/'formal'/ARM/'TRAINER_EXIT.json');assert final['passed'] and final['generation']==4096
            atomic_json(BASE/'TRAINING_COMPLETE.json',final);return
        args=[sys.executable,'scripts/je07_bun_worker.py','formal']+(['--resume',str(cp)] if cp else [])
        old=read(BASE/'TRAIN_LAUNCH.json') if (BASE/'TRAIN_LAUNCH.json').exists() else {}
        if old.get('pid') and process_matches(old['pid'],'scripts/je07_bun_worker.py'):
            child=Adopted(old['pid'])
        else:
            while any(gpu_owners().values()):
                atomic_json(BASE/'WAIT_EXTERNAL_GPU.json',dict(no_takeover=True,unix=time.time()));time.sleep(10)
            with (BASE/f'formal_attempt{attempt:03d}.log').open('a') as log:
                child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(BASE/'TRAIN_LAUNCH.json',dict(pid=child.pid,devices=list(range(8)),attempt=attempt))
        while child.poll() is None:
            gpu_sample(child.pid)
            state=BASE/'formal'/ARM/'STATE.json'
            if state.exists() and not (BASE/'state/FIRST_NONZERO_UPDATE.json').exists():
                row=read(state)
                if row['finite'] and row['nonzero_update']:
                    atomic_json(BASE/'state/FIRST_NONZERO_UPDATE.json',row)
                    notify('first_update','JE07正式八卡finite非零更新，g'+str(row['generation'])+'，吞吐'+str(round(row['fresh_ticks_per_second']))+'ticks/s。')
            cp1=next((m.parent for m in sorted((BASE/'formal'/ARM).glob('gen000001*/COMMITTED.json'),reverse=True)),None) if not (BASE/'state/RESTORE_MONITOR.json').exists() else None
            if cp1 is not None:
                marker=BASE/'state/RESTORE_LAUNCH.json'
                old=read(marker).get('pid') if marker.exists() else None
                if not old or not alive(old):
                    with (BASE/'validation/fresh_restore.log').open('a') as log:
                        monitor=subprocess.Popen(['taskset','-c','112-115',sys.executable,'scripts/je07_bun_worker.py','restore-monitor','--resume',str(cp1)],cwd=ROOT,env=child_env('',True),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    atomic_json(marker,dict(pid=monitor.pid,generation=1,cpu_only=True,not_a_start_gate=True))
            time.sleep(10)
        final=BASE/'formal'/ARM/'TRAINER_EXIT.json'
        if child.returncode==0 and final.exists() and read(final)['generation']==4096:
            atomic_json(BASE/'TRAINING_COMPLETE.json',read(final));notify('training_complete','JE07 4096代训练完成，权威CPU评估继续。');return
        atomic_json(BASE/f'TECHNICAL_FAILURE_{attempt:03d}.json',dict(returncode=child.returncode,unix=time.time()))
    raise RuntimeError('technical recovery exhausted')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['seal-cpu','owner','prepare-inputs']);args=parser.parse_args()
    if args.command=='seal-cpu':seal_cpu()
    elif args.command=='prepare-inputs':
        assert predecessor_finished();prepare_inputs()
    else:owner()
