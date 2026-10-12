"""Native Bun transfer of the audited it831 backbone and frozen teaching recipe.

No battle input adapter, no dynamic opponents, and no modifications to the live
Bun192 runner. CPU smoke cannot authorize GPU training.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.population_run_state import atomic_json, sha, lock
from scripts.je03_bun_stage1_20261011 import (SOURCE as BUN_SOURCE, SOURCE_SHA,
    split_forward, digest, verify_checkpoint, gpu_owners, read)

# Checkpoints are external inputs; never import historical repository code.
SOURCE = Path(os.environ.get('BUN_IT831_CHECKPOINT', ROOT / 'checkpoints/params_it00000831_ema.pkl'))
RUN = Path(os.environ.get('BUN_MIGRATION_RUN', ROOT / 'runs/je03_bun_stage1_20261011/bun831_native_tracks_v1')).resolve()
ARMS = ['bun392_base', 'bun392_recipe', 'bun192_base']
POINTS = [0, 32, 64, 128, 192, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
RETENTION_V2 = os.environ.get('BUN_RETENTION_V2') == '1'
if RETENTION_V2:
    RUN = Path(os.environ['BUN_RETENTION_RUN']).resolve()
    ARMS = ['bun392_retention_control', 'bun392_retention_teacher']
    POINTS = [0, 8, 16, 32, 64, 128, 192, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
SOURCES = ['scripts/je03_bun831_20261011.py', 'scripts/bun_checkpoint_state.py',
    'scripts/je03_bun831_evaluator_20261011.py', 'scripts/je03_bun_stage1_20261011.py',
    'scripts/je03_bun_evaluator_20261011.py', 'jax_bomb/bun_env.py',
    'jax_bomb/jax_train.py', 'jax_bomb/jax_net.py', 'web/sim.js', 'web/bun_hunter_bot.js']


def recipe(arm):
    assert arm in ARMS
    if RETENTION_V2:
        return dict(
            curriculum='danger_arena=.45,full_ambush=.20,full_contact=.10,combat=.10,full=.10,carry_home=.03,near_steal=.02',
            reward_profile='retention_v2', lr=.0002, entropy=.005, adv_top_frac=1.,
            crate_coef=0., brick_coef=0.,
            retention_replay=.45 if arm.endswith('teacher') else 0.,
            teacher_kl_coef=.08 if arm.endswith('teacher') else 0.,
            opponents='idle=0,roam=0,rule_combat=.35,weak=0,old=.65,recent=0,history=0')
    base = dict(curriculum='danger_arena=.35,full_ambush=.25,full=.2,carry_home=.2',
        reward_profile='combat_evolution', lr=.0003, entropy=.01, adv_top_frac=1.,
        crate_coef=0., brick_coef=0.,
        opponents='idle=0,roam=0,rule_combat=.3,weak=0,old=.7,recent=0,history=0')
    if arm == 'bun392_recipe':
        base.update(curriculum='full=.3,carry_home=.15,near_steal=.1,route_break=.1,combat=.1,full_contact=.1,full_ambush=.1,danger_arena=.05',
            reward_profile='legacy', lr=.0002, entropy=.008, adv_top_frac=.25,
            crate_coef=.25, brick_coef=.2,
            opponents='idle=.15,roam=.03,rule_combat=.12,weak=0,old=.4,recent=.3,history=0')
    return base


def plan():
    if RETENTION_V2:
        return dict(
            experiment_id='JE03-BUN392-RETENTION-V2-20261011-v1', stage=1,
            stage2_enabled=False,
            architecture=dict(embed=392, depth=4, heads=4, patch=3, obs=[24,13,15],
                               state=24, ability_actions=3, move_actions=5,
                               independent_critic=True, value_bins=128),
            initialization=dict(
                both='exact shared frozen native Bun392 g0 from prior sealed checkpoint; no winner carryover'),
            tracks=dict(
                control=dict(arm='bun392_retention_control', variable='retention_v2 reward plus combat-heavy stratified curriculum'),
                treatment=dict(arm='bun392_retention_teacher', variable='control plus fixed g0 actor KL=.08 and 45% combat replay resets'),
                attribution='shared reward/course/opponents; teacher retention+replay is the sole within-round treatment variable'),
            arms=ARMS, seed=2026101121, generations=4096, envs_per_device=1024,
            steps=128, minibatch_per_device=4096, epochs=2,
            ticks_per_generation=524288, ticks_per_arm=2147483648,
            checkpoints=POINTS, recovery_every=64, gamma=.995, lam=.95, clip=.2,
            value_coef=.5, ema_decay=.995, optimizer_eps=1e-8, hp=1, rule='bun',
            native_map=806,
            inputs='native Bun24/global24; full item/revive/ability semantics retained',
            reward_spec=dict(
                safety='-2 own_hit -12 death -16 self_kill -12 mutual_death',
                escape='+.50 safe_bomb_escape -1.50 unsafe_detonation',
                kill='+1 enemy_hit +16 surviving_kill +4 causal_kill',
                task_progress='+.05 clip(progress,-1,1), one-step bounded',
                task_success='+.75 enter_enemy_half +1 enter_enemy_base +2 steal +1 carry_cross_home +4 capture +1 recover +1 return_bun',
                task_penalty='-2 drop; no crate/brick/explore bonus',
                terminal='+6 win -6 loss; all event flags are one-shot environment events'),
            curriculum=dict(
                weights='danger_arena .45, full_ambush .20, full_contact .10, combat .10, full .10, carry_home .03, near_steal .02',
                teacher_replay='45% reset override to danger_arena/combat/combat_moving/full_contact; control 0%'),
            recipes={a:recipe(a) for a in ARMS},
            opponents=dict(dynamic=False, refreshed=False,
                           composition='rule_combat .35 + frozen old .65',
                           no_live_learner=True),
            evaluation=dict(protocol='async real web Bun Hunter hard/normal/easy + Tactical + carry/steal/full/route; 128+ games at checkpoints',
                            cpu_async=True, cuda_disabled=True),
            telemetry=dict(reward_components=['safety','escape','kill','progress','task_success','task_penalty','terminal','total'],
                           lesson_counts=True, teacher_kl=True, stage2_enabled=False),
            no_cutoff=True, gpu_smoke_required=True)
    return dict(experiment_id='JE03-BUN831-NATIVE-TRACKS-20261011-v1', stage=1, stage2_enabled=False,
        architecture=dict(embed=392, depth=4, heads=4, patch=3, obs=[24,13,15], state=24,
            ability_actions=3, move_actions=5, independent_critic=True, value_bins=128),
        initialization=dict(bun392_base='common frozen it831 EMA, explicit native Bun semantic transfer v1',
            bun392_recipe='common frozen it831 EMA, explicit native Bun semantic transfer v1',
            bun192_base='original Bun it4000; independent identical-initialized Critic'),
        sources=dict(it831=str(SOURCE), it831_sha256=sha(SOURCE), it4000=str(BUN_SOURCE), it4000_sha256=SOURCE_SHA),
        tracks=dict(teaching=dict(arms=['bun392_base','bun392_recipe'],
                variable='complete frozen teaching recipe package with identical Bun392 initialization/capacity/Critic',
                scope='Bun curriculum, full objective rewards, fixed opponent composition, PPO top25/Adam2e-4/entropy.008; integrated reproduction, not individual attribution'),
            capacity_initialization=dict(arms=['bun192_base','bun392_base'], variable='Actor/Critic capacity and source initialization package; identical Bun teaching',
                limitation='joint package: cannot separate capacity from initialization; no winner carryover'),
            followup=dict(condition='positive real Hunter AND Bun-task/safety confirmation',
                experiment='same392 same gen0; opponent composition only, then fresh independent seed and >=256 games/tier')),
        arms=ARMS, seed=2026101111, generations=4096, envs_per_device=1024, steps=128,
        minibatch_per_device=4096, epochs=2, ticks_per_generation=524288, ticks_per_arm=2147483648,
        checkpoints=POINTS, recovery_every=128, gamma=.995, lam=.95, clip=.2, value_coef=.5,
        ema_decay=.995, optimizer_eps=1e-8, hp=1, rule='bun', native_map=806,
        inputs='native Bun24/global24 throughout rollout, policy export and authority; no masking/battle adapter',
        migration='shared spatial0:14 and global0:11 only; new Bun planes/global11:24 and ability2 retain seeded native initialization; time normalization differs and is disclosed',
        opponents=dict(dynamic=False, refreshed=False, it4000='public frozen original Bun',
            transfer='public frozen Bun831 gen0; never a live learner',
            historical_gap='No archived executable A* league or snapshot loading found; fixed idle/roam/reactive rules and two frozen native policies are an explicit controlled substitute; no claim exact reproduction'),
        recipes={a:recipe(a) for a in ARMS},
        evaluation=dict(protocol='same real web Bun Hunter64/tier,300+64 residual and Tactical64',
            task_reference='native web items/traps/rescue6 episodes + native Bun task16 each carry_home/near_steal/route_break/full,300tick; separately reported',
            reference='original Bun it4000 and shared g0; it831 ONLY cross-mode diagnostic',
            promotion='>=2 late checkpoints with >=6 safe kills/192 and >=2/64 hard or normal; Hunter death/self/trade noninferiority within4/tier, Bun-task/safety noninferiority; then256/tier and independent seed; no automatic stage2',
            cpu_async=True, cuda_disabled=True),
        no_cutoff=True, gpu_smoke_required=True, schedule='first released4 cards: Bun392_base; next released4: Bun392_recipe; then matched Bun192_base, independent runs')


def migrate(old, fresh):
    """Bun core planes share battle14; battle global tail has different semantics."""
    p = copy.deepcopy(old)
    weight = np.asarray(fresh['tok'][0]).copy()
    assert weight.shape == (216,392) and np.asarray(old['tok'][0]).shape == (126,392)
    weight[:126] = old['tok'][0]
    p['tok'] = (weight, np.asarray(old['tok'][1]).copy())
    state = np.asarray(fresh['state_w']).copy()
    state[:11] = old['state_w'][:11]
    p['state_w'] = state
    w, b = [np.asarray(x).copy() for x in fresh['heads']['wb']]
    w[:,:2], b[:2] = old['heads']['wb']
    p['heads']['wb'] = (w,b)
    return p


def save_pickle(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        pickle.dump(value,f,protocol=5); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def prepare():
    os.environ['JAXBOMB_RULE'] = 'bun'
    import jax
    from jax_bomb import jax_net as net
    assert net.N_BOMB == 3
    p = plan(); RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'plan.json').exists():
        assert read(RUN/'plan.json') == p
        verify_seal(); return
    if RETENTION_V2:
        source_run = Path(os.environ['BUN_RETENTION_SOURCE_RUN']).resolve()
        source_input = source_run / 'inputs/bun392_gen0.pkl'
        assert source_input.exists()
        with source_input.open('rb') as f: transferred = pickle.load(f)
        save_pickle(RUN/'inputs/bun392_gen0.pkl', transferred)
        atomic_json(RUN/'plan.json',p)
        atomic_json(RUN/'SOURCE_SEAL.json',dict(
            sources={s:sha(ROOT/s) for s in SOURCES},
            plan_sha256=sha(RUN/'plan.json'),
            inputs={'bun392_gen0.pkl':sha(RUN/'inputs/bun392_gen0.pkl')},
            source_run=str(source_run), source_input_sha256=sha(source_input)))
        atomic_json(RUN/'PREPARED_CPU.json',dict(passed=True, platform=jax.default_backend(), native_bun=True,
            plan_sha256=sha(RUN/'plan.json'), gpu_ready=False, source_input_sha256=sha(source_input), unix=time.time()))
        return
    assert sha(BUN_SOURCE) == SOURCE_SHA
    with SOURCE.open('rb') as f: old = pickle.load(f)
    with BUN_SOURCE.open('rb') as f: bun = pickle.load(f)
    fresh = net.init_transformer(jax.random.PRNGKey(p['seed']),24,13,15,embed=392,depth=4,patch=3,state_dim=24)
    fresh = jax.device_get(fresh); transferred = migrate(old,fresh)
    save_pickle(RUN/'inputs/bun392_gen0.pkl',transferred)
    save_pickle(RUN/'inputs/bun192_gen0.pkl',bun)
    atomic_json(RUN/'plan.json',p)
    atomic_json(RUN/'SOURCE_SEAL.json',dict(sources={s:sha(ROOT/s) for s in SOURCES},
        plan_sha256=sha(RUN/'plan.json'), inputs={f:sha(RUN/'inputs'/f) for f in ['bun392_gen0.pkl','bun192_gen0.pkl']}))
    atomic_json(RUN/'PREPARED_CPU.json',dict(passed=True, platform=jax.default_backend(), native_bun=True,
        plan_sha256=sha(RUN/'plan.json'), gpu_ready=False, unix=time.time()))


def verify_seal():
    seal = read(RUN/'SOURCE_SEAL.json')
    assert seal['plan_sha256'] == sha(RUN/'plan.json')
    assert all(sha(ROOT/f)==h for f,h in seal['sources'].items())
    assert all(sha(RUN/'inputs'/f)==h for f,h in seal['inputs'].items())
    if RETENTION_V2:
        assert seal.get('source_run') == str(Path(os.environ['BUN_RETENTION_SOURCE_RUN']).resolve())


def require_gpu_gate(doc):
    assert doc['passed'] and doc['platform']=='gpu'
    assert doc['finite'] and doc['nonzero'] and doc['full_restore']
    assert doc['smoke_digest'] == doc['replay_digest']


def decide(baseline, rows):
    # Frozen prospectively before treatment768. Earlier observed evidence is
    # explicitly calibration, never retrospectively called preregistration.
    if 768 not in rows: return dict(action='WAIT_COMMON_768')
    if not all(g in rows for g in (256,512,768)): return dict(action='WAIT_COMPLETE_WINDOW')
    degraded = [rows[g]['safe_win'] < baseline['safe_win']+6
        and rows[g]['death_episode'] >= baseline['death_episode']+20
        and rows[g]['self_episodes'] >= baseline['self_episodes']+12 for g in (256,512,768)]
    return dict(action='STOP_ATOMIC' if all(degraded) else 'CONTINUE', window=[256,512,768],
        reason='sustained Hunter safety regression without useful safe-kill gain' if all(degraded) else 'retain scientific value; no early winner')


def worker(arm, phase, devices, resume=None, cpu=False):
    os.environ['JAXBOMB_RULE'] = 'bun'
    import jax
    import jax.numpy as jnp
    import optax
    from jax_bomb import jax_train as train, bun_env as env
    verify_seal(); p = read(RUN/'plan.json'); r = p['recipes'][arm]
    assert train.IS_BUN and train.N_BOMB == 3 and env.N_OBS_CH == 24
    count = len(jax.local_devices())
    assert (cpu and count==1 and jax.default_backend()=='cpu' and os.environ['CUDA_VISIBLE_DEVICES']=='') or (not cpu and count==4 and jax.default_backend()=='gpu')
    env.prepare(); env.configure_training(
        r['curriculum'], p['hp'], reward_profile=r['reward_profile'],
        retention_replay=r.get('retention_replay', 0.0))
    forward = train.net_forward
    train.net_forward = lambda q,a,o,s=None: split_forward(q,a,o,s,forward)
    def load(name):
        with (RUN/'inputs'/name).open('rb') as f:return jax.tree.map(jnp.asarray,pickle.load(f))
    if RETENTION_V2:
        frozen = transfer = load('bun392_gen0.pkl')
    else:
        frozen = load('bun192_gen0.pkl'); transfer = load('bun392_gen0.pkl')
    # Kinds are sampled at reset by the existing Bun rollout. Frozen networks
    # are pure Actor policies; P1 transitions never train either Critic.
    def opponent(states, obs, gv, masks, arch, key, kinds, weak, old, recent):
        n=states.pos.shape[0]; mm,bm=masks[0][n:],masks[1][n:]
        kr,kf,kt,kroam=jax.random.split(key,4)
        rule=train.flee_bot_actions(states.pos[:,1],states.pos[:,0],mm,bm,kr,idle_ratio=0.,roam_ratio=0.,pure_flee_ratio=0.)
        oldacts,_,_=train.sample_actions(frozen,arch,obs[n:],(mm,bm),kf,state=gv[n:])
        if arm=='bun392_recipe':
            newacts,_,_=train.sample_actions(transfer,arch,obs[n:],(mm,bm),kt,state=gv[n:])
            idle=jnp.tile(jnp.array([[4,0]],jnp.int32),(n,1))
            move=jnp.argmax(jnp.where(mm,jax.random.uniform(kroam,mm.shape),-1),-1)
            roam=jnp.stack([move,jnp.zeros(n,jnp.int32)],-1)
            return train.select_bun_opponent_actions(kinds,jnp.stack([idle,roam,rule,oldacts,oldacts,newacts,newacts]))
        return jnp.where((kinds==2)[:,None],rule,oldacts)
    train.bun_opponent_actions = opponent
    weights = jnp.asarray(train.parse_bun_opponent_weights(r['opponents']))
    n, steps, mb = (2,2,4) if cpu else (p['envs_per_device'],p['steps'],p['minibatch_per_device'])
    opt = optax.adam(r['lr'],eps=p['optimizer_eps'])
    def rollout(params, states, key):
        result=train.collect_rollout(params,'transformer',states,key,steps,crate_coef=r['crate_coef'],brick_coef=r['brick_coef'],
            flee_bot_ratio=0.,bun_opponent_weights=weights,bun_opponent_old_params=frozen,
            return_components=RETENTION_V2, return_lesson_stats=RETENTION_V2)
        states,batch=result[:2]; obs,st,acts,lps,vals,rew,done,masks=batch
        select=lambda x:x[:,:n]
        obs,st,acts,lps,vals,rew,done=map(select,(obs,st,acts,lps,vals,rew,done)); masks=tuple(select(x) for x in masks)
        all_masks=train.both_masks(states)
        _,_,tail=train.sample_actions(params,'transformer',train.both_perspectives(states)[:n],tuple(x[:n] for x in all_masks),jax.random.split(key)[0],state=train.both_states(states)[:n])
        adv=train.compute_gae(rew,vals,jnp.concatenate([vals[1:],tail[None]]),done,p['gamma'],p['lam'])
        base_stats=jnp.array([jnp.mean(rew),jnp.mean(done),jnp.sum(result[3])])
        if RETENTION_V2:
            component_stats=result[4].reshape(-1)
            lesson_stats=result[5]
            stats=jnp.concatenate([base_stats,component_stats,lesson_stats])
        else:
            stats=base_stats
        return states,(obs,st,acts,lps,adv,adv+vals,masks),stats
    teacher_params = None
    if RETENTION_V2:
        teacher_params=dict(actor=transfer, critic=jax.tree.map(lambda x:x.copy(),transfer))
    def update(params, adam, batch, key, ema):
        params,adam,loss=train.ppo_update(
            params,opt,adam,'transformer',batch,key,mb,p['clip'],p['value_coef'],
            r['entropy'],p['epochs'],axis_name='dev',return_loss=True,
            adv_top_frac=r['adv_top_frac'], reference_params=teacher_params,
            reference_kl_coef=r.get('teacher_kl_coef', 0.0))
        d=p['ema_decay']
        return params,adam,jax.random.split(key)[0],jax.tree.map(lambda e,x:d*e+(1-d)*x,ema,params),loss
    roll=jax.pmap(rollout); learn=jax.pmap(update,axis_name='dev')
    rep=lambda q:jax.tree.map(lambda x:jnp.stack([x]*count),q)
    out=RUN/phase/arm; out.mkdir(parents=True,exist_ok=True); own=lock(out/'worker.lock')
    if resume:
        m=verify_checkpoint(resume,sha(RUN/'plan.json'))
        with (Path(resume)/'state.pkl').open('rb') as f:saved=pickle.load(f)
        assert saved['shape']==[count,n,steps,mb], 'resume shape mismatch'
        live=jax.tree.map(jnp.asarray,saved['live']); generation=saved['generation']
        assert digest(live)==m['full_state_digest']
    else:
        init=frozen if (arm=='bun192_base' or RETENTION_V2) else transfer
        params=dict(actor=init,critic=jax.tree.map(lambda x:x.copy(),init))
        key=jax.random.split(jax.random.PRNGKey(p['seed']),count)
        states=jax.pmap(lambda k:env.init_batch(k,n))(key)
        live=(rep(params),rep(opt.init(params)),states,key,rep(params)); generation=0
    atomic_json(out/'WORKER.json',dict(pid=os.getpid(),arm=arm,phase=phase,devices=devices,platform=jax.default_backend(),restore=resume,initial_digest=digest(live),unix=time.time()))
    stopping=[False]; signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    def checkpoint(g):
        cp=out/f'gen{g:06d}'; cp.mkdir(exist_ok=True)
        state_digest=digest(live)
        if (cp/'COMMITTED.json').exists():
            assert verify_checkpoint(cp,sha(RUN/'plan.json'))['full_state_digest']==state_digest; return cp
        save_pickle(cp/'state.pkl',dict(generation=g,live=jax.device_get(live),shape=[count,n,steps,mb]))
        save_pickle(cp/'policy.pt',jax.tree.map(lambda x:np.asarray(x[0]),live[0]['actor']))
        save_pickle(cp/'ema_policy.pt',jax.tree.map(lambda x:np.asarray(x[0]),live[4]['actor']))
        atomic_json(cp/'COMMITTED.json',dict(passed=True,generation=g,arm=arm,platform=jax.default_backend(),
            fresh_ticks=g*count*n*steps,state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),
            ema_policy_sha256=sha(cp/'ema_policy.pt'),full_state_digest=state_digest,plan_sha256=sha(RUN/'plan.json')))
        return cp
    if not resume:checkpoint(0)
    begin=time.time(); finish=2 if phase!='formal' else p['generations']; compute=0
    for g in range(generation+1,finish+1):
        oldparams=live[0]; start=time.time()
        states,batch,stats=roll(live[0],live[2],live[3]); jax.block_until_ready(batch); rt=time.time()-start; t=time.time()
        params,adam,key,ema,loss=learn(live[0],live[1],batch,live[3],live[4]); jax.block_until_ready(params)
        ut=time.time()-t; live=(params,adam,states,key,ema); del batch
        arrays=jax.tree.leaves(params)
        finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(live)) and np.isfinite(np.asarray(loss)).all()
        replicas=all(np.array_equal(np.asarray(x)[0],np.asarray(x)[i]) for x in arrays for i in range(1,count))
        norms={k:float(np.sqrt(sum(float(jnp.sum((x-y)**2)) for x,y in zip(jax.tree.leaves(params[k]),jax.tree.leaves(oldparams[k]))))) for k in ['actor','critic']}
        assert finite and replicas and all(v>0 for v in norms.values()),(g,finite,replicas,norms)
        dt=time.time()-start; compute+=dt; ticks=count*n*steps; updates=p['epochs']*n*steps//(mb//2)
        row=dict(generation=g,pid=os.getpid(),platform=jax.default_backend(),finite=bool(finite),replicas_exact=replicas,
            actor_update_l2=norms['actor'],critic_update_l2=norms['critic'],parameter_update_l2=sum(norms.values()),
            fresh_ticks=g*ticks,fresh_ticks_per_second=ticks/dt,optimizer_updates=updates,optimizer_updates_per_second=updates/dt,
            rollout_seconds=rt,update_seconds=ut,seconds=dt,loss=np.asarray(loss).tolist(),stats=np.asarray(stats).tolist(),unix=time.time())
        if RETENTION_V2:
            stats_array=np.asarray(stats)
            row['reward_components']=stats_array[:,3:19].tolist()
            row['lesson_counts']=stats_array[:,19:35].tolist()
            row['teacher_kl_coef']=r.get('teacher_kl_coef',0.0)
            row['retention_replay']=r.get('retention_replay',0.0)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n'); f.flush()
        if phase!='formal' or g in POINTS or g%p['recovery_every']==0 or stopping[0]:checkpoint(g)
        atomic_json(out/'STATE.json',dict(row,compute_seconds=compute,wall_seconds=time.time()-begin)); print(json.dumps(row),flush=True)
        if stopping[0]:break
    checkpoint(g)
    atomic_json(out/'TRAINER_EXIT.json',dict(passed=True,generation=g,pid=os.getpid(),platform=jax.default_backend(),
        final_state_digest=digest(live),fresh_ticks=g*count*n*steps,compute_seconds=compute,wall_seconds=time.time()-begin,stopped_at_update_boundary=stopping[0]))


def child_env(devices, cpu=False):
    return dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else devices,JAXBOMB_RULE='bun',
        JAX_PLATFORMS='cpu' if cpu else 'cuda,cpu',JAX_PLATFORM_NAME='cpu' if cpu else 'cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')


def gate(arm, devices, cpu=False):
    verify_seal(); phase='cpu_smoke' if cpu else 'smoke'; replay='cpu_replay' if cpu else 'replay'
    for mode,resume in [(phase,None),(replay,str(RUN/phase/arm/'gen000001'))]:
        cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--arm',arm,'--phase',mode,'--devices',devices]
        if cpu:cmd+=['--cpu']
        if resume:cmd+=['--resume',resume]
        with (RUN/f'{mode}_{arm}.log').open('a') as log:
            child=subprocess.Popen(cmd,env=child_env(devices,cpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(RUN/f'{mode}_{arm}_LAUNCH.json',dict(pid=child.pid,unix=time.time()))
            assert child.wait()==0, f'{mode} failed; preserved log'
    a,b=[read(RUN/m/arm/'TRAINER_EXIT.json') for m in [phase,replay]]
    rows=[json.loads(x) for x in (RUN/phase/arm/'events.jsonl').read_text().splitlines()]
    assert len(rows)==2 and a['final_state_digest']==b['final_state_digest']
    assert all(x['finite'] and x['replicas_exact'] and x['actor_update_l2']>0 and x['critic_update_l2']>0 for x in rows)
    doc=dict(passed=True,platform='cpu' if cpu else 'gpu',finite=True,nonzero=True,full_restore=True,
        smoke_digest=a['final_state_digest'],replay_digest=b['final_state_digest'],smoke=a,replay=b,
        shape='tiny CPU correctness only' if cpu else 'production4GPU',devices=devices,
        plan_sha256=sha(RUN/'plan.json'),source_seal_sha256=sha(RUN/'SOURCE_SEAL.json'),unix=time.time())
    if not cpu:require_gpu_gate(doc)
    atomic_json(RUN/f'READY_{"CPU_SMOKE" if cpu else "GPU"}_{arm}.json',doc)


def arm_owner(arm, devices):
    verify_seal(); own=lock(RUN/f'owner_{arm}.lock')
    atomic_json(RUN/f'OWNER_{arm}.json',dict(pid=os.getpid(),devices=devices,unix=time.time()))
    assert all(not gpu_owners()[int(i)] for i in devices.split(',')), 'no takeover of occupied GPU'
    ready=RUN/f'READY_GPU_{arm}.json'
    if not ready.exists():gate(arm,devices)
    doc=read(ready); require_gpu_gate(doc)
    assert doc['plan_sha256']==sha(RUN/'plan.json') and doc['source_seal_sha256']==sha(RUN/'SOURCE_SEAL.json')
    for attempt in range(1,9):
        checkpoints=sorted((RUN/'formal'/arm).glob('gen*/COMMITTED.json'))
        resume=str(checkpoints[-1].parent) if checkpoints else None
        if resume:
            m=verify_checkpoint(resume,sha(RUN/'plan.json'))
            if m['generation']==read(RUN/'plan.json')['generations']:
                result=read(RUN/'formal'/arm/'TRAINER_EXIT.json')
                assert result['passed'] and result['final_state_digest']==m['full_state_digest']
                atomic_json(RUN/f'TRAINING_COMPLETE_{arm}.json',dict(result,unix=time.time())); return
            events=RUN/'formal'/arm/'events.jsonl'
            if events.exists():
                text=events.read_text(); archive=events.with_name(f'events_before_resume_{time.time_ns()}.jsonl'); archive.write_text(text)
                temporary=events.with_suffix('.tmp'); temporary.write_text(''.join(x+'\n' for x in text.splitlines() if json.loads(x)['generation']<=m['generation'])); temporary.replace(events)
        cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--arm',arm,'--phase','formal','--devices',devices]
        if resume:cmd+=['--resume',resume]
        with (RUN/f'formal_{arm}_attempt{attempt}_{time.time_ns()}.log').open('a') as log:
            child=subprocess.Popen(cmd,env=child_env(devices),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(RUN/f'LAUNCH_{arm}.json',dict(pid=child.pid,devices=devices,resume=resume,phase='FORMAL',unix=time.time()))
            while child.poll() is None:
                with (RUN/f'GPU_{arm}.jsonl').open('a') as f:
                    sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
                    f.write(json.dumps(dict(pid=child.pid,gpu=sample,unix=time.time()))+'\n')
                time.sleep(5)
        if child.returncode==0:
            result=read(RUN/'formal'/arm/'TRAINER_EXIT.json')
            assert result['passed'] and result['generation']==read(RUN/'plan.json')['generations']
            atomic_json(RUN/f'TRAINING_COMPLETE_{arm}.json',dict(result,unix=time.time())); return
        atomic_json(RUN/f'TECHNICAL_STOP_{arm}_{child.pid}.json',dict(returncode=child.returncode,attempt=attempt,unix=time.time()))
    raise RuntimeError('bounded restart budget exhausted; evidence retained')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['prepare','worker','cpu-gate','arm-owner'])
    ap.add_argument('--arm',choices=ARMS); ap.add_argument('--phase'); ap.add_argument('--devices',default=''); ap.add_argument('--resume'); ap.add_argument('--cpu',action='store_true'); args=ap.parse_args()
    try:
        if args.command=='prepare':prepare()
        elif args.command=='worker':worker(args.arm,args.phase,args.devices,args.resume,args.cpu)
        elif args.command=='cpu-gate':gate(args.arm,'',True)
        else:arm_owner(args.arm,args.devices)
    except BaseException as e:
        atomic_json(RUN/f'TECHNICAL_STOP_{args.command}_{os.getpid()}.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time())); raise
