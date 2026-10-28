"""H5 per-generation learner collection, oracle queries, aggregation and PPO."""
import argparse
import os
import signal
os.environ.setdefault('JAXBOMB_RULE','bun')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.je08_bun_recipe import *
from scripts import je09_stage_h5c_dagger as dagger

def initial_funnel():
    from scripts.je09_bun_ops import FUNNEL
    return np.zeros((10,len(FUNNEL)),np.float32)

def restore_monitor(cp):
    """Fresh CPU process loads EVERY saved field, verifies complete digest and
    next RNG transition. It never waits for or occupies a training device."""
    import jax
    import jax.numpy as jnp
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and jax.default_backend()=='cpu'
    proof=verify(cp)
    with (Path(cp)/'state.pkl').open('rb') as f:s=pickle.load(f)
    assert digest(s['live'])==proof['full_state_digest']
    assert s['shape']==SHAPE
    required={'params','adam','states','aux','key','ema','pool','funnel','pool_counts','course','pfsp','assist_level','assist_proof_hashes','dagger_buffer'}
    if 'distance_stage' in s['live']:required.add('distance_stage')
    assert set(s['live'])==required
    keys=s['live']['key']
    next_keys=np.asarray(jax.vmap(lambda k:jax.random.split(k)[0])(jnp.asarray(keys)))
    assert digest(next_keys)==proof['next_rng_digest']
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(s['live']))
    atomic_json(RUN/'state/RESTORE_MONITOR.json',dict(passed=True,fresh_process=True,cpu_only=True,
        cuda_visible_devices='',generation=proof['generation'],full_state_digest=proof['full_state_digest'],
        next_rng_digest=proof['next_rng_digest'],complete_fields=sorted(required),not_a_start_gate=True,unix=time.time()))


def worker(resume=None,replay=False):
    import jax
    import jax.numpy as jnp
    import optax
    from jax_bomb import je07_bun_env as env,jax_train as train
    from scripts import je09_stage_h5c_ops as ops, je07_bun_curriculum as curriculum
    from scripts.je03_bun_stage1_20261011 import split_forward
    p=read(RUN/'plan.json');count=len(jax.local_devices());n=p['envs_per_device'];steps=p['steps'];mb=p['minibatch_per_device']
    assert count*n*steps==p['fresh_ticks_per_generation']
    assert n*steps//(mb//2*p['gradient_accumulation'])==p['updates_per_generation']
    assert count==DEVICE_COUNT and jax.default_backend()=='gpu' and train.IS_BUN and env.N_OBS_CH==24 and train.N_BOMB==3
    env.prepare();env.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1)
    from scripts import je09_bun_scenes as scenes
    env._DANGER_TRACKING=True
    scenes.bind_training()
    original=train.net_forward;train.net_forward=lambda q,a,o,s=None:split_forward(q,a,o,s,original)
    with (RUN/'inputs/initial.pkl').open('rb') as f:initial=pickle.load(f)
    with (RUN/'inputs/frozen.pkl').open('rb') as f:fixed=pickle.load(f)
    retention_parent=initial['actor']
    if p.get('return_expert'):
        with (RUN/'inputs/retention_parent.pkl').open('rb') as f: retention_parent=pickle.load(f)['actor']
    frozen=jax.tree.map(jnp.asarray,dict(g0=fixed['g0'],stage1_teacher=fixed['stage1_teacher'],parent=retention_parent))
    def rep(q):return jax.tree.map(lambda x:jnp.stack([jnp.asarray(x)]*count),q)
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(p['lr'],eps=p['optimizer_eps']))
    assert sha(BASE/'inputs/return_replay.pkl')==p['return_replay_sha256']
    from scripts.je10_causal_replay import load_verified
    return_replay=jax.tree.map(jnp.asarray,load_verified(BASE/'inputs/return_replay.pkl',BASE/'inputs/RETURN_REPLAY.json'))
    distance_enabled='distance_curriculum' in p
    roll=jax.pmap(lambda params,st,aux,key,pool,course,pfsp,assist,stage,dagger_round:ops.collect(params,st,aux,key,pool,frozen,course,pfsp,steps,assist_probability=assist,expert_hazard_mode=p["drop_expert"].get("hazard_mode","global"),bridge_config=p.get("stage_b_reward"),return_config=p.get("return_expert"),chase_stage=stage if distance_enabled else None,dagger_round=dagger_round))
    learn=jax.pmap(lambda params,adam,batch,key,ema,replay_data:ops.ppo(params,adam,batch,key,ema,opt,frozen,mb,gradient_accumulation=p['gradient_accumulation'],expert_coefficient=p['drop_expert']['coefficient'],expert_temperature=p['drop_expert']['temperature'],return_coefficient=p.get('return_expert',{}).get('coefficient',0.),return_temperature=p.get('return_expert',{}).get('temperature',1.),return_replay=return_replay,long_fraction=p['return_coverage']['long_return_fraction'],dagger_replay=replay_data),axis_name='dev')
    aggregate=jax.pmap(dagger.aggregate)
    def summaries(params,old):
        norms=jnp.stack([jnp.sqrt(sum(jnp.sum((x-y)**2) for x,y in zip(jax.tree.leaves(params[k]),jax.tree.leaves(old[k])))) for k in ['actor','critic']])
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(params)]))
        return norms,finite
    summarize=jax.pmap(summaries)
    anchor=jax.tree.map(jnp.asarray,initial['actor'])
    def actor_drift(q):
        distance=lambda ref:jnp.sqrt(sum(jnp.sum((x-y)**2) for x,y in zip(jax.tree.leaves(q['actor']),jax.tree.leaves(ref))))
        return jnp.stack([distance(anchor),distance(frozen['g0'])])
    measure_drift=jax.pmap(actor_drift)
    out=RUN/'formal'/ARM;out.mkdir(parents=True,exist_ok=True);worker_lock=lock(out/'worker.lock')
    if resume:
        m=verify(resume)
        with (Path(resume)/'state.pkl').open('rb') as f:saved=pickle.load(f)
        assert saved['shape']==[count,n,steps,mb]
        assert digest(saved['live'])==m['full_state_digest']
        live={k:(rep(v) if k=='pool' else jax.tree.map(jnp.asarray,v)) for k,v in saved['live'].items()}
        generation=saved['generation'];del saved
    else:
        pool=jax.tree.map(lambda *xs:np.stack(xs),*[initial['actor'] if name in ['current','ring0','ring1','ring2','ring3'] else fixed[name] for name in POOL])
        if p.get('return_expert'):
            with (RUN/'inputs/initial_pool.pkl').open('rb') as f: pool=jax.tree.map(jnp.asarray,pickle.load(f))
        key=jax.random.split(jax.random.PRNGKey(p['seed']),count)
        course=ops.curriculum_weights(jnp.zeros((10,len(ops.FUNNEL))),p.get("combat_fraction",.4),p.get("bridge_fraction",.35))
        states=jax.pmap(lambda k:scenes.init_batch(k,n,course,jnp.int32(0) if distance_enabled else None))(key)
        live=dict(params=rep(initial),adam=rep(opt.init(jax.tree.map(jnp.asarray,initial))),states=states,aux=rep(ops.reward_state(n)),key=key,
            ema=rep(initial),pool=rep(pool),funnel=rep(initial_funnel()),pool_counts=rep(jnp.zeros((11,6))),
            course=rep(course),pfsp=rep(ops.pfsp_weights(jnp.zeros((11,6)))),assist_level=rep(jnp.int32(0)),assist_proof_hashes=rep(jnp.zeros((3,32),jnp.uint8)))
        generation=0
        live['dagger_buffer']=rep(dagger.buffer())
        if distance_enabled:live['distance_stage']=rep(jnp.int32(0))
    atomic_json(out/('REPLAY_WORKER.json' if replay else 'WORKER.json'),dict(pid=os.getpid(),devices=list(range(count)),backend=jax.default_backend(),shape=[count,n,steps,mb],resume=str(Path(resume).relative_to(ROOT)) if resume else None,unix=time.time()))
    def checkpoint(g):
        cp=out/f'gen{g:06d}';cp.mkdir(exist_ok=True)
        if (cp/'COMMITTED.json').exists():
            try:verify(cp,force=True);return
            except Exception as error:
                atomic_json(RUN/'state'/f'PRESERVED_CORRUPT_{g:06d}_{time.time_ns()}.json',dict(checkpoint=str(cp.relative_to(ROOT)),error=repr(error)))
                cp=out/f'gen{g:06d}_recovery_{time.time_ns()}';cp.mkdir()

        start=time.time();host=jax.device_get(live)
        for name in ['params','adam','ema']:
            for x in jax.tree.leaves(host[name]):assert all(np.array_equal(x[0],x[i]) for i in range(1,count))
        host['pool']=jax.tree.map(lambda x:x[0],host['pool'])
        adam_steps=np.asarray(host['adam'][1][0].count)
        assert np.all(adam_steps==g*p['updates_per_generation'])
        full=digest(host)
        save_pickle(cp/'state.pkl',dict(generation=g,shape=[count,n,steps,mb],live=host,pool_storage='unreplicated; reconstructed across4'))
        save_pickle(cp/'policy.pt',jax.tree.map(lambda x:x[0],host['params']['actor']))
        save_pickle(cp/'ema_policy.pt',jax.tree.map(lambda x:x[0],host['ema']['actor']))
        next_rng=np.asarray(jax.vmap(lambda k:jax.random.split(k)[0])(live['key']))
        atomic_json(cp/'COMMITTED.json',dict(passed=True,generation=g,arm=ARM,devices=count,shape=[count,n,steps,mb],
            state_schema=STATE_SCHEMA,fresh_ticks=g*count*n*steps,plan_sha256=sha(RUN/'plan.json'),state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),
            adam_steps=adam_steps.tolist(),gradient_accumulation=p['gradient_accumulation'],
            ema_policy_sha256=sha(cp/'ema_policy.pt'),full_state_digest=full,next_rng_digest=digest(next_rng),replicas_exact=True,
            input_encoding='bun24-trap-timer-v2',regions_sha256=sha(RUN/'inputs/regions.pkl'),checkpoint_seconds=time.time()-start,pool_versions=pool_versions(g),unix=time.time()))
        atomic_json(out/'LATEST_CHECKPOINT.json',dict(generation=g,checkpoint=str(cp.relative_to(ROOT)),committed_sha256=sha(cp/'COMMITTED.json')))
    ring_interval=p.get('ring_interval',128);ring_offset=p.get('ring_generation_offset',0)
    def pool_versions(g):return dict(current=g,ring_interval=ring_interval,ring_writes=(g+ring_offset)//ring_interval,immutable=POOL[1:7])
    if not resume:checkpoint(0)
    begin=time.time();compute=0.;stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    target=p['generations']
    for g in range(generation+1,target+1):
        start=time.time();previous=live['params']
        stage=live.get('distance_stage',rep(jnp.int32(0)))
        if distance_enabled:stage=rep(jnp.int32(min(3,(g-1)//4)))
        states,aux,key,batch,stats=roll(live['params'],live['states'],live['aux'],live['key'],live['pool'],live['course'],live['pfsp'],jnp.array(curriculum.ASSIST_PROBABILITIES)[live['assist_level']],stage,jnp.full(count,g,jnp.int32))
        jax.block_until_ready(batch)
        ctx=np.asarray(batch[-1]);assert not np.any((ctx[...,18]>0)&(ctx[...,23]==0)), 'free enemy chase label'
        if not replay and (g==1 or not (RUN/'validation/ITEM_PRODUCTION_GRADIENT.json').exists()):
            from scripts.je09_stage_h_proof import production_proof
            production_proof(live['params'],batch,RUN)
        new_buffer,dagger_replay,dagger_stats=aggregate(live['dagger_buffer'],batch,key,jnp.full(count,g,jnp.int32))
        if not replay:
            from scripts.je09_stage_h5c_proof import prove
            if not (RUN/'validation/H5_PRODUCTION_GRADIENT.json').exists():prove(live['params'],batch,dagger_replay,RUN)
        roll_seconds=time.time()-start;t=time.time()
        params,adam,ema,loss=learn(live['params'],live['adam'],batch,key,live['ema'],dagger_replay);jax.block_until_ready(params);update_seconds=time.time()-t
        norms,finite=summarize(params,previous);norms=np.asarray(norms);drift=np.asarray(measure_drift(params))
        assert bool(np.asarray(finite).all()) and np.isfinite(np.asarray(loss)).all(),'nonfinite update'
        counts,poolcounts,component,means=map(np.asarray,stats)
        funnel=.95*np.asarray(live['funnel'][0])+.05*counts.sum(0).sum(1)
        pool_counts=.95*np.asarray(live['pool_counts'][0])+.05*poolcounts.sum(0)
        course=ops.curriculum_weights(jnp.asarray(funnel),p.get("combat_fraction",.4),p.get("bridge_fraction",.35));pfsp=ops.pfsp_weights(jnp.asarray(pool_counts))
        pool=jax.tree.map(lambda a,b:a.at[:,0].set(b),live['pool'],params['actor'])
        if (g+ring_offset)%ring_interval==0:
            slot=7+((g+ring_offset)//ring_interval-1)%4
            pool=jax.tree.map(lambda a,b:a.at[:,slot].set(b),pool,params['actor'])
            atomic_json(RUN/'state'/f'LEAGUE_{g:06d}.json',dict(generation=g,slot=slot,actor_digest=digest(jax.tree.map(lambda x:np.asarray(x[0]),params['actor'])),pool_versions=pool_versions(g)))
        assist_level=live['assist_level'];assist_proofs=live['assist_proof_hashes']
        for proof_file in sorted((BASE/'assist_probe_v1').glob('*/PROBE_COMPLETE.json')):
            proof=read(proof_file);level=int(np.asarray(assist_level)[0]);identity=proof.get('policy_sha256','')
            if len(identity)!=64:continue
            token=np.frombuffer(bytes.fromhex(identity),np.uint8)
            if any(np.array_equal(token,x) for x in np.asarray(assist_proofs)[0]):continue
            update=curriculum.update_assist_course(dict(level=level,proofs=[],manifest_sha256=curriculum.PROBE_MANIFEST_SHA),proof)
            if update['level']>level:
                assist_proofs=assist_proofs.at[:,level].set(jnp.asarray(token));assist_level=rep(jnp.int32(update['level']))
                atomic_json(RUN/'state'/f'ASSIST_ANNEAL_{g:06d}.json',dict(generation=g,level=update['level'],proof_sha256=sha(proof_file),probability=curriculum.assist_probability(update)))
        live=dict(params=params,adam=adam,ema=ema,states=states,aux=aux,key=jax.vmap(lambda k:jax.random.split(k)[0])(key),pool=pool,
            funnel=rep(funnel),pool_counts=rep(pool_counts),course=rep(course),pfsp=rep(pfsp),assist_level=assist_level,assist_proof_hashes=assist_proofs)
        live['dagger_buffer']=new_buffer
        if distance_enabled:live['distance_stage']=stage
        elapsed=time.time()-start;compute+=elapsed;ticks=count*n*steps
        row=dict(generation=g,finite=True,nonzero_update=bool(np.all(norms>0)),actor_update_l2=norms[:,0].tolist(),critic_update_l2=norms[:,1].tolist(),actor_from_initial_l2=drift[:,0].tolist(),actor_from_g0_l2=drift[:,1].tolist(),
            gradient_accumulation=p['gradient_accumulation'],effective_global_actor_batch=p['effective_global_actor_batch'],
            fresh_ticks=g*ticks,fresh_ticks_per_second=ticks/elapsed,optimizer_updates=128,optimizer_updates_per_second=128/elapsed,
            rollout_seconds=roll_seconds,update_seconds=update_seconds,seconds=elapsed,compute_seconds=compute,wall_seconds=time.time()-begin,
            loss=np.asarray(loss).mean(0).tolist(),loss_names=['loss','gradient_preclip','policy','value','entropy','ppo_approx_kl','teacher_loss','g0_move_kl','g0_bomb_kl','g0_ability_kl_diagnostic','stage1_move_kl','stage1_bomb_kl','stage1_ability_kl_diagnostic','parent_noncarry_kl','expert_loss','expert_ce','expert_eligible_fraction','return_expert_loss','return_expert_ce','return_expert_eligible_fraction'],
            component_names=list(ops.COMPONENTS),reward_components=component.sum(0).tolist(),mean_reward=float(means[:,0].mean()),
            funnel_names=list(ops.FUNNEL),funnel_by_lesson_family=counts.sum(0).tolist(),opponent_families=['responsive_rule','g0','stage1_teacher','pfsp_league'],
            actual_opponent_valid_ticks=counts.sum((0,1))[:,0].tolist(),scene_names=list(scenes.NAMES),actual_scene_valid_ticks=counts.sum((0,2))[:,0].tolist(),
            curriculum_weights=np.asarray(course).tolist(),pfsp_weights=np.asarray(pfsp).tolist(),pfsp_counts=pool_counts.tolist(),
            pool_versions=pool_versions(g),learner_seat_counts=[count*n//2*steps,count*n//2*steps],unix=time.time())
        row['dagger']={name:float(np.asarray(value).sum()) for name,value in dagger_stats.items()}
        for name in ['history_fraction','new_fraction']:row['dagger'][name]/=count
        row['dagger']['source_policy_digest']=digest(jax.tree.map(lambda x:np.asarray(x[0]),previous['actor']))
        row['dagger']['learner_generation']=g-1
        row['dagger']['native_transitions']=dict(zip(['own_window_enter','own_kill','enemy_base_entry',
            'loose_pickup','inventory_or_native_steal','attributed_custody_start','capture','native_transfer'],
            ctx[...,60:68].sum((0,1,2)).tolist()))
        role=ctx[...,68].astype(np.int32);tier=ctx[...,69].astype(np.int32);natural=ctx[...,0]==8
        eligible=ctx[...,55]>0;query=ctx[...,53]>0;label=ctx[...,54].astype(np.int32)
        row['dagger']['opponent_course']={str(role_id):dict(visited=int(((role==role_id)&natural).sum()),
            eligible=int(((role==role_id)&eligible).sum()),query=int(((role==role_id)&query).sum()),
            path_a=int(((role==role_id)&query&((label==1)|(label==4))).sum()),
            path_b=int(((role==role_id)&query&(label==2)).sum()),
            carry_return=int(((role==role_id)&query&(label==3)&(ctx[...,57]<=0)).sum()),
            cage_delivery=int(((role==role_id)&query&(label==3)&(ctx[...,57]>0)).sum()),
            candidate_a=int(((role==role_id)&eligible&((label==1)|(label==4))).sum()),
            candidate_b=int(((role==role_id)&eligible&(label==2)).sum()),
            candidate_carry_return=int(((role==role_id)&eligible&(label==3)&(ctx[...,57]<=0)).sum()),
            candidate_cage_delivery=int(((role==role_id)&eligible&(label==3)&(ctx[...,57]>0)).sum()),
            opponent_native_custody_ticks=int(((role==role_id)&natural&(ctx[...,70]>0)).sum()),
            native_transitions=ctx[...,60:68][(role==role_id)&natural].sum(0).tolist()) for role_id in [0,1,2]}
        row['dagger']['difficulty']=int(tier.max())
        row['dagger']['strong_full_families']={str(i):int((natural&(role==0)&(ctx[...,71]==i)).sum()) for i in range(4)}
        if ctx.shape[-1] > 72:
            target_fragment=ctx[...,72].astype(np.int32)
            target_rows={}
            for target_id,target_name in enumerate(['none','a_loose','b_stock','carry_return','cage_delivery']):
                target_run_ids=(target_fragment==target_id)&natural
                target_rows[str(target_id)]=dict(name=target_name,
                    visited=int(target_run_ids.sum()), candidate=int((target_run_ids&eligible).sum()),
                    query=int((target_run_ids&query).sum()),
                    weak_courier=int((target_run_ids&(role==1)).sum()),
                    weak_guard=int((target_run_ids&(role==2)).sum()),
                    difficulty={str(level):int((target_run_ids&(tier==level)).sum()) for level in range(4)},
                    native_transitions=ctx[...,60:68][target_run_ids].sum(0).tolist())
            row['dagger']['targeted_course']=target_rows
        assert row['dagger']['noneligible_label']==0
        row['assist_probability']=curriculum.ASSIST_PROBABILITIES[int(np.asarray(live['assist_level'])[0])]
        if distance_enabled:row['distance_stage']=int(np.asarray(stage)[0])
        row['actual_assist_ticks']=float(counts[...,ops.FUNNEL.index('assist_tick')].sum())
        row['actual_assist_valid_fraction']=float(counts[...,ops.FUNNEL.index('assist_tick')].sum()/max(1,counts[...,0].sum()))
        row['xla_memory']=[device.memory_stats() for device in jax.local_devices()]
        if not replay:
            with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        if not replay:atomic_json(out/'STATE.json',row)
        print(json.dumps(row),flush=True)
        if replay:
            expected=read(out/f'gen{g:06d}'/'COMMITTED.json');host=jax.device_get(live);host['pool']=jax.tree.map(lambda x:x[0],host['pool'])
            assert digest(host)==expected['full_state_digest'],'fresh-process next-update mismatch'
            atomic_json(RUN/'state/FRESH_GPU_REPLAY.json',dict(passed=True,fresh_process=True,platform='gpu',generation=g,full_state_digest=digest(host),next_update_exact=True,shape=SHAPE))
            return
        if g==1 or g in POINTS or g%p['recovery_every']==0 or stopping[0]:checkpoint(g)
        if stopping[0]:break
    checkpoint(g)
    atomic_json(out/'TRAINER_EXIT.json',dict(passed=True,generation=g,fresh_ticks=g*count*n*steps,compute_seconds=compute,wall_seconds=time.time()-begin,atomic_stop=stopping[0],unix=time.time()))
    if g==target and not stopping[0]:
        atomic_json(RUN/'TRAINING_COMPLETE.json',dict(passed=True,generation=g,fresh_ticks=g*count*n*steps,unix=time.time()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['formal','restore-monitor','replay']);ap.add_argument('--resume');args=ap.parse_args()
    try:
        if args.command=='restore-monitor':restore_monitor(args.resume)
        else:worker(args.resume,replay=args.command=='replay')
    except Exception:
        atomic_json(RUN/'state'/f'TECHNICAL_{args.command}_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),unix=time.time()));raise
