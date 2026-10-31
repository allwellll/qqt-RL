"""Authorized rollback to original recipe, with policy/new-Critic warmup init."""
import argparse,json,os,pickle,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def formal_configuration(spec):
    """Restore the authorized original mixture without either A2/A3 add-on."""
    assert spec.get('iter2_fallback') is True
    assert spec['curriculum_fraction']==.25
    assert spec['opponent_weights']==[.2,.25,.2,.15,.1,.1]
    return dict(curriculum_fraction=spec['curriculum_fraction'],opponent_weights=spec['opponent_weights'],
        entropy_anneal=False,shaping_multiplier=1.,shaping_cap=.2)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('protocol');parser.add_argument('arm');parser.add_argument('seed',type=int)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--verify-restore',action='store_true');args=parser.parse_args()
    protocol=json.loads(Path(args.protocol).read_text());spec={**protocol['common'],**protocol['arms'][args.arm]}
    formal=formal_configuration(spec)
    out=ROOT/protocol['namespace']/args.arm;out.mkdir(parents=True,exist_ok=True);guard=lock(out/'worker.lock')
    for p,h in protocol['source_sha256'].items():assert sha(ROOT/p)==h,p
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda',
        JAXBOMB_RULE='bun',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
        JAX_COMPILATION_CACHE_DIR=str(BASE/'iter2/cache'),JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0')
    import jax,jax.numpy as j,numpy as np,optax,copy
    from scripts import ppo_scale_longrun_train as P,bomb_threat_collector as M,bomb_threat_reward as R
    from scripts import bomb_threat_iter2_recipe as recipe
    from scripts.bomb_threat_runtime import install
    from scripts.carrier_mechanism_analysis import bank,merged,shard
    from scripts.combat_mix_opponents import fixed_allocation,combat_scene_mask
    from scripts.campaign_paired_train_canonical import canonical_collector
    from scripts.checkpoint_policy_artifact import load_actor
    from scripts.terminal_checkpoint_restore import digest
    assert jax.default_backend()=='gpu' and jax.local_device_count()==1
    P.T.configure();P.C.install_provenance();install()
    items=[v for v in bank() if v['record']['split']=='train'];bs,ba=merged(items,'states'),merged(items,'aux')
    n=spec['envs_per_device'];batch_size=n*spec['rollout_ticks'];steps_per_update=batch_size//spec['global_minibatch']
    rep=lambda tree:jax.tree.map(lambda a:j.asarray(a)[None],tree)
    seed=args.seed;policy=ROOT/protocol['policy_source']['checkpoint']/'policy.pt'
    assert sha(policy)==protocol['policy_source']['policy_sha256']
    actor=load_actor(policy.parent)
    opt=optax.adam(spec['lr'],eps=1e-8)
    start=0;warm_rows=[];env_steps=0;optimizer_steps=0;ring_cursor=0;formal_initialized=False
    cps=sorted(out.glob('u*/COMMITTED.json'))
    source=None
    if cps:source=cps[-1]
    elif protocol.get('continuation_source'):source=ROOT/protocol['continuation_source']['checkpoint']/'COMMITTED.json'
    if source:
        cm=json.loads(source.read_text());cp=source.parent
        assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']
        payload=pickle.load((cp/'state.pkl').open('rb'))
        params=payload['params'];adam=payload['adam'];st=payload['states'];ax=payload['aux'];keys=payload['keys']
        ring=payload['combat_ring'];ring_cursor=payload['combat_ring_cursor'];fixed=payload['combat_fixed_actor']
        start=payload['update'];warm_rows=payload['warmup_rows'];env_steps=payload['environment_steps'];optimizer_steps=payload['optimizer_steps']
        assert payload['iter2_arm']==spec['iter2_arm']
        assert payload['iter2_fallback'] is True
        formal_initialized=payload['formal_distribution_initialized']
    else:
        params=dict(actor=actor,critic=copy.deepcopy(actor))
        adam={k:opt.init(v) for k,v in params.items()}
        keys=jax.random.split(jax.random.PRNGKey(seed),n)
        # A1 distribution for all three warmups; certified Bank states only.
        spawns=json.loads((ROOT/protocol['spawn_manifest']).read_text())['spawn_cells']
        designated=j.asarray(combat_scene_mask(n,seed^0x806,.10));kinds=fixed_allocation(n,seed,protocol['A1_weights'])
        fresh,aux=jax.jit(M.arena_fresh)(keys,bs,ba,designated,j.asarray(spawns))
        aux=dict(aux,combat_mix=M.metadata(fresh,kinds,designated))
        st,ax,keys=shard(fresh,1),shard(aux,1),shard(keys,1)
        ring=[copy.deepcopy(actor) for _ in range(4)]
        fixed_source=ROOT/protocol['fixed_actor_source']['checkpoint']
        assert sha(fixed_source/'policy.pt')==protocol['fixed_actor_source']['policy_sha256'];fixed=load_actor(fixed_source)
        # Eight actual fresh states, both Actor paths must be bitwise identical.
        ss=jax.tree.map(lambda a:j.asarray(a[0,:8]),st)
        o,g,*_=M.E.inputs(ss)
        a=M.E.train.net_forward(actor,'transformer',o,g)
        b=M.E.train.net_forward(params,'transformer',o,g)
        for x,y in zip(a[:2],b[:2]):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
        atomic_json(out/'INIT.json',dict(status='PASS',authorization=protocol['initialization_decision'],
            policy_source=protocol['policy_source'],source_actor_digest=digest(actor),actor_logits_first8_bitwise=True,
            critic='independent deep copy of homologous policy network/value head; not historical trained Critic',
            Adam='new zero moments, separate Actor/Critic counters; actor counter held at0 during warmup',
            Adam_zero_initial=True,seed=seed,environment='new seed, native reset or certified real-acquisition train Bank',
            ring='four initial Actor copies, refresh every16 PPO updates',fixed_opponent=protocol['fixed_actor_source'],
            initial_sources={'real_bank':int(np.sum(np.asarray(aux['provenance_source'])>=0)),
                'natural_empty_hand':int(np.sum((np.asarray(aux['provenance_source'])<0)&(np.asarray(fresh.bun_carried[:,0])<0)))},
            initial_full_env_digest=digest((st,ax,keys)),unix=time.time()))
    params=jax.tree.map(j.asarray,params);adam=jax.tree.map(j.asarray,adam)
    if (spec['iter2_arm']=='A3' and not spec.get('iter2_fallback')) and len(warm_rows)>=20:R.settle=recipe.settlement(5.,.6)
    par,adm=rep(params),rep(adam)
    spawns=json.loads((ROOT/protocol['spawn_manifest']).read_text())['spawn_cells']
    collect=canonical_collector(M.collector(spec['rollout_ticks'],n,seed,.10,0.,spawns,protocol['A1_weights']),out/'INITIAL_PLACEMENT.json')
    def objective(p,b,ent):
        loss,metrics=P.pure_ppo_objective(p,b)
        return loss+(.005-ent)*metrics[2],metrics
    update=recipe.updater(spec['global_minibatch'],opt,objective)
    first_kind=int(np.asarray(ax['combat_mix']['kind'])[0,0])
    probe_groups={name:np.array([0]) if first_kind==i else np.array([],int) for i,name in [(3,'current'),(4,'ring'),(5,'fixed')]}
    probe_groups['bot']=np.array([0]) if first_kind<3 else np.array([],int)
    probe_choose=jax.jit(lambda p,s,a,k,r:M.choose(p,r,fixed,s,a,k,probe_groups))
    probe_step=jax.jit(lambda s,a,ac,k:M.transition(s,a,ac,k,0.))
    def probe(p,s,a,k,index):
        ss,aa,kk=jax.tree.map(lambda x:j.asarray(x[0,:1]),(s,a,k))
        rr=ring[index%4]
        nk,ke,ac,_=probe_choose(p,ss,aa,kk,rr)
        step=probe_step(ss,aa,ac,ke)
        return digest((nk,ke,ac,step))
    def save(index,row):
        cp=out/f'u{index:06d}';cp.mkdir(parents=True,exist_ok=True)
        payload=dict(params=P.one(par),adam=P.one(adm),states=jax.device_get(st),aux=jax.device_get(ax),keys=jax.device_get(keys),
            update=index,environment_steps=env_steps,optimizer_steps=optimizer_steps,seed=seed,iter2_arm=spec['iter2_arm'],
            arm=args.arm,iter2_fallback=True,formal_distribution_initialized=formal_initialized,
            combat_ring=ring,combat_ring_cursor=ring_cursor,combat_fixed_actor=fixed,warmup_rows=warm_rows,
            protocol_sha256=sha(Path(args.protocol)))
        for name,value in [('state.pkl',payload),('policy.pt',payload['params']['actor'])]:
            p=cp/(name+'.tmp')
            with p.open('wb') as f:pickle.dump(value,f,protocol=5);f.flush();os.fsync(f.fileno())
            p.replace(cp/name)
        result=dict(passed=True,checkpoint=str(cp.relative_to(ROOT)),state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),
            update=index,formal_update=index,environment_steps=env_steps,optimizer_steps=optimizer_steps,metrics=row)
        atomic_json(cp/'COMMITTED.json',result)
        if index>0 and index%40==0:atomic_json(out/'eval_queue'/f'u{index:06d}.json',result)
        token=dict(digest=probe(payload['params'],st,ax,keys,index),Adam_digest=digest(payload['adam']),
            full_environment_digest=digest((st,ax,keys)),ring_digest=digest(ring))
        atomic_json(out/f'NEXT_STEP_u{index:06d}.json',token)
        return result
    if args.verify_restore:
        assert cps,'restore requires checkpoint saved by this worker'
        expected=json.loads((out/f'NEXT_STEP_u{start:06d}.json').read_text())
        assert probe(P.one(par),st,ax,keys,start)==expected['digest']
        assert digest(P.one(adm))==expected['Adam_digest'] and digest((st,ax,keys))==expected['full_environment_digest']
        assert digest(ring)==expected['ring_digest']
        token_cp=source.parent
        atomic_json(out/'RESTORE.json',dict(passed=True,fresh_process=True,next_step_exact=True,Adam_exact=True,
            full_env_aux_RNG_exact=True,ring_exact=True,checkpoint=str(token_cp.relative_to(ROOT)),unix=time.time()));return
    # Warmup 20 full updates is initialization, never policy evaluation/PBT.
    actor_digest=digest(P.one(par)['actor'])
    def paused():return any((b/f).exists() for b in (BASE,ROOT/'runs/iter_pbt_v1') for f in ('USER_PAUSE.json','USER_STOP.json'))
    if len(warm_rows)<20:
        for wi in range(len(warm_rows)+1,21):
            if paused():save(0,dict(warmup_paused=True));return
            began=time.monotonic();before=P.one(par)
            (st,ax,keys),batch,diag=collect(par,st,ax,keys,bs,ba,rep(ring[0]),rep(fixed));jax.block_until_ready(batch)
            random=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),0xB000+wi),1)
            (par,adm),loss=update(par,adm,batch,random,.005,0.,True);jax.block_until_ready(par)
            after=P.one(par);assert digest(after['actor'])==actor_digest
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(after))
            delta=P.norms(before,after);assert delta['actor']==0 and delta['critic']>0
            # On-policy pre-update critic EV against same rollout GAE returns;
            # this is training EV, not independent value accuracy proof.
            v=np.asarray(batch[5]-batch[4]);returns=np.asarray(batch[5]);var=float(np.var(returns))
            ev=1-float(np.var(returns-v))/var if var>1e-12 else None
            env_steps+=batch_size;optimizer_steps+=steps_per_update
            row=dict(warm_update=wi,value_loss=float(np.asarray(loss)[...,2].mean()),explained_variance=ev,
                Actor_frozen_exact=True,update_l2=delta,cycle_seconds=time.monotonic()-began,environment_steps=env_steps,optimizer_steps=optimizer_steps,
                distribution='A1',reward_variant='base bomb-threat-v1 during all warmups',unix=time.time())
            warm_rows.append(row);atomic_json(out/'WARMUP_PROGRESS.json',dict(rows=warm_rows,unix=time.time()))
            par,adm=rep(after),rep(P.one(adm))
        gate=recipe.warmup_ready(warm_rows);atomic_json(out/'WARMUP_GATE.json',dict(**gate,actor_frozen_exact=True,unix=time.time()))
        if not gate['passed']:
            save(0,dict(warmup_gate_failed=True));atomic_json(out/'TECHNICAL_FAILURE.json',dict(reason='20-update Critic warmup did not meet EV/plateau gate',gate=gate,unix=time.time()));return
        # A3 multiplier applies only after identical base-reward A1 warmup.
        # Warm compilation uses base settle; create a distinct formal graph.
    gate=recipe.warmup_ready(warm_rows)
    assert gate['passed'],'warmup gate was not passed'
    if (spec['iter2_arm']=='A3' and not spec.get('iter2_fallback')):
        R.settle=recipe.settlement(5.,.6)
        # Close/clear only reward credit ledgers at this documented reward
        # boundary, keeping native state. Warmup is not a PBT capability point.
        if not formal_initialized:
            ax=dict(ax,combat_mix=dict(ax['combat_mix'],bomb_threat=shard(R.state(n),1)))
        collect=canonical_collector(M.collector(spec['rollout_ticks'],n,seed,.10,0.,spawns,protocol['A1_weights']),out/'FORMAL_PLACEMENT.json')
    # The rollback arm's initialization follows the10:00 policy-only
    # convention; at formal boundary rebuild native/certified episodes using
    # the restored original curriculum/opponent mixture. No synthetic carry.
    if spec.get('iter2_fallback'):
        if not formal_initialized:
            flat_keys=jax.tree.map(lambda x:j.asarray(x[0]),keys)
            flat_keys=jax.vmap(lambda k:jax.random.fold_in(k,0xFABB))(flat_keys)
            designated=j.asarray(combat_scene_mask(n,seed^0x806,spec['curriculum_fraction']))
            fresh,aux=jax.jit(M.arena_fresh)(flat_keys,bs,ba,designated,j.asarray(spawns))
            kinds=fixed_allocation(n,seed,spec['opponent_weights'])
            aux=dict(aux,combat_mix=M.metadata(fresh,kinds,designated))
            st,ax,keys=shard(fresh,1),shard(aux,1),shard(flat_keys,1)
            formal_initialized=True
        first_kind=int(np.asarray(ax['combat_mix']['kind'])[0,0])
        probe_groups={name:np.array([0]) if first_kind==i else np.array([],int) for i,name in [(3,'current'),(4,'ring'),(5,'fixed')]}
        probe_groups['bot']=np.array([0]) if first_kind<3 else np.array([],int)
        probe_choose=jax.jit(lambda p,s,a,k,r:M.choose(p,r,fixed,s,a,k,probe_groups))
        probe_step=jax.jit(lambda s,a,ac,k:M.transition(s,a,ac,k,0.))
        collect=canonical_collector(M.collector(spec['rollout_ticks'],n,seed,formal['curriculum_fraction'],0.,spawns,formal['opponent_weights']),out/'ROLLBACK_PLACEMENT.json')
    if start==0 and not cps:
        save(0,dict(fresh_parent=True,warmup_complete=True,formal_evaluation_eligible=False))
    total=int(protocol['chunk_start_update'])+int(protocol['chunk_updates'])
    end=min(total,int(protocol['chunk_start_update'])+2) if args.smoke else total
    if not args.smoke:assert (out/'SMOKE.json').exists() and (out/'RESTORE.json').exists()
    receipt=None
    for index in range(start+1,end+1):
        if paused():save(index-1,dict(user_paused=True));return
        before=P.one(par);began=time.monotonic()
        (st,ax,keys),batch,diag=collect(par,st,ax,keys,bs,ba,rep(ring[(index-1)%4]),rep(fixed));jax.block_until_ready(batch)
        rollout=time.monotonic()-began
        random=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),index),1)
        (par,adm),loss=update(par,adm,batch,random,recipe.entropy(index,(spec['iter2_arm']=='A2' and not spec.get('iter2_fallback'))),recipe.actor_ramp(index),False)
        jax.block_until_ready(par);after=P.one(par);delta=P.norms(before,after)
        finite=all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(after)) and bool(np.isfinite(np.asarray(loss)).all())
        assert finite and delta['actor']>0 and delta['critic']>0,'nonfinite or zero real PPO update'
        if index%16==0:ring[ring_cursor]=after['actor'];ring_cursor=(ring_cursor+1)%4
        env_steps+=batch_size;optimizer_steps+=steps_per_update
        row=dict(passed=True,finite=True,new_update=index,formal_update=index,environment_steps=env_steps,optimizer_steps=optimizer_steps,
            update_l2=delta,cycle_seconds=time.monotonic()-began,rollout_seconds=rollout,actual_environment_steps_this_update=batch_size,
            actual_optimizer_steps_this_update=steps_per_update,loss=np.asarray(loss).mean((0,1)).tolist(),configuration=spec,
            actor_lr=float(spec['lr']*recipe.actor_ramp(index)),entropy_coefficient=float(recipe.entropy(index,(spec['iter2_arm']=='A2' and not spec.get('iter2_fallback')))),
            categorical=True,dagger=0,combat_kl=0,warmup_environment_steps=20*batch_size,
            exposure=jax.tree.map(lambda a:np.asarray(a).sum(0).tolist(),jax.device_get(diag)),unix=time.time())
        if (spec['iter2_arm']=='A3' and not spec.get('iter2_fallback')) and index%40==0:
            earlier=[json.loads(l)['exposure'] for l in (out/'updates.jsonl').read_text().splitlines()] if (out/'updates.jsonl').exists() else []
            exposures=earlier+[row['exposure']]
            paid=sum(e['chain_reward'] for e in exposures);total_reward=sum(e['total_reward'] for e in exposures)
            terminal_reward=sum(e['native_parts'][2]+e['native_parts'][9]+e['native_parts'][61] for e in exposures)
            cap_ok=total_reward>0 and paid<.3*total_reward and paid<terminal_reward
            atomic_json(out/'A3_PAYMENT_GUARD.json',dict(status='PASS' if cap_ok else 'BLOCKED',update=index,
                paid=paid,total_reward=total_reward,terminal_kill_delivery_reward=terminal_reward,per_episode_cap=.6,
                native_components=[2,9,61],partial_episode_prefix=True,unix=time.time()))
            if not cap_ok:
                receipt=save(index,row);atomic_json(out/'TECHNICAL_FAILURE.json',dict(reason='authorized A3 actual payment ceiling violated',last_atomic=receipt,unix=time.time()));return
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        if index in (int(protocol['chunk_start_update'])+1,int(protocol['chunk_start_update'])+2) or index%40==0 or index==end:
            receipt=save(index,row)
        atomic_json(out/'PROGRESS.json',dict(iterations=index,last=row))
        par,adm=rep(after),rep(P.one(adm))
    if args.smoke:atomic_json(out/'SMOKE.json',dict(passed=True,actor_critic_nonzero=True,warmup_gate=gate,
        checkpoint=receipt,policy_logits_first8_exact=True,env_per_second=batch_size/row['cycle_seconds'],unix=time.time()));return
    atomic_json(out/'COMPLETE.json',dict(**receipt,actual_environment_steps=env_steps,actual_optimizer_steps=optimizer_steps))


if __name__=='__main__':main()
