"""Conservative native Bun392 paired self-play worker; explicit full-state league."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.je03_bun_selfplay_20261012 import *
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
    from scripts import je03_bun_selfplay_ops_20261012 as ops
    forward = train.net_forward
    train.net_forward = lambda q,a,o,s=None: split_forward(q,a,o,s,forward)
    ops.net_forward = train.net_forward
    def load(name):
        with (RUN/'inputs'/name).open('rb') as f:return jax.tree.map(jnp.asarray,pickle.load(f))
    frozen = load('bun392_gen0.pkl')
    stage1 = load('stage1_actor.pkl')
    stage1_critic = load('stage1_critic.pkl')
    def opponent(states, obs, gv, masks, arch, key, kinds, weak, old, league):
        n=states.pos.shape[0]; rows=jnp.arange(n); seats=1-rows%2
        idx=rows+seats*n; mm,bm=masks[0][idx],masks[1][idx]
        kr,kf,ks,kn,kh=jax.random.split(key,5)
        rule=train.flee_bot_actions(states.pos[rows,seats],states.pos[rows,1-seats],mm,bm,kr,idle_ratio=0.,roam_ratio=0.,pure_flee_ratio=0.)
        sample=lambda q,k:train.sample_actions(q,arch,obs[idx],(mm,bm),k,state=gv[idx])[0]
        oldacts=sample(frozen,kf); fixedacts=sample(stage1,ks)
        if r.get('dynamic_enabled',arm.endswith('dynamic')):
            recentacts=sample(league[0],kn); historyacts=sample(league[1],kh)
        else:
            recentacts=historyacts=fixedacts
        return train.select_bun_opponent_actions(kinds,jnp.stack([rule,rule,rule,fixedacts,oldacts,recentacts,historyacts]))
    ops.bun_opponent_actions = opponent
    weights = jnp.asarray(train.parse_bun_opponent_weights(r['opponents']))
    n, steps, mb = (2,2,4) if cpu else (p['envs_per_device'],p['steps'],p['minibatch_per_device'])
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(r['lr'],eps=p['optimizer_eps']))
    def rollout(params, states, key, league):
        result=ops.collect_rollout(params,'transformer',states,key,steps,crate_coef=r['crate_coef'],brick_coef=r['brick_coef'],
            flee_bot_ratio=0.,bun_opponent_weights=weights,bun_opponent_old_params=frozen,bun_opponent_recent_params=league,
            return_components=RETENTION_V2, return_lesson_stats=RETENTION_V2)
        states,batch=result[:2]; obs,st,acts,lps,vals,rew,done,masks=batch
        idx=ops.balanced_indices(n)[0]
        select=lambda x:x[:,idx]
        obs,st,acts,lps,vals,rew,done=map(select,(obs,st,acts,lps,vals,rew,done)); masks=tuple(select(x) for x in masks)
        all_masks=train.both_masks(states)
        _,_,tail=train.sample_actions(params,'transformer',train.both_perspectives(states)[idx],tuple(x[idx] for x in all_masks),jax.random.split(key)[0],state=train.both_states(states)[idx])
        adv=train.compute_gae(rew,vals,jnp.concatenate([vals[1:],tail[None]]),done,p['gamma'],p['lam'])
        base_stats=jnp.array([jnp.mean(rew),jnp.mean(done),jnp.sum(result[3])])
        if RETENTION_V2:
            component_stats=result[4].reshape(-1)
            lesson_stats=result[5]
            stats=jnp.concatenate([base_stats,component_stats,lesson_stats,result[6],result[7]])
        else:
            stats=base_stats
        return states,(obs,st,acts,lps,adv,adv+vals,masks),stats
    teacher_params = None
    if RETENTION_V2:
        teacher_params=dict(actor=frozen, critic=frozen)
    def update(params, adam, batch, key, ema):
        params,adam,loss=ops.ppo_update(
            params,opt,adam,'transformer',batch,key,mb,p['clip'],p['value_coef'],
            r['entropy'],p['epochs'],axis_name='dev',return_loss=True,
            adv_top_frac=r['adv_top_frac'], reference_params=teacher_params,
            reference_kl_coef=r.get('teacher_kl_coef', 0.0))
        d=p['ema_decay']
        return params,adam,jax.random.split(key)[0],jax.tree.map(lambda e,x:d*e+(1-d)*x,ema,params),loss
    def diagnostics(params,previous,teacher,batch):
        o,st,_,_,_,_,masks=batch
        o=o.reshape(-1,*o.shape[2:])[:64]; st=st.reshape(-1,st.shape[-1])[:64]
        mm=masks[0].reshape(-1,5)[:64]; bm=masks[1].reshape(-1,3)[:64]
        def distribution(q):
            mv,ab,_,_=train.net_forward(q,'transformer',o,st)
            joint=ops.adjusted_joint_logits(mv,ab,mm,bm,jnp.zeros((len(o),5,3),jnp.bool_),'off',0.)
            return jax.nn.log_softmax(joint)
        lp=distribution(params); ref=distribution(teacher); oldlp=distribution(previous); prob=jnp.exp(lp)
        kl=lambda target:jnp.mean(jnp.sum(jnp.where(prob>0,prob*(lp-target),0.),axis=-1))
        ent=-jnp.mean(jnp.sum(jnp.where(prob>0,prob*lp,0.),axis=-1))
        return jnp.array([kl(ref),kl(oldlp),ent])
    diagnose=jax.pmap(diagnostics)
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
        init=stage1
        params=dict(actor=init,critic=stage1_critic)
        key=jax.random.split(jax.random.PRNGKey(p['seed']),count)
        states=jax.pmap(lambda k:env.init_batch(k,n))(key)
        live=(rep(params),rep(opt.init(params)),states,key,rep(params),(rep(stage1),rep(stage1))); generation=0
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
        states,batch,stats=roll(live[0],live[2],live[3],live[5]); jax.block_until_ready(batch); rt=time.time()-start; t=time.time()
        params,adam,key,ema,loss=learn(live[0],live[1],batch,live[3],live[4]); jax.block_until_ready(params)
        ut=time.time()-t
        diagnostic=diagnose(params,live[0],rep(teacher_params), batch)
        jax.block_until_ready(diagnostic)
        league=live[5]
        league=ops.advance_league(league,params['actor'],g,p['league_refresh_every'])
        live=(params,adam,states,key,ema,league); del batch
        arrays=jax.tree.leaves(params)
        finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(live)) and np.isfinite(np.asarray(loss)).all() and np.isfinite(np.asarray(diagnostic)).all()
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
            row['gradient_norm_preclip']=np.asarray(loss)[:,1].tolist()
            row['policy_diagnostics']=np.asarray(diagnostic).tolist()
            row['policy_diagnostic_names']=['kl_to_g0','kl_update','entropy']
            row['opponent_tick_counts']=stats_array[:,35:42].tolist()
            row['lesson_tick_counts']=stats_array[:,42:58].tolist()
            row['actor_drift_from_stage1_l2']=float(np.sqrt(sum(float(jnp.sum((x-rep(y))**2)) for x,y in zip(jax.tree.leaves(params['actor']),jax.tree.leaves(stage1)))))
            row['league_current_version']=g//p['league_refresh_every']
            row['league_history_version']=max(0,g//p['league_refresh_every']-1)
            # Actual teacher KL and update KL are telemetry, never startup gates.
            if g%p['league_refresh_every']==0:
                atomic_json(out/f'LEAGUE_VERSION_{g:06d}.json', dict(generation=g, recent_digest=digest(league[0]), history_digest=digest(league[1]), unix=time.time()))
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n'); f.flush()
        if phase!='formal' or g in POINTS or g%p['recovery_every']==0 or stopping[0]:checkpoint(g)
        atomic_json(out/'STATE.json',dict(row,compute_seconds=compute,wall_seconds=time.time()-begin)); print(json.dumps(row),flush=True)
        if stopping[0]:break
    checkpoint(g)
    atomic_json(out/'TRAINER_EXIT.json',dict(passed=True,generation=g,pid=os.getpid(),platform=jax.default_backend(),
        final_state_digest=digest(live),fresh_ticks=g*count*n*steps,compute_seconds=compute,wall_seconds=time.time()-begin,stopped_at_update_boundary=stopping[0]))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True,choices=ARMS);ap.add_argument('--phase',default='formal');ap.add_argument('--devices',default='');ap.add_argument('--resume');ap.add_argument('--cpu',action='store_true');args=ap.parse_args()
    try: worker(args.arm,args.phase,args.devices,args.resume,args.cpu)
    except Exception:
        atomic_json(RUN/f'TECHNICAL_STOP_worker_{os.getpid()}.json',dict(traceback=traceback.format_exc(),unix=time.time()));raise
