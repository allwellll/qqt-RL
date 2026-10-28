"""Frozen real-custody Monte Carlo, paired branches and bounded learning."""
import os,time,pickle,json,sys,subprocess
from pathlib import Path
import numpy as np
from scripts.carrier_mechanism_run import B,R,read,save,notify,authorize
from scripts.population_run_state import atomic_json,sha
from scripts.carrier_mechanism_stats import calibration,calibration_intervals,cluster_interval,discounted_returns,event_outcomes_from


def bank():
    frozen=read(B/'BANK_FROZEN.json');items=[]
    for name,h in frozen['files'].items():
        p=B/'bank'/name;assert sha(p)==h;items.extend(pickle.load(p.open('rb')))
    assert len({x['record']['source_episode'] for x in items})==len(items)
    assert len({(x['record']['layout'],x['record']['split']) for x in items})==len({x['record']['layout'] for x in items})
    if not items:raise RuntimeError('No genuine custody state: collection coverage failure, no fabricated substitute')
    return items


def setup():
    authorize()
    import jax
    from scripts import carrier_mechanism_engine as E
    E.configure();assert jax.default_backend()=='gpu'
    return jax,E


def merged(items,key):
    import jax
    return jax.tree.map(lambda *a:np.concatenate(a),*[x[key] for x in items])


def parity_reference(items,E,jax):
    from scripts.terminal_checkpoint_restore import digest
    out=[]
    step=jax.jit(E.transition);choose=jax.jit(E.choose)
    for i,x in enumerate(items):
        p=pickle.load((B/'inputs'/f"{x['record']['source_policy']}.pkl").open('rb'))['params']
        after,aux,reward,done,info,parts=step(x['before'],x['preaux'],x['acquisition_actions'],x['acquisition_envkeys'])
        assert digest(after)==digest(x['states']),'acquisition environment mismatch'
        assert digest(aux)==digest(x['aux']),'acquisition reward ledger mismatch'
        assert bool(E.causal_pickup(x['before'],after,info)[0])
        nk,ke,act,_=choose(p,x['states'],x['aux'],x['keys'])
        nxt=step(x['states'],x['aux'],act,ke)
        out.append(dict(source=x['record']['source_episode'],acquisition_info=digest(info),acquisition_reward=np.asarray(reward).tolist(),acquisition_components=np.asarray(parts).tolist(),next_hash=digest((nk,ke,act,nxt)),post_hash=digest(after),aux_hash=digest(aux)))
    return out


def parity():
    jax,E=setup();items=bank();out=parity_reference(items,E,jax)
    save(B/'parity_reference.pkl',out)
    # A new interpreter reloads the exact pickle and independently regenerates all values.
    env=dict(os.environ);code=subprocess.run([sys.executable,'-m','scripts.carrier_mechanism_analysis','parity-check'],env=env,stdout=(B/'parity_fresh.log').open('a'),stderr=subprocess.STDOUT)
    assert code.returncode==0,'fresh process parity failed'
    fresh=pickle.load((B/'parity_fresh.pkl').open('rb'));assert out==fresh
    # Replay complete actual walk-in, using native transitions and recorded joint actions.
    # No reward state is synthesized; acquisition aux parity above tests its full ledger.
    lengths=np.array([len(x['prefix_actions']) for x in items]);maxlen=int(lengths.max());n=len(items)
    initial=[]
    for x in items:
        r=x['record'];keys=jax.random.split(jax.random.PRNGKey(r['source_episode_seed']),128)
        st,_=jax.jit(E.initial,static_argnums=1)(keys[r['episode_index']:r['episode_index']+1],['train','dev','holdout'].index(r['split']))
        initial.append(jax.device_get(st))
    states=jax.tree.map(lambda *a:np.concatenate(a),*initial)
    acts=np.zeros((maxlen,n,2,2),np.int32);ek=np.zeros((maxlen,n,2),np.uint32)
    for i,x in enumerate(items):acts[:lengths[i],i]=x['prefix_actions'];ek[:lengths[i],i]=x['prefix_envkeys']
    import jax.numpy as jnp
    def replay(st,a,k,length):
        def body(s,data):
            a,k,t=data;nxt,_,_=jax.vmap(lambda s,a,k:E.env.step(s,a,k,auto_reset=False,return_info=True))(s,a,k)
            return E.select_tree(t<length,nxt,s),None
        return jax.lax.scan(body,st,(a,k,jnp.arange(a.shape[0])))[0]
    restored=jax.jit(replay)(states,acts,ek,lengths)
    from scripts.terminal_checkpoint_restore import digest
    for i,x in enumerate(items):
        assert digest(jax.tree.map(lambda a:a[i:i+1],restored))==digest(x['states']),'actual inbound walk replay mismatch'
    atomic_json(B/'PARITY_COMPLETE.json',dict(passed=True,states=n,fresh_process_next_step_exact=n,acquisition_aux_reward_exact=n,whole_inbound_replay_exact=n,bank_sha256=sha(B/'BANK_FROZEN.json'),reference_sha256=sha(B/'parity_reference.pkl'),finite_task_timeout=2400,residual_settlement=True,bootstrap_at_terminal=0,rollout_cutoff_is_truncation=True,unix=time.time()))
    notify(f'A parity实测通过：{n}状态完整入站重放、取包env/aux/reward及fresh-process下一步hash全部一致。')


def shard(x,devices):
    import jax
    return jax.tree.map(lambda a:np.asarray(a).reshape((devices,-1)+np.asarray(a).shape[1:]),x)


def expand(items,repeats,devices):
    import jax
    selected=[x for x in items for _ in range(repeats)]
    actual=len(selected);padded=((actual+devices-1)//devices)*devices
    selected+=selected[:1]*(padded-actual)
    st=merged(selected,'states');ax=merged(selected,'aux')
    # Same state/repetition stream across tested policies and action branches.
    keys=np.array([np.asarray(jax.random.fold_in(jax.random.PRNGKey(2026102317),i)) for i in range(padded)])
    return selected,actual,shard(st,devices),shard(ax,devices),shard(keys,devices)


def branch_moves(states,aux,E,jax):
    import jax.numpy as jnp
    from scripts.je10_route_oracle import continuous_distance,DELTA
    def directions(st,ax):
        _,_,mm,_=E.inputs(st);d=ax['dagger']['distance'][:,4];blocked=st.wall|st.brick|st.core.pushable
        here=continuous_distance(st.pos[:,0],d,blocked)
        there=continuous_distance((st.pos[:,0,None]+.18*DELTA).reshape(-1,2),jnp.repeat(d,5,0),jnp.repeat(blocked,5,0)).reshape(-1,5)
        gains=here[:,None]-there
        progress=mm&(gains>1e-5);regress=mm&(gains< -1e-5)
        f=jnp.argmax(jnp.where(progress,gains,-1e9),-1);b=jnp.argmin(jnp.where(regress,gains,1e9),-1)
        return jnp.stack([jnp.where(progress.any(-1),f,-1),jnp.where(regress.any(-1),b,-1),jnp.where(mm[:,4],4,-1)],-1),here
    return jax.device_get(jax.pmap(directions)(states,aux))


def run_paths(params,states,aux,keys,E,jax,first=None,horizon=2528,progress=None):
    import jax.numpy as jnp
    devices=jax.local_device_count();rep=jax.tree.map(lambda a:jnp.stack([jnp.asarray(a)]*devices),params)
    carry=(states,aux,keys,jnp.zeros(states.t.shape,bool));datas=[];tails={};steps=0;chunk=128
    normal=jax.pmap(lambda p,c:E.rollout(p,c[0],c[1],c[2],chunk,ended=c[3]))
    branch=jax.pmap(lambda p,c,f:E.rollout(p,c[0],c[1],c[2],chunk,first_moves=f,ended=c[3]))
    pred=jax.pmap(lambda p,s:E.prediction(p,s)[2])
    for ci in range((horizon+chunk-1)//chunk):
        if ci==0 and first is not None:carry,data=branch(rep,carry,first)
        else:carry,data=normal(rep,carry)
        jax.block_until_ready(carry[0].t);host=tuple(np.array(a,copy=True) for a in jax.device_get(data))
        # (device,time,per-device,...) -> (time,flat,...)
        host=tuple(np.swapaxes(a,0,1).reshape((a.shape[1],-1)+a.shape[3:]) for a in host)
        datas.append(host);steps+=chunk
        if steps in [256,512,1024]:tails[steps]=np.asarray(pred(rep,carry[0])).reshape(-1)
        if progress:atomic_json(progress,dict(stage='AB_ROLLOUT',steps=steps,terminal=int(np.asarray(carry[3]).sum()),total=int(carry[3].size),unix=time.time()))
        if np.asarray(carry[3]).all():break
    data=tuple(np.concatenate([x[i] for x in datas])[:horizon] for i in range(len(datas[0])))
    return data,tails,jax.tree.map(lambda a:np.array(a,copy=True),jax.device_get(carry))


def summarize_paths(data,items,repeats,actual,gamma=.995):
    r,v,term,flags,dist,acts,lp,valid,parts=[a[:,:actual] for a in data]
    done=term.any(0);ret=discounted_returns(r.T,term.T,gamma);n=len(items)
    delivery=(flags[:,:,0]&valid).any(0);death=(flags[:,:,1]&valid).any(0);drop=(flags[:,:,2]&valid).any(0)
    tick=np.arange(len(flags))[:,None];first_delivery=np.where(flags[:,:,0]&valid,tick,10**9).min(0)
    first_failure=np.where((flags[:,:,1]|flags[:,:,2])&valid,tick,10**9).min(0)
    direct=delivery&(first_delivery<first_failure)
    recovery=delivery&(first_failure<first_delivery)
    clusters=np.repeat([x['record']['source_episode'] for x in items],repeats)
    first=v[0].reshape(n,repeats).mean(-1);mc=ret.reshape(n,repeats)
    complete=done.reshape(n,repeats).all(-1)
    cal=calibration(first[complete],mc[complete]) if complete.any() else None
    if cal:cal['source_episode_cluster_CI']=calibration_intervals(first[complete],mc[complete])
    bias_ci=cluster_interval((v[0]-ret)[done],clusters[done]) if done.any() else [None,None]
    # Every source state is one cluster, including death/drop failures.
    summary=dict(states=n,rollouts=actual,terminal_MC_rollouts=int(done.sum()),censored_rollouts=int((~done).sum()),calibration_terminal_complete_states=cal,bias_source_cluster_CI=bias_ci,
        direct_custody_delivery=float(direct.mean()),direct_custody_delivery_CI=cluster_interval(direct,clusters),direct_safe_settled_delivery=float((direct&~death).mean()),eventual_delivery_after_death_or_drop=float(recovery.mean()),delivery_estimand='any eventual enemy-bun delivery in same full finite task; direct initial-custody success separate',delivery=float(delivery.mean()),delivery_cluster_CI=cluster_interval(delivery,clusters),death=float(death.mean()),drop=float(drop.mean()),
        mean_terminal_MC=float(ret[done].mean()) if done.any() else None,mean_partial_return_censored=float(ret[~done].mean()) if (~done).any() else None,
        valid_environment_steps=int(valid.sum()),positive_return_variance=float(ret[done].var()) if done.any() else None)
    for source in sorted({x['record']['source_policy'] for x in items}):
        ix=np.repeat([x['record']['source_policy']==source for x in items],repeats)
        summary.setdefault('by_source',{})[source]=dict(states=int(ix.sum()/repeats),delivery=float(delivery[ix].mean()),death=float(death[ix].mean()))
    return summary,dict(returns=ret,terminal=done,delivery=delivery,direct_custody_delivery=direct,recovery_delivery=recovery,death=death,drop=drop,v=v[0],clusters=clusters)


def analyze_one():
    jax,E=setup();assert read(B/'PARITY_COMPLETE.json')['passed'];items=bank();protocol=read(B/'PROTOCOL.json');repeats=protocol['AB']['repeats'];devices=jax.local_device_count()
    selected,actual,states,aux,keys=expand(items,repeats,devices);directions,dist=branch_moves(states,aux,E,jax)
    rows=[]
    for i,x in enumerate(items):
        st=x['states'];r=x['record'];d=float(np.asarray(dist).reshape(-1)[i*repeats]);rows.append(dict(source=r['source_episode'],split=r['split'],distance=d,distance_stratum='near' if d<=8 else 'mid' if d<=16 else 'far',enemy_free=bool(st.alive[0,1] and st.trapped[0,1]==0),enemy_dead=bool(not st.alive[0,1]),enemy_native_trap=int(st.trapped[0,1]),live_fuses=int((st.fuse>0).sum()),topology_no_route_flag=d>=32767))
    atomic_json(B/'BANK_STRATA.json',dict(states=rows,coverage_limitations=['map806 only','stock death/native-control only; drop/transfer separate and not represented by this Bank'],unexpected_unreachable=sum(r['topology_no_route_flag'] for r in rows)))
    allresults={}
    for model in [os.environ['CARRIER_AB_MODEL']]:
        assert model in protocol['AB']['models']
        p=pickle.load((B/'inputs'/f'{model}.pkl').open('rb'))['params'];models={}
        for branch in ['natural','forward','backward','wait']:
            target=B/'AB'/model/f'{branch}.pkl'
            if target.exists():result=pickle.load(target.open('rb'))
            else:
                first=None if branch=='natural' else directions[:,:,['forward','backward','wait'].index(branch)]
                data,tails,end=run_paths(p,states,aux,keys,E,jax,first,horizon=protocol['AB']['full_horizon'],progress=B/'AB_PROGRESS.json')
                summary,arrays=summarize_paths(data,items,repeats,actual)
                r,v,done,flags,d,act,lp,valid,parts=[a[:,:actual] for a in data]
                tailv=np.asarray(jax.pmap(lambda p,s:E.prediction(p,s)[2])(jax.tree.map(lambda a:np.stack([a]*devices),p),end[0])).reshape(-1)[:actual]
                # Exact training GAE with tested critic, explicitly a separate estimator.
                nv=np.concatenate([v[1:],tailv[None]])
                training_steps=int(read(R/protocol['models'][model]['plan'])['steps'])
                gae=np.concatenate([np.asarray(E.train.compute_gae(r[t:t+training_steps],v[t:t+training_steps],nv[t:t+training_steps],done[t:t+training_steps],.995,.95)) for t in range(0,len(r),training_steps)])
                summary['GAE_identity']=dict(implementation='jax_train.compute_gae',rollout_steps=training_steps,gamma=.995,lambda_gae=.95,values='tested Critic, actual sampled rewards and terminal masks',scope='restored Bank starts a fresh production-length rollout boundary; reconstructed GAE, not a historical saved optimizer batch')
                cuts={}
                for h,tv in tails.items():
                    stop=min(h,len(r));cutr=discounted_returns(r[:stop].T,done[:stop].T);bt=discounted_returns(r[:stop].T,done[:stop].T,tail=tv[:actual])
                    cuts[str(h)]=dict(mean_partial=float(cutr.mean()),mean_self_V_bootstrapped=float(bt.mean()),mean_full_terminal_MC=float(arrays['returns'][arrays['terminal']].mean()) if arrays['terminal'].any() else None,mean_self_bootstrap_error_vs_completed_tail=float((bt-arrays['returns'])[arrays['terminal']].mean()) if arrays['terminal'].any() else None,warning='self-V bootstrap is descriptive, never an independent calibration target')
                summary['truncation_sensitivity']=cuts
                active_first=np.ones(actual,bool) if first is None else first.reshape(-1)[:actual]>=0
                summary['branch_legal_coverage']=int(active_first.sum());summary['excluded_no_direction']=int((~active_first).sum())
                if first is not None:
                    summary['legal_branch_denominator']=dict(rollouts=int(active_first.sum()),states=int(active_first.sum()/repeats),delivery=float(arrays['delivery'][active_first].mean()) if active_first.any() else None,death=float(arrays['death'][active_first].mean()) if active_first.any() else None,drop=float(arrays['drop'][active_first].mean()) if active_first.any() else None,unavailable_direction_excluded=int((~active_first).sum()),all_legal_failures_retained=True)
                firstbad=[]
                for i,x in enumerate(items):
                    for k in range(repeats):
                        z=i*repeats+k;back=np.flatnonzero((np.diff(d[:,z])>1e-5)&valid[:-1,z]&flags[:-1,z,3])
                        if len(back):
                            t=int(back[0]);firstbad.append(dict(source=x['record']['source_episode'],repeat=k,tick=t,action=act[t,z].tolist(),distance_before=float(d[t,z]),distance_after=float(d[t+1,z]),reward=float(r[t,z]),components=parts[t,z].tolist(),value=float(v[t,z]),GAE=float(gae[t,z]),died_later=bool(arrays['death'][z]),dropped_later=bool(arrays['drop'][z])))
                result=dict(summary=summary,arrays=arrays,first_gae=gae[0],first_actions=act[0],first_ability=act[0,:,1],first_legal=active_first,first_bad=firstbad,trajectory=dict(rewards=r,values=v,terminal=done,flags=flags,distance=d,actions=act,logp=lp,valid=valid,reward_components=parts,GAE=gae),states=rows)
                save(target,result);atomic_json(target.with_suffix('.json'),summary)
                notify(f'A/B实测 {model}/{branch}：terminal {summary["terminal_MC_rollouts"]}/{actual}，交付{summary["delivery"]:.3f}，死亡{summary["death"]:.3f}；V校准见独立MC文件。')
            models[branch]=result
            if branch=='natural':
                cal=result['summary']['calibration_terminal_complete_states']
                if cal:notify(f'首批V实数 {model}: EV={cal["explained_variance"]}, RMSE={cal["rmse"]:.4f}, bias={cal["bias"]:.4f}, MC均值={cal["mean_return"]:.4f}；仅完整终局样本，非自举。')
        nat=models['natural']
        capture_file=B/'AB'/model/'REAL_FIRST_DEVIATION_BANK.pkl'
        if not capture_file.exists():
            targets=np.full(int(states.t.size),-1,np.int32)
            for row in nat['first_bad']:
                si=next(i for i,x in enumerate(items) if x['record']['source_episode']==row['source'])
                targets[si*repeats+row['repeat']]=row['tick']
            snapshots=capture_deviations(p,states,aux,keys,targets.reshape(states.t.shape),E,jax)
            save(capture_file,dict(targets=targets,full_causal_snapshots=snapshots,rows=nat['first_bad'],includes_death_drop=True,teleported=False))
        contrasts={};base=models['forward']
        for b in ['backward','wait']:
            q=models[b];eligible=base['first_legal']&q['first_legal']&base['arrays']['terminal']&q['arrays']['terminal'];assert np.array_equal(base['first_ability'],q['first_ability'])
            delta=base['arrays']['returns']-q['arrays']['returns'];clusters=base['arrays']['clusters']
            contrasts['forward_minus_'+b]=dict(estimand='subset Q contrasts',paired_rollouts=int(eligible.sum()),mean=float(delta[eligible].mean()) if eligible.any() else None,source_episode_cluster_CI=cluster_interval(delta[eligible],clusters[eligible]),GAE_first_delta_mean=float((base['first_gae']-q['first_gae'])[eligible].mean()) if eligible.any() else None,ability_identical=True,future_random_keys_identical=True)
        atomic_json(B/'AB'/model/'Q_CONTRASTS.json',contrasts)
        atomic_json(B/'AB'/model/'BANK_P.json',dict(executed_action_perturbations=True,teleported=False,selection='all fixed backward and wait branches; no survivor filtering',branches={b:dict(summary=models[b]['summary'],first_deviations=len(models[b]['first_bad'])) for b in ['backward','wait']},natural_first_deviations=nat['first_bad']))
        allresults[model]=dict(V=nat['summary'],Q=contrasts)
        notify(f'B {model} subset Q contrasts实数：'+json.dumps(contrasts,ensure_ascii=False))
    atomic_json(B/f'AB_MODEL_{model}_COMPLETE.json',dict(passed=True,models=allresults,states=len(items),rollouts_per_state=repeats,holdout_used_for_C_tuning=False,unix=time.time()))


def train():
    # Fresh processes bound host-registration and compilation resources.
    # No state, seed, objective, optimizer or sample budget changes.
    stages=[('accounting','AB_CUSTODY_ACCOUNTING.json'),('verified-deviations','B_VERIFIED_DEVIATIONS_COMPLETE.json'),('smoke','C_TECHNICAL_SMOKE.json'),('formal','C_TRAINING_COMPLETE.json'),('native-eval','C_NATIVE_BANK_COMPLETE.json')]
    for stage,marker in stages:
        if (B/marker).exists():continue
        authorize()
        if stage=='smoke':
            source_files=['scripts/carrier_mechanism_engine.py','scripts/carrier_mechanism_train.py','scripts/carrier_mechanism_analysis.py','scripts/carrier_mechanism_stats.py','scripts/carrier_mechanism_bank_eval.py']
            import shutil
            for f in source_files:
                target=B/'C_execution_sources'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/f,target)
            atomic_json(B/'C_EXECUTION_SOURCE.json',dict(sources={f:sha(R/f) for f in source_files},protocol_sha256=sha(B/'PROTOCOL.json'),AB_measurement_complete=True,holdout_not_used_to_select_C_recipe=True,old_continuous_resume=False,isolated_substage_processes=True,unix=time.time()))
        atomic_json(B/'C_SUBSTAGE.json',dict(stage=stage,state='RUNNING',unix=time.time()))
        code=subprocess.run([sys.executable,'-m','scripts.carrier_mechanism_analysis',stage],env=dict(os.environ),stdout=(B/f'{stage}.log').open('a'),stderr=subprocess.STDOUT).returncode
        if code or not (B/marker).exists():raise RuntimeError(f'isolated substage {stage} failed: {code}')
    atomic_json(B/'C_COMPLETE.json',dict(passed=True,gpu_complete=True,science_evaluation_complete=False,old_continuous_resume=False,unix=time.time()))

if __name__=='__main__':
    if sys.argv[1]=='parity-check':
        jax,E=setup();save(B/'parity_fresh.pkl',parity_reference(bank(),E,jax))


def capture_deviations(params,states,aux,keys,targets,E,jax):
    """Replay actual stochastic policy to first measured reversal; retain full causal state."""
    import jax.numpy as jnp
    rep=jax.tree.map(lambda a:np.stack([a]*jax.local_device_count()),params)
    def capture(p,st,ax,key,targets):
        n=st.t.size
        carry=(st,ax,key,jnp.zeros(n,bool),st,st,ax,ax,key,key,jnp.zeros((n,2,2),jnp.int32),key,jnp.zeros(n,bool))
        def body(c,t):
            st,ax,key,ended,pre,post,pa,po,pk,nk0,act0,ke0,got=c
            nk,ke,act,_=E.choose(p,st,ax,key);after,na,r,done,info,parts=E.transition(st,ax,act,ke)
            hit=(t==targets)&~ended
            pre=E.select_tree(hit,st,pre);post=E.select_tree(hit,after,post);pa=E.select_tree(hit,ax,pa);po=E.select_tree(hit,na,po);pk=E.select_tree(hit,key,pk);nk0=E.select_tree(hit,nk,nk0);act0=E.select_tree(hit,act,act0);ke0=E.select_tree(hit,ke,ke0)
            return (E.select_tree(ended,st,after),E.select_tree(ended,ax,na),E.select_tree(ended,key,nk),ended|done,pre,post,pa,po,pk,nk0,act0,ke0,got|hit),None
        c,_=jax.lax.scan(body,carry,jnp.arange(intmax))
        return c[4:]
    intmax=int(np.asarray(targets).max())+1
    if intmax<=0:return None
    return jax.device_get(jax.pmap(capture)(rep,states,aux,keys,targets))


def postprocess_AB_custody():
    marker=B/'AB_CUSTODY_ACCOUNTING.json'
    if marker.exists():return
    protocol=read(B/'PROTOCOL.json');complete=read(B/'AB_COMPLETE.json')
    jax,E=setup();import jax.numpy as jnp
    _,actual,st,ax,keys=expand(bank(),8,8)
    def first_logps(p,st):
        mv,ab,_,_,o,g,mm,bm=E.prediction(p,st)
        return jax.nn.log_softmax(jnp.where(mm,mv,-1e9)),jax.nn.log_softmax(jnp.where(bm,ab,-1e9))
    logits=jax.pmap(first_logps)
    for model in protocol['AB']['models']:
        params=pickle.load((B/'inputs'/f'{model}.pkl').open('rb'))['params'];rep=jax.tree.map(lambda a:jnp.stack([a]*8),params)
        lm,lb=jax.device_get(logits(rep,st));lm=lm.reshape(-1,5)[:actual];lb=lb.reshape(-1,3)[:actual]
        for branch in ['natural','forward','backward','wait']:
            p=B/'AB'/model/f'{branch}.pkl';r=pickle.load(p.open('rb'));t=r['trajectory'];flags=t['flags'];valid=t['valid'];n=flags.shape[1]
            executed=r['first_actions'];correct=lm[np.arange(actual),executed[:,0]]+lb[np.arange(actual),executed[:,1]]
            t['sampled_action_logp_before_intervention']=t['logp'][0].copy();t['logp'][0]=correct
            r['summary']['first_logp_identity']='executed action policy log-probability; original pre-intervention sample logp separate, never PPO training'
            tick=np.arange(len(flags))[:,None];fd=np.where(flags[:,:,0]&valid,tick,10**9).min(0);ff=np.where((flags[:,:,1]|flags[:,:,2])&valid,tick,10**9).min(0)
            direct=(fd<ff);recovery=(ff<fd)&(fd<10**9);death=r['arrays']['death']
            r['arrays']['direct_custody_delivery']=direct;r['arrays']['recovery_delivery']=recovery
            r['summary'].update(direct_custody_delivery=float(direct.mean()),direct_custody_delivery_CI=cluster_interval(direct,r['arrays']['clusters']),direct_safe_settled_delivery=float((direct&~death).mean()),eventual_delivery_after_death_or_drop=float(recovery.mean()),delivery_estimand='eventual enemy-bun delivery; direct initial-custody success separate')
            if branch in ['backward','wait']:
                outcomes=[event_outcomes_from(flags[:,z],valid[:,z]) for z in range(n)]
                survivors=np.array([not(x['immediate_death'] or x['immediate_drop']) for x in outcomes])
                r['summary']['executed_BankP']=dict(perturbation_ticks=1,total_rollouts=n,
                    immediate_deaths=sum(x['immediate_death'] for x in outcomes),immediate_drops=sum(x['immediate_drop'] for x in outcomes),
                    survived_initial_transition=int(survivors.sum()),all_failures_retained=True,
                    survivor_conditioned_eventual_delivery=float(r['arrays']['delivery'][survivors].mean()) if survivors.any() else None,
                    survivor_conditioning='alive and no drop on intervention transition only; unconditional outcomes above include all failures')
            items=bank();initial=r['arrays']['v'].reshape(len(items),8).mean(-1);returns=r['arrays']['returns'].reshape(len(items),8)
            strata=read(B/'BANK_STRATA.json')['states']
            groups={}
            for kind,labels in [('layout',[x['record']['layout'] for x in items]),('source_policy',[x['record']['source_policy'] for x in items]),('post_pickup_threat',['dead' if x['enemy_dead'] else 'native_trap' if x['enemy_native_trap']>0 else 'free' for x in strata]),('remaining_finite_task_ticks',['lt128' if 2400-int(x['states'].t[0])<128 else '128to512' if 2400-int(x['states'].t[0])<=512 else 'gt512' for x in items])]:
                for label in sorted(set(labels)):
                    ix=np.array([v==label for v in labels])&r['arrays']['terminal'].reshape(len(items),8).all(-1);cal=calibration(initial[ix],returns[ix]) if ix.any() else dict(states=0,all_censored=True);cal['source_episode_CI']=calibration_intervals(initial[ix],returns[ix]);groups[kind+':'+label]=cal
            r['summary']['calibration_strata']=groups
            source_index={x['record']['source_episode']:i for i,x in enumerate(items)}
            for row in r['first_bad']:
                z=source_index[row['source']]*8+row['repeat']
                row.update(event_outcomes_from(flags[:,z],valid[:,z],row['tick']))
                row['qualification']='raw topology regression only; use VERIFIED_FIRST_DEVIATIONS for safe-route failures'
                row['outcome_time_scope']='inclusive regression transition onward only'
            save(p,r);atomic_json(p.with_suffix('.json'),r['summary'])
            if branch=='natural':complete['models'][model]['V']=r['summary']
        bp_path=B/'AB'/model/'BANK_P.json';bp=read(bp_path)
        for branch in ['backward','wait']:
            bp['branches'][branch]['summary']=read(B/'AB'/model/f'{branch}.json')
        natural=pickle.load((B/'AB'/model/'natural.pkl').open('rb'))
        bp['natural_first_deviations']=natural['first_bad'];atomic_json(bp_path,bp)
        capture_path=B/'AB'/model/'REAL_FIRST_DEVIATION_BANK.pkl'
        capture=pickle.load(capture_path.open('rb'));capture['rows']=natural['first_bad'];save(capture_path,capture)
    atomic_json(B/'AB_COMPLETE.json',complete);atomic_json(marker,dict(passed=True,all_deaths_drops_retained=True,eventual_delivery_not_initial_custody_success=True,postprocessing_existing_full_trajectories_only=True))


def verify_first_deviation_banks():
    """Native safe-set qualification: a combat escape is not a route failure."""
    if (B/'B_VERIFIED_DEVIATIONS_COMPLETE.json').exists():return
    jax,E=setup();import jax.numpy as jnp
    from scripts.terminal_checkpoint_restore import digest
    items=bank();selected,actual,states,aux,keys=expand(items,8,8)
    def collect(p,c):
        def body(c,_):
            st,ax,key,ended,got,pre,post,pa,po,pk,nk0,act0,ke0,sets,tick,snap_tick=c
            nk,ke,act,data=E.choose(p,st,ax,key);teacher=E.safe_teacher(st,ax)
            move_set=teacher.any(-1);qualified=move_set.any(-1)
            mismatch=~move_set[jnp.arange(st.t.size),act[:,0,0]]
            after,na,r,done,info,parts=E.transition(st,ax,act,ke)
            hit=qualified&mismatch&~ended&~got
            pre=E.select_tree(hit,st,pre);post=E.select_tree(hit,after,post);pa=E.select_tree(hit,ax,pa);po=E.select_tree(hit,na,po);pk=E.select_tree(hit,key,pk);nk0=E.select_tree(hit,nk,nk0);act0=E.select_tree(hit,act,act0);ke0=E.select_tree(hit,ke,ke0);sets=E.select_tree(hit,teacher,sets);snap_tick=jnp.where(hit,tick,snap_tick)
            stop=ended|got|hit
            return (E.select_tree(stop,st,after),E.select_tree(stop,ax,na),E.select_tree(stop,key,nk),ended|done,got|hit,pre,post,pa,po,pk,nk0,act0,ke0,sets,tick+1,snap_tick),None
        return jax.lax.scan(body,c,None,length=128)[0]
    step=jax.pmap(collect)
    counts={}
    for model in read(B/'PROTOCOL.json')['AB']['models']:
        target=B/'AB'/model/'VERIFIED_FIRST_DEVIATIONS.pkl'
        if target.exists():counts[model]=read(target.with_suffix('.json'));continue
        p=pickle.load((B/'inputs'/f'{model}.pkl').open('rb'))['params'];rep=jax.tree.map(lambda a:jnp.stack([a]*8),p)
        zero=jnp.zeros(states.t.shape,bool)
        carry=(states,aux,keys,zero,zero,states,states,aux,aux,keys,keys,jnp.zeros(states.t.shape+(2,2),jnp.int32),keys,jnp.zeros(states.t.shape+(5,3),bool),jnp.zeros(states.t.shape,jnp.int32),jnp.full(states.t.shape,-1,jnp.int32))
        for ci in range(20):
            carry=step(rep,carry);jax.block_until_ready(carry[0].t)
            atomic_json(B/'B_VERIFIED_DEVIATION_STATUS.json',dict(model=model,ticks=(ci+1)*128,verified_first_route_deviations=int(np.asarray(carry[4]).sum()),ended_without_deviation=int(np.asarray(carry[3]&~carry[4]).sum()),unix=time.time()))
            if np.asarray(carry[3]|carry[4]).all():break
        host=jax.device_get(carry);flatten=lambda tr:jax.tree.map(lambda a:np.asarray(a).reshape((-1,)+a.shape[2:]),tr)
        h=tuple(map(flatten,host));rows=[];natural=pickle.load((B/'AB'/model/'natural.pkl').open('rb'));trajectory=natural['trajectory']
        for z in np.flatnonzero(h[4][:actual]):
            t=int(h[-1][z]);take=lambda tr:jax.tree.map(lambda a:a[z:z+1],tr)
            before=take(h[5]);after=take(h[6]);beforeaux=take(h[7]);afteraux=take(h[8])
            np.testing.assert_array_equal(h[11][z,0],trajectory['actions'][t,z])
            row=dict(source=selected[z]['record']['source_episode'],repeat=int(z%8),tick=t,state_hash=digest(before),next_state_hash=digest(after),action=h[11][z].tolist(),accepted_joint_actions=h[13][z].tolist(),phase=int(beforeaux['route_pbrs']['phase'][0,0]),target_identity=int(beforeaux['route_pbrs']['identity'][0,0]),phi_before=float(beforeaux['route_pbrs']['phi'][0,0]),phi_after=float(afteraux['route_pbrs']['phi'][0,0]),reward=float(trajectory['rewards'][t,z]),reward_component_names=list(E.ops.COMPONENTS),reward_components=trajectory['reward_components'][t,z].tolist(),GAE=float(trajectory['GAE'][t,z]),value=float(trajectory['values'][t,z]),distance_before=float(trajectory['distance'][t,z]),died_later=bool(natural['arrays']['death'][z]),dropped_later=bool(natural['arrays']['drop'][z]),eventual_delivery=bool(natural['arrays']['delivery'][z]))
            row.update(event_outcomes_from(trajectory['flags'][:,z],trajectory['valid'][:,z],t))
            row.update(absolute_tick=int(before.t[0]),source_acquisition_event=dict(source_episode=selected[z]['record']['source_episode'],event_tick=selected[z]['record']['event_tick'],native_control='death' if not selected[z]['record']['enemy_alive'] else 'native_trap'),outcome_time_scope='inclusive deviation transition onward only')
            rows.append(row)
        save(target,dict(rows=rows,full_causal_snapshots=h,includes_all_death_drop=True,teleported=False,qualification='first actual chosen movement outside nonempty native-safe set, current return phase gate; combat switches excluded'))
        summary=dict(passed=True,rollouts=actual,first_qualified_deviations=len(rows),raw_first_topology_regressions=len(natural['first_bad']),failed_in_bankP=sum(r['died_later'] or r['dropped_later'] for r in rows),positive_GAE_first_deviations=sum(r['GAE']>0 for r in rows),source_clusters=len({r['source'] for r in rows}),sha256=sha(target))
        atomic_json(target.with_suffix('.json'),summary);counts[model]=summary
        notify(f'B首偏航原生资格验收 {model}: {len(rows)}/{actual}条轨迹出现安全集合外首方向；死亡/掉包全保留，战斗避险未冒充路线失败。')
    atomic_json(B/'B_VERIFIED_DEVIATIONS_COMPLETE.json',dict(passed=True,models=counts,unix=time.time()))


def analyze():
    if os.environ.get('CARRIER_AB_MODEL'):
        analyze_one();return
    # Bound CUDA registration/compilation lifetime to a single fixed checkpoint.
    # The driver itself does not initialize JAX or take a GPU context.
    authorize();protocol=read(B/'PROTOCOL.json');results={}
    for model in protocol['AB']['models']:
        marker=B/f'AB_MODEL_{model}_COMPLETE.json'
        if not marker.exists():
            env=dict(os.environ,CARRIER_AB_MODEL=model)
            code=subprocess.run([sys.executable,'scripts/carrier_mechanism_run.py','AB'],env=env,stdout=(B/f'AB_{model}.log').open('a'),stderr=subprocess.STDOUT).returncode
            if code or not marker.exists():raise RuntimeError(f'isolated AB checkpoint {model} failed: {code}')
        result=read(marker);assert result['passed'];results.update(result['models'])
    atomic_json(B/'AB_COMPLETE.json',dict(passed=True,models=results,states=read(B/'BANK_FROZEN.json')['states'],rollouts_per_state=8,isolated_process_per_checkpoint=True,failed_partial_branches_not_used=True,holdout_used_for_C_tuning=False,unix=time.time()))


if __name__=='__main__' and sys.argv[1]!='parity-check':
    stage=sys.argv[1]
    if stage=='accounting':postprocess_AB_custody()
    elif stage=='verified-deviations':verify_first_deviation_banks()
    elif stage=='smoke':
        from scripts.carrier_mechanism_train import technical_smoke
        technical_smoke()
    elif stage=='formal':
        from scripts.carrier_mechanism_train import run
        run()
    elif stage=='native-eval':
        from scripts.carrier_mechanism_bank_eval import run
        run()
    else:raise ValueError('unknown diagnostic substage')
