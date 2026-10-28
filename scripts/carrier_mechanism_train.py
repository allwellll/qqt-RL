"""Bounded three-arm learning, exact common-parent Adam and learner-state labels."""
import os,time,pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from scripts.carrier_mechanism_run import B,R,read,save,notify,authorize
from scripts.carrier_mechanism_analysis import bank,merged,shard,setup,expand,run_paths,summarize_paths
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest


def objective(params,batch,arm,forward,value_loss):
    o,g,actions,old,ad,ret,mm,bm,teacher=batch
    move,ability,_,vl=forward(params,'transformer',o,g)
    lm=jax.nn.log_softmax(jnp.where(mm,move,-1e9));lb=jax.nn.log_softmax(jnp.where(bm,ability,-1e9))
    logp=lm[jnp.arange(len(old)),actions[:,0]]+lb[jnp.arange(len(old)),actions[:,1]]
    ratio=jnp.exp(logp-old);ppo=jnp.maximum(-ad*ratio,-ad*jnp.clip(ratio,.9,1.1)).mean()
    value=value_loss(vl,ret);entropy=-(jnp.sum(jnp.exp(lm)*lm,-1)+jnp.sum(jnp.exp(lb)*lb,-1)).mean()
    joint=lm[:,:,None]+lb[:,None,:];eligible=teacher.any((1,2))
    setmass=jax.nn.logsumexp(jnp.where(teacher,joint,-1e9),(1,2))
    supervised=jnp.where(eligible,-setmass,0).sum()/jnp.maximum(1,eligible.sum())
    loss=(ppo+.5*value-.005*entropy if arm!='DAgger-only' else 0.)+(.2*supervised if arm!='PPO-only' else 0.)
    return loss,jnp.array([ppo,value,entropy,supervised,eligible.mean(),jnp.mean((ratio-1)-(logp-old))])


def preserve_critic(adam,previous):
    # Adam momentum would move an unused Critic even with zero gradients.
    state=adam[1][0];old=previous[1][0]
    state=state._replace(mu=dict(state.mu,critic=old.mu['critic']),nu=dict(state.nu,critic=old.nu['critic']))
    return (adam[0],(state,adam[1][1]))


def collector(E,steps=128):
    def collect(params,states,aux,keys,bankstates,bankaux):
        def body(c,_):
            st,ax,key=c
            nk,ke,actions,data=E.choose(params,st,ax,key)
            o,g,mm,bm,own,lp,v=data;teacher=E.safe_teacher(st,ax)
            after,na,r,done,info,parts=E.transition(st,ax,actions,ke)
            # Real on-policy successor value is captured before a terminal reset.
            # True finite-task terminal has zero bootstrap; no phase pseudo-terminal.
            nval=E.prediction(params,after)[2]
            rkeys=jax.vmap(lambda k:jax.random.fold_in(k,0xCA))(nk)
            choice=jax.vmap(lambda k:jax.random.randint(k,(),0,bankstates.t.shape[0]))(rkeys)
            fresh=jax.tree.map(lambda a:a[choice],bankstates);fa=jax.tree.map(lambda a:a[choice],bankaux)
            after=E.select_tree(done,fresh,after);na=E.select_tree(done,fa,na)
            custody=(st.bun_carried[:,0]==1)&ax['dagger']['custody'][:,0]&st.alive[:,0]
            distance=E.aux_distance(st,ax)
            approved=teacher[jnp.arange(st.t.size),own[:,0],own[:,1]]
            phase=jnp.stack([custody,distance<=8,(distance>8)&(distance<=16),distance>16,approved,info['capture'][:,0],info['death'][:,0],info['drop'][:,0]|info['carrier_death'][:,0],info['own_bomb_defeat'][:,0]],-1)
            return (after,na,nk),(o,g,own,lp,v,nval,r[:,0],done,mm,bm,teacher,parts[:,0],phase)
        end,data=jax.lax.scan(body,(states,aux,keys),None,length=steps)
        o,g,act,lp,v,nv,r,done,mm,bm,teacher,parts,phase=data
        adv=E.train.compute_gae(r,v,nv,done,.995,.95)
        return end,(o,g,act,lp,adv,adv+v,mm,bm,teacher),(r,done,parts,v,adv,phase)
    return collect


def make_update(E,opt,arm):
    def update(params,adam,batch,key):
        # Four local envs *128 =512 rows, 32/device minibatch, 16 global minibatches.
        flatten=lambda a:a.reshape((-1,)+a.shape[2:])
        batch=tuple(map(flatten,batch));ad=batch[4]
        mean=jax.lax.pmean(ad.mean(),'device');sq=jax.lax.pmean(jnp.mean(ad**2),'device')
        ad=(ad-mean)/jnp.sqrt(jnp.maximum(sq-mean**2,0)+1e-8)
        batch=batch[:4]+(ad,)+batch[5:]
        ii=jax.random.permutation(key,len(ad)).reshape(-1,32)
        def body(c,index):
            p,a=c;mb=tuple(x[index] for x in batch)
            (loss,metrics),grad=jax.value_and_grad(lambda q:objective(q,mb,arm,E.train.net_forward,E.train.hl_gauss_value_loss),has_aux=True)(p)
            grad=jax.lax.pmean(grad,'device');loss=jax.lax.pmean(loss,'device');metrics=jax.lax.pmean(metrics,'device')
            changes,na=opt.update(grad,a,p);np_=optax.apply_updates(p,changes)
            if arm=='DAgger-only':np_=dict(np_,critic=p['critic']);na=preserve_critic(na,a)
            gn=optax.global_norm(grad)
            return (np_,na),jnp.concatenate([loss[None],metrics,gn[None]])
        return jax.lax.scan(body,(params,adam),ii)
    return jax.pmap(update,axis_name='device')


def evaluate_saved(cp,items,E,jax,split,label):
    out=cp/f'{label}_{split}.json'
    if out.exists():return read(out)
    chosen=[x for x in items if x['record']['split']==split]
    selected,actual,st,ax,keys=expand(chosen,8,jax.local_device_count())
    params=pickle.load((cp/'state.pkl').open('rb'))['params']
    data,_,_=run_paths(params,st,ax,keys,E,jax,horizon=2528,progress=B/'C_EVAL_PROGRESS.json')
    summary,arrays=summarize_paths(data,chosen,8,actual)
    save(cp/f'{label}_{split}.pkl',arrays);atomic_json(out,summary)
    return summary


def run():
    jax,E=setup();assert read(B/'AB_COMPLETE.json')['passed'];protocol=read(B/'PROTOCOL.json')['C'];items=bank();trainitems=[x for x in items if x['record']['split']=='train']
    assert trainitems and all(x['record']['split']=='train' for x in trainitems)
    source=pickle.load((B/'inputs/parent.pkl').open('rb'));initial_params=source['params'];initial_adam=source['adam'];devices=jax.local_device_count();assert devices==8
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(protocol['lr'],eps=1e-8))
    rep=lambda tr:jax.tree.map(lambda a:jnp.stack([jnp.asarray(a)]*devices),tr)
    collect=jax.pmap(collector(E),in_axes=(0,0,0,0,None,None))
    updates={arm:make_update(E,opt,arm) for arm in protocol['arms']}
    bankstates=merged(trainitems,'states');bankaux=merged(trainitems,'aux');records=[]
    control=B/'C/no_update/u000';control.mkdir(parents=True,exist_ok=True)
    if not (control/'COMMITTED.json').exists():
        save(control/'state.pkl',dict(params=initial_params,adam=initial_adam));save(control/'policy.pt',initial_params['actor'])
        atomic_json(control/'COMMITTED.json',dict(passed=True,update=0,policy_sha256=sha(control/'policy.pt'),state_sha256=sha(control/'state.pkl'),parent_optimizer_digest=digest(initial_adam),no_update=True))
    for seed in protocol['paired_seeds']:
        for arm in protocol['arms']:
            authorize();base=B/'C'/f'{arm}_s{seed}';base.mkdir(parents=True,exist_ok=True)
            if (base/'COMPLETE.json').exists():records.append(read(base/'COMPLETE.json'));continue
            params=rep(initial_params);adam=rep(initial_adam);rng=np.random.default_rng(seed)
            ids=rng.integers(0,len(trainitems),32);selected=[trainitems[i] for i in ids]
            states=shard(merged(selected,'states'),devices);aux=shard(merged(selected,'aux'),devices)
            keys=shard(np.asarray(jax.random.split(jax.random.PRNGKey(seed),32)),devices)
            start=0;last=sorted(base.glob('u*/COMMITTED.json'))
            if last:
                cp=last[-1].parent;receipt=read(cp/'COMMITTED.json');assert sha(cp/'state.pkl')==receipt['state_sha256'];saved=pickle.load((cp/'state.pkl').open('rb'));start=saved['update'];params=rep(saved['params']);adam=rep(saved['adam']);states=saved['states'];aux=saved['aux'];keys=saved['keys']
            def commit(u,metrics=None):
                cp=base/f'u{u:03d}';cp.mkdir(exist_ok=True)
                one=lambda tr:jax.tree.map(lambda a:np.asarray(a)[0],jax.device_get(tr))
                save(cp/'state.pkl',dict(params=one(params),adam=one(adam),states=jax.device_get(states),aux=jax.device_get(aux),keys=jax.device_get(keys),update=u,seed=seed,arm=arm))
                save(cp/'policy.pt',one(params)['actor'])
                atomic_json(cp/'COMMITTED.json',dict(passed=True,update=u,seed=seed,arm=arm,state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),parent_optimizer_digest=digest(initial_adam),environment_steps=u*4096,optimizer_steps=u*16,metrics=metrics,unix=time.time()))
                # Persistent CPU evaluation queue; no GPU training dependency.
                atomic_json(B/'CPU_QUEUE'/f'{arm}_s{seed}_u{u:03d}.json',dict(checkpoint=str(cp.relative_to(R)),seed=seed,arm=arm,update=u,final=u==protocol['updates']))
            if start==0 and not last:commit(0)
            began=time.time()
            for u in range(start+1,protocol['updates']+1):
                authorize();before=jax.tree.map(lambda a:np.asarray(a)[0],jax.device_get(params))
                (states,aux,keys),batch,diag=collect(params,states,aux,keys,bankstates,bankaux)
                batchkeys=np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),u),devices))
                (params,adam),losses=updates[arm](params,adam,batch,batchkeys);jax.block_until_ready(params)
                after=jax.tree.map(lambda a:np.asarray(a)[0],jax.device_get(params))
                norms={k:float(np.sqrt(sum(np.sum((np.asarray(a,dtype=float)-np.asarray(b,dtype=float))**2) for a,b in zip(jax.tree.leaves(before[k]),jax.tree.leaves(after[k]))))) for k in ['actor','critic']}
                losses=np.asarray(losses);assert np.isfinite(losses).all() and all(np.isfinite(a).all() for a in jax.tree.leaves(after))
                assert norms['actor']>0 and (norms['critic']>0 if arm!='DAgger-only' else norms['critic']==0)
                ph=np.asarray(diag[-1]);eligible=np.asarray(batch[-1]).any((-1,-2));custody=ph[...,0]>0
                exposure={name:dict(custody_rows=int((custody&(ph[...,k]>0)).sum()),teacher_rows=int((eligible&(ph[...,k]>0)).sum()),oracle_consistent_actions=int((custody&(ph[...,k]>0)&(ph[...,4]>0)).sum())) for name,k in [('near',1),('mid',2),('far',3)]}
                outcomes={name:int(ph[...,k].sum()) for name,k in [('delivery',5),('death',6),('drop',7),('self_defeat',8)]}
                actions=np.asarray(batch[2]);joint=np.asarray(batch[-1]);ar=np.indices(actions.shape[:-1])
                chosen_approved=joint[tuple(ar)+(actions[...,0],actions[...,1])]
                wrong=eligible&~chosen_approved;advantages=np.asarray(diag[4]);normalized=(advantages-advantages.mean())/(advantages.std()+1e-8)
                signal=dict(eligible_rows=int(eligible.sum()),set_disagreement_rows=int(wrong.sum()),disagreement_positive_raw_GAE=int((wrong&(advantages>0)).sum()),disagreement_positive_normalized_GAE=int((wrong&(normalized>0)).sum()),GAE_mean=float(advantages.mean()),GAE_variance=float(advantages.var()),return_mean=float(np.asarray(batch[5]).mean()),return_variance=float(np.asarray(batch[5]).var()))
                row=dict(update=u,seed=seed,arm=arm,finite=True,signal=signal,route_exposure=exposure,outcomes=outcomes,reward_component_names=list(E.ops.COMPONENTS),reward_component_sum=np.asarray(diag[2]).sum((0,1,2)).tolist(),update_l2=norms,loss_names=['total','ppo_surrogate','critic_hl_gauss','entropy','set_mass_teacher','teacher_fraction','approx_KL','gradient_norm_before_clip'],loss=losses.mean((0,1)).tolist(),teacher_rows=int(np.asarray(batch[-1]).any((-1,-2)).sum()),environment_steps=u*4096,optimizer_steps=u*16,seconds=time.time()-began)
                with (base/'updates.jsonl').open('a') as f:f.write(__import__('json').dumps(row)+'\n');f.flush();os.fsync(f.fileno())
                atomic_json(B/'STATUS.json',dict(stage='C_FORMAL_UPDATE',**row))
                if u in protocol['save_updates']:commit(u,row)
                if u==1:notify(f'C正式首更新 {arm}/seed{seed} finite，Actor Δ={norms["actor"]:.6g}，Critic Δ={norms["critic"]:.6g}，真实on-policy步4096；DAgger-only Critic按协议冻结。')
            row=dict(arm=arm,seed=seed,update=protocol['updates'],atomic_state_sha256=sha(base/f'u{protocol["updates"]:03d}/state.pkl'),environment_steps=protocol['updates']*4096,optimizer_steps=protocol['updates']*16,finite=True)
            atomic_json(base/'COMPLETE.json',row);records.append(row);notify(f'C固定run终态 {arm}/seed{seed} u16已原子保存，立即接续下一预声明臂。')
    atomic_json(B/'C_TRAINING_COMPLETE.json',dict(passed=True,runs=records,finite=True,old_continuous_resume=False,unix=time.time()))
    notify('C三臂×五配对seed固定更新预算全部结束；接续预声明保存点的有界native Bank评估，CPU/JS保留集独立运行；不恢复旧continuous。')
    # Evaluation is dispatched by a separate CPU owner. GPU process exits here.



def technical_smoke():
    jax,E=setup();items=[x for x in bank() if x['record']['split']=='train'];assert items
    source=pickle.load((B/'inputs/parent.pkl').open('rb'));devices=jax.local_device_count();assert devices==8
    selected=[items[i%len(items)] for i in range(32)]
    st=shard(merged(selected,'states'),devices);ax=shard(merged(selected,'aux'),devices)
    keys=shard(np.asarray(jax.random.split(jax.random.PRNGKey(2026102399),32)),devices)
    rep=lambda tr:jax.tree.map(lambda a:jnp.stack([jnp.asarray(a)]*devices),tr)
    p=rep(source['params']);a=rep(source['adam']);opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
    collect=jax.pmap(collector(E,steps=8),in_axes=(0,0,0,0,None,None))
    (st,ax,keys),batch,diag=collect(p,st,ax,keys,merged(items,'states'),merged(items,'aux'))
    save(B/'smoke/FROZEN_TRAIN_SCALE_BATCH.pkl',dict(batch=jax.device_get(batch),source_split='train',initial_parent_digest=digest(source['params']),same_batch_for_scale_audits=True))
    update=make_update(E,opt,'joint');(p,a),loss=update(p,a,batch,np.asarray(jax.random.split(jax.random.PRNGKey(91),8)));jax.block_until_ready(p)
    one=lambda tr:jax.tree.map(lambda a:np.asarray(a)[0],jax.device_get(tr))
    after=one(p);before=source['params'];norms={k:float(np.sqrt(sum(np.sum((np.asarray(x,dtype=float)-np.asarray(y,dtype=float))**2) for x,y in zip(jax.tree.leaves(after[k]),jax.tree.leaves(before[k]))))) for k in ['actor','critic']}
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(after)) and all(v>0 for v in norms.values()) and np.isfinite(np.asarray(loss)).all()
    snapshot=dict(params=after,adam=one(a),states=jax.tree.map(lambda a:np.asarray(a)[0],st),aux=jax.tree.map(lambda a:np.asarray(a)[0],ax),keys=np.asarray(keys)[0])
    save(B/'smoke/state.pkl',snapshot)
    nk,ke,act,_=jax.jit(E.choose)(snapshot['params'],snapshot['states'],snapshot['aux'],snapshot['keys'])
    result=jax.jit(E.transition)(snapshot['states'],snapshot['aux'],act,ke)
    atomic_json(B/'smoke/EXPECTED.json',dict(state_sha256=sha(B/'smoke/state.pkl'),optimizer_digest=digest(snapshot['adam']),next_step_hash=digest((nk,ke,act,result)),finite=True,nonzero=norms,environment_steps=256,optimizer_steps=1))
    code=__import__('subprocess').run([__import__('sys').executable,'-m','scripts.carrier_mechanism_train','smoke-restore'],env=dict(os.environ),stdout=(B/'smoke/fresh_restore.log').open('a'),stderr=__import__('subprocess').STDOUT).returncode
    assert code==0 and read(B/'smoke/RESTORE.json')['passed']
    atomic_json(B/'C_TECHNICAL_SMOKE.json',dict(passed=True,finite=True,nonzero=norms,fresh_process_restore=True,saved_optimizer=True,on_policy=True,environment_steps=256,scope='technical validity only; no capability admission gate',unix=time.time()))
    notify(f'C技术smoke通过：256真实on-policy步，Actor Δ={norms["actor"]:.6g}，Critic Δ={norms["critic"]:.6g}，finite；新进程env/aux/RNG/Adam恢复hash一致。立即进入正式固定预算。')

if __name__=='__main__':
    if __import__('sys').argv[1]=='smoke-restore':
        jax,E=setup();x=pickle.load((B/'smoke/state.pkl').open('rb'));expected=read(B/'smoke/EXPECTED.json');assert sha(B/'smoke/state.pkl')==expected['state_sha256'];assert digest(x['adam'])==expected['optimizer_digest']
        nk,ke,act,_=jax.jit(E.choose)(x['params'],x['states'],x['aux'],x['keys']);nxt=jax.jit(E.transition)(x['states'],x['aux'],act,ke)
        assert digest((nk,ke,act,nxt))==expected['next_step_hash'];atomic_json(B/'smoke/RESTORE.json',dict(passed=True,next_step_exact=True,optimizer_exact=True))
