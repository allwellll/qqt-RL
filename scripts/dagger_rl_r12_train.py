"""R12: only auxiliary DAgger rollout behavior differs; PPO stays categorical."""
import os,sys,time,json,pickle,inspect,subprocess
import numpy as np
import jax,jax.numpy as j,optax
from scripts import dagger_rl_r10_train as S
from scripts.dagger_rl_r12_run import B,R,read,authorize,notify,save
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest
from scripts.dagger_rl_greedy_source import auxiliary_action
C,T=S.C,S.T
rep,one,norms=C.rep,C.one,C.norms


def bind():
    T.B,T.R,T.read,T.authorize,T.notify=B,R,read,authorize,notify
    T.ACTIVE_MIX=False;T.bind()
    C.collector=lambda steps:S.collector(steps,qualified=False)


def auxiliary_choose(params,states,aux,keys,*,greedy):
    nk,ke,act,data=C.E.choose(params,states,aux,keys)
    if not greedy:return nk,ke,act,data
    o,g,mm,bm,own,lp,v=data
    mv,ab,_,_=C.E.train.net_forward(params,'transformer',o,g)
    selected,lp=auxiliary_action(mv,ab,mm,bm,own,lp,greedy=True)
    act=act.at[:,0].set(selected)
    return nk,ke,act,(o,g,mm,bm,selected,lp,v)


def _auxiliary_factory():
    source=inspect.getsource(S.collector)
    assert source.count('def collector(steps, qualified=True):')==1
    assert source.count('C.E.choose(p, st, ax, key)')==1
    source=source.replace('def collector(steps, qualified=True):',
        'def auxiliary_collector(steps, greedy=False):\n    qualified=False')
    source=source.replace('C.E.choose(p, st, ax, key)',
                          'auxiliary_choose(p, st, ax, key, greedy=greedy)')
    scope=dict(vars(S),auxiliary_choose=auxiliary_choose)
    exec(compile(source,'<R12-auxiliary-student-collector>','exec'),scope)
    return scope['auxiliary_collector']


auxiliary_collector=_auxiliary_factory()


def commit(base,u,p,a,st,ax,key,seed,arm,history=None,metrics=None,formal=True,*,auxiliary):
    cp=base/f'u{u:03d}';cp.mkdir(parents=True,exist_ok=True)
    ast,aax,akey=auxiliary
    state=dict(params=one(p),adam=one(a),states=jax.device_get(st),aux=jax.device_get(ax),
        keys=jax.device_get(key),history=history,update=u,seed=seed,arm=arm,
        auxiliary=dict(states=jax.device_get(ast),aux=jax.device_get(aax),keys=jax.device_get(akey)))
    save(cp/'state.pkl',state);save(cp/'policy.pt',one(p)['actor'])
    atomic_json(cp/'COMMITTED.json',dict(passed=True,update=u,arm=arm,seed=seed,state_sha256=sha(cp/'state.pkl'),
        policy_sha256=sha(cp/'policy.pt'),metrics=metrics,environment_steps=u*16384 if formal else 1024,
        ppo_environment_steps=u*8192 if formal else 512,auxiliary_environment_steps=u*8192 if formal else 512,
        optimizer_steps=u*32 if formal else 2,full_both_streams_saved=True))
    if formal:atomic_json(B/'CPU_QUEUE'/f'{base.name}_u{u:03d}.json',dict(checkpoint=str(cp.relative_to(R)),update=u,final=u==32))
    return cp


def probe(x,bs,ba):
    p,a=rep(x['params']),rep(x['adam']);ast=x['auxiliary']
    pend,batch,diag=C.collector(8)(p,x['states'],x['aux'],x['keys'],bs,ba)
    aend,aux_batch,aux_diag=auxiliary_collector(8,greedy=True)(p,ast['states'],ast['aux'],ast['keys'],bs,ba)
    h,rb,stats=T.aggregate(aux_batch,x['history'],x['seed'],x['update']+1);rb=tuple(v[:,:64] for v in rb)
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
    (pn,an),loss=C.updater('anchor',opt)(p,a,batch,rb,np.asarray(jax.random.split(jax.random.PRNGKey(81),8)))
    jax.block_until_ready(pn)
    return digest((one(pn),one(an),pend,aend,batch,aux_batch,h,rb,stats,np.asarray(loss)))


def smoke():
    bind();items,source=T.setup();st,ax,key,bs,ba=T.initial(2026102698,items)
    ast,aax,akey,_,_=T.initial(2026102698^0xAA12,items)
    p,a=rep(source['params']),rep(source['adam'])
    pend,batch,diag=C.collector(8)(p,st,ax,key,bs,ba)
    aend,aux_batch,aux_diag=auxiliary_collector(8,greedy=True)(p,ast,aax,akey,bs,ba)
    history,rb,stats=T.aggregate(aux_batch,None,2026102698,1);rb=tuple(v[:,:64] for v in rb)
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
    (pn,an),loss=C.updater('anchor',opt)(p,a,batch,rb,np.asarray(jax.random.split(jax.random.PRNGKey(8),8)))
    jax.block_until_ready(pn);delta=norms(source['params'],one(pn))
    ability=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,one(pn)['actor']['heads']['wb'],source['params']['actor']['heads']['wb'])))
    assert all(v>0 for v in delta.values()) and ability>0
    assert np.isfinite(np.asarray(loss)).all() and all(np.isfinite(v).all() for v in jax.tree.leaves((pn,an)))
    cp=commit(B/'smoke',1,pn,an,*pend,2026102698,'technical',history,formal=False,auxiliary=aend)
    x=pickle.load((cp/'state.pkl').open('rb'));expected=probe(x,bs,ba)
    atomic_json(B/'smoke/EXPECTED.json',dict(next_update=expected,state_sha256=sha(cp/'state.pkl')))
    code=subprocess.run([sys.executable,'-m','scripts.dagger_rl_r12_train','restore'],stdout=(B/'smoke/restore.log').open('a'),stderr=subprocess.STDOUT).returncode
    assert code==0 and read(B/'smoke/RESTORE.json')['passed']
    atomic_json(B/'TECHNICAL_SMOKE.json',dict(passed=True,finite=True,nonzero=delta,ability_head_update_l2=ability,
        ability_head_gradient_norm=float(np.asarray(loss)[...,-1].mean()),fresh_full_both_stream_restore=True,
        total_executed_environment_steps=3072,total_executed_optimizer_steps=6,
        formal_inherits_smoke=False,auxiliary_rows_only_enter_supervised_replay=True,capability_gate=False))
    notify('R12技术smoke通过：双collector完整env/aux/RNG/Adam/history新进程下一更新精确，Actor/Critic/ability非零finite；立即固定正式队列。')


def restore():
    bind();items,source=T.setup();_,_,_,bs,ba=T.initial(2026102698,items)
    x=pickle.load((B/'smoke/u001/state.pkl').open('rb'));expected=read(B/'smoke/EXPECTED.json')
    assert sha(B/'smoke/u001/state.pkl')==expected['state_sha256']
    assert probe(x,bs,ba)==expected['next_update']
    atomic_json(B/'smoke/RESTORE.json',dict(passed=True,full_next_update_hash=expected['next_update']))


def formal():
 bind()
 items,source=T.setup();pr=read(B/'PROTOCOL.json');opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(pr['lr'],eps=1e-8));collect=C.collector(128);aux_collectors={arm:auxiliary_collector(128,greedy=pr['auxiliary_greedy'][arm]) for arm in pr['arms']}
 control=B/'formal/no_update/u000';control.mkdir(parents=True,exist_ok=True)
 if not (control/'COMMITTED.json').exists():
  save(control/'state.pkl',source);save(control/'policy.pt',source['params']['actor']);atomic_json(control/'COMMITTED.json',dict(passed=True,update=0,no_update=True,state_sha256=sha(control/'state.pkl'),policy_sha256=sha(control/'policy.pt')))
 records=[]
 for seed in pr['paired_seeds']:
  for arm in pr['arms']:
   update=C.updater('anchor',opt);aux_collect=aux_collectors[arm]
   authorize();base=B/'formal'/f'{arm}_s{seed}';base.mkdir(parents=True,exist_ok=True)
   if (base/'COMPLETE.json').exists():records.append(read(base/'COMPLETE.json'));continue
   p=rep(source['params']);a=rep(source['adam']);st,ax,key,bs,ba=T.initial(seed,items);ast,aax,akey,_,_=T.initial(seed^0xAA12,items);auxiliary=(ast,aax,akey);start=0;history=None
   last=sorted(base.glob('u*/COMMITTED.json'))
   if last:
    cp=last[-1].parent;assert sha(cp/'state.pkl')==read(cp/'COMMITTED.json')['state_sha256'];x=pickle.load((cp/'state.pkl').open('rb'));start=x['update'];p=rep(x['params']);a=rep(x['adam']);st=x['states'];ax=x['aux'];key=x['keys'];history=x['history'];auxiliary=tuple(x['auxiliary'][k] for k in ['states','aux','keys'])
   else:commit(base,0,p,a,st,ax,key,seed,arm,auxiliary=auxiliary)
   if start==0:
    assert digest(one(p))==digest(source['params']) and digest(one(a))==digest(source['adam'])
    atomic_json(base/'FRESH_PARENT.json',dict(passed=True,parent_input_sha256=sha(B/'inputs/parent.pkl'),params_exact=True,Adam_exact=True,smoke_not_inherited=True))
   began=time.time()
   for u in range(start+1,33):
    authorize();before=one(p);mode='anchor'
    (st,ax,key),batch,diag=collect(p,st,ax,key,bs,ba);auxiliary,aux_batch,aux_diag=aux_collect(p,*auxiliary,bs,ba);history,rb,ag=T.aggregate(aux_batch,history,seed,u)
    (p,a),loss=update(p,a,batch,rb,np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),u),8)));jax.block_until_ready(p)
    after=one(p);n=norms(before,after);assert n['actor']>0 and (n['critic']==0 if mode=='dagger' else n['critic']>0)
    assert np.isfinite(np.asarray(loss)).all() and all(np.isfinite(v).all() for v in jax.tree.leaves(after))
    fn=['carry','banana_inventory','ability2_legal','sample_release','greedy_release','release_probability_sum','learner_any_item_contact','delivery','death','window_stock','free_stock','loose','transfer','self_defeat','empty_combat','empty_window']
    flags=np.asarray(diag[1]);od=np.asarray(diag[2]);item=np.asarray(batch[9]);carry=flags[...,0]>0
    row=dict(arm=arm,seed=seed,update=u,mode=mode,finite=True,update_l2=n,environment_steps=u*16384,ppo_environment_steps=u*8192,auxiliary_environment_steps=u*8192,optimizer_steps=u*32,aggregation=ag,elapsed_seconds=time.time()-began,effective_steps_this_process=(u-start)*16384,steps_per_second=(u-start)*16384/(time.time()-began),loss=np.asarray(loss).mean((0,1)).tolist(),ability_head_gradient_norm=float(np.asarray(loss)[...,-1].mean()),ability_head_update_l2=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,after['actor']['heads']['wb'],before['actor']['heads']['wb']))),events=dict(zip(fn,flags.sum((0,1,2)).tolist())),item_opportunities=dict(release=int(od[...,2].sum()),contact=int(od[...,3].sum()),mean_forecast_net=float(od[...,4][item].mean()) if item.any() else None),carry_inventory_ticks=int((carry&(flags[...,1]>0)).sum()),carry_release=int((carry&(flags[...,3]>0)).sum()),reward_components=np.asarray(diag[0]).sum((0,1,2)).tolist(),opponent_exposure=dict(current=0,ring=0,fixed=8192,kind='JAX native tier3'),GAE_mean=float(np.asarray(diag[-1]).mean()),GAE_variance=float(np.asarray(diag[-1]).var()))
    row['dagger_coefficient']=pr['dagger_coefficient_by_arm'][arm]
    row['learning_rate']=pr['learning_rate_by_arm'][arm]
    q=np.asarray(diag[8]);row['route_release_qualification']=dict(zip(['candidate','removed','emptied'],q.sum((0,1,2)).astype(int).tolist()))
    row['route_release_by_source_phase']={}
    for sn,sv in [('natural',np.asarray(diag[6])<0),('real_pickup_bank',np.asarray(diag[6])>=0)]:
     for ph,pv in [('carry',flags[...,0]>0),('empty_window',flags[...,15]>0),('empty_combat',flags[...,14]>0)]:
      ix=sv&pv;row['route_release_by_source_phase'][sn+'/'+ph]=dict(ticks=int(ix.sum()),**dict(zip(['candidate','removed','emptied'],q[ix].sum(0).astype(int).tolist())))
    exposure=np.asarray(diag[7]);selected=exposure[...,0]>0
    row['opponent_exposure']=dict(current=0,ring=0,fixed=8192,rule3=int((~selected).sum()),historical_parent=int(selected.sum()),kind='JAX rule3/frozen common-parent categorical; fixed historical, not current/ring/JS')
    row['combat_support']={name:dict(rule3=int(((exposure[...,i]>0)&~selected).sum()),historical_parent=int(((exposure[...,i]>0)&selected).sum())) for i,name in enumerate(['unused','live','combat','near_le4','mid_4to8','far_gt8']) if i>0}
    row['GAE_mean']=float(np.asarray(diag[5]).mean());row['GAE_variance']=float(np.asarray(diag[5]).var())
    row['anchor_KL']=float(np.asarray(loss)[...,7].mean());row['anchor_eligible_fraction']=float(np.asarray(loss)[...,8].mean())
    origins=np.asarray(diag[6]);row['source_exposure']={}
    for name,valid in [('natural',origins<0),('real_pickup_bank',origins>=0)]:
     row['source_exposure'][name]=dict(ticks=int(valid.sum()),events={k:float(flags[...,i][valid].sum()) for i,k in enumerate(fn)},item_release_labels=int(od[...,2][valid].sum()),item_contact_labels=int(od[...,3][valid].sum()))
    row['auxiliary_aggregation']=ag;row['auxiliary_greedy']=pr['auxiliary_greedy'][arm];row['auxiliary_events']=np.asarray(aux_diag[1]).sum((0,1,2)).tolist();row['auxiliary_item_opportunities']=dict(release=int(np.asarray(aux_diag[2])[...,2].sum()),contact=int(np.asarray(aux_diag[2])[...,3].sum()));row['PPO_batch_source']='fresh categorical learner stream only';row['auxiliary_advantages_discarded']=True
    row['auxiliary_source_exposure']={sn:dict(ticks=int(sv.sum()),item_release_labels=int(np.asarray(aux_diag[2])[...,2][sv].sum()),item_contact_labels=int(np.asarray(aux_diag[2])[...,3][sv].sum())) for sn,sv in [('natural',np.asarray(aux_diag[6])<0),('real_pickup_bank',np.asarray(aux_diag[6])>=0)]}
    row['opponent_exposure_scope']='PPO stream only';row['auxiliary_opponent_exposure']=dict(current=0,ring=0,fixed=8192,rule3=8192,historical_parent=0)
    with (base/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    # Every update is atomic for a safe user stop; evaluator consumes declared points only.
    cp=commit(base,u,p,a,st,ax,key,seed,arm,history,row,auxiliary=auxiliary)
    atomic_json(B/'STATUS.json',dict(stage='FORMAL',**row))
    if u==1:notify(f'R12正式首更新 {arm}/s{seed} u1 {mode} finite：Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}，8192 PPO+8192辅助真实学生步，{ag["new_item_rows"]}道具老师行，原子SHA {sha(cp/"state.pkl")[:16]}。')
    if u==17:notify(f'R12 {arm}/s{seed} u17：{mode}，Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}，中段邻点完整保存（本轮无阶段切换）；未使用历史数据作PPO。')
   final=base/'u032';receipt=dict(arm=arm,seed=seed,update=32,finite=True,state_sha256=sha(final/'state.pkl'),policy_sha256=sha(final/'policy.pt'),environment_steps=524288,optimizer_steps=1024)
   atomic_json(base/'COMPLETE.json',receipt);records.append(receipt);notify(f'DAgger/RL固定run终态 {arm}/s{seed} u32完整原子提交；524288环境步/1024优化步，立即接续冻结队列，CPU不阻塞。')
 atomic_json(B/'TRAINING_COMPLETE.json',dict(passed=True,runs=records,environment_steps=3145728,optimizer_steps=6144,old_continuous_resume=False))

if __name__=='__main__':globals()[sys.argv[1]]()
