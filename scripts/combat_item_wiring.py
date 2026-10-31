"""Frozen train-only full-state update replays; no model adopted or retuned."""
import json,pickle
import numpy as np
import jax
import jax.numpy as j
import optax
from scripts.combat_item_run import B,read,save,notify
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest
from scripts import combat_item_engine as T,carrier_mechanism_engine as E
from scripts.combat_item_train import rep,updater,objective
from scripts.carrier_mechanism_analysis import bank,merged

def audit_collector(arm,steps):
 def collect(p,st,ax,keys,bs,ba):
  def body(c,_):
   st,ax,key=c;nk,ke,act,data=E.choose(p,st,ax,key)
   o,g,mm,bm,own,lp,v=data;rt=T.route_teacher(st,ax);ct=T.combat_teacher(st,ax,key)
   after,na,r,done,info,parts=T.transition(st,ax,act,ke,arm)
   # Pure same-state reward evaluation avoids cancellation error in r-extra.
   legacy=T.OLD_TRANSITION(st,{k:v for k,v in ax.items() if k!='tactics'},act,ke)[2][:,0]
   nv=E.prediction(p,after)[2]
   fresh,fa=T.fresh(jax.vmap(lambda k:jax.random.fold_in(k,0xF3))(nk),bs,ba)
   after=E.select_tree(done,fresh,after);na=E.select_tree(done,fa,na)
   native=~st.alive[:,1]|(st.trapped[:,1]>0);carry=st.bun_carried[:,0]==1
   stock=info['steal'][:,0];pickup=stock|info['pickup_loose'][:,0]|info['native_transfer'][:,0]
   inventory=j.any(st.item_slots[:,0]>0,-1)
   flags=j.stack([~carry&~native,~carry&native,carry,stock&native,stock&~native,info['pickup_loose'][:,0],info['native_transfer'][:,0],info['capture'][:,0],info['death'][:,0],info['own_bomb_defeat'][:,0],inventory,act[:,0,1]==2,info['trap_hit'][:,0],info['trap_hit'][:,1]],-1)
   return (after,na,nk),(o,g,own,lp,v,nv,r[:,0],done,mm,bm,rt,ct,parts[:,0],flags,legacy)
  end,d=jax.lax.scan(body,(st,ax,keys),None,length=steps)
  o,g,act,lp,v,nv,r,done,mm,bm,rt,ct,parts,flags,legacy=d
  adv=E.train.compute_gae(r,v,nv,done,.995,.95)
  baseadv=E.train.compute_gae(legacy,v,nv,done,.995,.95)
  return end,(o,g,act,lp,adv,adv+v,mm,bm,rt,ct),(parts,flags,r,v,adv,baseadv,legacy)
 return jax.pmap(collect,in_axes=(0,0,0,0,None,None))


def deltas(x,y):
 return {k:float(np.sqrt(sum(np.sum((np.asarray(a,dtype=float)-np.asarray(b,dtype=float))**2) for a,b in zip(jax.tree.leaves(x[k]),jax.tree.leaves(y[k]))))) for k in ['actor','critic']}

def gradient_contrast():
 def contrast(p,b,baseadv,value,key):
  b=tuple(x.reshape((-1,)+x.shape[2:]) for x in b);baseadv=baseadv.reshape(-1);value=value.reshape(-1)
  def norm(a):
   mean=jax.lax.pmean(a.mean(),'device');sq=jax.lax.pmean((a*a).mean(),'device')
   return (a-mean)/j.sqrt(j.maximum(sq-mean*mean,0)+1e-8)
  index=jax.random.permutation(key,len(baseadv)).reshape(-1,32)[0]
  actual=b[:4]+(norm(b[4]),)+b[5:]
  zero=b[:4]+(norm(baseadv),baseadv+value)+b[6:]
  def grad(batch):
   batch=tuple(x[index] for x in batch)
   g=jax.grad(lambda q:objective(q,batch,0.,E.train.net_forward,E.train.hl_gauss_value_loss)[0])(p)
   return jax.lax.pmean(g,'device')
  a,c=grad(actual),grad(zero)
  out=[]
  for k in ['actor','critic']:
   aa=j.concatenate([x.ravel() for x in jax.tree.leaves(a[k])]);cc=j.concatenate([x.ravel() for x in jax.tree.leaves(c[k])])
   out.extend([j.linalg.norm(aa-cc),j.linalg.norm(aa),j.linalg.norm(cc),j.sum(aa*cc)/j.maximum(j.linalg.norm(aa)*j.linalg.norm(cc),1e-20)])
  return j.stack(out)
 return jax.pmap(contrast,axis_name='device')

def run():
 items=[x for x in bank() if x['record']['split']=='train'];bs=merged(items,'states');ba=merged(items,'aux')
 opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8));update=updater(0.,opt);contrast=gradient_contrast()
 collect={arm:audit_collector(arm,128) for arm in ['banana','glue']};rows=[]
 for seed in read(B/'PROTOCOL.json')['paired_seeds']:
  for arm in ['banana','glue']:
   base=B/'formal'/f'{arm}_s{seed}'
   logs={x['update']:x for x in [json.loads(line) for line in (base/'updates.jsonl').read_text().splitlines()]}
   for u in [0,1,8,16,24]:
    target=B/'wiring'/f'{arm}_s{seed}_from_u{u:03d}.json'
    if target.exists():rows.append(read(target));continue
    cp=base/f'u{u:03d}';marker=read(cp/'COMMITTED.json');assert sha(cp/'state.pkl')==marker['state_sha256']
    x=pickle.load((cp/'state.pkl').open('rb'));p=rep(x['params'])
    end,batch,diag=collect[arm](p,x['states'],x['aux'],x['keys'],bs,ba);jax.block_until_ready(batch)
    parts,flags,r,v,ad,cfad,legacy=[np.asarray(a) for a in jax.device_get(diag)]
    actual=logs[u+1];names=list(E.ops.COMPONENTS[:-1])+list(T.events.METRICS)+['combined_total']
    sums=parts.sum((0,1,2));expected=np.array([actual['reward_components'][name] for name in names])
    replay_close=bool(np.allclose(sums,expected,rtol=2e-4,atol=2e-3));flag_counts=flags.sum((0,1,2))
    event_exact=flag_counts.tolist()==list(actual['events'].values())
    acts=np.asarray(batch[2]);mm=np.asarray(batch[6]);bm=np.asarray(batch[7])
    joint_legal=mm[..., :,None]&bm[...,None,:]
    chosen_legal=bool(np.take_along_axis(mm,acts[...,0,None],-1).all() and np.take_along_axis(bm,acts[...,1,None],-1).all())
    keys=np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),u+1),8))
    gs=np.asarray(contrast(p,batch,diag[5],diag[3],keys))[0]
    row=dict(arm=arm,seed=seed,source_update=u,next_update=u+1,source_state_sha256=marker['state_sha256'],train_only=True,
      no_parameter_selection=True,environment_steps=8192,training_optimizer_steps=0,actual_training_component_replay_close=replay_close,actual_training_event_replay_exact=event_exact,
      max_component_absolute_error=float(np.max(np.abs(sums-expected))),replacement_reward_sum=float(parts[...,-2].sum()),replacement_nonzero_rows=int((parts[...,-2]!=0).sum()),
      chosen_actions_legal=chosen_legal,teachers_within_legal_masks=all(not np.any(np.asarray(batch[i])&~joint_legal) for i in [8,9]),
      actual_reward_equals_final_component=bool(np.array_equal(r,parts[...,-1])),actual_GAE_equals_optimizer_input=bool(np.array_equal(ad,np.asarray(batch[4]))),
      GAE_changed_rows=int((ad!=cfad).sum()),GAE_delta_L2=float(np.linalg.norm(ad-cfad)),
      first_actual_minibatch_gradient=dict(actor_delta_L2=float(gs[0]),actor_actual_norm=float(gs[1]),actor_counterfactual_norm=float(gs[2]),actor_cosine=float(gs[3]),critic_delta_L2=float(gs[4]),critic_actual_norm=float(gs[5]),critic_counterfactual_norm=float(gs[6]),critic_cosine=float(gs[7])),
      scope='same on-policy actions, masks, Critic and terminal masks; strip only replacement delta before actual GAE and normalized PPO/value objective. First production permutation minibatch before clipping. Gradient cosine is not harmful-interference proof.',batch_digest=digest(jax.device_get(batch)))
    # Fixed device0/env0 only: no search for a flattering or failing trajectory.
    route=np.asarray(batch[8])[0,:,0];act=acts[0,:,0]
    disagrees=route.any((-1,-2))&~route[np.arange(len(route)),act[:,0],act[:,1]]
    indices=np.flatnonzero(disagrees)
    fixed=dict(selection='fixed device0/env0 in each frozen128-step train replay',first_tick=None,absence_is_not_success=True)
    if len(indices):
     t=int(indices[0]);fixed.update(first_tick=t,action=act[t].tolist(),safe_route_joint_set=route[t].tolist(),
       model_input_digest=digest((np.asarray(batch[0])[0,t,0],np.asarray(batch[1])[0,t,0])),hash_scope='model inputs, not full simulator state',
       phase='carry' if flags[0,t,0,2] else 'native_window',reward=float(r[0,t,0]),components=parts[0,t,0].tolist(),
       actual_GAE=float(ad[0,t,0]),legacy_reward_GAE=float(cfad[0,t,0]),
       actual_GAE_next=float(ad[0,t+1,0]) if t+1<len(route) else None,
       actual_GAE_change=float(ad[0,t+1,0]-ad[0,t,0]) if t+1<len(route) else None)
    row['fixed_first_safe_route_disagreement']=fixed
    if u==0:
     (pp,aa),loss=update(p,rep(x['adam']),batch,keys);jax.block_until_ready(pp)
     one=lambda tr:jax.tree.map(lambda a:np.asarray(a)[0],jax.device_get(tr))
     truth=pickle.load((base/'u001/state.pkl').open('rb'));difference=deltas(one(pp),truth['params'])
     zero=batch[:4]+(diag[5],diag[5]+diag[3])+batch[6:]
     (cf,cfadam),_=update(p,rep(x['adam']),zero,keys);jax.block_until_ready(cf)
     control=pickle.load((B/'formal'/f'control_s{seed}'/'u001/state.pkl').open('rb'))
     row['full_first_update_replay']=dict(actual_vs_saved_param_L2=difference,counterfactual_vs_matched_A0_param_L2=deltas(one(cf),control['params']),
       actual_vs_counterfactual_param_L2=deltas(one(pp),one(cf)),actual_digest_exact=digest(one(pp))==digest(truth['params']),actual_Adam_digest_exact=digest(one(aa))==digest(truth['adam']),counterfactual_digest_matches_A0=digest(one(cf))==digest(control['params']),diagnostic_replayed_optimizer_steps=64,no_model_adopted=True)
    save(target.with_suffix('.pkl'),dict(diag=jax.device_get(diag),actions=np.asarray(batch[2]),legal_move=np.asarray(batch[6]),legal_ability=np.asarray(batch[7]),route_teacher=np.asarray(batch[8]),combat_teacher=np.asarray(batch[9]),batch_digest=row['batch_digest']))
    atomic_json(target,row);rows.append(row)
   notify(f'训练信号审计批终态 {arm}/s{seed}：固定u0/1/8/16/24完整状态重放，模型不采用，无holdout调参。')
 atomic_json(B/'WIRING_AUDIT_COMPLETE.json',dict(passed=True,rows=rows,train_only_environment_steps=245760,diagnostic_replayed_optimizer_steps=384,training_budget_unchanged=True,no_adopted_updates=True,all_training_component_replays_close=all(r['actual_training_component_replay_close'] for r in rows),all_training_event_replays_exact=all(r['actual_training_event_replay_exact'] for r in rows)))
