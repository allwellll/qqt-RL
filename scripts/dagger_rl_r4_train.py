"""Isolated eviction-only screen; both arms keep the same joint+parent-KL math."""
import os,time,pickle,json
import numpy as np
import jax
import jax.numpy as j
import optax
from scripts.dagger_rl_r4_run import B,R,read,save,notify,authorize
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest
from scripts.carrier_mechanism_analysis import bank,merged,shard
from scripts import combat_item_engine as T,carrier_mechanism_engine as E,dagger_rl_oracle as O
from scripts.carrier_mechanism_train import preserve_critic
from scripts.dagger_rl_anchor import combat_rows,reference_kl
REFERENCE_PARAMS=None
MEMORY_STRATEGY="reservoir"
BASE_FRESH=T.fresh
BASE_TRANSITION=T.transition
PROVENANCE_INSTALLED=False
BANK_PROBABILITY=.25


def install_provenance(bank_probability=.25):
 """Common passive metadata; never part of observation, labels or rewards."""
 global PROVENANCE_INSTALLED,BANK_PROBABILITY
 if not 0 <= bank_probability <= 1:raise ValueError('Invalid Bank probability')
 if PROVENANCE_INSTALLED:
  if bank_probability!=BANK_PROBABILITY:raise ValueError('Bank probability cannot change within a worker')
  return
 BANK_PROBABILITY=bank_probability
 def fresh(keys,bs,ba):
  st,ax=BASE_FRESH(keys,bs,ba,bank_probability=bank_probability)
  split=jax.vmap(lambda k:jax.random.split(k,4))(keys)
  choice=jax.vmap(lambda k:jax.random.randint(k,(),0,bs.t.size))(split[:,1])
  use=jax.vmap(lambda k:jax.random.bernoulli(k,bank_probability))(split[:,2])
  return st,dict(ax,provenance_source=j.where(use,choice,-1))
 def transition(st,ax,actions,keys,arm='control'):
  source=ax['provenance_source'];old={k:v for k,v in ax.items() if k!='provenance_source'}
  af,na,r,done,info,parts=BASE_TRANSITION(st,old,actions,keys,arm)
  return af,dict(na,provenance_source=source),r,done,info,parts
 T.fresh=fresh;T.transition=transition;PROVENANCE_INSTALLED=True


def setup():
 authorize();
 if (B/'SOURCE_SEAL.json').exists():
  for name,h in read(B/'SOURCE_SEAL.json')['sources'].items():assert sha(R/name)==h,'sealed source changed: '+name
 T.configure();install_provenance();assert jax.default_backend()=='gpu' and jax.local_device_count()==8
 items=[x for x in bank() if x['record']['split']=='train'];assert len(items)==32
 from scripts.carrier_mechanism_stats import validate_bank_record
 for x in items:validate_bank_record(x['record'])
 source=pickle.load((B/'inputs/parent.pkl').open('rb'))
 assert read(B/'PROTOCOL.json')['anchor_coefficient']==.05
 global REFERENCE_PARAMS
 REFERENCE_PARAMS=source['params']
 return items,source

def rep(x):return jax.tree.map(lambda a:j.stack([j.asarray(a)]*8),x)
def one(x):return jax.tree.map(lambda a:np.asarray(a)[0],jax.device_get(x))
def norms(x,y):return {k:float(np.sqrt(sum(np.sum((np.asarray(a,dtype=float)-np.asarray(b,dtype=float))**2) for a,b in zip(jax.tree.leaves(x[k]),jax.tree.leaves(y[k]))))) for k in ['actor','critic']}

def initial(seed,items):
 keys=np.asarray(jax.random.split(jax.random.PRNGKey(seed),64));bs=merged(items,'states');ba=merged(items,'aux')
 st,ax=jax.jit(T.fresh)(keys,bs,ba)
 return shard(st,8),shard(ax,8),shard(keys,8),bs,ba

def collector(steps,reset_telemetry=False):
 def collect(p,st,ax,keys,bs,ba):
  def body(c,_):
   st,ax,key=c;nk,ke,act,data=E.choose(p,st,ax,key);o,g,mm,bm,own,lp,v=data
   route=T.route_teacher(st,ax);item,od=O.teacher(st,ax);teacher=O.combine(route,item)
   after,na,r,done,info,parts=T.transition(st,ax,act,ke,'control');nv=E.prediction(p,after)[2]
   fresh,fa=T.fresh(jax.vmap(lambda k:jax.random.fold_in(k,0xF3))(nk),bs,ba)
   after=E.select_tree(done,fresh,after);na=E.select_tree(done,fa,na)
   carry=st.bun_carried[:,0]==1;window=~st.alive[:,1]|(st.trapped[:,1]>0)
   _,ab,_,_=E.train.net_forward(p,'transformer',o,g);prob=jax.nn.softmax(j.where(bm,ab,-1e9))[:,2]
   flags=j.stack([carry,st.held_item[:,0]==1,bm[:,2],own[:,1]==2,j.argmax(j.where(bm,ab,-1e9),-1)==2,prob,info['trap_hit'][:,0],info['capture'][:,0],info['death'][:,0],info['steal'][:,0]&window,info['steal'][:,0]&~window,info['pickup_loose'][:,0],info['native_transfer'][:,0],info['own_bomb_defeat'][:,0],~carry&~window,~carry&window],-1)
   # BunState has no stored done flag. The collector's durable terminal latch
   # covers native termination/horizon while residual hazards settle.
   live=~ax['terminal']
   output=(o,g,own,lp,v,nv,r[:,0],done,mm,bm,teacher,item.any((1,2)),live,parts[:,0],flags,od,ax['provenance_source'])
   if reset_telemetry:output=output+(j.stack((done,done&(fa['provenance_source']>=0)),-1),)
   return (after,na,nk),output
  end,d=jax.lax.scan(body,(st,ax,keys),None,length=steps)
  o,g,act,lp,v,nv,r,done,mm,bm,teacher,isitem,live,parts,flags,od,origins=d[:17]
  adv=E.train.compute_gae(r,v,nv,done,.995,.95)
  diagnostics=(parts,flags,od,r,v,adv,origins)
  if reset_telemetry:diagnostics=diagnostics+(d[17],)
  return end,(o,g,act,lp,adv,adv+v,mm,bm,teacher,isitem,live),diagnostics
 return jax.pmap(collect,in_axes=(0,0,0,0,None,None))

def supervised(p,b):
 o,g,mm,bm,mask,isitem=b
 mv,ab,_,_=E.train.net_forward(p,'transformer',o,g)
 mv=j.where(mm,mv,-1e9);ab=j.where(bm,ab,-1e9)
 # Equal weight per present route/item family, not drowned by route row count.
 a=O.set_loss(mv,ab,mask&isitem[:,None,None]);c=O.set_loss(mv,ab,mask&~isitem[:,None,None])
 return (a+c)/j.maximum(j.any(isitem).astype(j.int32)+j.any(mask.any((1,2))&~isitem).astype(j.int32),1)

def objective(p,b,replay,mode):
 o,g,act,old,ad,ret,mm,bm,teacher,isitem=b[:10]
 mv,ab,_,vl=E.train.net_forward(p,'transformer',o,g)
 lm=jax.nn.log_softmax(j.where(mm,mv,-1e9));lb=jax.nn.log_softmax(j.where(bm,ab,-1e9))
 lp=lm[j.arange(len(old)),act[:,0]]+lb[j.arange(len(old)),act[:,1]];ratio=j.exp(lp-old)
 ppo=j.maximum(-ad*ratio,-ad*j.clip(ratio,.9,1.1)).mean();value=E.train.hl_gauss_value_loss(vl,ret)
 ent=-(j.sum(j.exp(lm)*lm,-1)+j.sum(j.exp(lb)*lb,-1)).mean()
 sup=.5*(supervised(p,(o,g,mm,bm,teacher,isitem))+supervised(p,replay))
 base=ppo+.5*value-.005*ent+.2*sup
 gate=combat_rows(o,g)&(b[10] if len(b)>10 else j.ones(len(g),bool))
 # Frozen parent has no policy gradient. Cargo/window/drop rows are unconstrained.
 if mode=='anchor':
  pm,pa,_,_=E.train.net_forward(REFERENCE_PARAMS,'transformer',o,g)
  anchor=reference_kl(mv,ab,pm,pa,mm,bm,gate)
  loss=base+.05*anchor
 else:
  anchor=j.array(0.)
  loss=base
 return loss,j.array([ppo,value,ent,sup,teacher.any((1,2)).mean(),isitem.mean(),anchor,gate.mean()])

def updater(mode,opt):
 def update(p,a,b,replay,key):
  b=tuple(x.reshape((-1,)+x.shape[2:]) for x in b);ad=b[4]
  mean=jax.lax.pmean(ad.mean(),'device');sq=jax.lax.pmean((ad**2).mean(),'device');ad=(ad-mean)/j.sqrt(j.maximum(sq-mean**2,0)+1e-8);b=b[:4]+(ad,)+b[5:]
  order=jax.random.permutation(key,len(ad)).reshape(-1,32)
  def body(c,ix):
   p,a=c;mb=tuple(x[ix] for x in b);rb=tuple(x[ix] for x in replay)
   (loss,metrics),grad=jax.value_and_grad(lambda q:objective(q,mb,rb,mode),has_aux=True)(p)
   grad=jax.lax.pmean(grad,'device');changes,na=opt.update(grad,a,p);pn=optax.apply_updates(p,changes)
   if mode=='dagger':pn=dict(pn,critic=p['critic']);na=preserve_critic(na,a)
   return (pn,na),jax.lax.pmean(j.concatenate([loss[None],metrics,optax.global_norm(grad)[None],optax.global_norm(grad['actor']['heads']['wb'])[None]]),'device')
  return jax.lax.scan(body,(p,a),order)
 return jax.pmap(update,axis_name='device')

def aggregate(batch,history,seed,u):
 from scripts.dagger_rl_memory import aggregate_rows
 b=tuple(np.asarray(x).reshape((-1,)+x.shape[3:]) for x in jax.device_get(batch))
 new=(b[0],b[1],b[6],b[7],b[8],b[9]);eligible=b[8].any((1,2))
 accepted=tuple(x[eligible] for x in new)
 history,replay,stats=aggregate_rows(accepted,history,seed=seed,update=u,strategy=MEMORY_STRATEGY,fallback=new)
 replay=tuple(x.reshape((8,1024)+x.shape[1:]) for x in replay)
 return history,replay,stats


def commit(base,u,p,a,st,ax,key,seed,arm,history=None,metrics=None,formal=True):
 cp=base/f'u{u:03d}';cp.mkdir(parents=True,exist_ok=True)
 save(cp/'state.pkl',dict(params=one(p),adam=one(a),states=jax.device_get(st),aux=jax.device_get(ax),keys=jax.device_get(key),history=history,update=u,seed=seed,arm=arm))
 save(cp/'policy.pt',one(p)['actor'])
 atomic_json(cp/'COMMITTED.json',dict(passed=True,update=u,arm=arm,seed=seed,state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),metrics=metrics,environment_steps=u*8192 if formal else 512,optimizer_steps=u*32 if formal else 2))
 if formal:atomic_json(B/'CPU_QUEUE'/f'{base.name}_u{u:03d}.json',dict(checkpoint=str(cp.relative_to(R)),update=u,final=u==32))
 return cp


def audit():
 if (B/'inputs/prior_audit_student_opportunities.pkl').exists():return corrected_audit()
 items,source=setup();st=merged(items,'states');ax=dict(merged(items,'aux'),tactics=T.events.state(32));key=np.asarray(jax.random.split(jax.random.PRNGKey(2026102490),32))
 # Preserve fixed parent/student trajectories. No action intervention here.
 def collect(st,ax,key):
  def body(c,_):
   s,a,k=c;nk,ke,act,data=E.choose(source['params'],s,a,k);label,diag=O.teacher(s,a)
   af,na,r,d,info,parts=T.transition(s,a,act,ke,'control')
   return (af,na,nk),(s,a,k,act,label,diag,data[3],data[1],data[0])
  return jax.lax.scan(body,(st,ax,key),None,length=128)
 collect=jax.jit(collect);examples=[];stats=[]
 for i in range(4):
  (st,ax,key),d=collect(st,ax,key);jax.block_until_ready(st.t)
  ss,aa,kk,acts,labels,diags,bm,gv,obs=jax.device_get(d);valid=labels.any((-1,-2));stats.append(dict(chunk=i,eligible=int(valid.sum()),release=int(diags[...,2].sum()),contact=int(diags[...,3].sum()),legal_inventory=int((bm[...,2]&(gv[...,21]>0)).sum()),actual_sampled_release=int((acts[:,:,0,1]==2).sum())))
  for t,n in np.argwhere(valid):
   if len(examples)>=32:break
   take=lambda tree:jax.tree.map(lambda x:x[t,n:n+1],tree)
   examples.append(dict(states=take(ss),aux=take(aa),keys=kk[t,n:n+1],label=labels[t,n],diag=diags[t,n],observed_action=acts[t,n],source=items[n]['record']['source_episode'],tick=int(ss.t[t,n])))
  atomic_json(B/'AUDIT_PROGRESS.json',dict(chunks=stats,examples=len(examples)))
 save(B/'audit_student_opportunities.pkl',examples)
 assert examples,'banana intervention unreachable on frozen train student paths: protocol correction required'
 # Controlled branches use native env, no hidden future keys in labels.
 parity=[];step=jax.jit(lambda s,a,k:jax.vmap(lambda st,ac,key:E.env.step(st,ac,key,False,True))(s,a,k))
 for ex in examples:
  s=jax.tree.map(j.asarray,ex['states']);a=ex['aux'];accepted=np.argwhere(ex['label']);move,ability=accepted[0]
  row=dict(source=ex['source'],tick=ex['tick'],initial_hash=digest((s,a,ex['keys'])),joint_set=accepted.tolist(),forecast_net=float(ex['diag'][4]))
  paths=[]
  for use in [True,False]:
   st=s;origin=np.asarray(st.pos[0,0]).astype(int);field=origin
   owns=np.argwhere((np.asarray(st.field_item[0])==1)&(np.asarray(st.field_owner[0])==0))
   if ability==0 and len(owns):field=owns[np.abs(owns+.5-np.asarray(st.pos[0,0])).sum(-1).argmin()]
   # First accepted move is away on release, home on contact.
   home=int(np.array([1,0,3,2])[move]) if ability==2 else int(move)
   armed=bool(st.field_armed[0,field[0],field[1]]) if ability==0 else False
   contact=False;finished=False;bad=False;steps=0;trace=[]
   for t in range(64):
    if use:
     d=int(move) if not armed and not contact else home
     ab=int(ability) if t==0 else 0
    else:d=home;ab=0
    ac=j.array([[[d,ab],[4,0]]]);oldslide=int(st.move_status[0,0])==2
    st,done,info=step(st,ac,jax.random.split(jax.random.fold_in(jax.random.PRNGKey(2026102491),t),1))
    armed|=not np.all(np.asarray(st.pos[0,0]).astype(int)==field)
    hit=bool(info['trap_hit'][0,0]) and int(info['trap_owner'][0,0])==0
    contact|=hit;steps=t+1;bad|=not bool(st.alive[0,0]) or int(st.trapped[0,0])>0 or int(st.bun_carried[0,0])!=1
    trace.append(dict(t=t,action=[d,ab],pos=np.asarray(st.pos[0,0]).tolist(),slide=int(st.move_status[0,0]),contact=hit))
    if use and contact and oldslide and int(st.move_status[0,0])!=2:finished=True;break
   distance=a['carry_items']['distance'][:,0];blocked=s.wall|s.brick|s.core.pushable
   from scripts.je10_route_oracle import continuous_distance
   gain=float(continuous_distance(s.pos[:,0],distance,blocked)[0]-continuous_distance(st.pos[:,0],distance,blocked)[0]-.18*steps)
   paths.append(dict(use=use,finished=finished,owned_contact=contact,bad=bad,steps=steps,net_distance=gain,final_hash=digest(st),trace=trace))
  row['paths']=paths;parity.append(row)
 save(B/'audit_native_branches.pkl',parity)
 actual=[r['paths'][0] for r in parity];positive=sum(x['owned_contact'] and x['finished'] and not x['bad'] and x['net_distance']>.5 for x in actual)
 assert positive==len(actual),'forecast opportunities failed native causal validation; no invalid teacher labels may enter formal'
 # Audit actual ability gradient on real input, not a toy-only graph.
 ex=examples[0];o,g,mm,bm=E.inputs(ex['states']);mask=j.asarray(ex['label'][None])
 def f(p):
  mv,ab,_,_=E.train.net_forward(p,'transformer',o,g)
  return O.set_loss(j.where(mm,mv,-1e9),j.where(bm,ab,-1e9),mask)
 grad=jax.grad(f)(source['params']);save(B/'audit_ability_gradient.pkl',jax.device_get(grad['actor']))
 mv,ab,_,_=E.train.net_forward(source['params'],'transformer',o,g);probs=np.asarray(jax.nn.softmax(j.where(bm,ab,-1e9)))[0]
 atomic_json(B/'ACTION_CHAIN_AUDIT.json',dict(passed=True,chunks=stats,examples=len(examples),native_positive_complete=positive,native_rejected=len(actual)-positive,causal_branches_scope='enemy wait physical causal microtest, not deployment opponent; no intervention samples used for training',release_probability=probs.tolist(),greedy_ability=int(probs.argmax()),ability_head_gradient_norm=float(optax.global_norm(grad['actor']['heads']['wb'])),actor_gradient_norm=float(optax.global_norm(grad['actor'])),critic_gradient_norm=float(optax.global_norm(grad['critic'])),no_future_rng_in_teacher=True,environment_steps=16384,raw_examples_sha256=sha(B/'audit_student_opportunities.pkl'),raw_branches_sha256=sha(B/'audit_native_branches.pkl')))
 notify(f'动作链审计完成：真实学生路径{sum(x["eligible"] for x in stats)}个老师机会，固定前{len(actual)}条原生完整分支中{positive}条自放自吃且净收益>0.5；不把物理干预混入训练。')


def corrected_audit():
 items,source=setup()
 original=pickle.load((B/'inputs/prior_audit_student_opportunities.pkl').open('rb'))
 branches=pickle.load((B/'inputs/prior_audit_native_branches.pkl').open('rb'))
 st=merged(original,'states');ax=merged(original,'aux');masks,diags=jax.jit(O.teacher)(st,ax)
 masks,diags=jax.device_get((masks,diags));accepted=[];kept=[];excluded=[]
 for i,(ex,branch) in enumerate(zip(original,branches)):
  if not masks[i].any():
   excluded.append(dict(source=ex['source'],tick=ex['tick'],reason='current-public-timer complete-slide safety abstention',original_forecast=float(ex['diag'][4])));continue
  assert np.array_equal(masks[i],ex['label']),'changed accepted plan needs new native branch, cannot reuse mismatched path'
  actual=branch['paths'][0];assert actual['finished'] and actual['owned_contact'] and not actual['bad'] and actual['net_distance']>.5
  accepted.append(dict(ex,label=masks[i],diag=diags[i]));kept.append(branch)
 assert accepted,'corrected intervention is absent; do not train equivalent arms'
 save(B/'audit_student_opportunities.pkl',accepted);save(B/'audit_native_branches.pkl',kept)
 ex=accepted[0];o,g,mm,bm=E.inputs(ex['states']);mask=j.asarray(ex['label'][None])
 def f(p):
  mv,ab,_,_=E.train.net_forward(p,'transformer',o,g)
  return O.set_loss(j.where(mm,mv,-1e9),j.where(bm,ab,-1e9),mask)
 grad=jax.grad(f)(source['params']);ability_grad=float(optax.global_norm(grad['actor']['heads']['wb']));assert ability_grad>0
 mv,ab,_,_=E.train.net_forward(source['params'],'transformer',o,g)
 prior=read(B/'inputs/prior_ACTION_CHAIN_AUDIT.json')
 atomic_json(B/'ACTION_CHAIN_AUDIT.json',dict(passed=True,protocol_version=2,original_fixed_states=len(original),accepted_states=len(accepted),excluded=excluded,examples=len(accepted),unique_sources=len({x['source'] for x in accepted}),native_positive_complete=len(accepted),native_rejected=0,reused_native_physical_paths=True,new_environment_steps=0,prior_parent_student_environment_steps=prior['environment_steps'],release_probability=np.asarray(jax.nn.softmax(j.where(bm,ab,-1e9)))[0].tolist(),ability_head_gradient_norm=ability_grad,actor_gradient_norm=float(optax.global_norm(grad['actor'])),critic_gradient_norm=float(optax.global_norm(grad['critic'])),raw_examples_sha256=sha(B/'audit_student_opportunities.pkl'),raw_branches_sha256=sha(B/'audit_native_branches.pkl'),no_future_rng_in_teacher=True,scope='same frozen train student states, deterministic visible-revival correction; known-timer exclusion before closed-loop outcomes, not capability selection'))
 notify(f'修正版动作审计：固定{len(original)}状态中{len(accepted)}安全标签，{len(excluded)}已知复活威胁明确弃权；ability真实梯度{ability_grad:.6g}，不重复旧无效smoke。')


def next_update_probe(x,bankstates,bankaux):
 # Continue the restored policy on fresh on-policy states, then execute the
 # actual memory draw and optimizer. Reference/replay branches are technical
 # diagnostics only and are never inherited by formal training.
 p,a=rep(x['params']),rep(x['adam'])
 (st,ax,key),batch,_=collector(8)(p,x['states'],x['aux'],x['keys'],bankstates,bankaux)
 history,replay,stats=aggregate(batch,x['history'],2026102498,2)
 replay=tuple(v[:,:64] for v in replay)
 opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
 (p,a),loss=updater('anchor',opt)(p,a,batch,replay,np.asarray(jax.random.split(jax.random.PRNGKey(9),8)))
 jax.block_until_ready(p)
 assert np.isfinite(np.asarray(loss)).all()
 return dict(env_aux_RNG=digest((st,ax,key)),params_Adam=digest((one(p),one(a))),
             memory=digest(history),replay=digest(replay),seen=history['seen'],
             finite=True,environment_steps=512,optimizer_steps=2)


def smoke():
 items,source=setup();assert read(B/'ACTION_CHAIN_AUDIT.json')['passed'] and read(B/'ACTION_CHAIN_AUDIT.json')['native_rejected']==0;st,ax,key,bs,ba=initial(2026102498,items)
 opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8));p=rep(source['params']);a=rep(source['adam'])
 (st,ax,key),b,d=collector(8)(p,st,ax,key,bs,ba)
 # Small technical shape replay matches each device's64 rows, not formal1024.
 history,rb,ar=aggregate(b,None,2026102498,1);rb=tuple(x[:,:64] for x in rb)
 (p,a),loss=updater('anchor',opt)(p,a,b,rb,np.asarray(jax.random.split(jax.random.PRNGKey(8),8)));jax.block_until_ready(p)
 n=norms(source['params'],one(p));ability_update=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,one(p)['actor']['heads']['wb'],source['params']['actor']['heads']['wb'])));assert ability_update>0;assert all(v>0 for v in n.values()) and np.isfinite(np.asarray(loss)).all()
 flatb=tuple(v.reshape((-1,)+v.shape[3:]) for v in b)
 def raw_anchor(q):
  o,g,mm,bm=flatb[0],flatb[1],flatb[6],flatb[7]
  mv,ab,_,_=E.train.net_forward(q,'transformer',o,g);pm,pa,_,_=E.train.net_forward(source['params'],'transformer',o,g)
  return reference_kl(mv,ab,pm,pa,mm,bm,combat_rows(o,g)&flatb[10])
 ag=jax.jit(jax.grad(raw_anchor))(one(p))
 raw_ability=float(optax.global_norm(ag['actor']['heads']['wb']));assert raw_ability>0
 assert float(optax.global_norm(ag['critic']))==0
 atomic_json(B/'SMOKE_ANCHOR_GRADIENT.json',dict(passed=True,raw_actor_gradient=float(optax.global_norm(ag['actor'])),raw_ability_head_gradient=raw_ability,raw_critic_gradient=0.,scope='new real learner smoke observations after actual update; no adopted extra optimizer step'))
 cp=commit(B/'smoke',1,p,a,st,ax,key,2026102498,'technical',history,formal=False)
 x=pickle.load((cp/'state.pkl').open('rb'))
 expected=next_update_probe(x,bs,ba)
 atomic_json(B/'smoke/EXPECTED.json',dict(next_update=expected,adam=digest(x['adam']),history=digest(history),nonzero=n))
 import subprocess,sys
 code=subprocess.run([sys.executable,'-m','scripts.dagger_rl_r4_train','restore'],stdout=(B/'smoke/restore.log').open('a'),stderr=subprocess.STDOUT).returncode
 assert code==0 and read(B/'smoke/RESTORE.json')['passed']
 assert read(B/'TEACHER_CLOSED_LOOP.json')['passed'],'native re-queried teacher sequence invalid'
 atomic_json(B/'TECHNICAL_SMOKE.json',dict(passed=True,finite=True,nonzero=n,fresh_process_restore=True,parity_environments=64,ability_head_update_l2=ability_update,ability_head_gradient_norm=float(np.asarray(loss)[...,-1].mean()),environment_steps=512,optimizer_steps=2,parity_diagnostic_environment_steps=1024,parity_diagnostic_optimizer_steps=4,total_executed_environment_steps=1536,total_executed_optimizer_steps=6,formal_inherits_smoke=False,capability_gate=False))
 notify(f'新DAgger/RL技术smoke通过：Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}；fresh恢复next-step/Adam/aggregate一致，立即正式训练。')

def restore():
 items,_=setup();x=pickle.load((B/'smoke/u001/state.pkl').open('rb'));e=read(B/'smoke/EXPECTED.json')
 assert digest(x['adam'])==e['adam'] and digest(x['history'])==e['history']
 actual=next_update_probe(x,merged(items,'states'),merged(items,'aux'))
 assert actual==e['next_update'],'fresh collector/memory/replay/optimizer continuation mismatch'
 atomic_json(B/'smoke/RESTORE.json',dict(passed=True,next_step_exact=True,optimizer_exact=True,aggregation_exact=True,next_full_update_exact=True,replay_exact=True,reservoir_seen_exact=True,parity_environments=64,full_saved_environments=64,continuation=actual))


def formal():
 global MEMORY_STRATEGY
 items,source=setup();pr=read(B/'PROTOCOL.json');opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(pr['lr'],eps=1e-8));updates={m:updater(m,opt) for m in ['joint','anchor']};collect=collector(128)
 control=B/'formal/no_update/u000';control.mkdir(parents=True,exist_ok=True)
 if not (control/'COMMITTED.json').exists():
  save(control/'state.pkl',source);save(control/'policy.pt',source['params']['actor']);atomic_json(control/'COMMITTED.json',dict(passed=True,update=0,no_update=True,state_sha256=sha(control/'state.pkl'),policy_sha256=sha(control/'policy.pt')))
 records=[]
 for seed in pr['paired_seeds']:
  for arm in pr['arms']:
   MEMORY_STRATEGY='fifo' if arm=='control' else 'reservoir'
   authorize();base=B/'formal'/f'{arm}_s{seed}';base.mkdir(parents=True,exist_ok=True)
   if (base/'COMPLETE.json').exists():records.append(read(base/'COMPLETE.json'));continue
   p=rep(source['params']);a=rep(source['adam']);st,ax,key,bs,ba=initial(seed,items);start=0;history=None
   last=sorted(base.glob('u*/COMMITTED.json'))
   if last:
    cp=last[-1].parent;assert sha(cp/'state.pkl')==read(cp/'COMMITTED.json')['state_sha256'];x=pickle.load((cp/'state.pkl').open('rb'));start=x['update'];p=rep(x['params']);a=rep(x['adam']);st=x['states'];ax=x['aux'];key=x['keys'];history=x['history']
   else:commit(base,0,p,a,st,ax,key,seed,arm)
   if start==0:
    assert digest(one(p))==digest(source['params']) and digest(one(a))==digest(source['adam'])
    atomic_json(base/'FRESH_PARENT.json',dict(passed=True,parent_input_sha256=sha(B/'inputs/parent.pkl'),params_exact=True,Adam_exact=True,smoke_not_inherited=True))
   began=time.time()
   for u in range(start+1,33):
    authorize();before=one(p);mode='anchor'
    (st,ax,key),batch,diag=collect(p,st,ax,key,bs,ba);history,rb,ag=aggregate(batch,history,seed,u)
    (p,a),loss=updates[mode](p,a,batch,rb,np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),u),8)));jax.block_until_ready(p)
    after=one(p);n=norms(before,after);assert n['actor']>0 and (n['critic']==0 if mode=='dagger' else n['critic']>0)
    assert np.isfinite(np.asarray(loss)).all() and all(np.isfinite(v).all() for v in jax.tree.leaves(after))
    fn=['carry','banana_inventory','ability2_legal','sample_release','greedy_release','release_probability_sum','self_contact','delivery','death','window_stock','free_stock','loose','transfer','self_defeat','empty_combat','empty_window']
    flags=np.asarray(diag[1]);od=np.asarray(diag[2]);item=np.asarray(batch[9]);carry=flags[...,0]>0
    row=dict(arm=arm,seed=seed,update=u,mode=mode,finite=True,update_l2=n,environment_steps=u*8192,optimizer_steps=u*32,aggregation=ag,elapsed_seconds=time.time()-began,effective_steps_this_process=(u-start)*8192,steps_per_second=(u-start)*8192/(time.time()-began),loss=np.asarray(loss).mean((0,1)).tolist(),ability_head_gradient_norm=float(np.asarray(loss)[...,-1].mean()),ability_head_update_l2=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,after['actor']['heads']['wb'],before['actor']['heads']['wb']))),events=dict(zip(fn,flags.sum((0,1,2)).tolist())),item_opportunities=dict(release=int(od[...,2].sum()),contact=int(od[...,3].sum()),mean_forecast_net=float(od[...,4][item].mean()) if item.any() else None),carry_inventory_ticks=int((carry&(flags[...,1]>0)).sum()),carry_release=int((carry&(flags[...,3]>0)).sum()),reward_components=np.asarray(diag[0]).sum((0,1,2)).tolist(),opponent_exposure=dict(current=0,ring=0,fixed=8192,kind='JAX native tier3'),GAE_mean=float(np.asarray(diag[-1]).mean()),GAE_variance=float(np.asarray(diag[-1]).var()))
    row['GAE_mean']=float(np.asarray(diag[5]).mean());row['GAE_variance']=float(np.asarray(diag[5]).var())
    row['anchor_KL']=float(np.asarray(loss)[...,7].mean());row['anchor_eligible_fraction']=float(np.asarray(loss)[...,8].mean())
    origins=np.asarray(diag[6]);row['source_exposure']={}
    for name,valid in [('natural',origins<0),('real_pickup_bank',origins>=0)]:
     row['source_exposure'][name]=dict(ticks=int(valid.sum()),events={k:float(flags[...,i][valid].sum()) for i,k in enumerate(fn)},item_release_labels=int(od[...,2][valid].sum()),item_contact_labels=int(od[...,3][valid].sum()))
    with (base/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    # Every update is atomic for a safe user stop; evaluator consumes declared points only.
    cp=commit(base,u,p,a,st,ax,key,seed,arm,history,row)
    atomic_json(B/'STATUS.json',dict(stage='FORMAL',**row))
    if u==1:notify(f'R4正式首更新 {arm}/s{seed} u1 {mode} finite：Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}，8192真实学生步，{ag["new_item_rows"]}道具老师行，原子SHA {sha(cp/"state.pkl")[:16]}。')
    if u==17:notify(f'R4 {arm}/s{seed} u17：{mode}，Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}，中段邻点完整保存（本轮无阶段切换）；未使用历史数据作PPO。')
   final=base/'u032';receipt=dict(arm=arm,seed=seed,update=32,finite=True,state_sha256=sha(final/'state.pkl'),policy_sha256=sha(final/'policy.pt'),environment_steps=262144,optimizer_steps=1024)
   atomic_json(base/'COMPLETE.json',receipt);records.append(receipt);notify(f'DAgger/RL固定run终态 {arm}/s{seed} u32完整原子提交；262144环境步/1024优化步，立即接续冻结队列，CPU不阻塞。')
 atomic_json(B/'TRAINING_COMPLETE.json',dict(passed=True,runs=records,environment_steps=1572864,optimizer_steps=6144,old_continuous_resume=False))

if __name__=='__main__':
 import sys
 globals()[sys.argv[1]]()
