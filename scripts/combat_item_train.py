"""Paired four-arm on-policy screen, exact parent/Adam and full-state atomic resume."""
import os,time,pickle,json
import numpy as np
import jax
import jax.numpy as j
import optax
from scripts.combat_item_run import B,R,read,save,notify,authorize
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest
from scripts.carrier_mechanism_analysis import bank,merged,shard
from scripts import combat_item_engine as T,carrier_mechanism_engine as E


def objective(p,b,combat_coefficient,forward,value_loss):
 o,g,act,old,ad,ret,mm,bm,route,combat=b
 mv,ab,_,vl=forward(p,'transformer',o,g)
 lm=jax.nn.log_softmax(j.where(mm,mv,-1e9));lb=jax.nn.log_softmax(j.where(bm,ab,-1e9))
 lp=lm[j.arange(len(old)),act[:,0]]+lb[j.arange(len(old)),act[:,1]]
 ratio=j.exp(lp-old);ppo=j.maximum(-ad*ratio,-ad*j.clip(ratio,.9,1.1)).mean()
 value=value_loss(vl,ret);entropy=-(j.sum(j.exp(lm)*lm,-1)+j.sum(j.exp(lb)*lb,-1)).mean()
 joint=lm[:,:,None]+lb[:,None,:]
 def ce(mask):
  valid=mask.any((1,2));val=-jax.nn.logsumexp(j.where(mask,joint,-1e9),(1,2))
  return j.where(valid,val,0).sum()/j.maximum(1,valid.sum())
 route_loss=ce(route);combat_loss=ce(combat)
 loss=ppo+.5*value-.005*entropy+.2*route_loss+combat_coefficient*combat_loss
 return loss,j.array([ppo,value,entropy,route_loss,combat_loss,route.any((1,2)).mean(),combat.any((1,2)).mean()])


def collector(arm,steps):
 def collect(p,st,ax,keys,bs,ba):
  def body(c,_):
   st,ax,key=c;nk,ke,act,data=E.choose(p,st,ax,key)
   o,g,mm,bm,own,lp,v=data;rt=T.route_teacher(st,ax);ct=T.combat_teacher(st,ax,key)
   after,na,r,done,info,parts=T.transition(st,ax,act,ke,arm)
   nv=E.prediction(p,after)[2]
   fresh,fa=T.fresh(jax.vmap(lambda k:jax.random.fold_in(k,0xF3))(nk),bs,ba)
   after=E.select_tree(done,fresh,after);na=E.select_tree(done,fa,na)
   native=~st.alive[:,1]|(st.trapped[:,1]>0);carry=st.bun_carried[:,0]==1
   stock=info['steal'][:,0];pickup=stock|info['pickup_loose'][:,0]|info['native_transfer'][:,0]
   inventory=j.any(st.item_slots[:,0]>0,-1)
   flags=j.stack([~carry&~native,~carry&native,carry,stock&native,stock&~native,info['pickup_loose'][:,0],info['native_transfer'][:,0],info['capture'][:,0],info['death'][:,0],info['own_bomb_defeat'][:,0],inventory,act[:,0,1]==2,info['trap_hit'][:,0],info['trap_hit'][:,1]],-1)
   return (after,na,nk),(o,g,own,lp,v,nv,r[:,0],done,mm,bm,rt,ct,parts[:,0],flags)
  end,d=jax.lax.scan(body,(st,ax,keys),None,length=steps)
  o,g,act,lp,v,nv,r,done,mm,bm,rt,ct,parts,flags=d
  adv=E.train.compute_gae(r,v,nv,done,.995,.95)
  return end,(o,g,act,lp,adv,adv+v,mm,bm,rt,ct),(parts,flags,r,v,adv)
 return jax.pmap(collect,in_axes=(0,0,0,0,None,None))


def updater(coefficient,opt):
 def update(p,a,b,key):
  b=tuple(x.reshape((-1,)+x.shape[2:]) for x in b);ad=b[4]
  mean=jax.lax.pmean(ad.mean(),'device');sq=jax.lax.pmean((ad**2).mean(),'device')
  ad=(ad-mean)/j.sqrt(j.maximum(sq-mean**2,0)+1e-8);b=b[:4]+(ad,)+b[5:]
  order=jax.random.permutation(key,len(ad)).reshape(-1,32)
  def body(c,ix):
   p,a=c;mb=tuple(x[ix] for x in b)
   (loss,metrics),grad=jax.value_and_grad(lambda q:objective(q,mb,coefficient,E.train.net_forward,E.train.hl_gauss_value_loss),has_aux=True)(p)
   grad=jax.lax.pmean(grad,'device');changes,a=opt.update(grad,a,p);p=optax.apply_updates(p,changes)
   return (p,a),jax.lax.pmean(j.concatenate([loss[None],metrics,optax.global_norm(grad)[None]]),'device')
  return jax.lax.scan(body,(p,a),order)
 return jax.pmap(update,axis_name='device')


def setup():
 authorize();T.configure();assert jax.default_backend()=='gpu' and jax.local_device_count()==8
 items=bank();items=[x for x in items if x['record']['split']=='train']
 assert len(items)==32
 for x in items:
  from scripts.carrier_mechanism_stats import validate_bank_record
  validate_bank_record(x['record'])
 source=pickle.load((B/'inputs/parent.pkl').open('rb'))
 return items,source

def rep(x):return jax.tree.map(lambda a:j.stack([j.asarray(a)]*8),x)
def one(x):return jax.tree.map(lambda a:np.asarray(a)[0],jax.device_get(x))
def norms(before,after):return {k:float(np.sqrt(sum(np.sum((np.asarray(x,dtype=float)-np.asarray(y,dtype=float))**2) for x,y in zip(jax.tree.leaves(before[k]),jax.tree.leaves(after[k]))))) for k in ['actor','critic']}

def initial(seed,items):
 keys=np.asarray(jax.random.split(jax.random.PRNGKey(seed),64));bs=merged(items,'states');ba=merged(items,'aux')
 st,ax=jax.jit(T.fresh)(keys,bs,ba)
 return shard(st,8),shard(ax,8),shard(keys,8),bs,ba

def commit(base,u,p,a,st,ax,key,seed,arm,metrics=None):
 cp=base/f'u{u:03d}';cp.mkdir(parents=True,exist_ok=True)
 save(cp/'state.pkl',dict(params=one(p),adam=one(a),states=jax.device_get(st),aux=jax.device_get(ax),keys=jax.device_get(key),update=u,seed=seed,arm=arm))
 save(cp/'policy.pt',one(p)['actor'])
 atomic_json(cp/'COMMITTED.json',dict(passed=True,update=u,arm=arm,seed=seed,state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt'),metrics=metrics,environment_steps=u*8192,optimizer_steps=u*32))
 atomic_json(B/'CPU_QUEUE'/f'{base.name}_u{u:03d}.json',dict(checkpoint=str(cp.relative_to(R)),update=u,final=u==32))
 return cp


def smoke():
 items,source=setup();st,ax,key,bs,ba=initial(2026102398,items);opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
 p=rep(source['params']);a=rep(source['adam']);c=collector('control',8)
 (st,ax,key),batch,diag=c(p,st,ax,key,bs,ba)
 # Frozen train-only batch audits magnitudes and masks; no holdout access.
 save(B/'smoke/FROZEN_TRAIN_BATCH.pkl',dict(batch=jax.device_get(batch),diagnostics=jax.device_get(diag)))
 (p,a),loss=updater(.05,opt)(p,a,batch,np.asarray(jax.random.split(jax.random.PRNGKey(8),8)));jax.block_until_ready(p)
 n=norms(source['params'],one(p));assert all(v>0 for v in n.values()) and np.isfinite(np.asarray(loss)).all()
 for x in jax.tree.leaves(one(p)):assert np.isfinite(x).all()
 cp=commit(B/'smoke',1,p,a,st,ax,key,2026102398,'technical')
 # Entire full-state snapshot, fresh interpreter parity without claiming capability.
 x=pickle.load((cp/'state.pkl').open('rb'));ss=one(st);aa=one(ax);kk=one(key)
 nk,ke,act,_=jax.jit(E.choose)(x['params'],ss,aa,kk);out=jax.jit(lambda s,a,ac,k:T.transition(s,a,ac,k,'control'))(ss,aa,act,ke)
 atomic_json(B/'smoke/EXPECTED.json',dict(next_hash=digest((nk,ke,act,out)),adam=digest(x['adam']),nonzero=n))
 import subprocess,sys
 code=subprocess.run([sys.executable,'-m','scripts.combat_item_train','restore'],stdout=(B/'smoke/restore.log').open('a'),stderr=subprocess.STDOUT).returncode
 assert code==0 and read(B/'smoke/RESTORE.json')['passed']
 atomic_json(B/'TECHNICAL_SMOKE.json',dict(passed=True,finite=True,nonzero=n,fresh_process_restore=True,environment_steps=512,optimizer_steps=2,capability_gate=False))
 notify(f'新实验技术smoke通过：512真实步，Actor Δ={n["actor"]:.6g}、Critic Δ={n["critic"]:.6g}，fresh全状态下一步一致；立即正式训练。')


def restore():
 setup();x=pickle.load((B/'smoke/u001/state.pkl').open('rb'));e=read(B/'smoke/EXPECTED.json')
 st=jax.tree.map(lambda a:np.asarray(a)[0],x['states']);ax=jax.tree.map(lambda a:np.asarray(a)[0],x['aux']);key=np.asarray(x['keys'])[0]
 nk,ke,act,_=jax.jit(E.choose)(x['params'],st,ax,key);out=jax.jit(lambda s,a,ac,k:T.transition(s,a,ac,k,'control'))(st,ax,act,ke)
 assert digest((nk,ke,act,out))==e['next_hash'] and digest(x['adam'])==e['adam']
 atomic_json(B/'smoke/RESTORE.json',dict(passed=True,env_aux_reward_rng_exact=True,optimizer_exact=True))


def formal():
 items,source=setup();protocol=read(B/'PROTOCOL.json');opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
 updates={arm:updater(.05 if arm=='combat' else 0.,opt) for arm in protocol['arms']}
 collects={arm:collector(arm,128) for arm in protocol['arms']}
 control=B/'formal/no_update/u000';control.mkdir(parents=True,exist_ok=True)
 if not (control/'COMMITTED.json').exists():
  save(control/'state.pkl',source);save(control/'policy.pt',source['params']['actor']);atomic_json(control/'COMMITTED.json',dict(passed=True,update=0,no_update=True,state_sha256=sha(control/'state.pkl'),policy_sha256=sha(control/'policy.pt')))
 records=[]
 for seed in protocol['paired_seeds']:
  for arm in protocol['arms']:
   authorize();base=B/'formal'/f'{arm}_s{seed}';base.mkdir(parents=True,exist_ok=True)
   if (base/'COMPLETE.json').exists():records.append(read(base/'COMPLETE.json'));continue
   p=rep(source['params']);a=rep(source['adam']);st,ax,key,bs,ba=initial(seed,items);start=0
   last=sorted(base.glob('u*/COMMITTED.json'))
   if last:
    cp=last[-1].parent;assert sha(cp/'state.pkl')==read(cp/'COMMITTED.json')['state_sha256'];x=pickle.load((cp/'state.pkl').open('rb'));start=x['update'];p=rep(x['params']);a=rep(x['adam']);st=x['states'];ax=x['aux'];key=x['keys']
    for family in ['banana','glue']:
     ax['tactics'].setdefault(family+'_frontier',np.full_like(ax['tactics'][family+'_start'],32767.))
   else:commit(base,0,p,a,st,ax,key,seed,arm)
   began=time.time()
   for u in range(start+1,33):
    authorize();before=one(p)
    (st,ax,key),batch,diag=collects[arm](p,st,ax,key,bs,ba)
    k=np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),u),8))
    (p,a),loss=updates[arm](p,a,batch,k);jax.block_until_ready(p);after=one(p);n=norms(before,after)
    assert all(v>0 for v in n.values()) and np.isfinite(np.asarray(loss)).all()
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(after))
    names=list(E.ops.COMPONENTS[:-1])+list(T.events.METRICS)+['combined_total']
    flag_names=['empty_combat','native_window','carry','window_stock','free_stock','loose','transfer','delivery','death','self_defeat','inventory','release_attempt','self_item_contact','enemy_item_contact']
    row=dict(arm=arm,seed=seed,update=u,finite=True,update_l2=n,environment_steps=u*8192,optimizer_steps=u*32,
      elapsed_seconds=time.time()-began,steps_per_second=u*8192/(time.time()-began),loss=np.asarray(loss).mean((0,1)).tolist(),
      reward_components=dict(zip(names,np.asarray(diag[0]).sum((0,1,2)).tolist())),
      events=dict(zip(flag_names,np.asarray(diag[1]).sum((0,1,2)).tolist())),
      route_teacher_rows=int(np.asarray(batch[-2]).any((-1,-2)).sum()),combat_teacher_rows=int(np.asarray(batch[-1]).any((-1,-2)).sum()),
      opponent_exposure=dict(current=0,ring=0,fixed=8192,fixed_kind='JAX native rule tier3; not Python or JS Hunter'),
      GAE_mean=float(np.asarray(diag[-1]).mean()),GAE_variance=float(np.asarray(diag[-1]).var()))
    with (base/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    atomic_json(B/'STATUS.json',dict(stage='FORMAL',**row))
    if u in protocol['checkpoints']:cp=commit(base,u,p,a,st,ax,key,seed,arm,row)
    if u==1:notify(f'正式首更新 {arm}/s{seed} u1：finite，Actor Δ={n["actor"]:.6g}，Critic Δ={n["critic"]:.6g}；8192真实on-policy步，{row["steps_per_second"]:.1f}步/秒（含首编译），原子保存。')
   final=base/'u032';receipt=dict(arm=arm,seed=seed,update=32,finite=True,state_sha256=sha(final/'state.pkl'),policy_sha256=sha(final/'policy.pt'),environment_steps=262144,optimizer_steps=1024)
   atomic_json(base/'COMPLETE.json',receipt);records.append(receipt)
   notify(f'固定run终态 {arm}/s{seed} u32已原子保存；按有界队列接续，不等待CPU评估。')
 atomic_json(B/'TRAINING_COMPLETE.json',dict(passed=True,runs=records,environment_steps=3145728,optimizer_steps=12288,old_continuous_resume=False))


if __name__=='__main__':
 import sys
 stage=sys.argv[1]
 if stage=='bank':
  from scripts.combat_item_bank import run
  run()
 else:globals()[stage]()
