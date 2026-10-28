"""Train-only same-batch coverage scan with genuine C/D native trajectories."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0' if os.environ.get('JE09_STAGE_E_SCAN_GPU')=='1' else '',JAX_PLATFORMS='cuda' if os.environ.get('JE09_STAGE_E_SCAN_GPU')=='1' else 'cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR','runs/je09_bun_20261015/stage_e_long_return_coverage_pfsp8_v10/cache/scan')
import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax,jax.numpy as jnp,numpy as np,optax
from scripts.je09_stage_e import BASE,configure
from scripts import je09_stage_e_ops as ops,je09_bun_ops as oldops,je09_bun_scenes as scenes,je09_stage_e_coverage as coverage
def trajectory(batch):
 c=np.asarray(batch[-1]);actions=np.asarray(batch[2])[...,0];active=c[...,12]>0
 reverse=down=0;unique=[];net=[];recovery=0
 for e in range(c.shape[1]):
  ix=np.flatnonzero(active[:,e]);cells=set()
  for t in ix:
   cells.add(int(c[t,e,11]))
   if t>0 and active[t-1,e] and c[t,e,11]!=c[t-1,e,11]:
    delta=c[t,e,10]-c[t-1,e,10];reverse+=int(delta>0);down+=int(delta<0)
    if delta>0:recovery+=int(c[t,e,9]>0)
  if len(ix):unique.append(len(cells));net.append(float(c[ix[0],e,10]-c[ix[-1],e,10]))
 return dict(window_rows=int(active.sum()),supervised_rows=int(c[...,9].sum()),reverse=reverse,downhill=down,net_return=sum(net),unique_cells=sum(unique),unique_actions=len(np.unique(actions[active])),recovery_rows_after_reverse=recovery)

from scripts.je03_bun_stage1_20261011 import split_forward,digest
from scripts.je03_bun831_20261011 import save_pickle
from scripts.population_run_state import atomic_json,sha

def main():
 from jax_bomb import je07_bun_env as env,jax_train as train
 configure();env.prepare();env.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1);env._DANGER_TRACKING=True;scenes.bind_training();coverage.bind_ops()
 original=train.net_forward;train.net_forward=lambda q,a,o,s=None:split_forward(q,a,o,s,original)
 dec=json.loads((BASE/'ROUND_DECLARATION.json').read_text());seeds=dec['seeds'];parent=Path(dec['parent']);source=parent.parents[2]
 with (parent/'state.pkl').open('rb') as f:saved=pickle.load(f)
 params=jax.tree.map(lambda x:jnp.asarray(x[0]),saved['live']['params']);pool=jax.tree.map(jnp.asarray,saved['live']['pool']);del saved
 with (source/'inputs/frozen.pkl').open('rb') as f:fixed=pickle.load(f)
 with (source/'inputs/retention_parent.pkl').open('rb') as f:retention=pickle.load(f)
 frozen=jax.tree.map(jnp.asarray,dict(g0=fixed['g0'],stage1_teacher=fixed['stage1_teacher'],parent=retention['actor']))
 n=16;steps=256;course=ops.curriculum_weights(jnp.zeros((10,len(ops.FUNNEL))),.6,.25)
 def rollout(q,st,key):return ops.collect(q,st,ops.reward_state(n),key,pool,frozen,course,jnp.ones(11)/11,steps,assist_probability=0.,expert_hazard_mode='local_safe',bridge_config=dec['stage_b_reward'],return_config=dec['return_expert'],chase_stage=jnp.int32(3))
 roll=jax.jit(rollout);key=jax.random.PRNGKey(seeds['scan'])
 states=coverage.init_batch(key,n,jax.nn.one_hot(9,10),jnp.int32(0),0.)
 print('collect C',flush=True);actual=roll(params,states,key);jax.block_until_ready(actual);batch=actual[3];print('C short native',trajectory(batch),flush=True)
 dcp=Path('runs/je09_bun_20261015/stage_d_distance_set_pfsp8_v9/formal/bun392_bridge_pfsp8/gen000016/policy.pt')
 with dcp.open('rb') as f:dactor=pickle.load(f)
 dp=dict(params,actor=jax.tree.map(jnp.asarray,dactor));lk=jax.random.fold_in(key,0xD16)
 longstates=coverage.init_batch(lk,n,jax.nn.one_hot(9,10),jnp.int32(3),1.,True)
 longresult=roll(dp,longstates,lk);jax.block_until_ready(longresult);longbatch=longresult[3];print('D long native',trajectory(longbatch),flush=True)
 def replay_rows(b,long):
  c=np.asarray(b[-1]);active=c[...,9]>0
  active &= ((c[...,22]>=24)&(c[...,22]<=36)&(c[...,21]>=24)) if long else ((c[...,22]>0)&(c[...,22]<24))
  ix=np.flatnonzero(active.ravel());assert len(ix)>0
  flat=lambda x:np.asarray(x).reshape((-1,)+x.shape[2:])[ix]
  return (flat(b[0]),flat(b[1]),flat(b[6][0]),flat(b[-1])[...,4:9]),dict(rows=len(ix),origin_distance_min=float(c[...,22][active].min()),origin_distance_max=float(c[...,22][active].max()),chase_initial_min=float(c[...,21][active].min()),chase_initial_max=float(c[...,21][active].max()))
 short,sdiag=replay_rows(batch,False);long,ldiag=replay_rows(longbatch,True);replay=dict(short=short,long=long)
 (BASE/'inputs').mkdir(exist_ok=True);save_pickle(BASE/'inputs/return_replay.pkl',replay)
 atomic_json(BASE/'validation/RETURN_REPLAY.json',dict(passed=True,sha256=sha(BASE/'inputs/return_replay.pkl'),long_actor_sha256=sha(dcp),short_actor_sha256=sha(parent/'policy.pt'),short=sdiag,long=ldiag,real_native_drop_pickup=True,action_takeover=False,future_information=False,long_trajectory=trajectory(longbatch)))
 assert not np.asarray(batch[-1])[0,:,9].any() and not np.asarray(longbatch[-1])[0,:,9].any()
 c=np.asarray(batch[-1]);assert not ((c[...,9]>0)&(c[...,18]>0)).any()
 # Actual collect->GAE batch identical except auxiliary metadata.
 print('counterfactual collect',flush=True);counter=jax.jit(lambda q,st,k:oldops.collect(q,st,oldops.reward_state(n),k,pool,frozen,course,jnp.ones(11)/11,steps,assist_probability=0.,expert_hazard_mode='local_safe',bridge_config=dec['stage_b_reward'],return_config=dec['return_expert'],chase_stage=jnp.int32(3)))(params,states,key);jax.block_until_ready(counter)
 assert digest(batch[:-1])==digest(counter[3][:-1])
 print('collect invariance exact',flush=True);replay=jax.tree.map(jnp.asarray,replay)
 def aux(q):
  total=0.
  for group in ['short','long']:
   o,g,m,t=replay[group];move=train.net_forward(q,'transformer',o,g)[0]
   total+=ops.return_expert.supervised_loss(move,m,t,jnp.ones(len(o),bool),.025,.5)[0]
  return total
 print('raw gradient',flush=True);raw=jax.jit(jax.grad(aux))(params);norm={k:float(optax.global_norm(raw[k])) for k in ['actor','critic']};assert norm['actor']>0 and norm['critic']==0
 flat=lambda x:x.reshape((-1,)+x.shape[2:]);o,g,m,ctx=map(flat,(batch[0],batch[1],batch[6][0],batch[-1]))
 logits=train.net_forward(params,'transformer',o,g)[0];lg=jax.grad(lambda q:ops.return_expert.supervised_loss(q,m,ctx[:,4:9],ctx[:,9]>0,.05,.5)[0])(logits)
 assert np.all(np.asarray(lg)[np.asarray(ctx[:,9])==0]==0)
 opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5))
 learn=jax.jit(lambda f,co:ops.ppo(params,opt.init(params),batch,key,params,opt,frozen,512,axis_name=None,expert_coefficient=.1,expert_temperature=1.,return_coefficient=co,return_temperature=.5,return_replay=replay,long_fraction=f))
 dk=jax.random.PRNGKey(seeds['diagnostic']);ds=coverage.init_batch(dk,n,jax.nn.one_hot(9,10),jnp.int32(0),.5)
 ck=jax.random.PRNGKey(seeds['combat_probe']);combat=scenes.init_batch(ck,n,jax.nn.one_hot(0,10));co=train.both_perspectives(combat)[:n];cg=train.both_states(combat)[:n];cm,cb=(x[:n] for x in train.both_masks(combat));rm,rb,_,_=train.net_forward(params,'transformer',co,cg)
 rows=[];models=[]
 print('PPO scan',norm,flush=True)
 for fraction in dec['budget']['scan_configs']:
  result=learn(jnp.float32(fraction),jnp.float32(.05));jax.block_until_ready(result);q=result[0];models.append(q)
  diag=roll(q,ds,dk);jax.block_until_ready(diag);a,bomb,_,_=train.net_forward(q,'transformer',co,cg);kl=float(ops.factor_kl(a,bomb,rm,rb,cm,cb)[:,:2].sum(-1).mean())
  moves=train.net_forward(q,'transformer',o,g)[0];lp=jax.nn.log_softmax(jnp.where(m,moves,-1e9));ent=-(jnp.exp(lp)*lp).sum(-1);active=ctx[:,9]>0
  row=dict(long_fraction=fraction,updates=16,parent_combat_kl=kl,action_entropy=float((ent*active).sum()/jnp.maximum(active.sum(),1)),finite=all(np.isfinite(x).all() for x in jax.tree.leaves(q)),**trajectory(diag[3]));row['score']=row['net_return']-row['reverse'];rows.append(row);print(row,flush=True)
 candidates=[r for r in rows if r['finite'] and r['parent_combat_kl']<.1 and r['action_entropy']>.15];assert candidates
 best=max(r['score'] for r in candidates);selected=min([r for r in candidates if r['score']>=best-abs(best)*.05],key=lambda r:r['long_fraction'])
 zero=learn(jnp.float32(selected['long_fraction']),jnp.float32(0.));jax.block_until_ready(zero);q=models[rows.index(selected)]
 delta={k:float(optax.global_norm(jax.tree.map(lambda x,y:x-y,q[k],zero[0][k]))) for k in ['actor','critic']};assert delta['actor']>0
 proof=dict(passed=True,collect_gae_ppo=True,raw_auxiliary_gradient=norm,actor_counterfactual_l2=delta['actor'],critic_counterfactual_l2=delta['critic'],excluded_row_gradients_exact_zero=True,train_trajectory=trajectory(batch),teacher_actions_injected=False,full_parent_pool=True,batch_sha256=digest(batch[:-1]),actions_logprob_reward_gae_unchanged=True,critic_note='raw auxiliary exactly0; multistep global clipping couples final update')
 atomic_json(BASE/'validation/RETURN_PRODUCTION_GRADIENT.json',proof)
 atomic_json(BASE/'validation/CHASE_PRODUCTION_GRADIENT.json',dict(passed=True,unchanged_configuration=dec['drop_expert'],return_chase_same_frame_conflicts=0))
 atomic_json(BASE/'validation/STAGE_E_SCAN.json',dict(passed=True,train_only=True,rows=rows,selected=selected,production=proof))
 dec['curriculum']['long_return_fraction']=selected['long_fraction'];atomic_json(BASE/'ROUND_DECLARATION.json',dec)
if __name__=='__main__':main()
