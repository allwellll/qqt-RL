"""Small production PPO scale scan on new train/diagnostic seeds only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax,jax.numpy as jnp,numpy as np,optax
from scripts import je09_bun_ops as ops,je09_bun_scenes as scenes
from scripts.je09_bun_migration import NAMESPACE,SOURCE_CHECKPOINT,SOURCE_NAMESPACE
from scripts.population_run_state import atomic_json
from scripts.je03_bun_stage1_20261011 import split_forward,digest
BASE=Path(NAMESPACE)

def main():
 from jax_bomb import je07_bun_env as env,jax_train as train
 env.prepare();env.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1);env._DANGER_TRACKING=True;scenes.bind_training()
 original=train.net_forward;train.net_forward=lambda q,a,o,s=None:split_forward(q,a,o,s,original)
 with open(Path(SOURCE_CHECKPOINT)/'state.pkl','rb') as f:saved=pickle.load(f)
 params=jax.tree.map(lambda x:jnp.asarray(x[0]),saved['live']['params']);del saved
 with open(Path(SOURCE_NAMESPACE)/'inputs/frozen.pkl','rb') as f:fixed=pickle.load(f)
 frozen=jax.tree.map(jnp.asarray,dict(g0=fixed['g0'],stage1_teacher=fixed['stage1_teacher'],parent=params['actor']));del fixed
 pool=jax.tree.map(lambda x:jnp.stack([x]*11),params['actor']);course=ops.curriculum_weights(jnp.zeros((10,len(ops.FUNNEL))))
 # Full scene9 trajectory begins empty, receives native real drop, then pickup.
 key=jax.random.PRNGKey(2026101601);states=scenes.init_batch(key,8,jax.nn.one_hot(9,10))
 roll=jax.jit(lambda q,st,k,scale:ops.collect(q,st,ops.reward_state(8),k,pool,frozen,course,jnp.ones(11)/11,64,assist_probability=0.,expert_hazard_mode='local_safe',bridge_config=dict(window=120,scale=scale)))
 zero=roll(params,states,key,jnp.float32(0));jax.block_until_ready(zero);one=roll(params,states,key,jnp.float32(1));jax.block_until_ready(one)
 assert digest(zero[3][:4])==digest(one[3][:4]),'reward intervention changed inputs/actions/logprob'
 batch0=zero[3];delta=one[3][4]-batch0[4];signal=float(jnp.linalg.norm(delta));assert signal>0
 print('real collect GAE scale signal',signal,flush=True)
 opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5))
 learn=jax.jit(lambda batch:ops.ppo(params,opt.init(params),batch,key,params,opt,frozen,128,axis_name=None,expert_coefficient=.05,expert_temperature=.5))
 ck=jax.random.PRNGKey(2026101603);combat=scenes.init_batch(ck,8,jax.nn.one_hot(0,10));co=train.both_perspectives(combat)[:8];cg=train.both_states(combat)[:8];cm,cb=(x[:8] for x in train.both_masks(combat));rm,rb,_,_=train.net_forward(params,'transformer',co,cg)
 dk=jax.random.PRNGKey(2026101602);ds=scenes.init_batch(dk,8,jax.nn.one_hot(9,10));rows=[];models=[]
 for scale in [0.,.5,1.,2.]:
  batch=list(batch0);batch[4]=batch0[4]+scale*delta;batch[5]=batch0[5]+scale*delta
  learned=learn(tuple(batch));jax.block_until_ready(learned);q=learned[0];models.append(q)
  diag=roll(q,ds,dk,jnp.float32(1));jax.block_until_ready(diag)
  a,b,_,_=train.net_forward(q,'transformer',co,cg);kl=float(ops.factor_kl(a,b,rm,rb,cm,cb)[:,:2].sum(-1).mean())
  counts=np.asarray(diag[4][2]);get=lambda k:float(counts[ops.COMPONENTS.index(k)])
  row=dict(scale=scale,window=120,updates=8,combat_parent_kl=kl,diagnostic_pickups=get('bridge_pickup'),diagnostic_switches=get('bridge_switch'),diagnostic_bridge_gross=get('bridge_reward'),diagnostic_carry_net=get('carry_home_net'),finite=all(np.isfinite(x).all() for x in jax.tree.leaves(q)))
  rows.append(row);print(row,flush=True)
 candidates=[r for r in rows if r['scale']>0 and r['combat_parent_kl']<.1 and r['finite']]
 assert candidates
 selected=max(candidates,key=lambda r:(r['diagnostic_switches'],r['diagnostic_carry_net'],r['diagnostic_bridge_gross'],-r['scale']))
 diff=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,models[rows.index(selected)]['actor'],models[0]['actor'])))
 assert diff>0
 atomic_json(BASE/'validation/STAGE_B_SCAN.json',dict(passed=True,parent=SOURCE_CHECKPOINT,train_seed=2026101601,diagnostic_seed=2026101602,rows=rows,selected=selected,production_collect_gae_signal_l2=signal,actor_counterfactual_l2=diff,selection='train-only production PPO micro scan; maximize switches then carry net at combat KL<.1; not learner authority success',teacher_actions_injected=False,shape=[1,8,64,128]))
 decl=dict(experiment_id='JE09-STAGE-B-WINDOW-PFSP8-20261015-v6',seed=2026101611,round=1,stage_b_reward=dict(window=120,scale=selected['scale']),change='finite post-real-pickup milestone window120; scan only reward scale; exclude already-home bonus')
 atomic_json(BASE/'ROUND_DECLARATION.json',decl)
 print('selected',selected,flush=True)
if __name__=='__main__':main()
