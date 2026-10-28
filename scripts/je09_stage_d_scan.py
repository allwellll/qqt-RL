"""Same-batch train-only chase scan and production auxiliary attribution."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,pickle,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax,jax.numpy as jnp,numpy as np,optax
from scripts import je09_bun_ops as ops,je09_bun_scenes as scenes,je09_chase_expert as expert
from scripts.je09_bun_migration import NAMESPACE,SOURCE_CHECKPOINT,SOURCE_NAMESPACE
from scripts.population_run_state import atomic_json
from scripts.je03_bun_stage1_20261011 import split_forward,digest
BASE=Path(NAMESPACE)

def trajectory(batch):
    c=np.asarray(batch[-1]);active=c[...,18]>0;actions=np.asarray(batch[2])[...,0]
    reverse=down=net=unique=recovery=0
    for e in range(c.shape[1]):
        ix=np.flatnonzero(active[:,e]);cells=set()
        for t in ix:
            cells.add(int(c[t,e,11]))
            if t>0 and active[t-1,e] and c[t,e,11]!=c[t-1,e,11]:
                delta=c[t,e,19]-c[t-1,e,19];reverse+=int(delta>0);down+=int(delta<0)
                recovery+=int(delta>0)
        if len(ix):net+=float(c[ix[0],e,19]-c[ix[-1],e,19]);unique+=len(cells)
    return dict(supervised_rows=int(active.sum()),reverse=reverse,downhill=down,net_pursuit=net,unique_cells=unique,unique_actions=len(np.unique(actions[active])),recovery_rows_after_reverse=recovery)

def main():
    from jax_bomb import je07_bun_env as env,jax_train as train
    env.prepare();env.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1);env._DANGER_TRACKING=True;scenes.bind_training()
    original=train.net_forward;train.net_forward=lambda q,a,o,s=None:split_forward(q,a,o,s,original)
    dec=json.loads((BASE/'ROUND_DECLARATION.json').read_text());seeds=dec['seeds']
    with open(Path(SOURCE_CHECKPOINT)/'state.pkl','rb') as f:saved=pickle.load(f)
    params=jax.tree.map(lambda x:jnp.asarray(x[0]),saved['live']['params']);pool=jax.tree.map(jnp.asarray,saved['live']['pool']);del saved
    with open(Path(SOURCE_NAMESPACE)/'inputs/frozen.pkl','rb') as f:fixed=pickle.load(f)
    with open(Path(SOURCE_NAMESPACE)/'inputs/retention_parent.pkl','rb') as f:parent=pickle.load(f)
    frozen=jax.tree.map(jnp.asarray,dict(g0=fixed['g0'],stage1_teacher=fixed['stage1_teacher'],parent=parent['actor']))
    course=ops.curriculum_weights(jnp.zeros((10,len(ops.FUNNEL))),.60,.25)
    def rollout(q,st,key,enabled=True):return ops.collect(q,st,ops.reward_state(8),key,pool,frozen,course,jnp.ones(11)/11,128,assist_probability=0.,expert_hazard_mode='local_safe',bridge_config=dec['stage_b_reward'],return_config=dec['return_expert'],chase_stage=jnp.int32(0),chase_supervision=enabled)
    roll=jax.jit(rollout);key=jax.random.PRNGKey(seeds['scan']);states=scenes.init_batch(key,8,jax.nn.one_hot(9,10),jnp.int32(0))
    actual=roll(params,states,key);jax.block_until_ready(actual);batch=actual[3];print('actual',trajectory(batch),flush=True)
    counter=roll(params,states,key,False);jax.block_until_ready(counter)
    assert digest(batch[:-1])==digest(counter[3][:-1]),'labels modified observations/actions/logprob/reward/GAE'
    atomic_json(BASE/'validation/CHASE_INVARIANCE.json',dict(passed=True,batch_sha256=digest(batch[:-1]),actions_logprob_reward_gae_unchanged=True))
    if '--invariance-only' in sys.argv:return
    c=np.asarray(batch[-1]);assert c[...,18].sum()>0 and not c[0,:,18].any()
    flat=lambda x:x.reshape((-1,)+x.shape[2:]);o,g,m,ctx=map(flat,(batch[0],batch[1],batch[6][0],batch[-1]))
    def auxloss(q):
        move=train.net_forward(q,'transformer',o,g)[0]
        return expert.supervised_loss(move,m,ctx[:,13:18],ctx[:,18]>0,.1,.5)[0]
    raw=jax.grad(auxloss)(params);norm={k:float(optax.global_norm(raw[k])) for k in ['actor','critic']}
    assert norm['actor']>0 and norm['critic']==0
    logits=train.net_forward(params,'transformer',o,g)[0]
    lg=jax.grad(lambda q:expert.supervised_loss(q,m,ctx[:,13:18],ctx[:,18]>0,.1,.5)[0])(logits)
    assert np.all(np.asarray(lg)[np.asarray(ctx[:,18])==0]==0)
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5))
    learn=jax.jit(lambda strength,temp:ops.ppo(params,opt.init(params),batch,key,params,opt,frozen,128,axis_name=None,expert_coefficient=strength,expert_temperature=temp,return_coefficient=.05,return_temperature=.5))
    dk=jax.random.PRNGKey(seeds['diagnostic']);ds=scenes.init_batch(dk,8,jax.nn.one_hot(9,10),jnp.int32(0))
    ck=jax.random.PRNGKey(seeds['combat_probe']);combat=scenes.init_batch(ck,8,jax.nn.one_hot(0,10));co=train.both_perspectives(combat)[:8];cg=train.both_states(combat)[:8];cm,cb=(x[:8] for x in train.both_masks(combat));rm,rb,_,_=train.net_forward(params,'transformer',co,cg)
    rows=[];models=[]
    for strength,temp in dec['budget']['scan_configs']:
        learned=learn(jnp.float32(strength),jnp.float32(temp));jax.block_until_ready(learned);q=learned[0];models.append(q)
        diag=roll(q,ds,dk);jax.block_until_ready(diag)
        a,b,_,_=train.net_forward(q,'transformer',co,cg);kl=float(ops.factor_kl(a,b,rm,rb,cm,cb)[:,:2].sum(-1).mean())
        moves=train.net_forward(q,'transformer',o,g)[0];lp=jax.nn.log_softmax(jnp.where(m,moves,-1e9));ent=-(jnp.exp(lp)*lp).sum(-1);active=ctx[:,18]>0
        row=dict(coefficient=strength,temperature=temp,updates=16,parent_combat_kl=kl,action_entropy=float((ent*active).sum()/jnp.maximum(active.sum(),1)),finite=all(np.isfinite(x).all() for x in jax.tree.leaves(q)),**trajectory(diag[3]));rows.append(row);print(row,flush=True)
    candidates=[r for r in rows if r['coefficient']>0 and r['parent_combat_kl']<.1 and r['action_entropy']>.15 and r['finite']]
    assert candidates;best=max(r['net_pursuit'] for r in candidates);near=[r for r in candidates if r['net_pursuit']>=best-abs(best)*.05]
    selected=min(near,key=lambda r:(r['coefficient'],r['reverse'],r['temperature']))
    delta={n:float(optax.global_norm(jax.tree.map(lambda x,y:x-y,models[rows.index(selected)][n],models[0][n]))) for n in ['actor','critic']};assert delta['actor']>0
    proof=dict(passed=True,collect_gae_ppo=True,raw_auxiliary_gradient=norm,excluded_row_gradients_exact_zero=True,actor_counterfactual_l2=delta['actor'],critic_counterfactual_l2=delta['critic'],critic_note='raw auxiliary gradient zero; shared clipping/multistep Adam may couple final updates',train_trajectory=trajectory(batch),teacher_actions_injected=False,batch_sha256=digest(batch[:-1]),actions_logprob_reward_gae_unchanged=True,full_parent_pool=True)
    atomic_json(BASE/'validation/CHASE_PRODUCTION_GRADIENT.json',proof)
    atomic_json(BASE/'validation/STAGE_D_SCAN.json',dict(passed=True,rows=rows,selected=selected,production=proof,train_only=True))
    dec['drop_expert'].update(coefficient=selected['coefficient'],temperature=selected['temperature']);atomic_json(BASE/'ROUND_DECLARATION.json',dec)

if __name__=='__main__':main()
