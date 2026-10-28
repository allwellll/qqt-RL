"""Train-only native expert validation, strength/temperature scan and PPO proof."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,pickle,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax,jax.numpy as jnp,numpy as np,optax
from scripts import je09_bun_ops as ops,je09_bun_scenes as scenes,je09_drop_expert as expert
from scripts.population_run_state import atomic_json
from scripts.je09_bun_migration import NAMESPACE, STAGE_B, STAGE_C
BASE=Path(NAMESPACE)

def dataset(seed,n=32):
    from jax_bomb import je07_bun_env as env,jax_train as train
    key=jax.random.PRNGKey(seed);states=scenes.init_batch(key,n,jax.nn.one_hot(9,10));aux=ops.reward_state(n)
    rows=jnp.arange(n);seat=rows%2;idx=rows+seat*n
    def tick(c,_):
        st,ax,key=c;key,k=jax.random.split(key)
        mm,bm=train.both_masks(st);label,active=expert.labels(st,ax,mm[idx],seat)
        obs=train.both_perspectives(st)[idx];gv=train.both_states(st)[idx]
        actions=jnp.tile(jnp.array([[[4,0],[4,0]]]),(n,1,1)).at[rows,seat,0].set(jnp.where(active,label,4))
        ns,done,info=jax.vmap(lambda s,a,kk:env.step(s,a,kk,False,True))(st,actions,jax.random.split(k,n))
        _,na,_,metrics=ops.combined_reward(st,ns,info,ax,jnp.ones(n,bool))
        return (ns,na,key),(obs,gv,mm[idx],label,active,metrics['drop_recovery'][rows,seat])
    _,data=jax.jit(lambda:jax.lax.scan(tick,(states,aux,key),None,length=96))()
    data=jax.device_get(data);obs,gv,mm,label,active,pickup=data
    success=pickup.any(0);keep=active&success[None]
    print('native expert',seed,int(success.sum()),'/',n,'labels',int(keep.sum()),flush=True)
    assert success.sum()>=n*.9,'expert not sufficiently successful; fix geometry before learning'
    return tuple(jnp.asarray(x[keep]) for x in (obs,gv,mm,label)),dict(seed=seed,games=n,first_pickup=int(success.sum()),labels=int(keep.sum()),ticks=96)

def main():
    from jax_bomb import je07_bun_env as env,jax_train as train
    env.prepare();env.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1);env._DANGER_TRACKING=True;scenes.bind_training()
    ds,proof=dataset(2026101541);val,vproof=dataset(2026101542,16)
    np.savez(BASE/'validation/TRAIN_EXPERT_DATA.npz',**{str(i):np.asarray(x) for i,x in enumerate(ds)})
    parent=Path('runs/je08_bun_20261014/h200_dp8_continuation_g1120_v1/formal/bun392_topology_potential_pfsp8/gen002048/policy.pt')
    with parent.open('rb') as f:p=jax.tree.map(jnp.asarray,pickle.load(f))
    # Fixed train-only combat replay protects the same parent actions in scan.
    st=scenes.init_batch(jax.random.PRNGKey(2026101561),32,jax.nn.one_hot(0,10))
    co=train.both_perspectives(st)[:32];cg=train.both_states(st)[:32];cm,cb=(x[:32] for x in train.both_masks(st))
    rm,rb,_,_=train.net_forward(p,'transformer',co,cg)
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5))
    def metrics(q,data):
        o,g,m,y=data;move,_,_,_=train.net_forward(q,'transformer',o,g)
        ce=-jax.nn.log_softmax(jnp.where(m,move,-1e9))[jnp.arange(len(y)),y].mean()
        a,b,_,_=train.net_forward(q,'transformer',co,cg)
        kl=ops.factor_kl(a,b,rm,rb,cm,cb)[:,:2].sum(-1).mean()
        acc=(jnp.argmax(jnp.where(m,move,-1e9),-1)==y).mean()
        return ce,kl,acc
    def update(q,a,strength,temp,key):
        ix=jax.random.randint(key,(64,),0,len(ds[3]));o,g,m,y=(x[ix] for x in ds)
        def loss(q):
            move,_,_,_=train.net_forward(q,'transformer',o,g)
            ce,_=expert.supervised_loss(move,m,y,jnp.ones(64,bool),strength,temp)
            a,b,_,_=train.net_forward(q,'transformer',co,cg)
            kl=ops.factor_kl(a,b,rm,rb,cm,cb)[:,:2].sum(-1).mean()
            return ce+.05*kl
        grad=jax.grad(loss)(q);u,a=opt.update(grad,a,q);return optax.apply_updates(q,u),a
    update=jax.jit(update);measure=jax.jit(metrics);rows=[]
    for strength,temp in [(0.,1.)]+[(s,t) for s in [.05,.2,1.] for t in [.5,1.]]:
        q=p;a=opt.init(p)
        for i in range(24):q,a=update(q,a,strength,temp,jax.random.PRNGKey(2026101570+i))
        ce,kl,acc=map(float,measure(q,val));row=dict(strength=strength,temperature=temp,diagnostic_ce=ce,combat_parent_kl=kl,diagnostic_accuracy=acc,updates=24);rows.append(row);print(row,flush=True)
    candidates=[x for x in rows if x['strength']>0 and x['combat_parent_kl']<.1]
    selected=min(candidates,key=lambda x:x['diagnostic_ce'])
    atomic_json(BASE/'validation/EXPERT_SCAN.json',dict(passed=True,train=proof,diagnostic=vproof,rows=rows,selected=selected,selection='lowest diagnostic CE with combat parent KL < .1; no authority/holdout seeds consumed',native_success_is_teacher_only=True))
    production(selected)

def production(selected):
    from jax_bomb import je07_bun_env as env,jax_train as train
    env.prepare();env.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1);env._DANGER_TRACKING=True;scenes.bind_training()
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5))
    # Production collect -> GAE -> actual PPO causal auxiliary Actor update.
    from jax_bomb import jax_net as net
    from scripts.je03_bun_stage1_20261011 import split_forward
    original=train.net_forward;train.net_forward=lambda q,a,o,s=None:split_forward(q,a,o,s,original)
    key=jax.random.PRNGKey(2035000001 if STAGE_C else 2026101662 if STAGE_B else 2026101552 if BASE.name.endswith('_v5') else 2026101562);actor=net.init_transformer(key,24,13,15,embed=16,depth=1,heads=4,state_dim=24)
    params=dict(actor=actor,critic=actor);frozen=dict(g0=actor,stage1_teacher=actor,parent=actor);pool=jax.tree.map(lambda x:jnp.stack([x]*11),actor)
    course=jax.nn.one_hot(9,10);st=scenes.init_batch(key,4,course)
    result=jax.jit(lambda:ops.collect(params,st,ops.reward_state(4),key,pool,frozen,course,jnp.ones(11)/11,4,assist_probability=0.,expert_hazard_mode='local_safe' if STAGE_B or BASE.name.endswith('_v5') else 'global'))()
    batch=result[3];assert float(batch[-1][...,3].sum())>0
    def learn(c):return ops.ppo(params,opt.init(params),batch,key,params,opt,frozen,8,axis_name=None,expert_coefficient=c,expert_temperature=selected['temperature'])
    a=jax.jit(learn)(selected['strength']);b=jax.jit(learn)(0.)
    delta={n:float(optax.global_norm(jax.tree.map(lambda x,y:x-y,a[0][n],b[0][n]))) for n in ['actor','critic']}
    flatten=lambda x:x.reshape((-1,)+x.shape[2:])
    obs,gv,mm,context=map(flatten,(batch[0],batch[1],batch[6][0],batch[-1]))
    def auxiliary(q):
        move,_,_,_=train.net_forward(q,'transformer',obs,gv)
        return expert.supervised_loss(move,mm,context[:,2].astype(jnp.int32),context[:,3]>0,selected['strength'],selected['temperature'])[0]
    grads=jax.grad(auxiliary)(params)
    raw={n:float(optax.global_norm(grads[n])) for n in ['actor','critic']}
    assert delta['actor']>0 and raw['actor']>0 and raw['critic']==0
    assert not bool(batch[-1][0,:,3].any()),'initial no-drop state must have no labels'
    atomic_json(BASE/'validation/EXPERT_PRODUCTION_GRADIENT.json',dict(passed=True,collect_gae_ppo=True,eligible_learner_rows=int(batch[-1][...,3].sum()),actor_counterfactual_l2=delta['actor'],critic_counterfactual_l2=delta['critic'],raw_auxiliary_gradient=raw,critic_update_note='global gradient clipping may couple final Adam scaling; raw auxiliary critic gradient must be exactly zero',coefficient=selected['strength'],temperature=selected['temperature']))
    print('production gradient',delta,flush=True)
if __name__=='__main__':
    if '--production-only' in sys.argv:production(dict(strength=.05,temperature=.5) if STAGE_C else json.loads((BASE/'validation/EXPERT_SCAN.json').read_text())['selected'])
    else:main()
