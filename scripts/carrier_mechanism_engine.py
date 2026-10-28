"""Isolated real-acquisition diagnostic, frozen native JAX reward transitions."""
import os
os.environ.setdefault('JAXBOMB_RULE','bun')
import jax
import jax.numpy as jnp
import numpy as np
from jax_bomb import je07_bun_env as env,jax_train as train,bun_jax_bots
from scripts import je09_bun_scenes as scenes,je10_ops as ops
from scripts import je09_stage_h5c_dagger as dagger
from scripts.je03_bun_stage1_20261011 import split_forward

ORIGINAL_FORWARD=train.net_forward


def configure():
    env.prepare();env.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1)
    env._DANGER_TRACKING=True;scenes.bind_training();ops.install()
    from scripts.retention_phase_conditioning import install as phase_install
    phase_install()
    train.net_forward=lambda q,a,o,s=None:split_forward(q,a,o,s,ORIGINAL_FORWARD)


def initial(keys,layout):
    states=jax.vmap(lambda k:env.init_batch(k,1,jax.nn.one_hot(0,16)))(keys)
    states=jax.tree.map(lambda x:x[:,0],states)
    rr,cc=jnp.indices((13,15));remove=jnp.where(layout==1,True,jnp.where(layout==2,(rr+cc)%2==0,False))
    states=states._replace(core=states.core._replace(brick=states.brick&~remove[None]),scene=jnp.full(keys.shape[0],8,jnp.int32))
    return states,ops.reward_state(keys.shape[0])


def inputs(states):
    obs=train.both_perspectives(states)[:states.t.size]
    gv=train.both_states(states)[:states.t.size]
    mm,bm=train.both_masks(states);return obs,gv,mm[:states.t.size],bm[:states.t.size]


def prediction(params,states):
    o,g,mm,bm=inputs(states)
    mv,ab,v,vl=train.net_forward(params,'transformer',o,g)
    return mv,ab,v,vl,o,g,mm,bm


def choose(params,states,aux,keys):
    keys,ka,kb,ko,ke=jnp.swapaxes(jax.vmap(lambda k:jax.random.split(k,5))(keys),0,1)
    mv,ab,v,vl,o,g,mm,bm=prediction(params,states)
    bm=bm.at[:,1:].set(bm[:,1:]&~aux['terminal'][:,None])
    lm=jax.nn.log_softmax(jnp.where(mm,mv,-1e9));lb=jax.nn.log_softmax(jnp.where(bm,ab,-1e9))
    move=jax.vmap(jax.random.categorical)(ka,lm);ability=jax.vmap(jax.random.categorical)(kb,lb)
    own=jnp.stack([move,ability],-1);logp=lm[jnp.arange(len(keys)),move]+lb[jnp.arange(len(keys)),ability]
    allmm,allbm=jax.vmap(env.legal_mask)(states)
    enemybm=allbm[:,1].at[:,1:].set(allbm[:,1,1:]&~aux['terminal'][:,None])
    def rule(s,m,b,k):
        st=jax.tree.map(lambda x:x[None],s)
        return bun_jax_bots.rule_bot_actions(st,jnp.array([1]),m[None],b[None],k,jnp.array([3]))[0]
    enemy=jax.vmap(rule)(states,allmm[:,1],enemybm,ko)
    return keys,ke,jnp.stack([own,enemy],1),(o,g,mm,bm,own,logp,v)


def transition(states,aux,actions,envkeys):
    # This follows the production collector reward order, including residual settlement.
    from scripts.prestock_training_state import update
    aux=dict(aux,prestock=update(states,aux,aux['prestock']))
    after,natural_done,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(states,actions,envkeys)
    horizon=after.t>=scenes.HORIZONS[states.scene.astype(jnp.int32)]
    route_terminal=natural_done|info['capture'].any(-1)|horizon
    info=dict(info,h_route_terminal=route_terminal,blocked_own_route=(actions[:,:,1]==2)&(states.held_item==env.ITEM_SLOW_GLUE)&(states.bun_carried>=0),h_joint_actions=actions)
    settled=~(jnp.any(after.fuse>0,(1,2))|jnp.any(after.core.blast_linger>0,(1,2))|jnp.any(after.trapped>0,1))
    deferred=aux['terminal']|natural_done|horizon
    done=deferred&settled
    reward,new,parts,metrics=ops.combined_reward(states,after,info,aux,settled,1.,dict(window=120,scale=2.0))
    new['dagger']=dagger.update(states,after,info,new,aux['dagger'])
    new['dagger_course']=aux['dagger_course'];new['terminal']=deferred
    return after,new,reward,done,info,parts


def causal_pickup(before,after,info):
    return ((before.bun_carried[:,0]<0)&(after.bun_carried[:,0]==1)&before.alive[:,0]&after.alive[:,0]
        &info['steal'][:,0]&(~before.alive[:,1]|(before.trapped[:,1]>0)))


def select_tree(mask,new,old):
    return jax.tree.map(lambda a,b:jnp.where(mask.reshape(mask.shape+(1,)*(a.ndim-mask.ndim)),a,b),new,old)


def collection_chunk(params,carry,steps=128):
    def body(c,_):
        st,aux,key,ended,got,pre,post,preaux,postaux,prekey,postkey,action,ekey=c
        nk,ke,act,data=choose(params,st,aux,key)
        after,na,r,d,info,parts=transition(st,aux,act,ke)
        acquired=causal_pickup(st,after,info)&~got&~ended
        pre=select_tree(acquired,st,pre);post=select_tree(acquired,after,post)
        preaux=select_tree(acquired,aux,preaux);postaux=select_tree(acquired,na,postaux)
        prekey=select_tree(acquired,key,prekey);postkey=select_tree(acquired,nk,postkey)
        action=select_tree(acquired,act,action);ekey=select_tree(acquired,ke,ekey)
        got|=acquired
        st=select_tree(ended,st,after);aux=select_tree(ended,aux,na)
        key=select_tree(ended,key,nk)
        return (st,aux,key,ended|d,got,pre,post,preaux,postaux,prekey,postkey,action,ekey),(act,ke,acquired,r[:,0],d)
    return jax.lax.scan(body,carry,None,length=steps)


def rollout(params,states,aux,keys,steps,first_moves=None,training=False,ended=None):
    n=states.t.size
    def body(c,t):
        st,ax,key,ended=c
        nk,ke,actions,data=choose(params,st,ax,key)
        o,g,mm,bm,own,lp,v=data
        if first_moves is not None:
            forced=first_moves>=0
            actions=actions.at[:,0,0].set(jnp.where((t==0)&forced,first_moves,actions[:,0,0]))
            own=actions[:,0]
            # First branch is an intervention; its PPO logprob is recorded but never used as on-policy evidence.
        after,na,r,done,info,parts=transition(st,ax,actions,ke)
        cell=jnp.clip(st.pos[:,0].astype(jnp.int32),jnp.array([0,0]),jnp.array([12,14]))
        dist=aux_distance(st,ax)
        flags=jnp.stack([info['capture'][:,0],info['death'][:,0],info['drop'][:,0]|info['carrier_death'][:,0],st.bun_carried[:,0]==1,after.alive[:,0]],-1)
        transition_valid=~ended
        after=select_tree(ended,st,after);na=select_tree(ended,ax,na);nk=select_tree(ended,key,nk)
        rewards=jnp.where(ended,0.,r[:,0]);terminal=done|ended
        teacher=safe_teacher(st,ax) if training else jnp.zeros((n,5,3),bool)
        diagnostics=(rewards,v,terminal,flags,dist,own,lp,transition_valid,parts[:,0])
        data=(diagnostics,o,g,mm,bm,teacher) if training else diagnostics
        return (after,na,nk,ended|done),data
    end,data=jax.lax.scan(body,(states,aux,keys,jnp.zeros(n,bool) if ended is None else ended),jnp.arange(steps))
    return end,data


def aux_distance(states,aux):
    from scripts.je10_route_oracle import continuous_distance
    d=aux['dagger']['distance'][:,4]
    return continuous_distance(states.pos[:,0],d,states.wall|states.brick|states.core.pushable)


def safe_teacher(states,aux):
    """Existing set-valued return directions plus native joint safety, no action takeover."""
    from jax_bomb import bun_safety
    from scripts.je10_route_oracle import action_set,route_allowed
    o,g,mm,bm=inputs(states)
    distance=aux['dagger']['distance'][:,4];blocked=states.wall|states.brick|states.core.pushable
    target,valid=action_set(states.pos[:,0],distance,mm,o[:,5],blocked,states.fuse)
    safe=jax.vmap(bun_safety.analyze_actions)(states).survivable[:,0]
    joint=(target>0)[:,:,None]&safe&mm[:,:,None]&bm[:,None,:]
    enemy_distance=jnp.abs(states.pos[:,0]-states.pos[:,1]).sum(-1)
    allowed=route_allowed(jnp.ones(states.t.shape,bool),states.alive[:,1],states.trapped[:,1],enemy_distance)
    eligible=allowed&(states.bun_carried[:,0]==1)&states.alive[:,0]&(states.trapped[:,0]==0)&aux['dagger']['custody'][:,0]&~aux['terminal']
    return joint&eligible[:,None,None]
