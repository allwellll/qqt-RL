"""Candidate bounded native teacher, with explicit joint payload return.

Uses exactly the parent's stratified slot selection, deterministic one-step
forks and native temporal safety analysis. No labels or tracer values escape
through Python side effects. A future collector must explicitly split payload.
"""
import jax
import jax.numpy as jnp

JOINT_OFFSET=95
EXTRA_SIZE=19


def verified_labels(states,aux,legal,seat,danger,original_labels,slots):
    from scripts.je09_stage_h6_ops import select_verifier_slots
    from scripts.native_route_certificate import certify_native_positions
    from scripts.prestock_native_teacher import navigation,targets
    from scripts.prestock_joint_payload import merge
    from jax_bomb import bun_safety,je07_bun_env as env
    labels=original_labels(states,aux,legal,seat,danger)
    n=seat.size;rows=jnp.arange(n);kind=labels[:,6].astype(jnp.int32)
    chosen=jax.tree.map(lambda x:x[rows,seat],aux['prestock'])
    preparing=chosen['active']&~chosen['ready']&(kind==2)
    eligible=labels[:,7]>0
    cells=states.pos[rows,seat].astype(jnp.int32)
    score=jnp.mod(cells[:,0]*73856093+cells[:,1]*19349663+states.t*83492791+rows*506952113,2147483647)
    fragment=jnp.where((kind==3)&(labels[:,9]>0),4,jnp.where(kind==3,3,
        jnp.where(kind==2,2,jnp.where((kind==1)|(kind==4),1,0))))
    ids,selected_slots,selected=select_verifier_slots(eligible,fragment,score,slots)
    sample=jax.tree.map(lambda x:x[ids],states);who=seat[ids]
    count=4*slots;r=jnp.arange(count)
    safety=jax.vmap(bun_safety.analyze_actions)(sample).survivable[r,who]
    actions=jnp.zeros((count,5,2,2),jnp.int32).at[:,:,:,0].set(4)
    actions=actions.at[r[:,None],jnp.arange(5)[None],who[:,None],0].set(jnp.arange(5))
    repeated=jax.tree.map(lambda x:jnp.repeat(x,5,0),sample)
    nxt,_,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
        repeated,actions.reshape(count*5,2,2),jax.random.split(jax.random.PRNGKey(6106),count*5))
    nr=jnp.arange(count*5);ns=jnp.repeat(who,5)
    one_step=(nxt.alive[nr,ns]&(nxt.trapped[nr,ns]==0)&~info['own_bomb_defeat'][nr,ns]).reshape(count,5)
    native_kind=kind[ids]
    channel=jnp.where((native_kind==1)|(native_kind==4),who,jnp.where(native_kind==2,2+who,4+who))
    distance=aux['dagger']['distance'][ids,channel];blocked=aux['dagger']['blocked'][ids]
    before=sample.pos[r,who];after=nxt.pos[nr,ns].reshape(count,5,2)
    cell=jnp.clip(jnp.floor(before).astype(jnp.int32),jnp.array([0,0]),jnp.array([12,14]))
    goal=distance[r,cell[:,0],cell[:,1]]==0
    stable=jnp.all((nxt.wall|nxt.brick|nxt.core.pushable).reshape(count,5,13,15)==blocked[:,None],(-2,-1))
    legacy,_=certify_native_positions(before,after,distance,blocked,legal[ids],safety[:,:,0]&one_step,goal,stable)
    prep_chosen=jax.tree.map(lambda x:x[ids],chosen)
    prep_chosen['active']&=selected_slots&(native_kind==2)
    nav,nav_valid=navigation(sample,nxt,who,prep_chosen,legal[ids],safety&one_step[:,:,None])
    prep_joint,_=targets(sample,who,prep_chosen,nav,nav_valid,safety)
    _,all_ability=jax.vmap(env.legal_mask)(states)
    joint,_=merge(legacy,selected_slots,native_kind,legal[ids],all_ability[ids,who],preparing[ids],prep_joint)
    joint=jnp.zeros((n,5,3),bool).at[ids].max(joint&selected_slots[:,None,None])
    movement=joint.any(-1).astype(labels.dtype)
    movement/=jnp.maximum(movement.sum(-1,keepdims=True),1)
    accepted=selected&joint.any((-2,-1))
    updated=jnp.concatenate([movement,(labels[:,7]*accepted)[:,None],labels[:,6:8],
        selected[:,None].astype(labels.dtype),labels[:,9:]],-1)
    opening=joint[:,:,1].any(-1)&preparing
    extra=jnp.concatenate([joint.reshape(n,15).astype(labels.dtype),
        preparing[:,None],chosen['revision'][:,None],chosen['objective'][:,None],opening[:,None]],-1)
    return jax.lax.stop_gradient(jnp.concatenate([updated,extra],-1))
