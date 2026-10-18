"""Candidate first-deviation authority for joint pre-stock labels.

Ordinary fragments retain their existing quotas. A selected preparation
revision has one navigation and one opening quota per rollout episode chunk;
returning to either subphase cannot mint another query for that revision.
"""
import jax
import jax.numpy as jnp


def joint_mismatch(target,actions,accepted):
    pair=jnp.clip(actions[...,0],0,4)*3+jnp.clip(actions[...,1],0,2)
    chosen=jnp.take_along_axis(target.reshape(target.shape[:-2]+(15,)),pair[...,None],-1)[...,0]
    return accepted&target.any((-2,-1))&~chosen


def first_deviations(mismatch,fragment,terminal,preparing,revision,opening):
    def episode(m,frag,end,prep,rev,op):
        def step(carry,values):
            failed,last_revision,prep_failed=carry
            mismatch,fragment,terminal,preparing,revision,opening=values
            prep_failed=jnp.where(revision!=last_revision,jnp.zeros_like(prep_failed),prep_failed)
            idx=jnp.clip(fragment-1,0,3);sub=opening.astype(jnp.int32)
            special=preparing&(fragment==2)
            seen=jnp.where(special,prep_failed[sub],failed[idx])
            first=mismatch&(fragment>0)&~seen
            failed=failed.at[idx].set(failed[idx]|(mismatch&~special))
            prep_failed=prep_failed.at[sub].set(prep_failed[sub]|(mismatch&special))
            failed=jnp.where(terminal,jnp.zeros_like(failed),failed)
            prep_failed=jnp.where(terminal,jnp.zeros_like(prep_failed),prep_failed)
            last_revision=jnp.where(terminal,-1,revision)
            return (failed,last_revision,prep_failed),first
        initial=(jnp.zeros(4,bool),jnp.int32(-1),jnp.zeros(2,bool))
        return jax.lax.scan(step,initial,(m,frag,end,prep,rev,op))[1]
    return jax.vmap(episode,in_axes=(1,1,1,1,1,1),out_axes=1)(
        mismatch,fragment,terminal,preparing,revision,opening)
