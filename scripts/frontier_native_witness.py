"""Proposal-only deterministic native opening witnesses; no training hook.

Restricted to an initially clear bomb field and a native control window longer
than the 40-step witness. Enumerate five legal first moves with placement and
four straight escape continuations (wait when a continuation is blocked).
Only a witnessed safe, resolved, learner-owned required-brick opening with
positive topology gain and an empty learner throughout can label the initial
joint action. This prevents escape paths from acquiring inventory before the
preparation is complete. This finite search
is incomplete and is not a certificate against an active adversary or all RNG.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from scripts.je08_bun_potential import shortest

HORIZON=40


class OpeningWitness(NamedTuple):
    joint_set:object
    accepted:object
    safe:object
    resolved:object
    opened:object
    topology_gain:object
    first_moves:object
    escape_moves:object


def placement_witness(state,seat,goals,first_bricks):
    from jax_bomb import je07_bun_env as env
    count=20;rows=jnp.arange(count);enemy=1-seat
    moves=jnp.repeat(jnp.arange(5),4);escape=jnp.tile(jnp.arange(4),5)
    move_mask,ability_mask=env.legal_mask(state)
    legal=move_mask[seat,moves]&ability_mask[seat,1]
    window=((~state.alive[enemy])&(state.bun_respawn[enemy]>HORIZON))|(
        state.alive[enemy]&(state.trapped[enemy]>HORIZON))
    clear=(state.core.owner<0).all()&(state.fuse==0).all()
    clear&=(state.core.blast_linger==0).all()&(state.core.brick_linger==0).all()
    eligible=window&clear&state.alive[seat]&(state.trapped[seat]==0)&goals.any()
    eligible&=(first_bricks&state.brick&~state.wall).any()
    live=jax.tree.map(lambda x:jnp.broadcast_to(x,(count,)+x.shape),state)
    valid=legal&eligible
    opened=jnp.zeros(count,bool)
    owned=jnp.zeros((count,13,15),bool)
    keys=jax.random.split(jax.random.PRNGKey(6106),count)

    def step(carry,tick):
        states,safe,opened,owned,keys=carry
        mm,am=jax.vmap(env.legal_mask)(states)
        follow=jnp.where(mm[rows,seat,escape],escape,4)
        move=jnp.where(tick==0,moves,follow)
        ability=jnp.where(tick==0,1,0)
        joint=jnp.zeros((count,2,2),jnp.int32).at[:,:,0].set(4)
        joint=joint.at[rows,seat,0].set(move).at[rows,seat,1].set(ability)
        keys,used=jax.vmap(lambda k:jax.random.split(k))(keys).transpose(1,0,2)
        after,done,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
            states,joint,used)
        still_controlled=~after.alive[:,enemy]|(after.trapped[:,enemy]>0)
        safe&=mm[rows,seat,move]&am[rows,seat,ability]
        safe&=after.alive[:,seat]&(after.trapped[:,seat]==0)&~info['own_bomb_defeat'][:,seat]
        safe&=after.bun_carried[:,seat]<0
        safe&=still_controlled&~done
        owned|=(after.blast_owner_linger[:,seat]>0).any(1)
        removed=states.brick&~after.brick
        opened|=(removed&first_bricks[None]&owned).any((-2,-1))
        return (after,safe,opened,owned,keys),None

    (after,safe,opened,owned,keys),_=jax.lax.scan(step,(live,valid,opened,owned,keys),jnp.arange(HORIZON))
    resolved=(after.fuse==0).all((-2,-1))&(after.core.blast_linger==0).all((-2,-1))
    resolved&=(after.core.brick_linger==0).all((-2,-1))
    before_d=shortest(state.wall,state.brick,goals)
    after_d=shortest(after.wall,after.brick,jnp.broadcast_to(goals,(count,13,15)))
    cell=jnp.clip(jnp.floor(state.pos[seat]).astype(jnp.int32),jnp.array([0,0]),jnp.array([12,14]))
    topology_gain=after_d[:,cell[0],cell[1]]<before_d[cell[0],cell[1]]
    accepted=safe&resolved&opened&topology_gain
    joint=jnp.zeros((5,3),bool).at[moves,1].max(accepted)
    return jax.lax.stop_gradient(OpeningWitness(joint,accepted,safe,resolved,opened,topology_gain,moves,escape))
