"""Read-only native-action opponents for learner-visited DAgger sampling."""
import jax
import jax.numpy as jnp
from scripts.je09_return_expert import action_set


def level(generation):return jnp.minimum(3,(jnp.asarray(generation)-1)//8)
def weak_fraction(generation):
    if isinstance(generation,int):return [.5,.35,.2,.1][min(3,(generation-1)//8)]
    return jnp.array([.5,.35,.2,.1])[level(generation)]


def roles(scene,uniform,generation):
    weak=weak_fraction(generation)
    return jnp.where(scene==8,jnp.where(uniform<weak/2,1,jnp.where(uniform<weak,2,0)),0).astype(jnp.int32)


def targeted_roles(scene,role_uniform,target_uniform,generation):
    """Assign weak native opponents independently across four lesson targets."""
    weak=weak_fraction(generation)
    is_weak=(scene==8)&(role_uniform<weak)
    target=jnp.minimum((target_uniform*4).astype(jnp.int32)+1,4)
    courier=(target==1)|(target==3)
    role=jnp.where(is_weak,jnp.where(courier,1,2),0).astype(jnp.int32)
    return role,jnp.where(is_weak,target,0).astype(jnp.int32)


def acquisition_kind(origin,learner):
    # An enemy stealing OUR stock carries our-origin bun. Recovering it is
    # valid A practice, but cannot be counted as enemy acquisition/capture.
    return jnp.where(origin==learner,4,1)


def actions(states,aux,opponent,proposal,legal,ability,key,role,generation,danger):
    from jax_bomb import je07_bun_env as env
    from scripts import je09_bun_scenes as scenes
    n=opponent.size;r=jnp.arange(n);tier=level(generation)
    carrying=states.bun_carried[r,opponent]>=0
    # Courier A walks into the learner cage itself before slow carrying.
    # Guard B stays at its own cage. Both use only native move/ability actions.
    channel=jnp.where(role==1,2+opponent,4+opponent)
    distance=aux['dagger']['distance'][r,channel]
    target,reachable=action_set(states.pos[r,opponent],distance,legal,
        jnp.zeros_like(danger),states.wall|states.brick|states.core.pushable,states.fuse)
    directed=jnp.argmax(target,-1)
    cell=jnp.clip(states.pos[r,opponent].astype(jnp.int32),jnp.array([0,0]),jnp.array([12,14]))
    neighbors=jnp.clip(cell[:,None]+jnp.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]]),jnp.array([0,0]),jnp.array([12,14]))
    goal_team=jnp.where(role==1,1-opponent,opponent)
    variant=scenes.OPEN_SCENE[states.scene].astype(jnp.int32)
    static=scenes.DISTANCE[variant[:,None],goal_team[:,None],neighbors[...,0],neighbors[...,1]]
    fallback=jnp.argmin(jnp.where(legal,static,1e8),-1)
    move=jnp.where(reachable,directed,fallback)
    at_base=jnp.stack([env._base_mask(0),env._base_mask(1)])[opponent,cell[:,0],cell[:,1]]
    random=jax.random.uniform(key,(n,3))
    slow=jnp.array([0.,.15,.35,.6])[tier]
    guard=jnp.array([0.,.1,.25,.5])[tier]
    duty=jnp.where(role==1,jnp.where(carrying,slow,1.),jnp.where(at_base,guard,1.))
    move=jnp.where(random[:,0]<duty,move,4)
    # Later tiers increasingly use the existing native bot escape response.
    hazard=jnp.any(danger>0,(-2,-1))|jnp.any(states.fuse>0,(-2,-1))
    escape=hazard&(random[:,1]<jnp.array([0.,.25,.5,1.])[tier])
    move=jnp.where(escape,proposal[:,0],move)
    bombcount=((states.core.owner==opponent[:,None,None])&(states.fuse>0)).sum((-2,-1))
    capacity=jnp.array([1,1,2,2])[tier]
    period=jnp.array([64,32,16,8])[tier]
    # A's approach may dig with the existing safe bot proposal; no fabricated
    # openings, bun ownership, movement speed, death, or drop state changes.
    period=jnp.where((role==1)&~carrying,4,period)
    use=(proposal[:,1]==1)&ability[:,1]&(bombcount<capacity)&((states.t+r)%period==0)
    result=jnp.stack([jnp.where(legal[r,move],move,4),use.astype(jnp.int32)],-1)
    return jnp.where((role>0)[:,None],result,proposal)
