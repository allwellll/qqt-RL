"""Proposal-only carrier threat memory; not installed in r03."""
import jax
import jax.numpy as jnp


def advance(previous,carrying,alive,trapped,enemy_alive,enemy_trapped,distance,
            enter_distance=4.,resume_distance=5.):
    if resume_distance<=enter_distance:raise ValueError('resume separation must exceed enter separation')
    available=alive&(trapped==0)
    controlled=~enemy_alive|(enemy_trapped>0)
    carrier=carrying&available
    threat=~controlled&((distance<=enter_distance)|(previous&(distance<=resume_distance)))
    latched=carrier&threat
    allowed=available&jnp.where(carrying,~latched,controlled)
    return jax.lax.stop_gradient(latched),jax.lax.stop_gradient(allowed)


def current_gate(states,aux):
    distance=jnp.abs(states.pos[:,0]-states.pos[:,1]).sum(-1)[:,None]
    previous=aux.get('carrier_phase_latched',jnp.zeros_like(states.alive))
    return advance(previous,states.bun_carried>=0,states.alive,states.trapped,
        states.alive[:,::-1],states.trapped[:,::-1],distance)[1]


def transition_gates(states,after,aux,newaux,terminal=None):
    """Pair PBRS endpoints using the same persisted, two-seat phase memory."""
    def endpoint(state,previous,values):
        distance=jnp.abs(state.pos[:,0]-state.pos[:,1]).sum(-1)[:,None]
        latched,allowed=advance(previous,state.bun_carried>=0,state.alive,
            state.trapped,state.alive[:,::-1],state.trapped[:,::-1],distance)
        return dict(values,carrier_phase_latched=latched,
            routing=dict(values['routing'],carrier_route_allowed=allowed))
    previous=aux.get('carrier_phase_latched',jnp.zeros_like(states.alive))
    before=endpoint(states,previous,aux)
    following=endpoint(after,before['carrier_phase_latched'],newaux)
    if terminal is not None:
        following['carrier_phase_latched'] &= ~terminal[:,None]
    return before,following
