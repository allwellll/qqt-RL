"""JAX event ledger: stable episode/victim/life/entry IDs and causal ownership.

Source matrices are [batch, victim, actor]. Trigger provenance must never
replace first-control causal ownership. Gross credit is not refunded on loss.
"""
import jax
import jax.numpy as jnp

ENTRY_REWARD = 4.
RESIDUAL_REWARD = .25
EPISODE_CAP = 8.
METRICS = ('control_entry_reward', 'control_residual_reward', 'control_clawback',
           'control_entry_count', 'control_residual_count')


def state(n, players=2):
    shape = (n, players)
    return dict(episode=jnp.zeros(n,jnp.int32),life=jnp.zeros(shape,jnp.int32),
        event_id=jnp.zeros(shape,jnp.int32),owner=jnp.full(shape,-1,jnp.int32),
        last_tick=jnp.full(shape,-1,jnp.int32),residual_paid=jnp.zeros(shape,bool),
        gross=jnp.zeros(shape),balance=jnp.zeros(shape))


def reset(old, done):
    empty=state(old['life'].shape[0],old['life'].shape[1])
    empty['episode']=old['episode']+done.astype(jnp.int32)
    return jax.tree.map(lambda a,b:jnp.where(done.reshape(done.shape+(1,)*(a.ndim-1)),b,a),old,empty)


def transition(e,s):
    n,players=e['alive'].shape
    ids=jnp.arange(players)
    source=e['source']
    unique=source.sum(-1)==1
    proposed=jnp.argmax(source,-1)
    owner_team=jnp.take_along_axis(e['teams'],proposed,-1)
    hostile=owner_team!=e['teams']
    edge=e['alive']&~e['controlled']&e['next_controlled']
    edge &= e['tick'][:,None]>s['last_tick']
    edge &= ~e['terminal'][:,None]
    attributed=edge&unique&hostile
    owner=jnp.where(edge,jnp.where(attributed,proposed,-1),s['owner'])
    event_id=s['event_id']+edge.astype(jnp.int32)
    owner_valid=owner>=0
    owner_hot=jax.nn.one_hot(jnp.maximum(owner,0),players)*owner_valid[...,None]
    entry=(owner_hot*attributed[...,None]).sum(1)*ENTRY_REWARD
    # Only a real controlled->dead edge with an active hostile trigger gets
    # residual credit; passive timeout is not an active trigger event.
    trigger_unique=e['trigger'].sum(-1)==1
    trigger_owner=jnp.argmax(e['trigger'],-1)
    trigger_hostile=jnp.take_along_axis(e['teams'],trigger_owner,-1)!=e['teams']
    death=e['alive']&~e['next_alive']&e['death']
    residual=death&e['controlled']&owner_valid&~s['residual_paid']
    residual &= trigger_unique&trigger_hostile&~e['terminal'][:,None]
    residual_hot=jax.nn.one_hot(trigger_owner,players)*residual[...,None]
    extra=residual_hot.sum(1)*RESIDUAL_REWARD
    alive=e['next_alive']&~e['death']
    entry=jnp.where(alive,jnp.minimum(entry,jnp.maximum(0.,EPISODE_CAP-s['gross'])),0.)
    extra=jnp.where(alive,jnp.minimum(extra,jnp.maximum(0.,EPISODE_CAP-s['gross']-entry)),0.)
    clawback=jnp.where(e['death'],-s['balance'],0.)
    reward=entry+extra+clawback
    respawn=~e['alive']&e['next_alive']
    clear=death|respawn|(e['controlled']&~e['next_controlled'])
    new=dict(s,life=s['life']+respawn.astype(jnp.int32),event_id=event_id,
        owner=jnp.where(clear,-1,owner),last_tick=jnp.where(edge,e['tick'][:,None],s['last_tick']),
        residual_paid=jnp.where(edge,False,s['residual_paid'])|residual,
        gross=s['gross']+entry+extra,
        balance=jnp.where(e['death'],0.,s['balance']+entry+extra))
    metrics=dict(zip(METRICS,(entry,extra,clawback,
        (owner_hot*attributed[...,None]).sum(1),residual_hot.sum(1))))
    metrics.update(event_id=event_id,owner=owner,episode=s['episode'],life=s['life'])
    return new,reward,metrics


def native_events(before,after,info,terminal):
    n=before.alive.shape[0]
    # Native trap entry captures original explosion causal provenance before
    # a contact pop can overwrite the simulator's death-source attribution.
    source=jnp.where(after.trapped_causal.any(-1,keepdims=True),
                     after.trapped_causal,after.trapped_source)
    popper=info.get('native_trap_popper',jnp.full(before.alive.shape,-1,jnp.int32))
    contact=jax.nn.one_hot(jnp.maximum(popper,0),before.alive.shape[1],dtype=jnp.bool_)&(popper>=0)[...,None]
    trigger=info['trigger_damage_source']|contact
    return dict(tick=before.t,alive=before.alive,next_alive=after.alive,
        controlled=before.trapped>0,next_controlled=after.trapped>0,
        source=source,trigger=trigger,death=info['death'],
        terminal=terminal,teams=jnp.broadcast_to(jnp.arange(2),(n,2)))


def parent_reward_events(before,info):
    """Control death pays residual only; preserve original native event metrics.

    Uncontrolled direct combat kills keep the inherited settled reward. A native
    sugar death must not accumulate another legacy16 after entry credit.
    """
    controlled_death=(before.trapped[:,::-1]>0)&info['death'][:,::-1]
    return dict(info,surviving_kill=info['surviving_kill']&~controlled_death)
