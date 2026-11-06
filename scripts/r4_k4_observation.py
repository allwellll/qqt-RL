"""Opt-in K4 mode ABI. Importing this module changes no shared forward path."""
import jax
import jax.numpy as j
import numpy as np

from scripts import r3_items_observation as parent

ENCODING='r4-k4-visible-mode01-v1'
PARENT_GLOBAL=37
MODE_GLOBAL=38


def append_mode(global_state,mode):
    global_state=j.asarray(global_state)
    if global_state.shape[-1]!=PARENT_GLOBAL:
        raise ValueError('K4 expects the frozen R3 global37 ABI')
    mode=j.broadcast_to(j.asarray(mode,global_state.dtype),global_state.shape[:-1])
    return j.concatenate([global_state,mode[...,None]],-1)


def migrate(tree):
    """Zero-extend Actor, Critic, ring/fixed weights and Adam moments only.

    Scalars and optimizer counters retain their original identities/values.
    Calling this twice is idempotent; unsupported state schemas are rejected.
    """
    if isinstance(tree,dict):
        result={k:migrate(v) for k,v in tree.items()}
        if 'state_w' in tree:
            weight=tree['state_w'];rows=weight.shape[-2]
            if rows in (37,42):
                zeros=np.zeros(weight.shape[:-2]+(1,weight.shape[-1]),weight.dtype)
                concat=np.concatenate if isinstance(weight,np.ndarray) else j.concatenate
                result['state_w']=concat([weight,zeros],-2)
            elif rows not in (38,43):
                raise ValueError('K4 mode migration requires R3 global37 or phase42')
        return result
    if isinstance(tree,tuple):
        values=[migrate(x) for x in tree]
        return type(tree)(*values) if hasattr(tree,'_fields') else tuple(values)
    if isinstance(tree,list):return [migrate(x) for x in tree]
    return tree


def forward(params,obs,state):
    """Mode0 reproduces the parent's raw inputs and existing phase ordering."""
    if obs.shape[-3:]!=(25,13,15) or state.shape[-1]!=MODE_GLOBAL:
        raise ValueError('K4 requires obs25/global38, including visible mode')
    rows=params['state_w'].shape[-2]
    if rows==38:encoded=state
    elif rows==43:
        encoded=j.concatenate([state[:,:24],parent.phase.features(obs[:,:24],state[:,:24]),
                               state[:,24:37],state[:,37:38]],-1)
    else:raise ValueError('K4 forward requires migrated38 or43 weights')
    return parent.phase._FORWARD(params,obs,encoded)


def actor_critic_forward(params,obs,state):
    move,ability,_,_=forward(params['actor'],obs,state)
    _,_,value,value_logits=forward(params['critic'],obs,state)
    return move,ability,value,value_logits


def bun_mode_zero_actor(actor):
    """Exact mode0 inference export: remove only the multiplied-zero row.

    The result has the existing R3 obs25/global37 ABI. This is a separate
    derived artifact, never a replacement for the full K4 checkpoint.
    """
    rows=actor['state_w'].shape[-2]
    if rows not in (38,43):raise ValueError('expected migrated K4 actor')
    return dict(actor,state_w=actor['state_w'][...,:-1,:])
