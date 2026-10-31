"""R3 visible-input expansion, zero-initialized old Actor/Critic migration."""
import copy
import jax
import jax.numpy as j
import numpy as np
from scripts import retention_phase_conditioning as phase

EXTRA_GLOBAL=13
ENCODING='r3-items-route-chain-visible-v1'


def features(states,seats,batches=None):
    rows=j.arange(len(seats));enemy=1-seats
    inventory=j.minimum(states.item_counts[rows,seats].sum(-1)/50.,1.)[:,None]
    enemy_types=j.stack([((states.item_slots[rows,enemy]==kind)&(states.item_counts[rows,enemy]>0)).any(-1) for kind in (1,2)],-1)
    status=j.concatenate([jax.nn.one_hot(states.move_status[rows,p],4) for p in (seats,enemy)],-1)
    ticks=j.stack([states.status_ticks[rows,p]/100. for p in (seats,enemy)],-1)
    extra=j.concatenate([inventory,enemy_types.astype(j.float32),status,ticks],-1)
    assert extra.shape[-1]==EXTRA_GLOBAL
    if batches is None:plane=j.zeros((len(seats),13,15),j.float32)
    else:
        own=(states.fuse>0)&(states.core.owner==seats[:,None,None])
        ids=batches['bomb_batch'];minimum=j.min(j.where(own,ids,1<<27),(-2,-1))
        # Rank relative to oldest live batch: visible placement history only.
        plane=j.where(own,(ids-minimum[:,None,None]+1)/8.,0.)
    return extra,plane[:,None]


def expand_inputs(obs,gv,states,seats,batches=None):
    extra,plane=features(states,seats,batches)
    return j.concatenate([obs,plane],1),j.concatenate([gv,extra],-1)


def migrate(tree):
    """Expand parameters AND Adam moments; retain all historical counters."""
    if isinstance(tree,dict):
        out={k:migrate(v) for k,v in tree.items()}
        if 'tok' in tree:
            w,b=tree['tok']
            if w.shape[-2]==216:
                out['tok']=(j.concatenate([j.asarray(w),j.zeros(w.shape[:-2]+(9,w.shape[-1]),w.dtype)],-2),b)
            elif w.shape[-2]!=225:raise ValueError('unexpected token input schema')
        if 'state_w' in tree:
            w=tree['state_w']
            if w.shape[-2] in (24,29):out['state_w']=j.concatenate([j.asarray(w),j.zeros(w.shape[:-2]+(EXTRA_GLOBAL,w.shape[-1]),w.dtype)],-2)
            elif w.shape[-2] not in (37,42):raise ValueError('R3 expects raw24 or phase5 parent')
        return out
    if isinstance(tree,tuple):
        v=[migrate(x) for x in tree]
        return type(tree)(*v) if hasattr(tree,'_fields') else tuple(v)
    if isinstance(tree,list):return [migrate(x) for x in tree]
    return tree


def forward(params,obs,state=None):
    if params['tok'][0].shape[-2]==216:
        return phase.forward(params,obs[:,:24],state[:,:24])
    assert obs.shape[-3]==25 and state.shape[-1]==37
    encoded=state if params['state_w'].shape[-2]==37 else j.concatenate([state[:,:24],phase.features(obs[:,:24],state[:,:24]),state[:,24:]],-1)
    return phase._FORWARD(params,obs,encoded)


def install():
    from jax_bomb import jax_net
    jax_net.transformer_forward=forward
