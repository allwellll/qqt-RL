"""Opt-in Critic-only reward history and delivery prediction interventions.

Importing this module does not install hooks or alter any existing run. Actor
observations/parameters always follow the original deployment-visible path.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as j
from jax_bomb import jax_net as net
from scripts import retention_phase_conditioning as phase

ARMS=('A0_baseline','A1_ledger_input','A2_ledger_plus_delivery_head')
DELIVERY_COEFFICIENT=.1
LEDGER_FIELDS=('transport_balance','transport_pending_capture','transport_episode_gross',
               'route_phi','route_offset','route_phase_none','route_phase_stock',
               'route_phase_drop','route_phase_carry','route_phase_delivery',
               'control_balance','control_gross','paid','shape_paid','terminal')
ENCODING='critic-reward-ledger15-v1'


def ledger_features(aux,seat=0):
    """Read current reward history only; missing fields fail rather than fill.

    All values belong to the transition *before* acting. No future outcome,
    return target, teacher action, or RNG enters this vector.
    """
    if seat not in (0,1):raise ValueError('Invalid learner seat')
    def scalar(root,key,scale):
        return j.arcsinh(j.asarray(aux[root][key][...,seat],j.float32)/scale)
    transport=[scalar('transport',k,24.) for k in ('balance','pending_capture','episode_gross')]
    route=[scalar('route_pbrs',k,8.) for k in ('phi','offset')]
    route_phase=jax.nn.one_hot(aux['route_pbrs']['phase'][...,seat],5)
    control=[scalar('control_credit',k,8.) for k in ('balance','gross')]
    paid=[j.arcsinh(j.asarray(aux[k][...,seat],j.float32)/24.) for k in ('paid','shape_paid')]
    return j.concatenate((j.stack(transport+route,-1),route_phase,
        j.stack(control+paid+[j.asarray(aux['terminal'],j.float32)],-1)),-1)


def migrate(params,arm):
    """Preserve all existing leaves; add only zero-initialized Critic leaves."""
    if arm not in ARMS:raise ValueError('Unknown arm')
    if set(params)!= {'actor','critic'}:raise ValueError('Independent Actor/Critic required')
    if 'ledger_input_w' in params['critic'] or 'delivery_head' in params['critic']:
        raise ValueError('Already migrated; never migrate a candidate twice')
    result=dict(actor=params['actor'],critic=dict(params['critic']))
    if arm==ARMS[0]:return result
    weight=params['critic']['state_w']
    if weight.ndim!=2 or weight.shape[0] not in (24,29):raise ValueError('Expected native/visible-phase Critic')
    result['critic']['ledger_input_w']=j.zeros((len(LEDGER_FIELDS),weight.shape[-1]),weight.dtype)
    if arm==ARMS[2]:
        result['critic']['delivery_head']=(j.zeros((weight.shape[-1],1),weight.dtype),j.zeros((1,),weight.dtype))
    return result


def actor_state(params,obs,state):
    return phase.conditioned_state(params['actor'],obs,state)


def critic_pool(params,obs,state,ledger):
    critic=params['critic']
    encoded=phase.conditioned_state(critic,obs,state)
    merged=critic
    if 'ledger_input_w' in critic:
        if ledger is None:raise ValueError('Ledger-augmented Critic requires current ledger')
        if ledger.shape!=state.shape[:-1]+(len(LEDGER_FIELDS),):raise ValueError('Critic ledger shape mismatch')
        # Equivalent to appending rows to the state projection, while keeping
        # old parameter/moment leaves byte-identical for unambiguous migration.
        merged=dict(critic,state_w=j.concatenate((critic['state_w'],critic['ledger_input_w']),0))
        encoded=j.concatenate((encoded,ledger),-1)
    return net._transformer_pool(merged,j.asarray(obs),encoded)


def forward(params,obs,state,ledger):
    move,ability,_,_=phase.forward(params['actor'],obs,state)
    features=critic_pool(params,obs,state,ledger)
    value,logits=net._value_head(features,*params['critic']['heads']['wv'])
    return move,ability,value,logits


def delivery_logits(params,obs,state,ledger):
    if 'delivery_head' not in params['critic']:raise ValueError('A2 delivery head absent')
    features=critic_pool(params,obs,state,ledger)
    w,b=params['critic']['delivery_head']
    return (features@w+b)[...,0]


def delivery_loss(logits,targets,mask):
    mask=j.asarray(mask,bool);labels=j.where(mask,j.asarray(targets),0.)
    per_row=j.logaddexp(0.,logits)-labels*logits
    return j.sum(j.where(mask,per_row,0.))/j.maximum(mask.sum(),1)


def ppo_objective(params,batch):
    """Same pure-PPO math in all arms; A2 adds Critic-only BCE.

    Caller supplies normalized on-policy advantages and same-custody targets.
    Ledger features are captured before the action, not recomputed from the
    current post-rollout environment. This function never reads a teacher.
    """
    from jax_bomb import jax_train as train
    move,ability,_,value_logits=forward(params,batch['obs'],batch['state'],batch['ledger'])
    lm=jax.nn.log_softmax(j.where(batch['move_mask'],move,-1e9))
    la=jax.nn.log_softmax(j.where(batch['ability_mask'],ability,-1e9))
    row=j.arange(len(move));act=batch['actions']
    lp=lm[row,act[:,0]]+la[row,act[:,1]]
    ratio=j.exp(lp-batch['old_logp']);adv=batch['advantage']
    ppo=j.maximum(-adv*ratio,-adv*j.clip(ratio,.9,1.1)).mean()
    value=train.hl_gauss_value_loss(value_logits,batch['returns'])
    entropy=-(j.sum(j.exp(lm)*lm,-1)+j.sum(j.exp(la)*la,-1)).mean()
    bce=j.array(0.,j.float32)
    if 'delivery_head' in params['critic']:
        # The collector must explicitly record pre-action carrying and known
        # outcome masks. An empty-handed row can never receive auxiliary loss.
        mask=batch['carrying']&batch['delivery_known']
        bce=delivery_loss(delivery_logits(params,batch['obs'],batch['state'],batch['ledger']),
                          batch['delivery_target'],mask)
    loss=ppo+.5*value-.005*entropy+DELIVERY_COEFFICIENT*bce
    return loss,dict(ppo=ppo,value=value,entropy=entropy,delivery_bce=bce,
        approx_kl=(batch['old_logp']-lp).mean(),clip_fraction=(j.abs(ratio-1)>.1).mean())


def custody_targets(ids,carrying,delivered,lost,terminal):
    """Resolve only the same custody observed in this on-policy rollout.

    Inputs have time-first shape (T, ...); custody IDs must be stable and unique
    within each environment stream. Unresolved tails remain masked, including
    rollout truncation. Transfer/drop followed by recovery requires a new ID.
    No stale policy observations are reintroduced into PPO.
    """
    arrays=[np.asarray(a) for a in (ids,carrying,delivered,lost,terminal)]
    ids,carrying,delivered,lost,terminal=arrays
    if not ids.ndim or any(a.shape!=ids.shape for a in arrays):raise ValueError('Invalid custody timeline')
    original=ids.shape
    ids,carrying,delivered,lost,terminal=[a.reshape(len(a),-1) for a in arrays]
    targets=np.zeros(ids.shape,np.float32);mask=np.zeros(ids.shape,bool)
    for column in range(ids.shape[1]):
        resolved={};closed=set()
        for t in range(len(ids)):
            identity=int(ids[t,column])
            if not carrying[t,column] or identity<0:continue
            if identity in closed:raise ValueError('Reused closed custody ID')
            if delivered[t,column] or lost[t,column] or terminal[t,column]:
                resolved[identity]=float(bool(delivered[t,column]));closed.add(identity)
        for t in range(len(ids)):
            identity=int(ids[t,column])
            if carrying[t,column] and identity in resolved:
                targets[t,column]=resolved[identity];mask[t,column]=True
    return targets.reshape(original),mask.reshape(original)


def _extend_moments(old,params):
    if isinstance(params,dict):
        return {k:_extend_moments(old[k],v) if k in old else jax.tree.map(j.zeros_like,v)
                for k,v in params.items()}
    if isinstance(params,(tuple,list)):
        if len(old)!=len(params):raise ValueError('Existing moment tuple mismatch')
        return type(params)(_extend_moments(a,b) for a,b in zip(old,params))
    if np.shape(old)!=np.shape(params):raise ValueError('Existing Adam leaf shape changed')
    return old


def migrate_adam(state,params):
    """Extend Adam mu/nu only; preserve existing moments, count and clipping."""
    if hasattr(state,'mu') and hasattr(state,'nu'):
        return state._replace(mu=_extend_moments(state.mu,params),nu=_extend_moments(state.nu,params))
    if isinstance(state,dict):return {k:migrate_adam(v,params) for k,v in state.items()}
    if isinstance(state,tuple):
        parts=[migrate_adam(v,params) for v in state]
        return type(state)(*parts) if hasattr(state,'_fields') else tuple(parts)
    if isinstance(state,list):return [migrate_adam(v,params) for v in state]
    return state
