"""Task-semantic-preserving reward deletion; separate from immutable D1."""
import jax.numpy as j

KEEP=('lesson_task','global_safety','settled_kill','transport_capture',
      'control_entry_reward','control_residual_reward','control_clawback','route_shaped_reward')

def select(parts,names,bad,paid):
    names=tuple(names)
    if parts.shape[-1]!=len(names) or any(k not in names for k in (*KEEP,'clawback','combined_total')):
        raise ValueError('Invalid fixed reward ABI')
    out=j.zeros_like(parts)
    for key in KEEP:out=out.at[...,names.index(key)].set(parts[...,names.index(key)])
    # Legacy paid includes shaping. Track only retained task and settled kill,
    # preserving their actual payment time and subsequent death refund.
    earned=parts[...,names.index('lesson_task')]+parts[...,names.index('settled_kill')]
    reclaim=j.where(bad,-paid,0.)
    out=out.at[...,names.index('clawback')].set(reclaim)
    reward=out.sum(-1)
    out=out.at[...,names.index('combined_total')].set(reward)
    return reward,j.where(bad,0.,paid+earned),out

def install(engine):
    """Wrap isolated collector transition and add explicit checkpointed ledger."""
    old_transition=engine.T.transition
    old_fresh=engine.T.fresh
    def fresh(keys,bs,ba):
        st,ax=old_fresh(keys,bs,ba)
        return st,dict(ax,campaign_task_paid=j.maximum(0.,ax['paid']-ax['shape_paid']))
    def transition(st,ax,actions,keys,arm='control'):
        paid=ax['campaign_task_paid'];core={k:v for k,v in ax.items() if k!='campaign_task_paid'}
        af,na,legacy,done,info,parts=old_transition(st,core,actions,keys,arm)
        names=(*engine.T.E.ops.COMPONENTS[:-1],*engine.T.events.METRICS,'combined_total')
        bad=info['death']|info['mutual_death'][:,None]
        reward,paid,out=select(parts,names,bad,paid)
        return af,dict(na,campaign_task_paid=paid),reward,done,info,out
    engine.T.fresh=fresh;engine.T.transition=transition
