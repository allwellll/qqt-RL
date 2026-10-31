"""Versioned ITER2-only optimizer/settlement; legacy workers stay untouched."""
import jax
import jax.numpy as j
import numpy as np
import optax


def actor_ramp(update):return j.minimum(j.maximum(j.asarray(update)/40.,0.),1.)


def entropy(update,anneal):
    return .005+(.0015-.005)*j.minimum(j.maximum(j.asarray(update)/320.,0.),1.) if anneal else j.asarray(.005)


def warmup_ready(rows):
    if len(rows)<20:return dict(passed=False,plateau=False,reason='requires 20 full updates')
    values=np.array([r['value_loss'] for r in rows[-5:]],float)
    scale=max(abs(float(np.mean(values))),1e-8)
    # Predeclared plateau: last five finite values span <=5% of their mean,
    # and absolute fitted drift is <=1% of mean per full update.
    plateau=bool(np.isfinite(values).all() and np.ptp(values)/scale<=.05 and abs(np.polyfit(np.arange(5),values,1)[0])/scale<=.01)
    ev=float(rows[-1]['explained_variance']) if rows[-1]['explained_variance'] is not None else float('nan')
    return dict(passed=bool(np.isfinite(ev) and ev>=.3 or plateau),plateau=plateau,
        explained_variance=ev,updates=len(rows),loss_last5=values.tolist(),value_loss_scale=scale)


def minibatch_step(optimizer,objective):
    """Actor Adam never advances during warmup; global gradient clip stays .5."""
    def step(params,adam,batch,ent,actor_scale,warm):
        def loss(p):return objective(p,batch,ent)
        (loss_value,metrics),grad=jax.value_and_grad(loss,has_aux=True)(params)
        grad=jax.tree.map(lambda x:x,grad)
        grad=dict(grad,actor=jax.tree.map(lambda x:j.where(warm,0.,x),grad['actor']))
        norm=optax.global_norm(grad);scale=j.minimum(1.,.5/(norm+1e-6))
        grad=jax.tree.map(lambda x:x*scale,grad)
        def change_actor(_):
            change,state=optimizer.update(grad['actor'],adam['actor'],params['actor'])
            change=jax.tree.map(lambda x:x*actor_scale,change)
            return optax.apply_updates(params['actor'],change),state
        actor,aa=jax.lax.cond(warm,lambda _:(params['actor'],adam['actor']),change_actor,operand=None)
        dc,ca=optimizer.update(grad['critic'],adam['critic'],params['critic'])
        return dict(actor=actor,critic=optax.apply_updates(params['critic'],dc)),dict(actor=aa,critic=ca),j.concatenate([loss_value[None],metrics,norm[None]])
    return step


def updater(minibatch,optimizer,objective):
    step=minibatch_step(optimizer,objective)
    def update(params,adam,batch,key,ent,actor_scale,warm):
        b=tuple(x.reshape((-1,)+x.shape[2:]) for x in batch)
        adv=b[4];adv=(adv-adv.mean())/j.sqrt(adv.var()+1e-8);b=b[:4]+(adv,)+b[5:]
        order=jax.random.permutation(key,len(adv)).reshape(-1,minibatch)
        def body(c,ids):
            p,a=c;p,a,loss=step(p,a,tuple(x[ids] for x in b),ent,actor_scale,warm)
            return (p,a),loss
        return jax.lax.scan(body,(params,adam),order)
    # Exactly one physical card per run; no inter-device gradient reduction.
    return jax.pmap(update,in_axes=(0,0,0,0,None,None,None))


def settlement(multiplier,cap):
    def settle(e,s):
        rows=j.arange(len(s['gross']));cell=j.clip(e['enemy_cell'],0,194);old=j.clip(e['old_cell'],0,194)
        valid=e['active']&~e['bad']&~e['terminal']
        threat=e['placed']&e['safe']&e['threat']&~s['enemy_seen'][rows,cell]&valid
        chain=e['placed']&e['safe']&e['chain']&(e['old_id']>s['old_seen'][rows,old])&valid
        bricks=e['owned_removed']&~s['brick_seen'];topology=e['improved']&bricks.any((1,2))&valid
        amount=(.02*multiplier)*threat+(.01*multiplier)*chain+(.04*multiplier)*topology
        pending=j.minimum(s['pending']+amount,j.maximum(0.,cap-s['gross']))
        pending=j.where(valid,pending,0.);pay=j.where(e['settled']&valid,pending,0.)
        n=dict(s,pending=j.where(e['settled'],0.,pending),gross=s['gross']+pay,
            enemy_seen=s['enemy_seen'].at[rows,cell].set(s['enemy_seen'][rows,cell]|threat),
            old_seen=s['old_seen'].at[rows,old].set(j.where(chain,e['old_id'],s['old_seen'][rows,old])),
            brick_seen=s['brick_seen']|bricks,brick_pending=j.where(valid,s['brick_pending']+topology.astype(j.int32),0))
        paid_bricks=j.where(e['settled']&valid,n['brick_pending'],0);n['brick_pending']=j.where(e['settled'],0,n['brick_pending'])
        return n,pay,dict(candidate=threat|chain|topology,settled_count=(pay>0).astype(j.int32),pay=pay,
            threat_events=threat.astype(j.int32),chain_events=chain.astype(j.int32),brick_events=topology.astype(j.int32),
            brick_paid_events=paid_bricks,threat_pending=pending)
    return settle
