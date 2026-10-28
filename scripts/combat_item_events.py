"""Independent replacement ledgers: no placement/tick payments or duplicate legacy pay."""
import jax.numpy as j

CAP=2.
METRICS=('banana_completed','banana_net_distance','banana_effective_pay','glue_completed',
         'glue_chase_delay','glue_effective_pay','item_replacement_delta')


def phase_allowed(carrying,enemy_alive,enemy_trapped,*unused):
    return carrying|~j.asarray(enemy_alive)|(j.asarray(enemy_trapped)>0)


def state(n):
    z=j.zeros((n,2));out={}
    for family in ['banana','glue']:
        out[f'{family}_frontier']=j.full((n,2),32767.)
        for name in ['active','valid']:out[f'{family}_{name}']=z.astype(bool)
        for name in ['start','cost','gross','balance','legacy_balance']:out[f'{family}_{name}']=z
    return out


def adjust(base,delta,coefficient):
    # Static zero treatment is bit-for-bit equivalent, including signed zeros.
    return base if coefficient==0 else base+coefficient*delta


def settle(e,s,family):
    new=dict(s);eligible=e['carry']&e['safe']&~e['bad']&~e['topology_changed']
    start=e['banana_start']&(e['banana_owner']==0) if family=='banana' else e['glue_hit']
    active=s[family+'_active'];begin=start&~active&eligible
    active=active|begin
    valid=j.where(begin,True,s[family+'_valid'])&eligible
    initial=j.where(begin,j.minimum(e.get('start_distance',e['distance']),s[family+'_frontier']),s[family+'_start'])
    # Historical closest approach never resets on death/reacquisition within one episode.
    # Different item ids must still not repay walking backwards over an old corridor.
    new[family+'_frontier']=j.where(e['carry']&(e['distance']<32767.),j.minimum(s[family+'_frontier'],e['distance']),s[family+'_frontier'])
    cost=j.where(begin,0.,s[family+'_cost'])
    if family=='banana':
        cost=cost+j.where(active,e['walk_budget'],0.)
        finish=active&~e['sliding']
        benefit=j.maximum(initial-e['distance']-cost,0.)
        candidate=j.minimum(.5,.2*benefit)
    else:
        cost=cost+j.where(active,e['chase_delay'],0.)
        finish=active&~e['slow']
        benefit=j.where(initial>e['distance']+1e-5,cost,0.)
        candidate=j.minimum(.5,.25*benefit)
    pay=j.where(finish&valid,j.minimum(candidate,j.maximum(0.,CAP-s[family+'_gross'])),0.)
    lost=e['bad']|e['lost'];balance=s[family+'_balance']+pay
    net=pay-j.where(lost,balance,0.)
    oldpay=e['legacy_'+family];oldbalance=s[family+'_legacy_balance']+oldpay
    oldnet=oldpay-j.where(lost,oldbalance,0.)
    reset=lost|e['capture']
    new.update({family+'_active':active&~finish&~lost,family+'_valid':valid&~finish&~lost,
                family+'_start':initial,family+'_cost':cost,family+'_gross':s[family+'_gross']+pay,
                family+'_balance':j.where(reset,0.,balance),family+'_legacy_balance':j.where(reset,0.,oldbalance)})
    return net-oldnet,new,dict(completed=finish.astype(j.float32),benefit=j.where(finish&valid,benefit,0.),pay=pay,net=net,legacy_net=oldnet)
