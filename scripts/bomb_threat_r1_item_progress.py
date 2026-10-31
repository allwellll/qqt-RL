"""Independent capped attribute progress ledger; no rule or policy mutation.

High-water is episode history. Restored real Bank states initialize at their
existing attributes, so pre-Bank growth is never paid again. Payout is delayed
30 ticks; terminal/control loss void pending entries, including the tail.
"""
import jax.numpy as j

INITIAL = (2., 2., 1.3)
MAXIMUM = (10., 8., 2.4)
WEIGHT = .08
DELAY = 30
VERSION = 'r1-attribute-highwater-v2'


def fraction(stats):
    low=j.asarray(INITIAL,j.float32);high=j.asarray(MAXIMUM,j.float32)
    return j.clip((stats-low)/(high-low),0.,1.)


def state(stats):
    n=len(stats)
    return dict(highwater=fraction(stats),pending=j.zeros((n,DELAY),j.float32),
        gross=j.zeros(n,j.float32),void=j.zeros(n,j.float32),scale=j.ones(n,j.float32),
        events=j.zeros((n,3),j.int32))


def step(s,before,after,grew,*,active,self_death,terminal,control_lost):
    """`before/after` are immediately around actual native _grow, not tick net.

    `grew` must be the engine's actual per-attribute growth evidence. An increase
    from changing/resetting a scene is not an eligible event. High-water does
    not roll back when a pending event is voided, on death, or on stat loss.
    """
    f=fraction(after)
    eligible=grew&(after>before+1e-6)&active[:,None]
    delta=j.where(eligible,j.maximum(f-s['highwater'],0.),0.)
    water=j.maximum(s['highwater'],j.where(eligible,f,s['highwater']))
    amount=WEIGHT*delta.sum(-1)*s['scale']
    invalid=self_death|terminal|control_lost
    # Invalidate before maturation: a death exactly30 ticks later still voids.
    pay=j.where(invalid,0.,s['pending'][:,0])
    queue=j.concatenate([s['pending'][:,1:],amount[:,None]],axis=1)
    void=j.where(invalid,s['pending'].sum(-1)+amount,0.)
    queue=j.where(invalid[:,None],0.,queue)
    out=dict(s,highwater=water,pending=queue,gross=s['gross']+pay,
        void=s['void']+void,events=s['events']+(delta>0).astype(j.int32))
    return out,pay,dict(progress=delta,candidate=amount,pay=pay,void=void,
        pending=queue.sum(-1),highwater=water,scale=s['scale'])
