"""R3 private collector. R1/R2 source files and live processes stay intact."""
import inspect
from types import ModuleType
import jax
import jax.numpy as j
from scripts import bun_native_v2_custody_collector as C
from scripts import r3_items_reward as R,r3_items_observation as O

E=C.E
_base=C._module
_reward=_base.reward
scope=dict(_reward.transition.__globals__)
settle_source=inspect.getsource(scope['settle'])
settle_source=settle_source.replace('topology=e[\'improved\']&bricks.any((1,2))&valid',
    "topology=e['improved']&bricks.any((1,2))&valid&~e['r3_road_improved']")
assert 'raw=.02*threat+.01*selected+.04*topology+.015*multi+approach' in settle_source
settle_source=settle_source.replace('raw=.02*threat+.01*selected+.04*topology+.015*multi+approach','raw=.04*topology+approach')
settle_source=settle_source.replace("CAP-s['gross']","CAP-s['gross']-s['r3']['gross'][:,5:].sum(-1)")
scope['CAP']=.45
exec(compile(settle_source,__file__,'exec'),scope)
source=C._reward_source
anchor="    n,r,m=settle(e,s);n['brick_owned']=owned&after.brick"
assert source.count(anchor)==1
source=source.replace(anchor,"    e=dict(e,r3_road_improved=info['r3_road_improved'])\n"+anchor)
exec(compile(source,__file__,'exec'),scope)
_legacy_transition=scope['transition']


def reward_transition(before,after,info,aux,meta,seats):
    ledger,candidates,metrics,road,bad,terminal,active=R.transition(before,after,info,aux,meta,seats)
    candidates=candidates.at[:,:5].multiply(ledger['item_scale'][:,None])
    old,pay,m,birth=_legacy_transition(before,after,dict(info,r3_road_improved=road),aux,meta,seats)
    ledger,payment,settlement=R.settle(ledger,candidates,bad=bad,terminal=terminal,active=active,
        attribute_matured=m['r1_item_pay']*1.5,attack_gross=old['gross'])
    # Attribute high-water legacy delay is already30; pay its scaled amount
    # once under the new joint cap, rather than duplicating .08 and .12.
    actual=pay-m['r1_item_pay']+payment.sum(-1)
    old=dict(old,r3=ledger)
    m.update(metrics,r3_pay=payment,r3_positive_pay=j.maximum(payment,0.),
        r3_inherited_positive_pay=j.maximum(pay-m['r1_item_pay'],0.),
        r3_candidate=candidates,r3_pending=settlement['pending'],r3_void=settlement['void'],r3_gross=ledger['gross'],pay=actual)
    return old,actual,m,birth


_reward.transition=reward_transition


def configure(attack=True,antifarm=True):
    assert attack and antifarm
    C.configure(attack,antifarm);O.install()


def metadata(*args,**kwargs):
    m=C.metadata(*args,**kwargs);n=len(m['learner_seat'])
    return dict(m,bomb_threat=dict(m['bomb_threat'],r3=R.state(n)))


def inputs(st,seats,ledger=None):
    obs,gv,mm,bm=C.inputs(st,seats)
    obs,gv=O.expand_inputs(obs,gv,st,seats,ledger)
    return obs,gv,mm,bm


def prediction(p,st,seats,ledger=None):
    o,g,mm,bm=inputs(st,seats,ledger);mv,ab,v,vl=E.train.net_forward(p,'transformer',o,g)
    return mv,ab,v,vl,o,g,mm,bm


src=_base._source.replace(_base._anchor,_base._repl)
src=src.replace("prediction(p,states,seats)","prediction(p,states,seats,aux['combat_mix']['bomb_threat']['r3'])")
src=src.replace("inputs(ss,enemyseat[ids])","inputs(ss,enemyseat[ids],jax.tree.map(lambda x:x[ids],aux['combat_mix']['bomb_threat']['r3']))")
src=src.replace('prediction(p,after,seat)[2]',"prediction(p,after,seat,na['combat_mix']['bomb_threat']['r3'])[2]")
src=src.replace("            diagnostics.update(policy_diag)","            diagnostics.update(policy_diag)\n            diagnostics.update({name:sm[name].sum(axis=0) for name in sm if name.startswith('r3_') or name.startswith('v2_')})")
anchor="                credit['items']=dict(credit['items'],scale=ax['combat_mix']['bomb_threat']['items']['scale'])"
assert src.count(anchor)==1
src=src.replace(anchor,anchor+"\n                credit['r3']=dict(credit['r3'],item_scale=ax['combat_mix']['bomb_threat']['r3']['item_scale'])")
_impl=ModuleType('r3_private_collector');exec(compile(src,__file__,'exec'),_impl.__dict__)
def transition(*args,**kwargs):
    after,aux,reward,done,info,parts,metrics=C.transition(*args,**kwargs)
    pending=aux['combat_mix']['bomb_threat']['r3']['queue'].sum((1,2))>0
    # Complete30-tick risk adjudication before resetting an ended game. This
    # is a ledger tail, explicitly separate from pre-terminal ability metrics.
    return after,aux,reward,done&~pending,info,parts,metrics


_impl.fresh=C.fresh;_impl.transition=transition;_impl.metadata=metadata;_impl.inputs=inputs;_impl.prediction=prediction
_base._impl=_impl


def collector(*args,**kwargs):
    _impl.ATTACK=True
    return _impl.collector(*args,**kwargs)


for name in ('banks','fresh','seat_allocation','bank_selection'):globals()[name]=getattr(C,name)
for name in ('choose','choose_with_diagnostics','route_teacher','combat_teacher'):globals()[name]=getattr(_impl,name)
