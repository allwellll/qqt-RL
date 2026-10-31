"""A3SR versioned A3S collector; reset randomization and passive exposure only."""
from pathlib import Path
from types import ModuleType
from scripts import bomb_threat_seat5050_collector as original
from scripts import bomb_threat_a3sr_spawn as spawn
import jax
import jax.numpy as j

# Keep A3S frozen bytes and all transition/GAE formulas. Compile a local module
# so changes never mutate the collector imported by an already running arm.
_source=Path('scripts/bomb_threat_seat5050_collector.py').read_text()
_from="diagnostics.update(policy_diag)"
_to="""diagnostics.update(policy_diag)
            diagnostics.update({name:(done*fa['a3sr'][name]).sum() for name in ('spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','spawn_eligible')})
            diagnostics.update(spawn_eligible_seat=(onehot*(done&fa['a3sr']['spawn_eligible'])[:,None]).sum(0),
                spawn_random_seat=(onehot*(done&(fa['a3sr']['spawn_mode']>0))[:,None]).sum(0),
                spawn_native_eligible=(done&fa['a3sr']['spawn_eligible']&(fa['a3sr']['spawn_mode']==0)).sum())"""
assert _source.count(_from)==1
_source=_source.replace('fst,fa=fresh(fkeys,bs,ba,designated,spawn_pairs,seats)','fst,fa=reset_fresh(fkeys,bs,ba,designated,spawn_pairs,seats,done)')
_impl=ModuleType('a3sr_seat_collector');exec(compile(_source.replace(_from,_to),'scripts/bomb_threat_a3sr_collector.py','exec'),_impl.__dict__)


def fresh(keys,bs,ba,designated,spawn_pairs,seats):
    states,aux=original.fresh(keys,bs,ba,designated,spawn_pairs,seats)
    sk=jax.vmap(lambda k:jax.random.fold_in(k,0xA3))(keys)
    states,meta=spawn.apply_batch(sk,states,aux,enabled=True)
    return states,dict(aux,a3sr=meta)


def reset_fresh(keys,bs,ba,designated,spawn_pairs,seats,done):
    states,aux=original.fresh(keys,bs,ba,designated,spawn_pairs,seats)
    sk=jax.vmap(lambda k:jax.random.fold_in(k,0xA3))(keys)
    states,meta=spawn.apply_batch(sk,states,aux,reset_mask=done,enabled=True)
    return states,dict(aux,a3sr=meta)


def transition(states,aux,actions,keys,coefficient):
    clean={k:v for k,v in aux.items() if k!='a3sr'}
    after,na,r,done,info,parts,metrics=original.transition(states,clean,actions,keys,coefficient)
    na=dict(na,a3sr=aux['a3sr'])
    return after,na,r,done,info,parts,metrics

_impl.fresh=fresh
_impl.reset_fresh=reset_fresh
_impl.transition=transition
for _name in ('seat_allocation','inputs','choose','choose_with_diagnostics','metadata','route_teacher','combat_teacher','prediction'):
    globals()[_name]=getattr(_impl,_name)
collector=_impl.collector
E=original.E
