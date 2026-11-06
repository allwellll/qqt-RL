"""L2_FTN2 combined, source-pinned process-local reward installer. No eager JAX import."""
from scripts import cx_fountain_continuation_v2 as B,cx_safe_frontier_v1 as F
VERSION='cx-ftn2-continuation-frontier-v1'
SOURCES={}
def fork_aux(aux,out=None):
 """Explicit one-time recipe fork: discard unpaid old attack, never refill caps.
 Actor/Adam/env/RNG and item queues are retained; new temporal flags start clean.
 """
 import jax
 import jax.numpy as j
 s=aux['combat_mix']['bomb_threat']['r3'];pending=s['queue'][:,:,5:].sum(1)
 ns=dict(s,queue=s['queue'].at[:,:,5:].set(0),void=s['void'].at[:,5:].add(pending),
         pressure_count=j.zeros_like(s['pressure_count']),last_explosion=j.full_like(s['last_explosion'],-1000),
         last_explosion_batch=j.full_like(s['last_explosion_batch'],-1))
 result=dict(aux,combat_mix=dict(aux['combat_mix'],bomb_threat=dict(aux['combat_mix']['bomb_threat'],r3=ns)))
 assert jax.tree.structure(result)==jax.tree.structure(aux)
 if out is not None:
  from scripts.population_run_state import atomic_json
  atomic_json(out/'FORK_REWARD_MIGRATION.json',dict(version=VERSION,old_unpaid_attack_voided=jax.device_get(pending.sum(0)).tolist(),
    gross_monotone_retained=True,item_queue_retained=True,temporal_sequence_reset=True,aux_schema_unchanged=True))
 return result
def install(R=None):
 R=B.install(R);R=F.install(R)
 SOURCES.update({'continuation_'+k:v for k,v in B.SOURCES.items()});SOURCES.update({'frontier_'+k:v for k,v in F.SOURCES.items()})
 return R
