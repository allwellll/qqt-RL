"""Explicit checkpointed JE07 drop provenance and carrier-phase assistance."""
import hashlib
import json
import jax
import jax.numpy as jnp

DROP_RECOVERY=8.
ASSIST_MAX_TICKS=600
ASSIST_PROBABILITIES=(.7,.5,.3,.1)
PROBE_MANIFEST=dict(schema='je07.unassisted.transport-probe/v1',seed=2026101307,
    games_per_bucket=16,buckets=['near','mid','far'],full_games=16,assist=False,
    bands=dict(near=[24,32],mid=[8,24],far=[0,8]),carry_ticks=600,full_ticks=900,
    opponent='real web Hunter normal',map=806,seat='team0 fixed same-protocol diagnostic')
PROBE_MANIFEST_SHA=hashlib.sha256(json.dumps(PROBE_MANIFEST,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def drop_state():
    # One conserved object per origin on map806. Episode-scoped cycle identity
    # deliberately never renews on drop/repickup, even with a new drop nonce.
    return dict(nonce=jnp.zeros(2,jnp.int32),credit=jnp.full(2,-1,jnp.int8),
        claimed_nonce=jnp.full((2,2),-1,jnp.int32),cycle_paid=jnp.zeros((2,2),bool),
        balance=jnp.zeros(2),pending_capture=jnp.zeros(2,bool))

def drop_reward(e,s,coefficient=1.):
    s=jax.tree.map(jnp.asarray,s)
    nonce=s['nonce'];credit=s['credit']
    # Drop precedes pickup in the authority tick. Multiple origins can drop
    # simultaneously; each nonce advances once, never on a custody transfer.
    for origin in range(2):
        happened=e['drop']&(e['drop_origin']==origin)
        source=jnp.argmax(happened)
        nonce=nonce.at[origin].add(happened.any().astype(jnp.int32))
        credit=credit.at[origin].set(jnp.where(happened.any(),e['drop_credit'][source],credit[origin]).astype(jnp.int8))
    team=jnp.arange(2);origin=jnp.clip(e['pickup_origin'],0,1)
    valid=e['pickup']&(e['pickup_origin']==1-team)&(e['carried']==origin)&~e['death']&~e['trade']
    eligible=valid&(nonce[origin]>0)&(credit[origin]==team)
    new=eligible&~s['cycle_paid'][team,origin]&(s['claimed_nonce'][team,origin]!=nonce[origin])
    paid=DROP_RECOVERY*new
    bad=e['death']|e['trade']|(e['drop']&(e['drop_origin']==1-team))
    balance=s['balance']+paid
    clawback=jnp.where(bad,-balance,0.)
    pending=(s['pending_capture']|e['capture'])&~bad
    settled=pending&e['settled']
    state=dict(s,nonce=nonce,credit=credit,
        claimed_nonce=s['claimed_nonce'].at[team,origin].set(jnp.where(new,nonce[origin],s['claimed_nonce'][team,origin])),
        cycle_paid=s['cycle_paid'].at[team,origin].set(s['cycle_paid'][team,origin]|new),
        balance=jnp.where(bad|settled,0.,balance),pending_capture=pending&~settled)
    reward=coefficient*(paid+clawback)
    return reward,state,dict(drop_recovery=new,drop_bonus=paid,drop_clawback=clawback,
        unattributed_pickup=valid&~eligible,repeat_recovery=eligible&~new,own_recover=e['pickup']&(e['pickup_origin']==team))

def assist_state(n):
    return dict(phase=jnp.zeros(n,jnp.int32),carrying=jnp.zeros(n,bool),
        selected=jnp.zeros(n,bool),ticks=jnp.zeros(n,jnp.int32))

def assist_gate(s,carried,alive,ended,key,probability):
    acquired=carried&~s['carrying']&alive&~ended
    selected=jnp.where(acquired,jax.random.uniform(key,carried.shape)<probability,s['selected'])
    ticks=jnp.where(acquired,0,s['ticks'])
    active=carried&alive&~ended&selected&(ticks<ASSIST_MAX_TICKS)
    next_state=dict(phase=s['phase']+acquired.astype(jnp.int32),carrying=carried&alive&~ended,
        selected=selected&carried&alive&~ended,ticks=jnp.where(carried,ticks+1,0))
    return active,next_state

def assisted_actions(original,legal_moves,active,key):
    # Uniform legal moves/wait; no bomb/use_item, no pursuit target. Original
    # responsive/neural/PFSP action is preserved on all unassisted rows.
    move=jax.random.categorical(key,jnp.where(legal_moves,0.,-1e9))
    random=jnp.stack([move,jnp.zeros_like(move)],-1).astype(original.dtype)
    return jnp.where(active[:,None],random,original)

def assist_course_state():
    return dict(level=0,proofs=[],manifest_sha256=PROBE_MANIFEST_SHA)

def assist_probability(s):
    return ASSIST_PROBABILITIES[int(s['level'])]

def update_assist_course(s,proof):
    # Only independent fixed-seed UNASSISTED probes inform monotone annealing.
    # No training/CPU gate: absent/weak probes leave current ratio in place.
    valid=(proof.get('seed_manifest_sha256')==PROBE_MANIFEST_SHA and proof.get('assist') is False and not proof.get('smoke',False)
        and proof.get('games_per_bucket')==16 and all(proof.get(k,0)>=13 for k in ['near','mid','far'])
        and proof.get('full_capture',0)>=1)
    if not valid:return dict(s)
    identity=proof.get('policy_sha256')
    if identity and any(p.get('policy_sha256')==identity for p in s['proofs']):return dict(s)
    return dict(s,level=min(3,s['level']+1),proofs=s['proofs']+[dict(proof)])
