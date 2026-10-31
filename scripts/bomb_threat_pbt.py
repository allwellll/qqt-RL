"""Audited PBT state handoff and stock-shaping-off arm helpers."""
import copy

WITH_STOCK_SHAPING='bomb-threat-v1-with-stock-shaping'
NATIVE_ONLY_ZERO_STOCK_SHAPING='bomb-threat-v1-native-only-zero-stock-shaping'
ZERO_SHAPING_WEIGHTS=('threat','topology','chain')


def training_reward_variant(protocol,run_name):
    """Resolve training-time shaping semantics from a run's frozen arm."""
    arm=protocol.get('arms',{}).get(run_name,{})
    explicit=protocol.get('training_reward_variant')
    if explicit:return explicit
    return NATIVE_ONLY_ZERO_STOCK_SHAPING if arm.get('zero_stock_shaping') else WITH_STOCK_SHAPING


def restore_probe_reward_variant(source_variant,target_variant,has_target_checkpoint):
    """Use source semantics for a fresh fork and target semantics when restoring."""
    return target_variant if has_target_checkpoint else source_variant

def verify_source_next_step(actual_digest,expected_digest):
    """Reject a fork unless its unmodified donor state reproduces NEXT_STEP."""
    if actual_digest!=expected_digest:
        raise ValueError('donor NEXT_STEP parity mismatch before target remap')
    return True

def remap_combat_mix_kind(aux,target_kind):
    """Apply a child-slot opponent assignment without mutating donor aux."""
    result=dict(aux);mix=dict(result['combat_mix']);mix['kind']=target_kind
    result['combat_mix']=mix
    return result


def validate_source_reward_contract(target_protocol,source_protocol,source_run,
                                    source_protocol_sha256,checkpoint_protocol_sha256):
    """Fail closed if copied checkpoint and target reward/rule provenance disagree."""
    expected_sha=target_protocol.get('source_protocol_sha256')
    if not expected_sha or source_protocol_sha256!=expected_sha:
        raise ValueError('source protocol SHA mismatch')
    if checkpoint_protocol_sha256!=source_protocol_sha256:
        raise ValueError('checkpoint protocol SHA mismatch')
    source_reward=source_protocol.get('reward_version')
    source_rule=source_protocol.get('rule_version')
    if not source_reward or source_reward!=target_protocol.get('source_reward_version') or \
       source_reward!=target_protocol.get('reward_version'):
        raise ValueError('source reward version does not match target reward version')
    if not source_rule or source_rule!=target_protocol.get('source_rule_version') or \
       source_rule!=target_protocol.get('rule_version'):
        raise ValueError('source rule version does not match target rule version')
    actual_source_variant=training_reward_variant(source_protocol,source_run)
    declared_source_variant=target_protocol.get('source_training_reward_variant')
    target_variant=target_protocol.get('training_reward_variant')
    if actual_source_variant!=declared_source_variant:
        raise ValueError('source training reward variant mismatch')
    expected_target_variant=(NATIVE_ONLY_ZERO_STOCK_SHAPING
        if target_protocol.get('zero_shaping_weights') else WITH_STOCK_SHAPING)
    if target_variant!=expected_target_variant:
        raise ValueError('target training reward variant mismatch')
    requires_rekey=actual_source_variant!=target_variant
    if bool(target_protocol.get('pbt_rekey_source'))!=requires_rekey:
        raise ValueError('source reward variant/rekey declaration mismatch')
    if requires_rekey and tuple(target_protocol.get('zero_shaping_weights',()))!=ZERO_SHAPING_WEIGHTS:
        raise ValueError('zero-shaping fork must disable exactly threat, topology and chain weights')
    return dict(source_reward_version=source_reward,target_reward_version=target_protocol['reward_version'],
        source_rule_version=source_rule,target_rule_version=target_protocol['rule_version'],
        source_training_reward_variant=actual_source_variant,
        target_training_reward_variant=target_variant,requires_rng_rekey=requires_rekey,
        source_protocol_sha256=source_protocol_sha256)


def rekey_keys(jax,keys,seed):
    import jax.numpy as j
    x=j.asarray(keys);flat=x.reshape((-1,2))
    return jax.vmap(lambda k:jax.random.fold_in(k,j.asarray(seed,j.uint32)))(flat).reshape(x.shape)

def reset_shaping_ledger(j,aux):
    result=dict(aux);meta=dict(result['combat_mix']);ledger=dict(meta['bomb_threat'])
    for name in ('pending','gross','brick_pending'):
        ledger[name]=j.zeros_like(ledger[name])
    meta['bomb_threat']=ledger;result['combat_mix']=meta
    return result

def zero_stock_transition(original):
    def transition(st,aux,actions,keys,coefficient):
        import jax.numpy as j
        after,new_aux,reward,done,info,parts,metrics=original(st,aux,actions,keys,coefficient)
        pay=metrics['pay'];reward=reward.at[:,0].add(-pay)
        parts=parts.at[:,0,-1].set(reward[:,0])
        meta=dict(new_aux['combat_mix']);ledger=dict(meta['bomb_threat'])
        for name in ('pending','gross','brick_pending'):
            ledger[name]=j.zeros_like(ledger[name])
        meta['bomb_threat']=ledger;new_aux=dict(new_aux,combat_mix=meta)
        metrics=dict(metrics,pay=j.zeros_like(pay),settled_count=j.zeros_like(metrics['settled_count']),
            threat_pending=j.zeros_like(metrics['threat_pending']),brick_paid_events=j.zeros_like(metrics['brick_paid_events']))
        return after,new_aux,reward,done,info,parts,metrics
    return transition

def _without_reward_accumulators(aux):
    result=copy.deepcopy(aux);ledger=result['combat_mix']['bomb_threat']
    for name in ('pending','gross','brick_pending'):ledger.pop(name,None)
    return result

def pbt_transfer_digests(digest,params_before,params_after,adam_before,adam_after,states_before,states_after,aux_before,keys_before,aux_after,keys_after,ring_before,ring_after,ring_cursor_before=None,ring_cursor_after=None):
    source_aux=_without_reward_accumulators(aux_before);target_aux=_without_reward_accumulators(aux_after)
    actor_exact=bool(digest(params_before['actor'])==digest(params_after['actor']))
    critic_exact=bool(digest(params_before['critic'])==digest(params_after['critic']))
    return dict(actor_exact=actor_exact,critic_exact=critic_exact,
        actor_critic_exact=actor_exact and critic_exact,Adam_exact=bool(digest(adam_before)==digest(adam_after)),
        environment_exact=bool(digest(states_before)==digest(states_after)),nonreward_aux_exact=bool(digest(source_aux)==digest(target_aux)),
        ring_exact=bool(digest(ring_before)==digest(ring_after)),
        ring_cursor_exact=ring_cursor_before==ring_cursor_after,
        source_rng_sha256=digest(keys_before),
        fork_rng_sha256=digest(keys_after),RNG_distinct=digest(keys_before)!=digest(keys_after),
        parent_aux_sha256=digest(source_aux),fork_aux_sha256=digest(target_aux))
