import jax
import jax.numpy as j
import numpy as np
import pytest
from scripts.bomb_threat_pbt import (
    NATIVE_ONLY_ZERO_STOCK_SHAPING,
    WITH_STOCK_SHAPING,
    rekey_keys,
    reset_shaping_ledger,
    validate_source_reward_contract,
    pbt_transfer_digests,
    restore_probe_reward_variant,
    zero_stock_transition,
    verify_source_next_step,
    remap_combat_mix_kind,
)


def test_rng_rekey_is_deterministic_slot_distinct_and_shape_preserving():
    keys=jax.random.split(jax.random.PRNGKey(17),8).reshape(2,4,2)
    a=np.asarray(rekey_keys(jax,keys,2026112903));b=np.asarray(rekey_keys(jax,keys,2026112903));c=np.asarray(rekey_keys(jax,keys,2026112904))
    assert a.shape==keys.shape and np.array_equal(a,b) and not np.array_equal(a,c) and not np.array_equal(a,keys)


def test_zero_shaping_clears_only_credit_accumulators_and_preserves_event_history():
    aux={'combat_mix':{'kind':j.array([1]),'bomb_threat':{'pending':j.array([.1]),'gross':j.array([.1]),
      'brick_pending':j.array([2]),'enemy_seen':j.array([[True,False]]),'old_seen':j.array([[3,-1]])}}}
    out=reset_shaping_ledger(j,aux);ledger=out['combat_mix']['bomb_threat']
    assert np.array_equal(ledger['pending'],[0]) and np.array_equal(ledger['gross'],[0]) and np.array_equal(ledger['brick_pending'],[0])
    assert np.array_equal(ledger['enemy_seen'],[[True,False]]) and np.array_equal(ledger['old_seen'],[[3,-1]])


def test_zero_stock_transition_removes_only_shaping_payment():
    def original(st,aux,actions,keys,coefficient):
        return st,aux,j.array([[5.,3.]]),j.array([False]),{},j.array([[[5.,3.,0.]]]),dict(pay=j.array([.75]),
            settled_count=j.array([1]),threat_pending=j.array([.2]),brick_paid_events=j.array([1]))
    fn=zero_stock_transition(original);r=fn(None,{'combat_mix':{'bomb_threat':{'pending':j.array([.2]),'gross':j.array([.2]),'brick_pending':j.array([1])}}},None,None,0)
    assert np.array_equal(r[2],[[4.25,3.]]) and np.array_equal(r[5][:,0,-1],[4.25])
    assert np.array_equal(r[6]['pay'],[0]) and np.array_equal(r[3],[False])


def test_source_reward_contract_recognizes_fork_and_rejects_version_drift():
    source={
        'rule_version':'stock-window-v1',
        'reward_version':'bomb-threat-v1',
        'arms':{'bt_slot00_c000003':{'lr':4.8e-5}},
    }
    target={
        'rule_version':'stock-window-v1',
        'reward_version':'bomb-threat-v1',
        'source_rule_version':'stock-window-v1',
        'source_reward_version':'bomb-threat-v1',
        'source_training_reward_variant':WITH_STOCK_SHAPING,
        'training_reward_variant':NATIVE_ONLY_ZERO_STOCK_SHAPING,
        'source_protocol_sha256':'source-protocol-sha',
        'pbt_rekey_source':True,
        'zero_shaping_weights':['threat','topology','chain'],
    }
    result=validate_source_reward_contract(
        target,source,'bt_slot00_c000003','source-protocol-sha','source-protocol-sha')
    assert result['source_training_reward_variant']==WITH_STOCK_SHAPING
    assert result['target_training_reward_variant']==NATIVE_ONLY_ZERO_STOCK_SHAPING
    assert result['requires_rng_rekey'] is True

    with pytest.raises(ValueError,match='reward version'):
        validate_source_reward_contract(
            target,dict(source,reward_version='unknown-v9'),'bt_slot00_c000003',
            'source-protocol-sha','source-protocol-sha')
    with pytest.raises(ValueError,match='checkpoint protocol'):
        validate_source_reward_contract(
            target,source,'bt_slot00_c000003','source-protocol-sha','different-protocol-sha')


def test_pbt_transfer_receipt_distinguishes_actor_critic_and_optimizer_state():
    def digest(value):return repr(value)
    params={'actor':{'w':[1]},'critic':{'w':[2]}}
    adam={'m':[3],'v':[4]}
    aux={'combat_mix':{'bomb_threat':{'pending':[1],'gross':[2],'brick_pending':[3],'enemy_seen':[4]}}}
    cleared={'combat_mix':{'bomb_threat':{'pending':[0],'gross':[0],'brick_pending':[0],'enemy_seen':[4]}}}
    row=pbt_transfer_digests(digest,params,params,adam,adam,[5],[5],aux,[6],cleared,[7],['r'],['r'])
    assert row['actor_exact'] and row['critic_exact'] and row['Adam_exact']
    assert row['environment_exact'] and row['nonreward_aux_exact'] and row['ring_exact']
    assert row['RNG_distinct'] and row['ring_cursor_exact']


def test_restore_probe_uses_source_variant_before_first_fork_and_target_after_resume():
    assert restore_probe_reward_variant(WITH_STOCK_SHAPING,NATIVE_ONLY_ZERO_STOCK_SHAPING,False)==WITH_STOCK_SHAPING
    assert restore_probe_reward_variant(NATIVE_ONLY_ZERO_STOCK_SHAPING,NATIVE_ONLY_ZERO_STOCK_SHAPING,False)==NATIVE_ONLY_ZERO_STOCK_SHAPING
    assert restore_probe_reward_variant(WITH_STOCK_SHAPING,NATIVE_ONLY_ZERO_STOCK_SHAPING,True)==NATIVE_ONLY_ZERO_STOCK_SHAPING

def test_source_parity_is_checked_before_child_combat_mix_remap():
    assert verify_source_next_step('donor','donor') is True
    with pytest.raises(ValueError,match='donor NEXT_STEP parity'):
        verify_source_next_step('child-kind','donor')
    donor={'combat_mix':{'kind':j.array([1]),'stable':j.array([7])}}
    child=remap_combat_mix_kind(donor,j.array([4]))
    assert np.array_equal(donor['combat_mix']['kind'],[1])
    assert np.array_equal(child['combat_mix']['kind'],[4])
    assert np.array_equal(child['combat_mix']['stable'],[7])
