import numpy as np
import jax.numpy as j
from scripts import campaign_reward_v2 as r

NAMES=('lesson_task','global_safety','settled_kill','settled_shaping','clawback','transport_capture','control_entry_reward','control_residual_reward','route_shaped_reward','blocked_route','control_clawback','combined_total')

def test_weights_and_timing_are_actual_legacy_components():
    parts=j.array([[[0.,0.,0.,4.,0.,0.,1.,.1,.2,-.25,0.,5.05]]])
    reward,ledger,out=r.select(parts,NAMES,j.array([[False]]),j.array([[3.]]))
    assert np.isclose(reward[0,0],1.3)
    # No native capture +2 replacement: delayed transport_capture pays +24.
    parts=parts.at[...,5].set(24.)
    reward,_,out=r.select(parts,NAMES,j.array([[False]]),ledger)
    assert np.isclose(reward[0,0],25.3)
    assert float(out[0,0,5])==24.

def test_removed_shaping_never_reappears_as_death_clawback():
    parts=j.zeros((1,1,len(NAMES))).at[...,2].set(16.).at[...,3].set(8.)
    reward,paid,_=r.select(parts,NAMES,j.array([[False]]),j.zeros((1,1)))
    assert float(reward[0,0])==16. and float(paid[0,0])==16.
    parts=j.zeros_like(parts).at[...,1].set(-12.).at[...,4].set(-24.)
    reward,paid,_=r.select(parts,NAMES,j.array([[True]]),paid)
    assert float(reward[0,0])==-28. and float(paid[0,0])==0.
    # A repeated bad tick cannot refund again.
    reward,_,_=r.select(j.zeros_like(parts),NAMES,j.array([[True]]),paid)
    assert float(reward[0,0])==0.

def test_zero_intervention_exact_legacy_and_shape_schema_validation():
    import pytest
    parts=j.zeros((1,1,len(NAMES)))
    with pytest.raises(ValueError):r.select(parts,NAMES[:-1],j.array([[False]]),j.zeros((1,1)))


def test_retained_control_credit_keeps_its_death_refund():
    names=NAMES
    parts=j.zeros((1,1,len(names))).at[...,names.index('control_clawback')].set(-4.)
    reward,_,out=r.select(parts,names,j.array([[True]]),j.zeros((1,1)))
    assert float(reward[0,0])==-4.
    assert float(out[0,0,names.index('control_clawback')])==-4.


def test_genuine_bank_fresh_preserves_inherited_task_kill_refund():
    from types import SimpleNamespace
    def old_fresh(keys,bs,ba):
        return None,dict(paid=j.array([[21.,5.]]),shape_paid=j.array([[8.,2.]]))
    engine=SimpleNamespace(T=SimpleNamespace(fresh=old_fresh,transition=lambda *args:None))
    r.install(engine)
    _,aux=engine.T.fresh(None,None,None)
    np.testing.assert_array_equal(aux['campaign_task_paid'],j.array([[13.,3.]]))
