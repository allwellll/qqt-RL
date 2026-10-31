"""R2 changes one loss only; source bookkeeping cannot change native dynamics."""
import pickle
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import pytest
from scripts import dagger_rl_train as old,dagger_rl_r2_train as new


def test_r2_control_matches_frozen_joint_and_anchor_is_offphase_zero(monkeypatch):
    p=dict(actor=j.arange(8,dtype=j.float32)/10,critic=j.ones(1))
    parent=dict(actor=j.zeros(8),critic=j.ones(1))
    def forward(q,arch,o,g):
        return j.tile(q['actor'][:5],(len(o),1)),j.tile(q['actor'][5:],(len(o),1)),j.zeros(len(o)),j.tile(q['critic'],(len(o),1))
    monkeypatch.setattr(new.E.train,'net_forward',forward)
    monkeypatch.setattr(new.E.train,'hl_gauss_value_loss',lambda v,r:j.mean((v[:,0]-r)**2))
    monkeypatch.setattr(new,'REFERENCE_PARAMS',parent)
    mask=j.zeros((2,5,3),bool).at[:,2,2].set(True)
    obs=j.zeros((2,24,13,15));g=j.zeros((2,24)).at[:,9:11].set(1)
    b=(obs,g,j.zeros((2,2),int),j.zeros(2),j.ones(2),j.ones(2),j.ones((2,5),bool),j.ones((2,3),bool),mask,j.ones(2,bool))
    replay=(b[0],b[1],b[6],b[7],mask,b[-1])
    f=lambda q:old.objective(q,b,replay,'joint')[0]
    c=lambda q:new.objective(q,b,replay,'joint')[0]
    np.testing.assert_array_equal(f(p),c(p))
    for a,z in zip(jax.tree.leaves(jax.grad(f)(p)),jax.tree.leaves(jax.grad(c)(p))):np.testing.assert_array_equal(a,z)
    delta=lambda q:new.objective(q,b,replay,'anchor')[0]-c(q)
    grad=jax.grad(delta)(p)
    assert np.linalg.norm(grad['actor'][:5])>0 and np.linalg.norm(grad['actor'][5:])>0
    np.testing.assert_array_equal(grad['critic'],0)
    # Actual enemy cargo remains free to learn item release even after revival.
    cargo=g.at[:,13:15].set(1);bc=b[:1]+(cargo,)+b[2:]
    # Differentiate each complete objective independently: differentiating their
    # subtraction accumulates common gradients in a different float32 order.
    off=lambda q:new.objective(q,bc,replay,'anchor')[0]
    baseline=lambda q:new.objective(q,bc,replay,'joint')[0]
    np.testing.assert_array_equal(off(p),baseline(p))
    for a,z in zip(jax.tree.leaves(jax.grad(off)(p)),jax.tree.leaves(jax.grad(baseline)(p))):np.testing.assert_array_equal(a,z)
    # Deferred settlement rows are not live combat even if actors are visible.
    terminal_batch=b+(j.zeros(2,bool),)
    terminal=lambda q:new.objective(q,terminal_batch,replay,'anchor')[0]
    terminal_control=lambda q:new.objective(q,terminal_batch,replay,'joint')[0]
    np.testing.assert_array_equal(terminal(p),terminal_control(p))
    for a,z in zip(jax.tree.leaves(jax.grad(terminal)(p)),jax.tree.leaves(jax.grad(terminal_control)(p))):np.testing.assert_array_equal(a,z)


def test_r2_origin_metadata_preserves_full_native_initial_and_transition(monkeypatch):
    if not Path('runs/carrier_mechanism_20261023/v1').exists():pytest.skip('local genuine pickup Bank required')
    from scripts.carrier_mechanism_analysis import bank,merged
    from scripts.terminal_checkpoint_restore import digest
    from scripts import combat_item_engine as T
    T.configure()
    items=[x for x in bank() if x['record']['split']=='train']
    bs,ba=jax.tree.map(j.asarray,merged(items,'states')),jax.tree.map(j.asarray,merged(items,'aux'))
    keys=jax.random.split(jax.random.PRNGKey(2026102411),8)
    expected=jax.jit(new.BASE_FRESH)(keys,bs,ba)
    monkeypatch.setattr(T,'fresh',T.fresh);monkeypatch.setattr(T,'transition',T.transition)
    monkeypatch.setattr(new,'PROVENANCE_INSTALLED',False)
    new.install_provenance()
    actual=jax.jit(T.fresh)(keys,bs,ba)
    st,ax=actual;clean={k:v for k,v in ax.items() if k!='provenance_source'}
    assert digest(jax.device_get(expected))==digest(jax.device_get((st,clean)))
    assert (np.asarray(ax['provenance_source'])>=0).any()
    actions=j.tile(j.array([[[4,0],[4,0]]]),(8,1,1))
    before=jax.jit(new.BASE_TRANSITION)(st,clean,actions,keys)
    after=jax.jit(T.transition)(st,ax,actions,keys)
    aa=after[1];after=(after[0],{k:v for k,v in aa.items() if k!='provenance_source'},*after[2:])
    assert digest(jax.device_get(before))==digest(jax.device_get(after))
    np.testing.assert_array_equal(aa['provenance_source'],ax['provenance_source'])
