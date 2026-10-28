from tests.external_artifacts import requires_artifacts
import inspect
import numpy as np
import pytest
import jax
import jax.numpy as j
from scripts import combat_item_engine as T
from scripts import dagger_rl_r4_train as P


def legacy(keys,bs,ba):
    split=jax.vmap(lambda k:jax.random.split(k,4))(keys)
    st,ax=T.OLD_INITIAL(split[:,0],0)
    choice=jax.vmap(lambda k:jax.random.randint(k,(),0,bs.t.size))(split[:,1])
    use=jax.vmap(lambda k:jax.random.bernoulli(k,.25))(split[:,2])
    item=jax.vmap(lambda k:jax.random.randint(k,(),0,3))(split[:,3])
    slots=st.item_slots.at[:,0,0].set(item.astype(st.item_slots.dtype))
    counts=st.item_counts.at[:,0,0].set((item>0).astype(st.item_counts.dtype))
    st=st._replace(item_slots=slots,item_counts=counts,held_item=slots[:,:,0])
    return T.E.select_tree(use,jax.tree.map(lambda a:a[choice],bs),st),dict(T.E.select_tree(use,jax.tree.map(lambda a:a[choice],ba),ax),tactics=T.events.state(len(keys)))


@requires_artifacts('runs/carrier_mechanism_20261023/v1/BANK_FROZEN.json')
def test_default_fresh_bitwise_and_provenance_probability(monkeypatch):
    T.configure()
    from scripts.carrier_mechanism_analysis import bank,merged
    items=[x for x in bank() if x['record']['split']=='train']
    bs,ba=merged(items,'states'),merged(items,'aux')
    keys=jax.random.split(jax.random.PRNGKey(2026103180),16)
    old=jax.jit(legacy)(keys,bs,ba)
    actual=jax.jit(T.fresh)(keys,bs,ba)
    for a,b in zip(jax.tree.leaves(old),jax.tree.leaves(actual)):np.testing.assert_array_equal(a,b)
    monkeypatch.setattr(P,'PROVENANCE_INSTALLED',False)
    monkeypatch.setattr(T,'fresh',T.fresh);monkeypatch.setattr(T,'transition',T.transition)
    P.install_provenance()
    default=jax.jit(T.fresh)(keys,bs,ba)
    split=jax.vmap(lambda k:jax.random.split(k,4))(keys)
    use25=jax.vmap(lambda k:jax.random.bernoulli(k,.25))(split[:,2])
    choice25=jax.vmap(lambda k:jax.random.randint(k,(),0,bs.t.size))(split[:,1])
    for a,b in zip(jax.tree.leaves(default[0]),jax.tree.leaves(old[0])):np.testing.assert_array_equal(a,b)
    for key in old[1]:
        for a,b in zip(jax.tree.leaves(default[1][key]),jax.tree.leaves(old[1][key])):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(default[1]['provenance_source'],np.where(use25,choice25,-1))
    monkeypatch.setattr(P,'PROVENANCE_INSTALLED',False)
    P.install_provenance(.125)
    reduced=jax.jit(T.fresh)(keys,bs,ba)
    split=jax.vmap(lambda k:jax.random.split(k,4))(keys)
    use=jax.vmap(lambda k:jax.random.bernoulli(k,.125))(split[:,2])
    choice=jax.vmap(lambda k:jax.random.randint(k,(),0,bs.t.size))(split[:,1])
    np.testing.assert_array_equal(reduced[1]['provenance_source'],np.where(use,choice,-1))
    for a,b in zip(jax.tree.leaves(reduced[0]),jax.tree.leaves(bs)):
        if np.any(use):np.testing.assert_array_equal(np.asarray(a)[use],np.asarray(b)[np.asarray(choice)[use]])
    with pytest.raises(ValueError,match='cannot change'):P.install_provenance(.25)


def test_probability_validation():
    with pytest.raises(ValueError):T.fresh(None,None,None,1.1)
    with pytest.raises(ValueError):P.install_provenance(-.1)
