from tests.external_artifacts import requires_artifacts
import jax
import jax.numpy as j
import numpy as np
from scripts import combat_mix_legacy as L


def test_legacy_curriculum_reuses_je10_exact_native_weights_and_excludes_fake_carry():
    got=np.asarray(L.course())
    assert np.allclose(got[[0,8,9]],[.4,.35,.25])
    assert np.all(got[[2,3,4,6]]==0)
    assert np.isclose(got.sum(),1)


def test_legacy_current_and_four_ring_slots_and_fixed_members_are_preserved():
    pool={'p':j.arange(11,dtype=j.float32).reshape(11,1)}
    parent={'p':j.array([19.])}
    pool=L.initial_pool(pool,parent)
    assert np.array_equal(np.asarray(pool['p'][1:7,0]),np.arange(1,7))
    assert np.all(np.asarray(pool['p'][j.array([0,7,8,9,10]),0])==19)
    changed=L.advance_pool(pool,{'p':j.array([21.])},16)
    assert float(changed['p'][0,0])==21 and float(changed['p'][7,0])==21
    assert np.array_equal(np.asarray(changed['p'][1:7]),np.asarray(pool['p'][1:7]))


def test_legacy_adapter_calls_historical_collector_and_does_not_rewrite_rule_dispatch():
    import inspect
    source=inspect.getsource(L.raw)
    assert 'ops.collect(' in source
    assert 'scenes.init_batch' not in source
    assert 'bun_carried=' not in source


@requires_artifacts('runs/combat_mix_v1/PROTOCOL.json', 'runs/je10_bun_20261018/control_route_v38_r10/inputs/frozen.pkl')
def test_legacy_real_collector_one_tick_shapes_and_finite():
    import pickle,json
    from pathlib import Path
    from scripts import carrier_mechanism_engine as E,combat_item_engine as T
    T.configure()
    doc=json.loads(Path('runs/combat_mix_v1/PROTOCOL.json').read_text())
    x=pickle.load((Path(doc['sources']['2026103051']['checkpoint'])/'state.pkl').open('rb'))
    st=jax.tree.map(lambda a:j.asarray(a[0,:1]),x['states'])
    ax=L.base_aux(jax.tree.map(lambda a:j.asarray(a[0,:1]),x['aux']))
    keys=j.asarray(x['keys'][0,:1])
    base=Path('runs/je10_bun_20261018/control_route_v38_r10/inputs')
    f=pickle.load((base/'frozen.pkl').open('rb'))
    f={k:f[k] for k in ('g0','stage1_teacher')};f['parent']=x['params']['actor']
    pool=L.initial_pool(pickle.load((base/'initial_pool.pkl').open('rb')),x['params']['actor'])
    pf=E.ops.pfsp_weights(j.zeros((11,6)))
    result=jax.jit(lambda p,s,a,k:L.raw(1,p,s,a,k,pool,f,pf,j.int32(161)))(x['params'],st,ax,keys)
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(result))
    end,batch,diag=result
    assert batch[0].shape[:2]==(1,1) and len(batch)==8
    assert 'scene_opponent_counts' in diag
