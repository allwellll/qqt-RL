"""A3SR reset validity, deterministic sampling and native-off equivalence."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import pytest
from scripts import bomb_threat_a3sr_spawn as S


def geometry():
    wall=np.ones((13,15),bool);wall[2:11,2:13]=False
    brick=np.zeros_like(wall);base=np.zeros_like(wall)
    return wall,brick,base


def test_closed_two_cells_and_one_ray_corridor_rejected():
    wall=np.ones((13,15),bool);wall[5,5:7]=False
    assert not S.validate_cell(wall,np.zeros_like(wall),np.zeros_like(wall),np.zeros_like(wall),(5,5),2,.39,30)['valid']
    wall[:]=True;wall[5,2:5]=False
    assert not S.validate_cell(wall,np.zeros_like(wall),np.zeros_like(wall),np.zeros_like(wall),(5,3),8,.39,30)['valid']


def test_crates_are_passable_and_escape_uses_speed_ticks():
    w,b,base=geometry();box=np.zeros_like(w)
    r=S.validate_cell(w,b,base,box,(6,7),2,.39,30)
    assert r['valid'] and r['component_size']>=6 and r['escape_ticks']<30
    assert not S.validate_cell(w,b,base,box,(6,7),2,.01,30)['valid']
    box[6,7]=True
    assert not S.validate_cell(w,b,base,box,(6,7),2,.39,30)['valid']


def test_sample_determinism_fixed_geometry_and_seat_exchangeability():
    level=S.load_level();a=S.sample_numpy(level,123,4096);b=S.sample_numpy(level,123,4096)
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    w=np.asarray(level['wall'],bool).reshape(13,15);br=np.asarray(level['brick'],bool).reshape(13,15)
    np.testing.assert_array_equal(a['wall'],np.broadcast_to(w,a['wall'].shape))
    assert not np.any(a['brick']&~br)
    assert not np.any(a['carried']!=-1)
    assert .45<float(a['spawn_mode'].mean())<.55
    assert np.max(a['brick_drop_rate'])<=.3
    rand=a['spawn_mode'].astype(bool)&~a['spawn_fallback'].astype(bool)
    cells=a['cells'][rand]
    assert np.all(np.abs(cells[:,0]-cells[:,1]).sum(-1)>=6)
    assert np.max(np.abs(cells[:,0].mean(0)-cells[:,1].mean(0)))<.7


def test_default_off_exact_and_jax_numpy_validation_agree():
    import jax,jax.numpy as j
    w,b,base=geometry();box=np.zeros_like(w)
    cells=np.array([[2,2],[6,7],[0,0],[10,12]],np.int32)
    got=np.asarray(jax.jit(jax.vmap(lambda c:S.valid_cell_jax(j.asarray(w),j.asarray(b),j.asarray(base),j.asarray(box),c,2,.39,30)))(j.asarray(cells)))
    want=np.array([S.validate_cell(w,b,base,box,tuple(c),2,.39,30)['valid'] for c in cells])
    np.testing.assert_array_equal(got,want)
    assert S.configure(False)['enabled'] is False


def test_jax_native_noop_and_actual_dropped_layout():
    import jax,jax.numpy as j
    from jax_bomb import je07_bun_env as env
    env.prepare();env.configure_training('full=1',1)
    s=jax.tree.map(lambda a:a[0],env.init_batch(jax.random.PRNGKey(101),1))
    step=jax.jit(S.sample_jax)
    c,m=step(jax.random.PRNGKey(9),s.core)
    assert np.all(np.asarray(c.wall)==np.asarray(s.core.wall))
    assert not np.any(np.asarray(c.brick)&~np.asarray(s.core.brick))
    assert np.all(np.asarray(c.pushable)==np.asarray(s.core.pushable))
    if not bool(m['spawn_mode']):
        for x,y in zip(jax.tree.leaves(c),jax.tree.leaves(s.core)):np.testing.assert_array_equal(x,y)
    for key in range(20):
        c,m=step(jax.random.PRNGKey(key),s.core)
        if bool(m['spawn_mode']) and not bool(m['spawn_fallback']):
            bases=S.base_mask(S.load_level())
            for cell in np.asarray(c.pos).astype(int):assert S.validate_cell(np.asarray(c.wall),np.asarray(c.brick),bases,np.asarray(c.pushable),tuple(cell),2,.39,30)['valid']
            assert np.abs(np.asarray(c.pos[0])-np.asarray(c.pos[1])).sum()>=6


def test_batch_default_off_bank_guard_reset_mask_and_partial_tile():
    import jax,jax.numpy as j
    from jax_bomb import je07_bun_env as env
    env.prepare();env.configure_training('full=1',1)
    n=19;states=env.init_batch(jax.random.PRNGKey(111),n)
    keys=jax.random.split(jax.random.PRNGKey(413),n)
    aux={'provenance_source':j.full(n,-1).at[2].set(0)}
    states=states._replace(bun_carried=states.bun_carried.at[3,0].set(1))
    S.configure(False)
    same,off=S.apply_batch(keys,states,aux)
    for a,b in zip(jax.tree.leaves(same),jax.tree.leaves(states)):np.testing.assert_array_equal(a,b)
    assert not np.asarray(off['spawn_eligible']).any()
    mask=j.arange(n)!=4
    changed,meta=jax.jit(lambda k,s:S.apply_batch(k,s,aux,mask,enabled=True))(keys,states)
    eligible=np.ones(n,bool);eligible[[2,3,4]]=False
    np.testing.assert_array_equal(meta['spawn_eligible'],eligible)
    for idx in (2,3,4):
        for a,b in zip(jax.tree.leaves(changed),jax.tree.leaves(states)):np.testing.assert_array_equal(a[idx],b[idx])
    np.testing.assert_array_equal(changed.bun_spawn_pos,changed.core.pos)
    assert np.all(np.asarray(changed.bun_carried)==np.asarray(states.bun_carried))
    assert not np.any(np.asarray(changed.core.brick)&~np.asarray(states.core.brick))
    # The17th eligible tile tail may never scatter into row0 a second time.
    for idx in np.flatnonzero(eligible & ~np.asarray(meta['spawn_mode'],bool)):
        for a,b in zip(jax.tree.leaves(changed),jax.tree.leaves(states)):np.testing.assert_array_equal(a[idx],b[idx])


def test_worker_two_smoke_updates_and_frozen_reward_path():
    from scripts.bomb_threat_a3sr_train import worker_source
    source=worker_source();compile(source,'a3sr-worker','exec')
    assert "+2) if args.smoke" in source
    assert "spec['iter2_arm'] in ('A3S','A3SR')" in source
    assert "recipe.settlement(5.,.6)" in source
    assert "below18000 env/s" in source
    assert "no eligible real resets in smoke" in source


def test_bounded_protocol_and_no_duplicate_target(tmp_path,monkeypatch):
    import json
    from scripts import bomb_threat_a3sr_schedule as A
    from scripts.population_run_state import atomic_json
    monkeypatch.setattr(A,'ROOT',tmp_path);base=tmp_path/'runs/bomb_threat_v1';monkeypatch.setattr(A,'BASE',base)
    b=base/'iter2/a3sr';monkeypatch.setattr(A,'B',b)
    atomic_json(b/'PROTOCOL.json',dict(status='AUTHORIZED_READY',authorization='task',versioned_sources={}))
    atomic_json(base/'iter2/TRAINING_PROTOCOL.json',dict(common={},source_sha256={},spawn_manifest='spawns',A1_weights=[],fixed_actor_source={}))
    recipe={'iter2_arm':'A3SR','fallback_policy_source':{'checkpoint':'policy','policy_sha256':'sha'}}
    path=A.make_protocol(5,25,None,recipe)
    frozen=json.loads(path.read_text())
    assert frozen['worker_module']=='scripts.bomb_threat_a3sr_train'
    assert frozen['chunk_start_update']==0 and frozen['chunk_updates']==320
    assert A.make_protocol(5,25,None,recipe)==path
    src={'update':320,'checkpoint':'atomic320'}
    second=json.loads(A.make_protocol(5,26,src,recipe).read_text())
    assert second['environment_steps_per_run']==640*262144
    with pytest.raises(AssertionError):A.make_protocol(5,27,{'update':640},recipe)
    assert A.apply_switch(4,{'run':'other'},20) is None


def test_native_verdict_requires_funnel_safety_both_seats():
    import copy
    from scripts.bomb_threat_a3sr_verdict import compare
    control=dict(by_seat={s:dict(win_minus_loss=3,zero_bomb_games=2,self_deaths=0) for s in ('P0','P1')},
        funnel={s:dict(window_pickup=5,same_chain_delivery=2) for s in ('P0','P1')})
    candidate=copy.deepcopy(control);candidate['by_seat']['P1']['win_minus_loss']=8
    assert compare(candidate,control)['status']=='PASS'
    for key in ('window_pickup','same_chain_delivery'):
        bad=copy.deepcopy(candidate);bad['funnel']['P1'][key]-=1
        assert compare(bad,control)['status']=='FAIL'
    bad=copy.deepcopy(candidate);bad['by_seat']['P0']['self_deaths']=1
    assert compare(bad,control)['status']=='FAIL'
    bad=copy.deepcopy(candidate);bad['by_seat']['P0']['zero_bomb_games']=3
    assert compare(bad,control)['status']=='FAIL'
