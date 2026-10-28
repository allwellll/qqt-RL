"""Additional native identity, topology batch and boundary regression cases."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAXBOMB_RULE='bun')
import jax
import jax.numpy as jnp
import numpy as np
from scripts import je08_bun_potential as p,je08_bun_ops as ops,je07_bun_scenes as scenes
from tests.native_bun_fixtures import fixture
from tests.test_je08_topology import event,started


def test_drop_retirement_and_carry_repickup_preserve_cap():
    s=started();_,s,_=p.phase_reward(event(10,8),s,.05,2)
    _,s,_=p.phase_reward(event(eligible=False,boundary=True),s,.05,2)
    r,again,m=p.phase_reward(event(),s,.05,2)
    assert not again['active'] and not m['phase_start'] and r==0
    r,s,m=p.phase_reward(event(identity=2),s,.05,2)
    assert m['phase_start'] and s['gross']==0
    s=started();_,s,_=p.phase_reward(event(10,8),s,.0625,4,True)
    _,s,_=p.phase_reward(event(eligible=False,bad=True,boundary=True),s,.0625,4,True)
    gross=s['gross'];r,s,m=p.phase_reward(event(),s,.0625,4,True)
    assert m['phase_start'] and r==0 and s['gross']==gross


def test_weighted_batched_maps_native_opened_partial():
    scenes.bind_training()
    from jax_bomb import je07_bun_env as env
    s=fixture({})
    walls=np.stack([s.wall]*3);bricks=np.stack([s.brick,np.zeros_like(s.brick),np.asarray(s.brick).copy()])
    bricks[2,::2]=False
    goals=np.broadcast_to(np.asarray(scenes.goals),(3,2,13,15))
    got=np.asarray(jax.jit(p.shortest)(jnp.asarray(walls[:,None]),jnp.asarray(bricks[:,None]),jnp.asarray(goals)))
    for b in range(3):
        for team in range(2):np.testing.assert_array_equal(got[b,team],p.cpu_shortest(walls[b],bricks[b],goals[b,team]))


def test_locked_target_extra_loose_and_other_player_pickup_ends():
    from jax_bomb import je07_bun_env as env
    s=fixture(dict(pos=[[5.5,4.5],[5.5,6.5]],loose=[[5,6,1,1],[5,4,1,1]]))
    batch=lambda q:jax.tree.map(lambda x:x[None],q)
    a=ops.reward_state(1)
    a['drop']['nonce']=a['drop']['nonce'].at[0,1].set(3)
    a['drop']['credit']=a['drop']['credit'].at[0,1].set(0)
    a['routing']['target']=a['routing']['target'].at[0,1].set(jnp.array([5,6]))
    # Opponent legally recovers the locked loose object while another object
    # of the same origin exists closer to learner. No nearest-target switching.
    nxt,_,info=env.step(s,jnp.array([[4,0],[4,0]]),jax.random.PRNGKey(9),False,True)
    _,a,_,m=ops.combined_reward(batch(s),batch(nxt),batch(info),a,jnp.ones(1,bool))
    np.testing.assert_array_equal(a['routing']['target'][0,1],[5,6])
    assert not a['potential']['drop']['active'][0,0] and m['drop_approach_net'][0,0]==0


def test_capture_boundary_no_delta_and_no_region_double_count():
    from jax_bomb import je07_bun_env as env
    s=fixture(dict(carry=[1,-1],pos=[[2.5,2.5],[5.5,9.5]]))
    s=s._replace(core=s.core._replace(pos=s.pos.at[0].set(env._BUN_BASES[0]+.5)))
    batch=lambda q:jax.tree.map(lambda x:x[None],q)
    nxt,_,info=env.step(s,jnp.array([[4,0],[4,0]]),jax.random.PRNGKey(4),False,True)
    _,a,_,m=ops.combined_reward(batch(s),batch(nxt),batch(info),ops.reward_state(1),jnp.ones(1,bool))
    assert info['capture'][0] and m['capture'][0,0]==24
    assert m['carry_home_net'][0,0]==0 and m['shaping'][0,0]==0


def test_combined_conditional_steal_shares_cap():
    from jax_bomb import je07_bun_env as env
    s=fixture(dict(carry=[1,-1],pos=[[5.5,6.02],[5.5,9.5]]))
    batch=lambda q:jax.tree.map(lambda x:x[None],q)
    wait=jnp.array([[4,0],[4,0]])
    nxt,_,info=env.step(s,wait,jax.random.PRNGKey(4),False,True)
    _,a,_,_=ops.combined_reward(batch(s),batch(nxt),batch(info),ops.reward_state(1),jnp.ones(1,bool))
    a['transport']['episode_gross']=a['transport']['episode_gross'].at[0,0].set(.75)
    a['potential']['carry']['gross']=a['potential']['carry']['gross'].at[0,0].set(3.25)
    after,_,info=env.step(nxt,jnp.array([[2,0],[4,0]]),jax.random.PRNGKey(4),False,True)
    _,a,_,m=ops.combined_reward(batch(nxt),batch(after),batch(info),a,jnp.ones(1,bool))
    assert m['carry_home_positive'][0,0]==0
    assert a['potential']['carry']['gross'][0,0]+a['transport']['episode_gross'][0,0]<=4


def test_phase_specific_funnel_metrics_are_separate():
    for phase in ['drop_approach','carry_home']:
        for name in ['stationary','phase_start','phase_end','cap_hit','topology_rebase','clawback']:
            assert phase+'_'+name in ops.POTENTIAL_METRICS
            assert phase+'_'+name in ops.FUNNEL
