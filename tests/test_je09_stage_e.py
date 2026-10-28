"""Stage E coverage and native provenance TDD, independent of old sources."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np

def fixture(n=12):
 from scripts.je09_bun_bridge import jax_bridge_state
 b=jax_bridge_state(n);b.update(active=jnp.ones((n,2),bool),paid=jnp.ones((n,2,7),bool),start_distance=jnp.full((n,2),34.),ticks=jnp.zeros((n,2),int))
 st=SimpleNamespace(scene=jnp.full(n,9),alive=jnp.ones((n,2),bool),bun_carried=jnp.tile(jnp.array([1,0]),(n,1)),trapped=jnp.zeros((n,2),int))
 aux=dict(bridge=b,terminal=jnp.zeros(n,bool),chase=dict(initial=jnp.full((n,2),32),nonce=jnp.ones((n,2),int)),drop=dict(nonce=jnp.ones((n,2),int),credit=jnp.tile(jnp.array([1,0]),(n,1))))
 return st,aux

def test_native_gate_distance_and_exact_exclusions():
 from scripts.je09_stage_e_coverage import gate
 from scripts.je09_return_expert import supervised_loss
 st,a=fixture();a['bridge']['paid']=a['bridge']['paid'].at[1,:,0].set(False).at[2,:,1].set(False)
 st.bun_carried=st.bun_carried.at[3,0].set(-1).at[4,0].set(0)
 a['bridge']['ticks']=a['bridge']['ticks'].at[5].set(120)
 a['terminal']=a['terminal'].at[6].set(True);st.scene=st.scene.at[7].set(0)
 st.alive=st.alive.at[8,0].set(False);st.trapped=st.trapped.at[9,0].set(1)
 a['bridge']['start_distance']=a['bridge']['start_distance'].at[10].set(37)
 a['drop']['credit']=a['drop']['credit'].at[11,1].set(-1)
 active=gate(st,a,jnp.zeros(12,int));np.testing.assert_array_equal(active,jnp.arange(12)==0)
 target=jnp.tile(jnp.array([.5,0,.5,0,0]),(12,1));mask=jnp.ones((12,5),bool)
 grad=jax.grad(lambda q:supervised_loss(q,mask,target,active,.05,.5)[0])(jnp.zeros((12,5)))
 assert grad[0,0]==grad[0,2] and grad[0,0]<0;np.testing.assert_array_equal(grad[1:],0)

def test_mixture_exact_and_disjoint():
 from scripts.je09_stage_e_coverage import coverage_choice
 for f in [.25,.4,.5]:
  long=coverage_choice(jnp.arange(1024),f);assert int(long.sum())==int(1024*f)
  assert float((~long).mean())>=.5

def test_distance_tables_native_initialization():
 from scripts.je09_stage_e_coverage import tables
 short,long,counts=tables()
 assert short.shape[:2]==(5,2) and long.shape[:2]==(5,2)
 assert np.all(np.asarray(counts)>0)

def test_integrated_safety_unreachable_and_zero_aux_gradient():
 from scripts.je09_stage_e_coverage import labels
 from scripts.je09_return_expert import supervised_loss
 from scripts.je08_bun_potential import shortest,INF
 st,a=fixture(4);st.pos=jnp.full((4,2,2),3.5)
 st.wall=jnp.zeros((4,7,7),bool);st.brick=jnp.zeros_like(st.wall);st.fuse=jnp.zeros((4,7,7))
 st.core=SimpleNamespace(blast_linger=jnp.zeros_like(st.fuse).at[3,3,3].set(1))
 goals=jnp.zeros_like(st.wall).at[:,1,1].set(True);d=shortest(st.wall,st.brick,goals)
 a['return_expert']=dict(distance=jnp.stack([d,d],1).at[2].set(INF))
 danger=jnp.zeros_like(st.fuse).at[1,3,3].set(1);mask=jnp.ones((4,5),bool)
 target,active=labels(st,a,mask,jnp.zeros(4,int),danger)
 np.testing.assert_array_equal(active,jnp.array([True,False,False,False]))
 grad=jax.grad(lambda q:supervised_loss(q,mask,target,active,.05,.5)[0])(jnp.zeros((4,5)))
 assert grad[0,0]==grad[0,2] and grad[0,0]<0;np.testing.assert_array_equal(grad[1:],0)

def test_production_adapter_source_differences_are_limited():
 from pathlib import Path
 original=Path('scripts/je09_bun_ops.py').read_text();current=Path('scripts/je09_stage_e_ops.py').read_text()
 # The native reward path is imported unchanged; no actor feature or override.
 assert 'def combined_reward' not in current
 assert "expert_coefficient=expert_coefficient" not in current or 'return_replay' in current
 assert "context=jnp.concatenate([context,aux['bridge']['start_distance']" in current
 for text in ["ratio=jnp.exp(logp-old[ii[:half]])","value=train.hl_gauss_value_loss(vl[half:],ret[ii[half:]])"]:
  assert text in current and text in original

def test_chase_and_return_never_label_the_same_frame():
 from tests.test_je09_chase_expert import fixture as chase_fixture
 from scripts.je09_stage_e_coverage import gate as return_gate
 from scripts.je09_chase_expert import gate as chase_gate
 from scripts.je09_bun_bridge import jax_bridge_state
 st,a=chase_fixture();n=len(st.scene);b=jax_bridge_state(n)
 b.update(active=jnp.ones((n,2),bool),paid=jnp.ones((n,2,7),bool),start_distance=jnp.full((n,2),34.))
 a['bridge']=b;a['chase']['initial']=jnp.full((n,2),32)
 for carried in [-1,0,1]:
  st.bun_carried=st.bun_carried.at[:,0].set(carried)
  r=return_gate(st,a,jnp.zeros(n,int));c=chase_gate(st,a,jnp.zeros(n,int))
  assert not bool((r&c).any())
