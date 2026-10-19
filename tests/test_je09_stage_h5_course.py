import os
os.environ.setdefault('JAXBOMB_RULE','bun')
import unittest
import jax.numpy as jnp
import jax
from scripts import je09_stage_h5_course as c


class CourseTests(unittest.TestCase):
    def test_annealing_keeps_strong_opponents(self):
        self.assertEqual([float(c.weak_fraction(g)) for g in [1,9,17,25]],
                         [.5,.35,.2,.1])
        self.assertEqual([int(c.level(g)) for g in [1,9,17,25]],[0,1,2,3])

    def test_roles_only_natural_full(self):
        role=c.roles(jnp.array([0,9,8,8,8]),jnp.array([0.,0.,.1,.3,.9]),1)
        self.assertEqual(role.tolist(),[0,0,1,2,0])

    def test_origin_accounting_is_not_capture(self):
        self.assertEqual(c.acquisition_kind(jnp.array([0,1]),jnp.array([0,0])).tolist(),[4,1])

    def test_opponent_is_readonly_actions_and_native_pickup(self):
        from scripts import je09_stage_h5c_dagger as d,je09_bun_scenes as scenes
        from jax_bomb import je07_bun_env as env
        env.prepare()
        st=scenes.init_batch(jax.random.PRNGKey(3),2,jnp.eye(10)[8])
        # Test fixture: the next native move crosses the real enemy-cage edge;
        # the course itself has no state writing API and never grants custody.
        base=env._BUN_BASES[0]
        pos=st.pos.at[:,1].set(base.astype(jnp.float32)+jnp.array([1.5,3.05]))
        core=st.core._replace(pos=pos,brick=jnp.zeros_like(st.brick),wall=jnp.zeros_like(st.wall))
        st=st._replace(core=core,scene=jnp.full(2,8,jnp.int8))
        old=st.bun_carried.copy();aux=dict(dagger=d.state(2))
        proposal=jnp.tile(jnp.array([2,0]),(2,1))
        action=c.actions(st,aux,jnp.ones(2,jnp.int32),proposal,jnp.ones((2,5),bool),
            jnp.ones((2,3),bool),jax.random.PRNGKey(2),jnp.ones(2,jnp.int32),1,jnp.zeros((2,13,15)))
        self.assertTrue(bool(jnp.array_equal(old,st.bun_carried)))
        self.assertEqual(action.shape,(2,2))
        # Authoritative Bun step performs stock decrement and carry assignment.
        joint=jnp.zeros((2,2,2),jnp.int32).at[:,0,0].set(4).at[:,1,0].set(2)
        after,_,info=jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True))(
            st,joint,jax.random.split(jax.random.PRNGKey(7),2))
        self.assertTrue(bool((after.bun_carried[:,1]==0).all()))
        self.assertTrue(bool(info['steal'][:,1].all()))


if __name__=='__main__':unittest.main()
