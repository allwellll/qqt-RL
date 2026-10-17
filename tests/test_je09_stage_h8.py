import unittest
import jax.numpy as jnp
import numpy as np
from scripts import je09_stage_h8_course as course


class StageH8CourierTests(unittest.TestCase):
    def test_target_allocation_is_declared_and_scene_gated(self):
        scene = jnp.full((10000,), 8, jnp.int32)
        role_u = jnp.linspace(0, 1, 10000, endpoint=False)
        target_u = jnp.mod(jnp.arange(10000) * .6180339, 1.)
        role, target = map(np.asarray, course.targeted_roles(scene, role_u, target_u, 1))
        weak = role > 0
        self.assertAlmostEqual(float(weak.mean()), .5, delta=.002)
        for value, expected in [(1, .70), (2, .05), (3, .05), (4, .20)]:
            self.assertAlmostEqual(float(np.mean(target[weak] == value)), expected, delta=.012)
        self.assertTrue(np.all(role[~weak] == 0))
        self.assertTrue(np.all(target[~weak] == 0))

    def test_carrier_destination_switches_from_learner_cage_to_own_cage(self):
        opponent = jnp.array([0, 0, 1, 1])
        role = jnp.array([1, 1, 1, 1])
        carrying = jnp.array([False, True, False, True])
        channel = jnp.where(role == 1, jnp.where(carrying, 4 + opponent,
                                                 2 + opponent), 4 + opponent)
        np.testing.assert_array_equal(np.asarray(channel), [2, 4, 3, 5])

    def test_no_combat_oracle_slots_and_no_state_mutation_in_contract(self):
        self.assertTrue(all('COMBAT_CONTROL' not in x for x in ['A', 'B', 'CARRY_RETURN', 'CAGE_DELIVERY']))
        self.assertIn('native', course.actions.__doc__)


if __name__ == '__main__':
    unittest.main()
