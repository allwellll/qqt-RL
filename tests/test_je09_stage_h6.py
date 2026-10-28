import unittest
from unittest.mock import patch
import jax.numpy as jnp
import numpy as np
from scripts import je09_stage_h6_ops as ops
from scripts import je09_stage_h5_course as opponent_course
from scripts import je09_stage_h5_course as opponent_course
from scripts.je09_stage_h6_fragments import Fragment, current_fragment, verify_teacher
from scripts.je09_stage_h6_declare import lineage_version, parent_version, successor_version
from scripts.je09_stage_h6 import schema_version


class StageH6Tests(unittest.TestCase):
    def test_declaration_lineage_advances_parent_and_successor_versions(self):
        self.assertEqual(lineage_version('JE09-STAGE-H6-v30'), 'v30')
        self.assertEqual(lineage_version('JE09-STAGE-H6-v31'), 'v31')
        self.assertEqual(lineage_version('JE09-STAGE-H6-v32'), 'v32')
        self.assertEqual(parent_version('JE09-STAGE-H6-v31'), 'v30')
        self.assertEqual(parent_version('JE09-STAGE-H6-v32'), 'v31')
        self.assertEqual(successor_version('JE09-STAGE-H6-v31'), 'v32')
        self.assertEqual(successor_version('JE09-STAGE-H6-v32'), 'v33')

    def test_v31_namespace_selects_its_own_state_schema(self):
        self.assertEqual(schema_version('runs/je09_bun_20261017/stage_h6_target_stratified_fragment_dagger_v31'), 'v31')
        self.assertEqual(schema_version('runs/je09_bun_20261017/stage_h6_target_stratified_fragment_dagger_v32'), 'v32')

    def test_targeted_course_balances_fragments_without_changing_weak_mass(self):
        scene = jnp.full((4096,), 8, jnp.int32)
        role_u = jnp.linspace(0, 1, 4096, endpoint=False)
        target_u = jnp.mod(jnp.arange(4096) * .6180339, 1.)
        role, target = opponent_course.targeted_roles(scene, role_u, target_u, 1)
        role = np.asarray(role); target = np.asarray(target)
        weak = role > 0
        self.assertAlmostEqual(float(weak.mean()), .5, delta=.01)
        for fragment in range(1, 5):
            self.assertGreater(int(np.sum(weak & (target == fragment))), 400)
        self.assertTrue(np.all(target[role == 0] == 0))
        self.assertEqual(set(np.unique(role[weak])), {1, 2})

    def test_verifier_slots_are_independent_and_ignore_missing_fragment_rows(self):
        eligible = jnp.array([1] * 12 + [1] + [0] * 3, dtype=jnp.bool_)
        fragment = jnp.array([1] * 4 + [2] * 4 + [3] * 4 + [4] + [0] * 3, dtype=jnp.int32)
        score = jnp.arange(16, dtype=jnp.float32)
        ids, selected_slots, selected_rows = ops.select_verifier_slots(
            eligible, fragment, score, slots_per_fragment=4)
        self.assertEqual(ids.shape, (16,))
        self.assertEqual(int(selected_slots.sum()), 13)
        self.assertEqual(int(selected_rows.sum()), 13)
        np.testing.assert_array_equal(np.asarray(fragment[ids[selected_slots]]),
            np.repeat(np.arange(1, 5), [4, 4, 4, 1]))
        with self.assertRaises(ValueError):
            ops.select_verifier_slots(eligible, fragment, score, slots_per_fragment=0)

    def test_declared_verifier_slot_count_is_read_from_worker_environment(self):
        with patch.dict('os.environ', {'JE09_H6_VERIFIER_SLOTS': '4'}):
            self.assertEqual(ops.configured_verifier_slots(), 4)

    def test_targeted_course_balances_fragments_without_changing_weak_mass(self):
        scene = jnp.full((4096,), 8, jnp.int32)
        ru = jnp.linspace(0, 1, 4096, endpoint=False)
        tu = jnp.mod(jnp.arange(4096) * 0.6180339, 1.)
        role, target = opponent_course.targeted_roles(scene, ru, tu, 1)
        role = np.asarray(role); target = np.asarray(target)
        weak = role > 0
        self.assertAlmostEqual(float(weak.mean()), .5, delta=.01)
        for fragment in range(1, 5):
            self.assertGreater(int(np.sum(weak & (target == fragment))), 400)
        self.assertTrue(np.all(target[role == 0] == 0))
        self.assertEqual(set(np.unique(role[weak])), {1, 2})

    def test_complete_games_get_declared_mass_without_reducing_bridge(self):
        weights = np.asarray(ops.curriculum_weights(jnp.zeros((10, 6))))
        self.assertAlmostEqual(float(weights[8]), .35, places=6)
        self.assertAlmostEqual(float(weights[9]), .25, places=6)
        self.assertAlmostEqual(float(weights[0]), .40, places=6)
        self.assertAlmostEqual(float(weights.sum()), 1., places=6)

    def test_fragment_labels_are_mutually_exclusive_and_objective_gated(self):
        ctx = jnp.zeros((6, 60), jnp.float32)
        ctx = ctx.at[:, 0].set(jnp.array([8, 8, 8, 8, 0, 8]))
        ctx = ctx.at[:, 55].set(jnp.array([1, 1, 1, 1, 1, 0]))
        ctx = ctx.at[:, 54].set(jnp.array([1, 2, 3, 3, 2, 2]))
        ctx = ctx.at[:, 57].set(jnp.array([0, 0, 0, 1, 0, 0]))
        got = np.asarray(ops.earliest_fragment(ctx))
        np.testing.assert_array_equal(got, [1, 2, 3, 4, 0, 0])

    def test_each_objective_fragment_gets_its_own_first_deviation(self):
        mismatch = jnp.array([[1], [1], [1], [1], [1], [1], [1]], dtype=jnp.bool_)
        fragment = jnp.array([[1], [1], [2], [3], [4], [2], [1]], dtype=jnp.int32)
        terminal = jnp.zeros_like(mismatch)
        got = np.asarray(ops.first_fragment_deviations(mismatch, fragment, terminal))
        np.testing.assert_array_equal(got[:, 0], [True, False, True, True, True, False, False])

    def test_combat_has_no_fragment_query_and_terminal_resets_episode_quota(self):
        mismatch = jnp.array([[1], [1], [1], [1]], dtype=jnp.bool_)
        fragment = jnp.array([[0], [2], [2], [2]], dtype=jnp.int32)
        terminal = jnp.array([[0], [0], [1], [0]], dtype=jnp.bool_)
        got = np.asarray(ops.first_fragment_deviations(mismatch, fragment, terminal))
        np.testing.assert_array_equal(got[:, 0], [False, True, False, True])

    def test_causal_current_state_machine_and_reject_first_cases(self):
        base = dict(terminal=False, learner_alive=True, learner_trapped=False,
            teammate_harmed=False, enemy_alive=True, enemy_trapped=False,
            attributed_window=False, carrying_enemy=False, loose_attributed=False,
            stock_in_enemy_cage=False, at_own_cage=False, reachable=True,
            legal_safe_actions=True, provenance_clear=True)
        cases = [
            (dict(), Fragment.COMBAT_CONTROL),
            (dict(attributed_window=True, stock_in_enemy_cage=True), Fragment.B_CAGE_STOCK),
            (dict(attributed_window=True, loose_attributed=True), Fragment.A_LOOSE_RECOVERY),
            (dict(carrying_enemy=True), Fragment.CARRY_RETURN),
            (dict(carrying_enemy=True, at_own_cage=True), Fragment.CAGE_DELIVERY),
            (dict(glue_only=True), Fragment.COMBAT_CONTROL),
            (dict(banana_only=True), Fragment.COMBAT_CONTROL),
            (dict(attributed_window=True), Fragment.AMBIGUOUS),
            (dict(enemy_alive=False), Fragment.AMBIGUOUS),
            (dict(reachable=False), Fragment.AMBIGUOUS),
            (dict(provenance_clear=False), Fragment.AMBIGUOUS),
            (dict(terminal=True), Fragment.AMBIGUOUS),
        ]
        for changes, expected in cases:
            self.assertEqual(current_fragment(**(base | changes)), expected)

    def test_counterfactual_verifier_rejects_incomplete_or_unsafe_teacher(self):
        safe = dict.fromkeys(['no_self_death', 'no_self_trap', 'no_teammate_harm',
            'no_route_block', 'time_expanded_hazard_clear', 'native_transition_consistent'], True)
        self.assertEqual(verify_teacher([1, 2], [0, 1, 2], {1: safe, 2: safe}), (True, 'accepted'))
        self.assertEqual(verify_teacher([1, 3], [0, 1, 2], {1: safe, 3: safe})[0], False)
        self.assertEqual(verify_teacher([1], [0, 1], {1: {'no_self_death': True}})[0], False)
        unsafe = dict(safe, no_teammate_harm=False)
        self.assertEqual(verify_teacher([1], [0, 1], {1: unsafe})[0], False)


if __name__ == '__main__':
    unittest.main()
