import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scripts import je03_bun831_20261011 as s


class Bun831Test(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        patcher = patch.object(s, 'sha', return_value='external-checkpoint-hash')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_actual_adam_updates_respect_top_fraction_and_rounding(self):
        from scripts.je03_bun831_accounting_20261011 import actual_updates
        self.assertEqual(actual_updates(1024,128,4096,2,1.),128)
        self.assertEqual(actual_updates(1024,128,4096,2,.25),32)
        self.assertEqual(actual_updates(2,2,4,2,.25),2)

    def test_evaluator_import(self):
        from scripts import je03_bun831_evaluator_20261011 as evaluator
        self.assertEqual(len(evaluator.TASKS), 4)

    def test_native_bun_and_two_attribution_tracks(self):
        p = s.plan()
        self.assertEqual(p['architecture']['obs'], [24, 13, 15])
        self.assertEqual(p['architecture']['state'], 24)
        self.assertEqual(p['architecture']['ability_actions'], 3)
        self.assertFalse(p['stage2_enabled'])
        self.assertEqual(p['generations'], 4096)
        self.assertEqual(p['ticks_per_arm'], 4096 * 524288)
        self.assertEqual(p['tracks']['teaching']['arms'], ['bun392_base', 'bun392_recipe'])
        self.assertEqual(p['tracks']['capacity_initialization']['arms'], ['bun192_base', 'bun392_base'])
        self.assertIn('joint package', p['tracks']['capacity_initialization']['limitation'])
        self.assertFalse(p['opponents']['dynamic'])

    def test_migration_copies_shared_semantics_not_battle_global_tail(self):
        old = dict(tok=(np.ones((126, 392), np.float32), np.ones(392)),
                   state_w=np.ones((24, 392)), state_b=np.ones(392),
                   heads=dict(wb=(np.ones((392, 2)), np.array([.1, .2]))))
        fresh = dict(tok=(np.full((216, 392), .03), np.zeros(392)),
                     state_w=np.full((24, 392), .02), state_b=np.zeros(392),
                     heads=dict(wb=(np.full((392, 3), .01), np.zeros(3))))
        q = s.migrate(old, fresh)
        np.testing.assert_array_equal(q['tok'][0][:126], old['tok'][0])
        np.testing.assert_array_equal(q['tok'][0][126:], fresh['tok'][0][126:])
        np.testing.assert_array_equal(q['state_w'][:11], old['state_w'][:11])
        np.testing.assert_array_equal(q['state_w'][11:], fresh['state_w'][11:])
        np.testing.assert_array_equal(q['heads']['wb'][0][:, :2], old['heads']['wb'][0])
        self.assertEqual(q['heads']['wb'][1][2], 0)
        self.assertEqual(old['tok'][0].shape, (126, 392))

    def test_decision_requires_future_common_confirmation_and_multiple_points(self):
        base = dict(safe_win=0, death_episode=41, self_episodes=10)
        bad = dict(safe_win=2, death_episode=80, self_episodes=40)
        self.assertEqual(s.decide(base, {256: bad, 512: bad})['action'], 'WAIT_COMMON_768')
        self.assertEqual(s.decide(base, {256: bad, 512: bad, 768: bad})['action'], 'STOP_ATOMIC')
        better = dict(safe_win=10, death_episode=45, self_episodes=11)
        self.assertEqual(s.decide(base, {256: bad, 512: bad, 768: better})['action'], 'CONTINUE')
        self.assertEqual(s.decide(base, {512: bad, 768: bad})['action'], 'WAIT_COMPLETE_WINDOW')

    def test_gate_cannot_promote_cpu_smoke_to_gpu_ready(self):
        doc = dict(passed=True, platform='cpu', finite=True, nonzero=True, full_restore=True)
        with self.assertRaises(AssertionError): s.require_gpu_gate(doc)

    def test_recovery_must_include_ema_rng_env_and_adam(self):
        doc = dict(passed=True, platform='gpu', finite=True, nonzero=True, full_restore=True,
                   smoke_digest='A', replay_digest='B')
        with self.assertRaises(AssertionError): s.require_gpu_gate(doc)
        doc['replay_digest'] = 'A'
        s.require_gpu_gate(doc)

    def test_inputs_have_same_recipe_pair_gen0(self):
        p = s.plan()
        self.assertEqual(p['initialization']['bun392_recipe'], p['initialization']['bun392_base'])
        self.assertEqual(s.recipe('bun392_base'), s.recipe('bun192_base'))
        self.assertNotEqual(s.recipe('bun392_base'), s.recipe('bun392_recipe'))


if __name__ == '__main__': unittest.main()
