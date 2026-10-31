import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import slot01_copy07_handoff_20261031 as H
from scripts.bomb_threat_slot01_copy07_train import worker_source


class Slot01Copy07Test(unittest.TestCase):
    def test_live_worker_never_handed_off(self):
        self.assertIsNone(H.apply_switch(1, {'run': 'active'}, {'active': 1}, 60))

    def test_other_slot_untouched(self):
        self.assertIsNone(H.apply_switch(7, {'run': 'active'}, {}, 60))

    def test_global_pause_wins(self):
        with patch.object(H, 'paused', return_value=True):
            self.assertIsNone(H.apply_switch(1, {'run': 'active'}, {}, 60))

    def test_only_source_handoff_rekeys(self):
        source = worker_source()
        compile(source, 'fork-worker', 'exec')
        handoff = source.index('        original_rng_digest=digest(keys)')
        restore = source.index('    if args.verify_restore:')
        self.assertLess(handoff, restore)
        self.assertEqual(source.count("if protocol.get('slot01_copy07_rekey',False):"), 1)
        self.assertIn("assert digest((params,adam,st,ax,ring,ring_cursor,fixed,warm_rows))==retained", source)

    def test_protocol_preserves_recipe_and_budget_and_rekeys_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'runs/bomb_threat_v1'
            (base / 'protocols').mkdir(parents=True)
            (root / 'scripts').mkdir()
            (root / 'scripts/bomb_threat_slot01_copy07_train.py').write_text('fixture')
            source = dict(checkpoint='runs/bomb_threat_v1/formal/bt_slot07_c000040/u005560',
                          update=5560, state_sha256='state', policy_sha256='policy')
            recipe = dict(iter2_arm='A3SR', r1_attack=False, r1_bank_fix=True, lr=7.5e-5)
            template = dict(arms={'bt_slot07_c000040': recipe}, common=dict(devices=1, envs_per_device=2048),
                            chunk_updates=640, round14_collector_seed=2026112905,
                            rule_version='stock-window-v1', reward_version='unchanged', source_sha256={})
            (base / 'protocols/bt_slot07_c000040.json').write_text(json.dumps(template))
            before = copy.deepcopy(template)
            with patch.object(H, 'ROOT', root), patch.object(H, 'BASE', base):
                path = H.protocol(source, 60, True)
                result = json.loads(path.read_text())
                self.assertEqual(result['arms']['bt_slot01_c000060'], recipe)
                self.assertEqual(result['common'], template['common'])
                self.assertEqual(result['round14_collector_seed'], template['round14_collector_seed'])
                self.assertEqual(result['chunk_updates'], 640)
                self.assertEqual(result['environment_steps_per_run'], (5560 + 640) * 262144)
                self.assertEqual(result['r1_seed'], H.TARGET_SEED)
                self.assertTrue(result['slot01_copy07_rekey'])
                self.assertEqual(json.loads((base/'protocols/bt_slot07_c000040.json').read_text()), before)
                self.assertEqual(H.protocol(source, 60, True), path)
                with self.assertRaises(AssertionError):
                    H.protocol(source, 60, False)


if __name__ == '__main__':
    unittest.main()
