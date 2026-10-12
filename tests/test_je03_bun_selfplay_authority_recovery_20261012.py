import os
import tempfile
import unittest
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAXBOMB_RULE='bun')
from scripts import je03_bun_selfplay_authority_recovery_20261012 as recovery
from scripts.population_run_state import atomic_json, lock, sha


class RecoveryTests(unittest.TestCase):
    def test_nested_manifest_is_real_and_has_64_paired_spawns(self):
        self.assertTrue(recovery.manifest().is_file())
        self.assertEqual(len(recovery.read(recovery.manifest())['spawn_cells']), 64)

    def test_decision_points_precede_early_backfill(self):
        self.assertEqual(recovery.PRIORITY[:5], [0, 512, 1024, 1536, 2048])
        self.assertEqual(set(recovery.PRIORITY), set(recovery.stage.POINTS))
        self.assertEqual(len(recovery.PRIORITY), len(set(recovery.PRIORITY)))

    def test_cpu_environment_overrides_inherited_gpu(self):
        env = recovery.cpu_env({'CUDA_VISIBLE_DEVICES': '0,1', 'JAX_PLATFORMS': 'cuda'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['JAX_PLATFORMS'], 'cpu')

    def test_legacy_lock_excludes_second_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            held = lock(Path(tmp) / 'owner.lock')
            with self.assertRaises(BlockingIOError):
                lock(Path(tmp) / 'owner.lock')
            held.close()

    def test_completion_rejects_tampered_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            atomic_json(out / 'task.json', {'result': 1})
            atomic_json(out / 'COMPLETE.json', {'policy_sha256': 'p', 'shards': {'task.json': sha(out / 'task.json')}})
            atomic_json(out / 'COMPLETE_WITH_BUN_TASKS.json', {
                'policy_sha256': 'p', 'authority_sha256': sha(out / 'COMPLETE.json'),
                'additional_shards': {}})
            recovery.validate_complete(out, 'p')
            atomic_json(out / 'task.json', {'result': 2})
            with self.assertRaises(AssertionError):
                recovery.validate_complete(out, 'p')


if __name__ == '__main__':
    unittest.main()
