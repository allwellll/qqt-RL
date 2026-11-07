"""Launch-level stop tests; no JAX, GPU, real checkpoint or persistent gate writes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import model_entry


class StopGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / '.hermes').mkdir()
        self.gate = self.root / '.hermes/TRAINING_STOP_STATE.json'
        self.gate.write_text(json.dumps(dict(automatic_resume_allowed=False, endpoint_restore_allowed=False)))
        self.config = self.root / 'parameters.json'
        self.marker = self.root / 'executed'
        self.script = self.root / 'sentinel.py'
        self.script.write_text('import os\nfrom pathlib import Path\n'
                               'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
                               'assert os.environ["JAX_PLATFORMS"] == "cpu"\n'
                               f'Path({str(self.marker)!r}).write_text("CPU only")\n')

    def run_entry(self, config, entry='model_diagnostics.py'):
        self.config.write_text(json.dumps(dict(script=str(self.script), **config)))
        with mock.patch.object(model_entry, '__file__', str(self.root / 'model_entry.py')), \
                mock.patch.object(model_entry.sys, 'argv', [entry, str(self.config)]), \
                mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '7', 'JAX_PLATFORMS': 'cuda'}):
            model_entry.main()

    def test_gpu_manifest_rejected_before_payload(self):
        with self.assertRaisesRegex(RuntimeError, 'explicit user resume'):
            self.run_entry({'gpu': 6})
        self.assertFalse(self.marker.exists())

    def test_training_role_rejected_even_with_cpu_environment(self):
        with self.assertRaises(RuntimeError):
            self.run_entry({'env': {'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'}}, 'model_training.py')
        self.assertFalse(self.marker.exists())

    def test_cuda_environment_in_diagnostics_is_rejected(self):
        for env in ({'CUDA_VISIBLE_DEVICES': '0,1'}, {'JAX_PLATFORMS': 'cuda,cpu'}, {'JAX_PLATFORM_NAME': 'gpu'}):
            with self.subTest(env=env), self.assertRaises(RuntimeError):
                self.run_entry({'env': env})
        self.assertFalse(self.marker.exists())

    def test_cpu_diagnostics_clear_inherited_gpu_environment(self):
        self.run_entry({})
        self.assertEqual(self.marker.read_text(), 'CPU only')

    def test_stale_restore_allowlist_cannot_override_immediate_stop(self):
        config = dict(script=str(self.script), gpu=6)
        self.config.write_text(json.dumps(config))
        digest = hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.gate.write_text(json.dumps(dict(automatic_resume_allowed=False, endpoint_restore_allowed=False,
                                            restore_config_sha256_allowlist=[digest])))
        with self.assertRaises(RuntimeError):
            self.run_entry({'gpu': 6})
        self.assertFalse(self.marker.exists())

    def test_tracked_policy_blocks_gpu_without_private_stop_state(self):
        self.gate.unlink()
        (self.root / 'training_launch_policy.json').write_text(json.dumps(dict(training_enabled=False)))
        with self.assertRaises(RuntimeError):
            self.run_entry({'gpu': 0})
        self.assertFalse(self.marker.exists())

    def test_private_resume_flag_cannot_override_tracked_policy(self):
        self.gate.write_text(json.dumps(dict(automatic_resume_allowed=True)))
        (self.root / 'training_launch_policy.json').write_text(json.dumps(dict(training_enabled=False)))
        with self.assertRaises(RuntimeError):
            self.run_entry({}, 'model_training.py')
        self.assertFalse(self.marker.exists())


if __name__ == '__main__':
    unittest.main()
