import os
import json
import tempfile
import unittest
from pathlib import Path

from scripts.je10_evaluate import unresolved_training_failure

VERIFIED = dict(formal_nonzero_verified=True, finite=True, devices=8, replicas_exact=True,
                generation=1, fresh_ticks=2097152, actor_update_l2=[1.] * 8, critic_update_l2=[2.] * 8)


class EvaluatorRecoveryTests(unittest.TestCase):
    def test_failure_without_formal_recovery_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'TECHNICAL_FAILURE.json').write_text('{}')
            self.assertTrue(unresolved_training_failure(directory))

    def test_failure_before_formal_recovery_is_historical(self):
        with tempfile.TemporaryDirectory() as directory:
            failure = Path(directory, 'TECHNICAL_FAILURE.json')
            recovery = Path(directory, 'FORMAL_NONZERO_CONFIRMED.json')
            failure.write_text('{}')
            recovery.write_text(json.dumps(VERIFIED))
            os.utime(failure, ns=(1_000_000_000, 1_000_000_000))
            os.utime(recovery, ns=(2_000_000_000, 2_000_000_000))
            self.assertFalse(unresolved_training_failure(directory))

    def test_failure_after_formal_recovery_blocks_again(self):
        with tempfile.TemporaryDirectory() as directory:
            failure = Path(directory, 'TECHNICAL_FAILURE.json')
            recovery = Path(directory, 'FORMAL_NONZERO_CONFIRMED.json')
            recovery.write_text(json.dumps(VERIFIED))
            failure.write_text('{}')
            os.utime(recovery, ns=(1_000_000_000, 1_000_000_000))
            os.utime(failure, ns=(2_000_000_000, 2_000_000_000))
            self.assertTrue(unresolved_training_failure(directory))

    def test_no_failure_does_not_require_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(unresolved_training_failure(directory))

    def test_invalid_recovery_never_suppresses_failure(self):
        for value in ['{}', 'broken-json', json.dumps(dict(VERIFIED, finite=False)),
                      json.dumps(dict(VERIFIED, actor_update_l2=[0.] * 8))]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                failure = Path(directory, 'TECHNICAL_FAILURE.json')
                recovery = Path(directory, 'FORMAL_NONZERO_CONFIRMED.json')
                failure.write_text('{}')
                recovery.write_text(value)
                os.utime(failure, ns=(1_000_000_000, 1_000_000_000))
                os.utime(recovery, ns=(2_000_000_000, 2_000_000_000))
                self.assertTrue(unresolved_training_failure(directory))


if __name__ == '__main__':
    unittest.main()
