import tempfile
import unittest
from pathlib import Path
from scripts.population_run_state import atomic_json, sha
from scripts.summarize_bun_stages import checked_complete


class EvidenceTests(unittest.TestCase):
    def test_valid_evidence_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            atomic_json(out / 'hunter.json', {'safe_win': 2})
            atomic_json(out / 'bun_tasks.json', {'capture': 14})
            atomic_json(out / 'COMPLETE.json', dict(passed=True, policy_sha256='p',
                hunter={}, tactical={}, shards={'hunter.json': sha(out / 'hunter.json')}))
            atomic_json(out / 'COMPLETE_WITH_BUN_TASKS.json', dict(passed=True,
                policy_sha256='p', authority_sha256=sha(out / 'COMPLETE.json'),
                additional_shards={'bun_tasks.json': sha(out / 'bun_tasks.json')},
                tasks={'full': {'death_episodes': 1, 'episodes': [[1]]}}))
            self.assertNotIn('episodes', checked_complete(out, 'p')['tasks']['full'])
            with self.assertRaises(AssertionError): checked_complete(out, 'other-policy')
            atomic_json(out / 'bun_tasks.json', {'capture': 16})
            with self.assertRaises(AssertionError): checked_complete(out, 'p')


if __name__ == '__main__':
    unittest.main()
