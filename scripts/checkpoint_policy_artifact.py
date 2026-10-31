"""Policy-only artifact verification; state SHA is historical commit metadata."""
import json
import pickle
from pathlib import Path

from scripts.population_run_state import sha


def verify_policy(checkpoint):
    checkpoint = Path(checkpoint)
    marker = json.loads((checkpoint / 'COMMITTED.json').read_text())
    for field in ('state_sha256', 'policy_sha256'):
        value = marker[field]
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid committed hash: ' + field)
    if sha(checkpoint / 'policy.pt') != marker['policy_sha256']:
        raise ValueError('Policy artifact hash mismatch')
    return marker


def load_actor(checkpoint):
    checkpoint = Path(checkpoint)
    verify_policy(checkpoint)
    with (checkpoint / 'policy.pt').open('rb') as stream:
        return pickle.load(stream)
