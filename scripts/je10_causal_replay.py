"""Reject fixed return supervision without hash-bound native acquisition proof."""
import hashlib
import json
import pickle
from pathlib import Path


def load_verified(replay_path, proof_path):
    try:
        raw = Path(replay_path).read_bytes()
        proof = json.loads(Path(proof_path).read_text())
        valid = (proof.get('passed') is True
                 and proof.get('real_native_drop_pickup') is True
                 and proof.get('action_takeover') is False
                 and proof.get('future_information') is False
                 and proof.get('sha256') == hashlib.sha256(raw).hexdigest())
        if not valid:
            raise ValueError('invalid proof')
        data = pickle.loads(raw)
        for group in ['short', 'long']:
            rows = proof[group]['rows']
            if rows <= 0 or len(data[group]) != 4 or any(len(x) != rows for x in data[group]):
                raise ValueError('invalid proof row count')
        return data
    except (OSError, KeyError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError('fixed return replay requires valid native acquisition proof') from error
