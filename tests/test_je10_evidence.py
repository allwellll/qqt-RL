import hashlib
import json

import pytest

from scripts.summarize_je10_evidence import verified_point, route_summary


def fixture(tmp_path):
    summary = dict(generation=8, policy_sha256='a' * 64)
    data = json.dumps(summary).encode()
    (tmp_path / 'SUMMARY.json').write_bytes(data)
    complete = dict(passed=True, generation=8, checkpoint_sha256='a' * 64,
                    shards={'SUMMARY.json': hashlib.sha256(data).hexdigest()})
    (tmp_path / 'COMPLETE.json').write_text(json.dumps(complete))
    return complete


def test_completed_point_rejects_mutated_shard(tmp_path):
    fixture(tmp_path)
    summary, proof = verified_point(tmp_path)
    assert summary['generation'] == 8 and len(proof['complete_sha256']) == 64
    (tmp_path / 'SUMMARY.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        verified_point(tmp_path)


def test_complete_cannot_cover_other_checkpoint_or_escape_directory(tmp_path):
    complete = fixture(tmp_path)
    complete['checkpoint_sha256'] = 'b' * 64
    (tmp_path / 'COMPLETE.json').write_text(json.dumps(complete))
    with pytest.raises(ValueError, match='identity mismatch'):
        verified_point(tmp_path)
    complete['shards'] = {'../outside.json': 'a' * 64}
    (tmp_path / 'COMPLETE.json').write_text(json.dumps(complete))
    with pytest.raises(ValueError, match='escapes'):
        verified_point(tmp_path)


def test_route_denominator_and_declared_seeds_are_required():
    with pytest.raises(ValueError, match='denominator'):
        route_summary(dict(games=15, episodes=[]))
    with pytest.raises(ValueError, match='seeds'):
        route_summary(dict(games=16, episodes=[{'seed': 1}] * 16))
