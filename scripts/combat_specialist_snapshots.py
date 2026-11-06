"""Hash-verified immutable league snapshots; metadata never contains host paths."""
import json
import pickle
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/combat_specialist_league_20261101')
SCHEMA = 'league-snapshot-v1'


def entry(checkpoint):
    cp = Path(checkpoint)
    if cp.is_absolute():
        raise ValueError('checkpoint must be repository relative')
    receipt = json.loads((cp/'COMMITTED.json').read_text())
    assert sha(cp/'policy.pt') == receipt['policy_sha256']
    result = dict(checkpoint=str(cp), policy=str(cp/'policy.pt'),
                  policy_sha256=receipt['policy_sha256'],
                  state_sha256=receipt['state_sha256'], update=receipt['update'])
    if (cp/'policy_bun_mode0.pt').exists():
        result.update(bun_mode_zero_policy=str(cp/'policy_bun_mode0.pt'),
                      bun_mode_zero_policy_sha256=sha(cp/'policy_bun_mode0.pt'))
    return result


def publish(family, checkpoint, fixed_checkpoint=None):
    if family not in ('G', 'C'):
        raise ValueError('unknown league family')
    path = BASE/'league_snapshots'/(family+'.json')
    previous = json.loads(path.read_text()) if path.exists() else {}
    current = entry(checkpoint)
    candidates = [current] + previous.get('ring', [])
    seen = set(); ring = []
    for candidate in candidates:
        key = candidate['policy_sha256']
        if key not in seen:
            seen.add(key); ring.append(candidate)
    fixed = previous.get('fixed', [])
    if fixed_checkpoint is not None:
        requested = entry(fixed_checkpoint)
        if fixed and fixed != [requested]:
            raise ValueError('fixed history is immutable')
        fixed = [requested]
    if not fixed:
        fixed = [current]
    manifest = dict(schema=SCHEMA, family=family, current=current,
                    ring=ring[:4], fixed=fixed,
                    ring_fallback='fewer than four distinct committed points; repeat available members explicitly')
    atomic_json(path, manifest)
    return manifest


def load(family):
    path = BASE/'league_snapshots'/(family+'.json')
    value = json.loads(path.read_text())
    assert value['schema'] == SCHEMA and value['family'] == family
    for item in [value['current']] + value['ring'] + value['fixed']:
        policy = Path(item['policy'])
        assert not policy.is_absolute() and sha(policy) == item['policy_sha256']
    return value


def actor(item, bun_mode_zero=False):
    name = 'bun_mode_zero_policy' if bun_mode_zero and 'bun_mode_zero_policy' in item else 'policy'
    path = Path(item[name])
    assert not path.is_absolute() and sha(path) == item[name+'_sha256']
    with path.open('rb') as stream:
        return pickle.load(stream)
