"""Read-only inventory of immutable v36 artifacts; never substitute checkpoint state for a historical rollout."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

RUN = Path('runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36_attempt02')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class InventoryTuple(tuple):
    """Inventory-only placeholder for unavailable environment namedtuple classes.

    This object is never stepped, passed to a Critic, or published as a state.
    Parameter ndarray subtrees are loaded normally and compared independently.
    """
    def __new__(cls, *values):
        return tuple.__new__(cls, values)


class InventoryUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('jax_bomb.je05_core_env', 'jax_bomb.je05_bun_env', 'jax_bomb.je07_bun_env'):
            return InventoryTuple
        return super().find_class(module, name)


def leaves(tree, prefix=''):
    if isinstance(tree, dict):
        for key in sorted(tree):
            yield from leaves(tree[key], prefix + '/' + str(key))
    elif isinstance(tree, (tuple, list)):
        for i, value in enumerate(tree):
            yield from leaves(value, prefix + '/' + str(i))
    else:
        yield prefix, np.asarray(tree)


def main():
    cp = RUN / 'formal/bun392_bridge_pfsp8/gen000032'
    receipt = json.loads((cp / 'COMMITTED.json').read_text())
    assert receipt['passed'] and receipt['generation'] == 32
    hashes = {name: digest(cp / name) for name in ('policy.pt', 'state.pkl')}
    assert hashes['policy.pt'] == receipt['policy_sha256']
    assert hashes['state.pkl'] == receipt['state_sha256']
    seal = json.loads((RUN / 'SOURCE_SEAL.json').read_text())
    checked = {}
    absent = []
    for name, expected in seal['sources'].items():
        file = RUN / 'inputs/sealed_sources' / name
        if not file.is_file():
            absent.append(name)
            continue
        actual = digest(file)
        assert actual == expected, name
        checked[name] = actual
    with (cp / 'state.pkl').open('rb') as stream:
        saved = InventoryUnpickler(stream).load()
    with (cp / 'policy.pt').open('rb') as stream:
        actor = pickle.load(stream)
    state_actor = dict(leaves(saved['live']['params']['actor']))
    for name, value in leaves(actor):
        np.testing.assert_array_equal(state_actor[name][0], value)
    critic = list(leaves(saved['live']['params']['critic']))
    h = hashlib.sha256()
    for name, value in critic:
        assert value.shape[0] == 8 and np.isfinite(value).all()
        for i in range(1, 8):
            np.testing.assert_array_equal(value[0], value[i])
        h.update(name.encode() + b'\0' + value[0].dtype.str.encode() + b'\0')
        h.update(json.dumps(value[0].shape).encode() + b'\0' + value[0].tobytes())
    candidates = sorted(str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file() and any(k in p.name.lower() for k in ('trajectory', 'rollout')))
    result = dict(schema='qqt.v36-input-audit/v1', policy_sha256=hashes['policy.pt'],
        state_sha256=hashes['state.pkl'], receipt_sha256=digest(cp/'COMMITTED.json'),
        plan_sha256=digest(RUN/'plan.json'), source_seal_sha256=digest(RUN/'SOURCE_SEAL.json'),
        state_fields=sorted(saved), live_fields=sorted(saved['live']),
        aux_fields=sorted(saved['live']['aux']), parameter_groups=sorted(saved['live']['params']),
        actor_replica0_exact=True, critic_sha256=h.hexdigest(), critic_tensor_count=len(critic),
        critic_replicas=8, critic_replicas_exact=True, critic_finite=True,
        critic_scope='same checkpoint PBRS-trained parameter evidence; no Critic forward/Q/A claim',
        source_hashes=checked, missing_sealed_source_files=absent,
        historical_trajectory_candidates=candidates,
        historical_training_trajectory_status='not preserved in checkpoint live fields; no trajectory/rollout files found in completed namespace',
        training_reward_blocker='Complete natural-Web-to-training aux/event/reward bridge absent; reward dependencies je09_stage_h8_ops and je09_stage_h5c_ops not present as readable sealed source. Unsealed bytecode is not a validated ledger.',
        inventory_method='Missing environment namedtuple types replaced only for inventory; actual ndarray Actor/Critic subtrees loaded and compared. No invented env state is simulated or evaluated.')
    out = Path('runs/replay_review_20261017/v36-audit.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('policy_sha256', 'state_sha256', 'critic_sha256', 'critic_tensor_count', 'parameter_groups', 'aux_fields', 'historical_trajectory_candidates')}))


if __name__ == '__main__':
    main()
