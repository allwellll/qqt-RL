"""Natural-boundary migration: preserve model/Adam, never import old midgames."""
import argparse
import json
import os
import pickle
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/rules_v2_critic_20261027/v1')
IMPLEMENTATION = Path('runs/authorized_campaign_20261026/protocols/rules_v2_critic_ab_v1_implementation01.json')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def export_parent():
    """Run in the unchanged old-rule reader process, before rule selection."""
    import jax
    import numpy as np
    from scripts.terminal_checkpoint_restore import digest
    parent = json.loads(IMPLEMENTATION.read_text())['parent']
    source = Path(parent['checkpoint'])
    for name in ('state', 'policy'):
        file = source / ('state.pkl' if name == 'state' else 'policy.pt')
        assert sha(file) == parent[name + '_sha256']
    with (source / 'state.pkl').open('rb') as stream:
        old = pickle.load(stream)
    value = dict(params=jax.tree.map(np.asarray, old['params']),
                 adam=jax.tree.map(np.asarray, old['adam']),
                 source_update=old.get('update'), source=parent)
    assert digest(value['params']) == digest(old['params'])
    assert digest(value['adam']) == digest(old['adam'])
    output = BASE / 'inputs/parent_model_adam.pkl'
    save(output, value)
    atomic_json(BASE / 'PARENT_EXPORT.json', dict(
        status='PASS', source=parent, output=str(output), sha256=sha(output),
        params_exact=True, Adam_exact=True, old_env_aux_RNG_excluded=True,
        source_update=value['source_update'],
        migration='segmented restart; all environment, aux and RNG rebuilt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['export-parent'])
    parser.parse_args()
    export_parent()


if __name__ == '__main__':
    main()
