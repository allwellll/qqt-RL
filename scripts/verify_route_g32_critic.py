"""CPU-only, read-only tensor comparison with the sealed same-generation state."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', JAXBOMB_RULE='bun')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.extract_h8_critic import load_legacy_modules
from scripts.export_je09_critic_npz import flatten


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    source = Path(sys.argv[1]).resolve()
    package = source / 'reports/je09_bun_20261017/route_pbrs_replay_g32'
    checkpoint = source / 'runs/je09_bun_20261017/stage_h8_route_pbrs_v35/formal/bun392_bridge_pfsp8/gen000032'
    receipt = json.loads((checkpoint / 'COMMITTED.json').read_text())
    expected = {'policy.pt': 'e141311268c6eaf419f1f651252bb27d46c106aeb407230c20eed4c8897a34e2',
                'state.pkl': 'cdc1e497c0fa83d3ba2f9a5986e365d3a3953ea63cbacd106016829ffdd98f12'}
    assert receipt['generation'] == 32 and receipt['passed']
    for file, value in expected.items():
        assert digest(checkpoint / file) == value
    assert digest(package / 'critic_params.npz') == '3174b62f39ea9d56f206c917d2107a8f5e31bda2fe693e88630f72715c0604d8'
    sys.path.insert(0, str(source))
    load_legacy_modules(source, Path('web/assets/maps/levels.json').resolve())
    with (checkpoint / 'state.pkl').open('rb') as f:
        state = pickle.load(f)
    params = flatten(state['live']['params']['critic'])
    with np.load(package / 'critic_params.npz') as exported:
        assert set(params) == set(exported.files)
        for key, value in params.items():
            assert np.array_equal(value, exported[key]), 'critic tensor equality'
            assert np.isfinite(value).all(), 'finite critic tensor'
    result = {'schema': 'qqt.route-g32-critic-audit/v1', 'generation': 32,
              'state_sha256': expected['state.pkl'], 'policy_sha256': expected['policy.pt'],
              'critic_sha256': digest(package / 'critic_params.npz'),
              'tensor_count': len(params), 'all_tensors_exact': True,
              'scope': 'same-generation parameter equality; no counterfactual Q/A measurement'}
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
