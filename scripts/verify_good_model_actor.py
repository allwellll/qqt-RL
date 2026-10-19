"""Verify exported Actor tensors directly against each immutable checkpoint."""
import argparse, hashlib, json, os, pickle, sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                  JAX_PLATFORM_NAME='cpu', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
p = argparse.ArgumentParser()
p.add_argument('--run', required=True, type=Path)
p.add_argument('--input', required=True, type=Path)
p.add_argument('--generation', required=True, type=int)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
sys.path.insert(0, str(a.input.resolve() / 'authority'))
from extract_h8_critic import load_legacy_modules
load_legacy_modules(a.input.resolve() / 'authority', Path('web/assets/maps/levels.json').resolve())
from export_bun_web_model import extract_transformer, pack_tensors

sha = lambda data: hashlib.sha256(data).hexdigest()
cp = a.run / f'formal/bun392_bridge_pfsp8/gen{a.generation:06d}'
receipt = json.loads((cp / 'COMMITTED.json').read_text())
data = (cp / 'policy.pt').read_bytes()
assert sha(data) == receipt['policy_sha256']
params = pickle.loads(data)
params = params.get('params', params)
flat, tensors = pack_tensors(extract_transformer(params))
model_data = (a.input / 'model.json').read_bytes()
model = json.loads(model_data)
assert model['flat'] == flat
assert model['tensors'] == {k: list(v) for k, v in tensors.items()}
assert model['meta']['checkpoint_sha256'] == receipt['policy_sha256']
proof = dict(schema='qqt.good-model-actor-parity/v1', passed=True,
             checkpoint_sha256=sha(data), model_sha256=sha(model_data),
             tensor_count=len(tensors), exact=True, device='cpu',
             scope='all exported float32 Actor tensors equal own policy.pt; CPU ONNX/JSON inference parity also checked during regeneration')
a.output.write_text(json.dumps(proof, indent=2) + '\n')
print('Own checkpoint Actor/export tensors exact', len(tensors))
