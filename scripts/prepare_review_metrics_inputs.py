"""Assemble a private immutable CPU authority; never writes into the source run."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True,type=Path);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--generation',type=int,default=8);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
seal=json.loads((a.run/'SOURCE_SEAL.json').read_text());plan=(a.run/'plan.json').read_bytes()
assert sha(plan)==seal['plan_sha256']
a.output.mkdir(parents=True,exist_ok=True)
for folder in ('scripts','jax_bomb'):
 shutil.copytree(a.baseline/folder,a.output/'authority'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name,h in seal['sources'].items():
 data=(a.run/'inputs/sealed_sources'/name).read_bytes();assert sha(data)==h
 target=a.output/'authority'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
target=a.output/'authority/web/assets/maps';target.mkdir(parents=True,exist_ok=True);shutil.copyfile('web/assets/maps/levels.json',target/'levels.json')
cp=a.run/f'formal/bun392_bridge_pfsp8/gen{a.generation:06d}';receipt=json.loads((cp/'COMMITTED.json').read_text());assert receipt['passed'] and receipt['generation']==a.generation
for file in ('model.json','model.onnx'):shutil.copyfile(a.run/f'authority_je10/g{a.generation:06d}/model'/file,a.output/file)
model=json.loads((a.output/'model.json').read_text());assert model['meta']['checkpoint_sha256']==receipt['policy_sha256']
proof=dict(schema='qqt.review-metrics-input/v1',checkpoint_sha256=receipt['policy_sha256'],state_sha256=receipt['state_sha256'],gamma=json.loads(plan)['gamma'],model_sha256=sha((a.output/'model.json').read_bytes()),onnx_sha256=sha((a.output/'model.onnx').read_bytes()),source_seal_sha256=sha((a.run/'SOURCE_SEAL.json').read_bytes()),sealed_source_count=len(seal['sources']))
assert proof['onnx_sha256']==model['meta']['onnx_sha256']
(a.output/'proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('CPU authority prepared; all sealed source hashes verified')
