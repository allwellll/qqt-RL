"""Reproduce a native Bun392 web asset from a hash-verified committed policy.

Only the browser policy and optional ONNX graph are exported, never optimizer
or training state. Public ancestry uses repository-relative references.
"""
import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_reference(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or '..' in p.parts or '\\' in value or ':' in value:
        raise ValueError('public source reference must be repository relative')
    return value


def validate_prefix(value):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', value):
        raise ValueError('asset prefix must be a portable lowercase name')
    return value


def native_encoding(committed):
    encoding = committed.get('input_encoding')
    if encoding not in (None, 'bun24-trap-timer-v2'):
        raise ValueError('unsupported native Bun encoding')
    return encoding


def validate_native(params):
    if params['tok'][0].shape != (216, 392) or params['state_w'].shape != (24, 392):
        raise ValueError('native Bun requires 24 planes, global24, embed392')
    if len(params['blocks']) != 4 or params['heads']['wm'][0].shape != (392, 5):
        raise ValueError('expected depth4 and five move actions')
    if params['heads']['wb'][0].shape != (392, 3):
        raise ValueError('native Bun requires three ability actions')


def export(checkpoint, output, source_ref, strategy, generated_at, name, onnx=False, asset_prefix=None):
    source_ref = validate_reference(source_ref)
    prefix = validate_prefix(asset_prefix if asset_prefix is not None else 'bun392-selfplay3072')
    checkpoint, output = Path(checkpoint), Path(output)
    committed_path = checkpoint / 'COMMITTED.json'
    committed = json.loads(committed_path.read_text())
    if not committed.get('passed'):
        raise ValueError('checkpoint is not committed')
    encoding = native_encoding(committed)
    policy_name = 'ema_policy.pt' if strategy == 'ema' else 'policy.pt'
    key = 'ema_policy_sha256' if strategy == 'ema' else 'policy_sha256'
    policy = checkpoint / policy_name
    policy_sha = sha(policy)
    if policy_sha != committed[key]:
        raise ValueError('policy hash differs from COMMITTED')
    # Pickle is executable: this API only accepts trusted, hash-verified policies.
    with policy.open('rb') as stream:
        params = pickle.load(stream)
    validate_native(params)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'model.unsealed.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts/export_bun_web_model.py'),
                    str(policy), '--output', str(temporary), '--name', name,
                    '--generated-at', generated_at], check=True, stdout=subprocess.DEVNULL)
    doc = json.loads(temporary.read_text())
    doc['meta'].update(source=f'{source_ref}/{policy_name}', checkpoint_sha256=policy_sha,
        committed_sha256=sha(committed_path), strategy=strategy, generation=committed['generation'],
        arm=committed['arm'], plan_sha256=committed['plan_sha256'], state_dim=24, attention_heads=4)
    if encoding:
        doc['meta']['bun_state_encoding'] = encoding
    if 'input_encoding' in doc['meta']:
        raise ValueError('input adapters cannot be published as native Bun')
    data = (json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()
    digest = hashlib.sha256(data).hexdigest()
    asset = output / f'{prefix}-{strategy}-{digest}.json'
    pending = asset.with_suffix('.tmp'); pending.write_bytes(data); pending.replace(asset)
    temporary.unlink()
    assets = [dict(file=asset.name, bytes=len(data), sha256=digest)]
    if onnx:
        graph = output / f'{prefix}.onnx'
        subprocess.run([sys.executable, str(ROOT / 'scripts/je03_native_vit_export_20261011.py'),
                        str(policy), str(graph)], check=True)
        graph_sha = sha(graph)
        if asset_prefix:
            target = output / f'{prefix}-{strategy}-{graph_sha}.onnx'
            graph.replace(target); graph = target
        assets.append(dict(file=graph.name, bytes=graph.stat().st_size, sha256=graph_sha))
    manifest = dict(schema='qqt.bun-release/v1', source_ref=source_ref, strategy=strategy,
        policy_sha256=policy_sha, raw_policy_sha256=committed['policy_sha256'],
        ema_policy_sha256=committed['ema_policy_sha256'], committed_sha256=sha(committed_path),
        state_sha256=committed['state_sha256'], plan_sha256=committed['plan_sha256'],
        generation=committed['generation'], arm=committed['arm'], generated_at=generated_at,
        input_contract=dict(obs=[24,13,15], global_dim=24, embed=392, depth=4, heads=4,
                            patch=3, move=5, ability=['none','bomb','use_item']), assets=assets)
    if encoding:
        manifest['input_contract']['bun_state_encoding'] = encoding
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['file']}\n" for r in assets)
                                     + f"{sha(output / 'manifest.json')}  manifest.json\n")
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--source-ref', required=True)
    p.add_argument('--strategy', required=True, choices=['raw', 'ema'])
    p.add_argument('--generated-at', required=True)
    p.add_argument('--name', default='Bun392动态自对弈3072')
    p.add_argument('--onnx', action='store_true')
    p.add_argument('--asset-prefix', help='portable prefix; new ONNX assets also use SHA256 names')
    a = p.parse_args()
    print(json.dumps(export(a.checkpoint, a.output, a.source_ref, a.strategy,
                            a.generated_at, a.name, a.onnx, a.asset_prefix), ensure_ascii=False, indent=2))
