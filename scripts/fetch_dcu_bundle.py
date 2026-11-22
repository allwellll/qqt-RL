"""Download the public bundle, optionally through an HTTPS GitHub mirror.

The manifest digest always comes directly from the official GitHub API over
HTTPS. Mirror payloads must match that manifest before installation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from scripts.fetch_training_bundle import BASE, PREFIX, TAG


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mirror', default='', help='Optional HTTPS prefix, e.g. https://ghfast.top/')
    args = parser.parse_args()
    if args.mirror and (not args.mirror.startswith('https://') or not args.mirror.endswith('/')):
        parser.error('--mirror must be an HTTPS prefix ending with /')
    root = Path(__file__).resolve().parents[1]
    staging = root / '.cache/bundle'
    staging.mkdir(parents=True, exist_ok=True)
    url = f'https://api.github.com/repos/allwellll/qqt-RL/releases/tags/{TAG}'
    with urllib.request.urlopen(url, timeout=30) as response:
        release = json.load(response)
    assets = {row['name']: row for row in release['assets']}
    expected = assets['MANIFEST.json'].get('digest', '')
    if not expected.startswith('sha256:'):
        raise ValueError('Official GitHub manifest SHA256 unavailable; refusing untrusted mirror data')

    def fetch(name, digest, size):
        target = staging / name
        def valid():
            if not target.is_file() or target.stat().st_size != size:
                return False
            with target.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest() == digest
        if not valid():
            partial = target.with_suffix(target.suffix + '.part')
            with urllib.request.urlopen(args.mirror + BASE + name, timeout=60) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output)
            partial.replace(target)
            if not valid():
                raise ValueError('Asset SHA256/size mismatch: ' + name)
        print('VERIFIED ' + name, flush=True)
        return target

    manifest_path = fetch('MANIFEST.json', expected.split(':')[1], assets['MANIFEST.json']['size'])
    manifest = json.loads(manifest_path.read_bytes())
    def payload(row):
        relative = Path(row['path'])
        if not row['path'].startswith(PREFIX) or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        target = root / relative
        if target.exists() or any(p.is_symlink() for p in [target, *target.parents]):
            raise ValueError('Refuse to overwrite or follow symlinks: ' + str(relative))
        name = '--'.join(relative.relative_to(PREFIX.rstrip('/')).parts)
        official = assets[name]
        if official.get('digest') != 'sha256:' + row['sha256'] or official['size'] != row['bytes']:
            raise ValueError('Official asset metadata differs from manifest: ' + name)
        local = fetch(name, row['sha256'], row['bytes'])
        return local, target
    with ThreadPoolExecutor(max_workers=3) as pool:
        ready = list(pool.map(payload, manifest['files']))
    # Publish only after every payload has passed integrity checks.
    for local, target in ready:
        target.parent.mkdir(parents=True, exist_ok=True)
        with local.open('rb') as source, target.open('xb') as output:
            shutil.copyfileobj(source, output)
    with (root / PREFIX / 'MANIFEST.json').open('xb') as output:
        output.write(manifest_path.read_bytes())
    print(f'PASS: {len(ready)} verified files installed; training remains disabled', flush=True)


if __name__ == '__main__':
    main()
