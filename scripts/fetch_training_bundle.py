"""Download and verify the published DUEL/SP data bundle, then safely extract it."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import tarfile
import tempfile
import urllib.request

TAG = 'training-bundle-20261107'
ASSET = 'duel-sp-policy-data-20261107.tar.gz'
BASE = 'https://github.com/allwellll/qqt-RL/releases/download/' + TAG + '/'
PREFIX = 'data/portable-training-20261107/'


def extract_verified(archive, destination, expected):
    archive, destination = Path(archive), Path(destination)
    with archive.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('Archive SHA256 mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        seen = set()
        for item in members:
            p = Path(item.name)
            if (not item.name.startswith(PREFIX) or p.is_absolute() or '..' in p.parts
                    or not item.isfile() or item.name in seen):
                raise ValueError('Unsafe or duplicate archive entry')
            seen.add(item.name)
            target = destination / p
            if target.exists() or any(parent.is_symlink() for parent in [target, *target.parents]):
                raise ValueError('Refuse to overwrite an existing file or follow a symlink')
        for item in members:
            target = destination / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(item) as source, target.open('xb') as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    return len(members)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--sha256', help='Required for an explicitly supplied local archive')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.archive:
        if not args.sha256:
            parser.error('--archive requires --sha256')
        count = extract_verified(args.archive, root, args.sha256)
    else:
        sums = urllib.request.urlopen(BASE + 'SHA256SUMS', timeout=60).read().decode()
        hashes = {line.split()[1]: line.split()[0] for line in sums.splitlines() if line.strip()}
        manifest_data = urllib.request.urlopen(BASE + 'MANIFEST.json', timeout=60).read()
        if hashlib.sha256(manifest_data).hexdigest() != hashes['MANIFEST.json']:
            raise ValueError('Manifest SHA256 mismatch')
        manifest = json.loads(manifest_data)
        count = 0
        with tempfile.TemporaryDirectory(prefix='qqt-training-download-') as folder:
            staged = []
            for row in manifest['files']:
                relative = Path(row['path'])
                if not row['path'].startswith(PREFIX) or relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe manifest path')
                target = root / relative
                if target.exists() or any(p.is_symlink() for p in [target, *target.parents]):
                    raise ValueError('Refuse to overwrite an existing file or follow a symlink')
                name = '--'.join(relative.relative_to(PREFIX.rstrip('/')).parts)
                if hashes.get(name) != row['sha256']:
                    raise ValueError('Asset checksum list differs from manifest')
                local = Path(folder) / name
                with urllib.request.urlopen(BASE + name, timeout=120) as source, local.open('wb') as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                with local.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != row['sha256'] or local.stat().st_size != row['bytes']:
                    raise ValueError('Asset SHA256/size mismatch: ' + name)
                staged.append((local, target)); count += 1
            for local, target in staged:
                target.parent.mkdir(parents=True, exist_ok=True)
                with local.open('rb') as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
            target_manifest = root / PREFIX / 'MANIFEST.json'
            with target_manifest.open('xb') as output:
                output.write(manifest_data)
    print(f'PASS: all SHA256 values verified; {count} runtime files installed; training remains disabled')


if __name__ == '__main__':
    main()
