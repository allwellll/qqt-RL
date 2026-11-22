"""Prepare verified carry-training inputs without importing JAX or starting training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TAG = 'dcu-carry-training-20261115'
ASSET = 'u930_actor.pt'
URL = f'https://github.com/allwellll/qqt-RL/releases/download/{TAG}/{ASSET}'
API_URL = 'https://api.github.com/repos/allwellll/qqt-RL/releases/assets/751461110?download=1'
DESTINATION = ROOT / 'data/carry-training-20261115' / ASSET
EXPECTED_SHA = '1b1079d213a971ed7013739868d3b9848484ecf48ade25418142e16fb3a06d27'
EXPECTED_BYTES = 30257447


def verify(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != EXPECTED_BYTES:
        raise ValueError('u930 model missing, linked, or wrong size: ' + str(path))
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != EXPECTED_SHA:
            raise ValueError('u930 SHA256 mismatch: ' + str(path))


def download(output):
    # Some platform proxies fail github.com downloads but allow the public
    # asset API. Both transports must return the exact same pinned bytes.
    for index, url in enumerate((URL, API_URL)):
        output.seek(0)
        output.truncate(0)
        request = urllib.request.Request(url, headers={
            'Accept': 'application/octet-stream', 'User-Agent': 'qqt-carry-training'})
        try:
            total, digest = 0, hashlib.sha256()
            with urllib.request.urlopen(request, timeout=30 if index == 0 else 120) as incoming:
                while chunk := incoming.read(1024 * 1024):
                    total += len(chunk)
                    if total > EXPECTED_BYTES:
                        raise ValueError('u930 download exceeds expected size')
                    digest.update(chunk)
                    output.write(chunk)
            if total != EXPECTED_BYTES or digest.hexdigest() != EXPECTED_SHA:
                raise ValueError('u930 downloaded size/SHA256 mismatch')
            return
        except (urllib.error.URLError, TimeoutError, ValueError):
            if index:
                raise


def install(source=None, destination=None):
    target = Path(destination) if destination is not None else DESTINATION
    if target.exists() or target.is_symlink():
        verify(target)
        return 'already_verified'
    if source is not None:
        verify(source)
    if any(parent.is_symlink() for parent in target.parents):
        raise ValueError('Refuse destination through a symbolic link')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.u930-', dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            if source is not None:
                with Path(source).open('rb') as incoming:
                    shutil.copyfileobj(incoming, output)
            else:
                download(output)
            output.flush()
            os.fsync(output.fileno())
        verify(temporary)
        # Publish the fully verified file atomically, refusing concurrent overwrite.
        os.link(temporary, target)
        return 'installed'
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--check', action='store_true', help='verify existing assets only, no network or writes')
    parser.add_argument('--u930-source', type=Path, help='install an existing trusted u930 actor; SHA is fixed')
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    from scripts import portable_training as portable
    if args.check:
        verify(DESTINATION)
        status = 'verified'
    else:
        if not (ROOT / portable.BUNDLE / 'MANIFEST.json').is_file():
            temporary = ROOT / '.cache/tmp'
            temporary.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, '-m', 'scripts.fetch_training_bundle'],
                           check=True, cwd=ROOT, env={**os.environ, 'TMPDIR': str(temporary)})
        portable.verify_bundle()
        status = install(args.u930_source)
    portable.verify_bundle()
    if 'jax' in sys.modules:
        raise RuntimeError('Asset preparation must not initialize JAX')
    print(json.dumps(dict(status=status, u930=str(DESTINATION.relative_to(ROOT)),
                          sha256=EXPECTED_SHA, bytes=EXPECTED_BYTES,
                          portable_bundle='verified', training_started=False)))


if __name__ == '__main__':
    main()
