#!/usr/bin/env bash
# Install only project-local Python additions and a checksum-verified Node binary.
set -euo pipefail
qqt_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$qqt_project"
python - <<'PY'
from importlib.metadata import version
expected='0.6.0+das.opt1.dtk2604'
for name in ('jax','jaxlib'):
    if version(name)!=expected:
        raise SystemExit(f'{name} is not the validated Hygon {expected}; use the matching platform image')
print('Hygon JAX build verified; preserving system dependencies')
PY
if [[ ! -x .venv/bin/python ]]; then python -m venv --system-site-packages .venv; fi
mkdir -p .cache/tmp .cache/pip .cache/downloads
TMPDIR="$qqt_project/.cache/tmp" PIP_CACHE_DIR="$qqt_project/.cache/pip" \
  .venv/bin/python -m pip install -r requirements-dcu.txt -e '.[test]'
if [[ ! -x .tools/node/bin/node ]]; then
  .venv/bin/python - <<'PY'
import hashlib,tarfile,urllib.request
from pathlib import Path
base='https://nodejs.org/dist/v22.16.0/'
name='node-v22.16.0-linux-x64.tar.xz'
cache=Path('.cache/downloads')
with urllib.request.urlopen(base+'SHASUMS256.txt',timeout=60) as r:
    sums=r.read().decode()
expected=next(row.split()[0] for row in sums.splitlines() if row.split()[-1]==name)
archive=cache/name
def digest():
    with archive.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if not archive.is_file() or digest()!=expected:
    with urllib.request.urlopen(base+name,timeout=120) as r,archive.open('wb') as f:
        while chunk:=r.read(1024*1024):f.write(chunk)
if digest()!=expected:raise SystemExit('Node SHA256 mismatch')
target=Path('.tools/node/bin/node');target.parent.mkdir(parents=True,exist_ok=True)
# Only the executable is needed; do not extract thousands of npm/header files.
with tarfile.open(archive,'r:xz') as t:
    with t.extractfile('node-v22.16.0-linux-x64/bin/node') as source,target.open('xb') as output:
        while chunk:=source.read(1024*1024):output.write(chunk)
target.chmod(0o755)
print('Installed official, verified Node 22.16.0 in .tools/node/bin')
PY
fi
.tools/node/bin/node --version
echo 'Environment ready. Download the training bundle if absent, then follow docs/dcu_kill_training.md.'
