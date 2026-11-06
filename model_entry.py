"""Run the source recorded in a private configuration within this process."""
import json
import os
import runpy
import sys
from pathlib import Path

def main():
    path = Path(sys.argv[1])
    os.chmod(path, 0o600)
    config = json.loads(path.read_text())
    if config.get('project_from_private_manifest'):
        project = path.resolve().parents[2]
        os.chdir(project)
        sys.path.insert(0, str(project))
    if 'gpu' in config:
        grant = json.loads(Path('.hermes/GPU_USER_RESERVATION.json').read_text())
        assert config['gpu'] in grant['allowed_gpus'] and config['gpu'] not in grant.get('disabled_gpus', []), 'only explicitly authorized devices'
        os.environ.update(CUDA_VISIBLE_DEVICES=str(config['gpu']), JAXBOMB_DEVICE_OVERRIDE=str(config['gpu']))
    os.environ.update(config.get('env', {}))
    sys.argv = [sys.argv[0]] + config.get('args', [])
    if 'module' in config:
        runpy.run_module(config['module'], run_name='__main__')
    else:
        runpy.run_path(config['script'], run_name='__main__')
