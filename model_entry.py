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
    project_root = Path(__file__).resolve().parent
    gate_path = project_root / '.hermes/TRAINING_STOP_STATE.json'
    policy_path = project_root / 'training_launch_policy.json'
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    if policy_path.exists() and not json.loads(policy_path.read_text()).get('training_enabled', False):
        gate = dict(automatic_resume_allowed=False, endpoint_restore_allowed=False)
    if gate is not None:
        if not gate.get('automatic_resume_allowed', False):
            env = config.get('env', {})
            gpu_environment = bool(env.get('CUDA_VISIBLE_DEVICES', '')) or any(
                value in str(env.get(key, '')).lower().split(',')
                for key in ('JAX_PLATFORMS', 'JAX_PLATFORM_NAME')
                for value in ('cuda', 'gpu')
            )
            if 'gpu' in config or Path(sys.argv[0]).name == 'model_training.py' or gpu_environment:
                import hashlib
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if not gate.get('endpoint_restore_allowed', False) or digest not in gate.get('restore_config_sha256_allowlist', []):
                    raise RuntimeError('Persistent user stop: GPU/training launches are disabled; explicit user resume required')
            if not gate.get('endpoint_restore_allowed', False):
                os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu')
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
