"""DCU selection and the retained stop gate, without initializing JAX or a DCU."""
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from qqt_rl.training.dcu import configure
from scripts.dcu_smoke import arguments
from scripts.dcu_training import adapted_train


def test_rocm_selection_replaces_inherited_cuda():
    with patch.dict(os.environ, {'JAX_PLATFORMS': 'cuda'}, clear=True):
        configure(0)
        assert os.environ['JAX_PLATFORMS'] == 'rocm'
        assert os.environ['JAX_PLATFORM_NAME'] == 'rocm'
        assert os.environ['HIP_VISIBLE_DEVICES'] == '0'
        assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'false'
        assert '--xla_gpu_autotune_level=0' in os.environ['XLA_FLAGS']
        assert 'constant_folding,multi_output_fusion' in os.environ['XLA_FLAGS']
        assert '--xla_backend_optimization_level=0' in os.environ['XLA_FLAGS']


def test_explicit_compiler_settings_are_preserved():
    with patch.dict(os.environ, {'XLA_FLAGS': '--xla_gpu_autotune_level=1'}, clear=True):
        configure(0)
        assert os.environ['XLA_FLAGS'] == '--xla_gpu_autotune_level=1'


@pytest.mark.parametrize('options', [
    ['--device', '-1'], ['--envs', '0'], ['--envs', '33'],
    ['--ticks', '0'], ['--ticks', '9'], ['--minibatch', '0'], ['--minibatch', '3'],
])
def test_smoke_rejects_unbounded_or_invalid_shapes(options):
    with pytest.raises(SystemExit):
        arguments(['--line', 'DUEL', '--device', '0', *options])


def test_rocm_training_preserves_gate_before_loading_jax():
    result = subprocess.run([sys.executable, '-m', 'scripts.dcu_training',
        '--line', 'DUEL', '--train', '--gpu', '0', '--updates', '1',
        '--out', 'runs/should-not-exist'], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'Persistent user stop' in result.stderr
    assert 'Traceback' in result.stderr
    assert 'xla_bridge' not in result.stderr


def test_adapter_keeps_runtime_authorization_and_uses_rocm():
    adapted = adapted_train()
    assert adapted.__globals__['require_training_authorization']
    assert adapted.__globals__['_configure_dcu'] is configure
    assert 'model_cuda_overlay' not in adapted.__code__.co_names
