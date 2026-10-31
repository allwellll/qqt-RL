from scripts import gpu_efficiency_observe as observe
import numpy as np
import pytest


def test_observer_sanitizes_role_identity_and_measures_interval_cpu():
    assert observe.role_for_command('python -m scripts.expert_reward_ablation_resume worker legacy 42') == 'D_worker'
    assert observe.role_for_command('python unrelated.py') is None
    previous = {123:dict(role='D_worker',cpu_ticks=100,read_bytes=1,write_bytes=2)}
    current = {123:dict(role='D_worker',cpu_ticks=110,rss_bytes=256,read_bytes=11,write_bytes=22)}
    result = observe.summarize_processes(current,previous,1.)
    assert result['D_worker']['count'] == 1
    assert result['D_worker']['interval_write_bytes'] == 20
    assert '123' not in str(result)
    assert result['D_worker']['cpu_percent_one_core'] > 0


def test_parity_requires_exact_discrete_fields_and_declared_float_tolerance():
    from scripts.gpu_efficiency_parity import compare_arrays
    assert compare_arrays(np.array([1],np.uint32),np.array([1],np.uint32),exact=True)['passed']
    assert not compare_arrays(np.array([1],np.uint32),np.array([2],np.uint32),exact=True)['passed']
    assert not compare_arrays(np.array([1.]),np.array([1.+1e-7]),exact=True)['passed']
    assert compare_arrays(np.array([1.]),np.array([1.+1e-7]),exact=False)['passed']
    assert not compare_arrays(np.array([np.nan]),np.array([np.nan]),exact=False)['passed']


def test_process_role_ignores_shell_commands_which_only_mention_a_module():
    assert observe.role_for_command("bash -c 'python -m scripts.gpu_efficiency_observe output'") is None
    assert observe.role_for_command('python -m scripts.expert_reward_ablation_resume owner') == 'D_owner'


def test_cpu_parity_has_no_training_or_queue_entrypoint():
    from pathlib import Path
    source=Path('scripts/gpu_efficiency_parity.py').read_text()
    assert 'train_seed(' not in source
    assert 'CUDA_VISIBLE_DEVICES' in source
    assert "JAX_PLATFORMS='cpu'" in source


def test_future_gpu_probe_refuses_to_run_before_D_fixed_budget_terminal(tmp_path,monkeypatch):
    from scripts import gpu_efficiency_performance_probe as probe
    monkeypatch.setattr(probe,'D',tmp_path)
    with pytest.raises(RuntimeError,match='six-run queue has not completed'):
        probe.verify_handoff()
