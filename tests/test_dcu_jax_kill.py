"""Admission and planning checks that never initialize DCUs."""
import subprocess
import sys

import pytest

from scripts.train_dcu_jax_kill import arguments, plan


@pytest.mark.parametrize('options', [
    ['--devices', '0,0'], ['--devices', '-1'], ['--envs', '8'],
    ['--ticks', '0'], ['--minibatch', '3'], ['--hours', '6'],
    ['--hours', 'nan'], ['--max-updates', '0'], ['--critic-warmup', '-1'],
])
def test_rejects_invalid_budget_or_geometry(options):
    with pytest.raises(SystemExit):
        arguments(['plan', *options])


def test_explicit_training_and_fresh_output_required(tmp_path):
    with pytest.raises(SystemExit):
        arguments(['train', '--out', str(tmp_path / 'new')])
    with pytest.raises(SystemExit):
        arguments(['train', '--out', str(tmp_path), '--acknowledge-training'])


def test_device_training_plan():
    row = plan(arguments(['plan']))
    assert row['global_environments'] == 1024
    assert row['global_samples'] == 65536
    assert row['global_minibatch'] == 4096
    assert row['rollout_gae_on_device'] and not row['host_callbacks']
    assert row['rule_bot_fraction'] + sum(row['opponent_model_fractions'].values()) == 1
    assert not row['optimizer_saved'] and not row['precise_resume']
    assert row['checkpoint_policy']['budget_gib'] == 4


def test_plan_is_read_only_and_does_not_initialize_jax(tmp_path):
    out = tmp_path / 'uncreated'
    code = ('import sys; from scripts.train_dcu_jax_kill import main; '
            f'main(["plan", "--out", {str(out)!r}]); '
            'assert "jax" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not out.exists()
