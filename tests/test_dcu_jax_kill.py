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
    assert row['envs_per_device'] == 2048
    assert row['global_environments'] == 16384
    assert row['global_samples'] == 1048576
    assert row['minibatch_per_device'] == 2048
    assert row['global_minibatch'] == 16384
    assert row['learner_inference_batch'] == 2048
    assert row['rollout_gae_on_device'] and not row['host_callbacks']
    assert abs(row['rule_bot_fraction'] + sum(row['opponent_model_fractions'].values()) - 1) < 1e-12
    assert row['initial_actor'] == 'u930'
    assert row['opponent_model_fractions'] == dict(
        current=.09, recent=.09, u930=.315, u840=.315, SP=.045, G=.045)
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


def test_league_allocation_balances_categories_and_seats():
    from qqt_rl.training.jax_kill_collect import opponent_allocation
    for count in (128, 512, 2048):
        bots, groups = opponent_allocation(count)
        assert bots + int(groups.sum()) == count
        assert bots % 2 == 0 and all(int(n) % 2 == 0 for n in groups)


def test_e512_migration_preserves_other_source_checks(tmp_path, monkeypatch):
    from scripts import train_dcu_jax_kill as runner
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    collector = tmp_path / 'collector.py'
    collector.write_text('reviewed collector')
    sources = {'scripts/train_dcu_jax_kill.py': runner.E512_RUNNER_SHA,
               runner.COLLECTOR_SOURCE: runner.UNBATCHED_COLLECTOR_SHA,
               'collector.py': runner.sha(collector)}
    runner.verify_checkpoint_sources(sources)
    collector.write_text('changed collector')
    with pytest.raises(RuntimeError, match='collector.py'):
        runner.verify_checkpoint_sources(sources)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        runner.verify_checkpoint_sources({'scripts/train_dcu_jax_kill.py': 'unreviewed'})
