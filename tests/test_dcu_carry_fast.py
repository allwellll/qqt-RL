"""CPU reset equivalence plus benchmark admission, without implicit DCU use."""
import os
from unittest.mock import patch

import pytest

from scripts.benchmark_dcu_carry import arguments


def test_benchmark_requires_explicit_ack(tmp_path):
    with pytest.raises(SystemExit):
        arguments(['--report', str(tmp_path/'out.json')])


@pytest.mark.parametrize('options', [
    ['--envs', '0'], ['--envs', '4097'], ['--device', '-1'],
    ['--ticks', '0'], ['--repeats', '21'], ['--minibatch', '3'],
    ['--value-chunk', '0'], ['--value-chunk', '3'],
    ['--policy-chunk', '0'], ['--policy-chunk', '3'],
    ['--pipeline-repeats', '1'], ['--pipeline-repeats', '6', '--ppo'],
])
def test_bounded_benchmark(options):
    with pytest.raises(SystemExit):
        arguments(['--report', 'reports/carry-test-uncreated.json',
                   '--acknowledge-benchmark', *options])


def test_fast_reset_matches_native_leaf_for_leaf():
    if os.environ.get('JAX_PLATFORMS') != 'cpu':
        pytest.skip('Explicit CPU backend required')
    import jax
    import numpy as np
    from qqt_rl.training import jax_carry_env as E
    from qqt_rl.training.jax_carry_fast import fresh_factory
    E.prepare()
    fresh = fresh_factory()
    keys = jax.random.split(jax.random.PRNGKey(791), 64)
    native = jax.jit(jax.vmap(E.fresh))(keys)
    optimized = jax.jit(jax.vmap(fresh))(keys)
    assert jax.default_backend() == 'cpu'
    for a, b in zip(jax.tree.leaves(native), jax.tree.leaves(optimized)):
        np.testing.assert_array_equal(a, b)


def test_default_training_uses_device_loop_and_bulk_critic():
    from scripts.train_dcu_jax_carry import arguments as training_args, plan
    row = plan(training_args(['plan']))
    assert row['collector'] == 'fused'
    assert row['envs_per_device'] == 2048
    assert row['global_environments'] == 16384
    assert row['global_samples'] == 1048576
    assert row['compiler_profile'] == 'throughput'
    assert row['policy_chunk'] == row['value_chunk'] == 512
    assert not row['python_tick_dispatch']
    assert not row['per_tick_host_tensor_reads']


def test_compiler_profile_preserves_explicit_flags():
    from scripts.train_dcu_jax_carry import configure_compiler
    with patch.dict(os.environ, {}, clear=True):
        configure_compiler('safe')
        assert 'XLA_FLAGS' not in os.environ
        configure_compiler('throughput')
        assert '--xla_backend_optimization_level=2' in os.environ['XLA_FLAGS']
    with patch.dict(os.environ, {'XLA_FLAGS': '--explicit-user-choice'}, clear=True):
        configure_compiler('throughput')
        assert os.environ['XLA_FLAGS'] == '--explicit-user-choice'


def test_kill_plan_uses_frozen_evaluated_history_pool():
    from scripts.train_dcu_jax_kill import arguments as kill_args, plan
    row = plan(kill_args(['plan', '--devices', '0,1,2,3,4,5,6,7', '--envs', '512']))
    assert row['initial_actor'] == 'u930'
    assert row['rule_bot_fraction'] == .1
    assert row['opponent_model_fractions'] == dict(
        current=.09, recent=.09, u930=.315, u840=.315, SP=.045, G=.045)
    assert row['opponent_pool']['candidate_auto_admission'] is False
