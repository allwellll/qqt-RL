"""The PPO normalization scope matters when diagnosing a wrong route move."""
import numpy as np
import jax
import jax.numpy as jnp


def test_device_moments_match_actual_ppo_normalization():
    from scripts.ppo_advantage_diagnostics import device_moments, normalized_diagnostic
    values=jnp.array([[[1.,2.],[3.,6.]], [[-3.,-1.],[1.,3.]]],jnp.float32)
    moments=np.asarray(jax.jit(device_moments)(values))
    assert moments.shape==(2,2)
    for device in range(2):
        ad=values[device]
        reference=np.asarray((ad-ad.mean())/(ad.std()+1e-8))
        for tick in range(2):
            for env in range(2):
                got=normalized_diagnostic(float(ad[tick,env]),moments[device])
                np.testing.assert_allclose(got['collector_ppo_normalized_advantage'],reference[tick,env],rtol=2e-7,atol=1e-7)


def test_positive_raw_advantage_can_have_negative_normalized_direction():
    from scripts.ppo_advantage_diagnostics import device_moments, normalized_diagnostic
    values=jnp.array([[[1.,2.],[3.,6.]]],jnp.float32)
    got=normalized_diagnostic(1.,np.asarray(device_moments(values))[0])
    assert got['collector_ppo_normalized_advantage']<0
    assert got['collector_advantage_device_mean']==3.
    assert 'not action Q' in got['collector_normalization_scope']


def test_device_populations_are_separate_and_constant_batch_is_finite():
    from scripts.ppo_advantage_diagnostics import device_moments, normalized_diagnostic
    values=jnp.array([[[2.,2.],[2.,2.]], [[20.,20.],[20.,20.]]],jnp.float32)
    moments=np.asarray(device_moments(values))
    np.testing.assert_array_equal(moments,[[2.,0.],[20.,0.]])
    for device in range(2):
        got=normalized_diagnostic(float(values[device,0,0]),moments[device])
        assert got['collector_ppo_normalized_advantage']==0.


def test_telemetry_does_not_mutate_inputs_and_reduces_to_two_scalars_per_device():
    from scripts.ppo_advantage_diagnostics import device_moments, normalized_diagnostic
    values=np.arange(24,dtype=np.float32).reshape(3,2,4)
    before=values.copy();moments=np.asarray(device_moments(values))
    assert moments.shape==(3,2)
    normalized_diagnostic(float(values[1,0,0]),moments[1])
    np.testing.assert_array_equal(values,before)
