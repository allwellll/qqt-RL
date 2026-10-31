"""A single supervised coefficient must leave PPO/value/entropy/anchor intact."""
import importlib
import jax
import jax.numpy as j
import numpy as np
import pytest


def test_control_is_identity_and_invalid_weight_rejected():
    m=importlib.import_module('scripts.dagger_rl_supervision_scale')
    base=j.array(-0.);metrics=j.arange(8.)
    assert m.scale_loss(base,metrics,.2) is base
    with pytest.raises(ValueError):m.scale_loss(base,metrics,-1.)
    with pytest.raises(ValueError):m.scale_loss(base,metrics,float('nan'))


def test_only_supervised_raw_gradient_changes_critic_and_anchor_preserved():
    m=importlib.import_module('scripts.dagger_rl_supervision_scale')
    def f(p,weight):
        actor,critic=p;ppo=actor**2;value=critic**2;entropy=actor*.1
        sup=(actor-2)**2;anchor=(actor+1)**2
        return m.scale_loss(ppo+.5*value-.005*entropy+.2*sup+.05*anchor,
            j.array([ppo,value,entropy,sup,0.,0.,anchor,0.]),weight)
    p=j.array([.4,.8]);a=jax.grad(lambda q:f(q,.2))(p);b=jax.grad(lambda q:f(q,.05))(p)
    np.testing.assert_allclose(a[1],b[1])
    np.testing.assert_allclose(a[0]-b[0],.3*(p[0]-2),rtol=1e-6)
