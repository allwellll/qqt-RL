"""The bounded step-size contrast preserves saved Adam and the data pipeline."""
import importlib
import json
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
import optax

ROOT=Path(__file__).resolve().parents[1]


def protocol():
    return json.loads((ROOT/'runs/dagger_rl_20261024/r9_step_scale/PROTOCOL.json').read_text())


def test_paired_budget_and_original_supervision_common():
    old=json.loads((ROOT/'runs/dagger_rl_20261024/r8_dagger_scale/PROTOCOL.json').read_text())
    p=protocol()
    assert p['learning_rate_by_arm']==dict(control=7.5e-5,lr_quarter=1.875e-5)
    assert p['dagger_coefficient_by_arm']==dict(control=.2,lr_quarter=.2)
    for k in ['parent','parent_input_sha256','optimizer','train_mixture','bank_sha256',
              'oracle','aggregation','envs','steps','updates','checkpoints','gamma',
              'gae_lambda','clip','value','entropy','gradient_clip','epochs','minibatch','anchor_coefficient']:
        assert p[k]==old[k],k
    assert p['total_environment_steps']==6*262144
    assert p['total_optimizer_steps']==6*1024
    assert set(p['paired_seeds']).isdisjoint(old['paired_seeds'])
    a=p['evaluation']['natural_holdout_seed'];b=old['evaluation']['natural_holdout_seed']
    assert set(range(a,a+64)).isdisjoint(range(b,b+64))


def test_actual_arm_updater_retains_nonzero_Adam_and_changes_only_step(monkeypatch):
    m=importlib.import_module('scripts.dagger_rl_r9_train')
    p=protocol()
    monkeypatch.setattr(m,'read',lambda _:p)
    # Exercise the actual arm dispatch with existing nonzero Adam moments/count.
    # Only the heavy model objective is replaced; the optimizer is unchanged.
    monkeypatch.setattr(m.C,'updater',lambda mode,optimizer:optimizer.update)
    opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(p['lr'],eps=1e-8))
    params={'actor':j.array([.3,-.2]),'critic':j.array([.4])}
    grad={'actor':j.array([.7,-.9]),'critic':j.array([.6])}
    saved=opt.init(params)
    for _ in range(7):
        _,saved=opt.update(grad,saved,params)
    outputs=[m.arm_updater(arm,opt)(grad,saved,params) for arm in p['arms']]
    a,b=outputs
    for x,y in zip(jax.tree.leaves(a[1]),jax.tree.leaves(b[1])):
        np.testing.assert_array_equal(x,y)
    assert int(a[1][1][0].count)==8
    assert np.any(np.asarray(a[1][1][0].mu['actor'])!=0)
    for x,y in zip(jax.tree.leaves(a[0]),jax.tree.leaves(b[0])):
        assert np.linalg.norm(x)>0
        np.testing.assert_allclose(y,.25*x,rtol=1e-6,atol=1e-12)


def test_frozen_collector_aggregation_and_objective_reused():
    m=importlib.import_module('scripts.dagger_rl_r9_train')
    old=importlib.import_module('scripts.dagger_rl_r6_train')
    assert m.collector is old.collector
    assert m.aggregate is old.aggregate
    assert m.ORIGINAL_OBJECTIVE is old.C.objective
