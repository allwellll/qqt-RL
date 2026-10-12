"""Real CPU optimizer updates and fresh-process full-state replay on a tiny batch."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu', JAXBOMB_RULE='bun')
from pathlib import Path
import pickle
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scripts import je03_bun_selfplay_ops_20261012 as ops
from scripts.je03_bun_stage1_20261011 import digest


def forward(params, arch, obs, state):
    x = state
    actor, critic = params['actor'], params['critic']
    return x @ actor['move'], x @ actor['ability'], jnp.zeros(len(x)), x @ critic


def fixture():
    params = dict(actor=dict(move=jnp.arange(10, dtype=jnp.float32).reshape(2, 5) / 100,
                            ability=jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 100),
                  critic=jnp.zeros((2, 128)))
    obs = jnp.zeros((2, 2, 1, 1, 1)); state = jnp.array([[[1., .2], [.3, 1.]], [[.5, 1.], [1., .8]]])
    actions = jnp.array([[[0, 0], [1, 1]], [[2, 0], [3, 1]]])
    masks = (jnp.ones((2, 2, 5), bool), jnp.ones((2, 2, 3), bool), jnp.zeros((2, 2, 5, 3), bool))
    batch = (obs, state, actions, jnp.full((2, 2), -2.7),
             jnp.array([[1., -.5], [.7, -1.]]), jnp.array([[2., -2.], [1., -1.]]), masks)
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(7.5e-5))
    return dict(params=params, adam=opt.init(params), ema=params,
                key=jax.random.PRNGKey(82), env=jnp.arange(8),
                league=(params['actor'], params['actor']), batch=batch)


def update(saved):
    ops.net_forward = forward
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(7.5e-5))
    key, next_key = jax.random.split(saved['key'])
    params, adam, loss = ops.ppo_update(saved['params'], opt, saved['adam'], 'tiny',
        saved['batch'], key, 4, .1, .5, .005, 1, return_loss=True,
        adv_top_frac=1., reference_params=fixture()['params'], reference_kl_coef=.1)
    assert np.isfinite(np.asarray(loss)).all()
    ema = jax.tree.map(lambda e, p: .995 * e + .005 * p, saved['ema'], params)
    return dict(saved, params=params, adam=adam, ema=ema, key=next_key,
                env=saved['env'] + jax.random.randint(key, (8,), 0, 4),
                league=ops.advance_league(saved['league'], params['actor'], 128))


def test_real_nonzero_update_and_fresh_process_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'net_forward', forward)
    original = fixture(); first = update(original)
    for key in ['actor', 'critic']:
        delta = sum(float(jnp.sum((x-y)**2)) for x, y in zip(
            jax.tree.leaves(first['params'][key]), jax.tree.leaves(original['params'][key])))
        assert delta > 0, key
    expected = digest(update(first))
    source = tmp_path / 'state.pkl'; target = tmp_path / 'digest.txt'
    with source.open('wb') as stream: pickle.dump(jax.device_get(first), stream)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), str(source), str(target)],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu'), check=True)
    assert target.read_text() == expected


def test_teacher_kl_has_no_critic_gradient(monkeypatch):
    monkeypatch.setattr(ops, 'net_forward', forward)
    saved = fixture(); obs, state, actions, old, adv, ret, masks = saved['batch']
    flatten = lambda x: x.reshape((-1,) + x.shape[2:])
    teacher = saved['params']
    params = jax.tree.map(lambda x: x + .1 * jnp.arange(x.size).reshape(x.shape), teacher)
    def difference(p):
        args = (p, 'tiny', flatten(obs), flatten(state), flatten(masks[0]), flatten(masks[1]),
                flatten(actions), old.reshape(-1), adv.reshape(-1), ret.reshape(-1), .1, .5, .005,
                flatten(masks[2]), 'off', 0., teacher)
        return ops._ppo_loss(*args, .1) - ops._ppo_loss(*args, 0.)
    grad = jax.grad(difference)(params)
    assert np.isfinite(np.asarray(jax.tree.leaves(grad['actor'])[0])).all()
    assert sum(float(jnp.sum(x**2)) for x in jax.tree.leaves(grad['actor'])) > 0
    np.testing.assert_array_equal(grad['critic'], 0)


if __name__ == '__main__':
    with Path(sys.argv[1]).open('rb') as stream: saved = pickle.load(stream)
    saved = jax.tree.map(jnp.asarray, saved)
    Path(sys.argv[2]).write_text(digest(update(saved)))
