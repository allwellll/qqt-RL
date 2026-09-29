"""The aux supervision heads must train the SHARED backbone, not a sidecar.

The root-cause audit found the old counterfactual/safety labels only ever
touched the critic or an NPZ file — they never entered the actor's gradient, so
the policy backbone was never taught to reason about danger. These tests pin the
fix: the safe-action / escape / margin heads read the same pooled transformer
feature as the policy head, and their BCE gradient is non-zero on backbone
weights (patch projection + attention blocks). If someone re-attaches the aux
loss to a detached feature, the backbone-gradient assertion goes RED.
"""

import jax
import jax.numpy as jnp

from jax_bomb import jax_net
from jax_bomb import bun_env


def _model(aux_heads):
    key = jax.random.PRNGKey(0)
    return jax_net.init_transformer(
        key, c=30, h=bun_env.H, w=bun_env.W, embed=64, depth=2, heads=4,
        patch=3, state_dim=24, aux_heads=aux_heads)


def _batch(n=4, c=30):
    obs = jax.random.normal(jax.random.PRNGKey(1), (n, c, bun_env.H, bun_env.W))
    state = jax.random.normal(jax.random.PRNGKey(2), (n, 24))
    return obs, state


def test_aux_heads_present_only_when_requested():
    assert "wsafe" not in _model(False)["heads"]
    heads = _model(True)["heads"]
    assert {"wsafe", "wesc", "wmargin"} <= set(heads)
    assert heads["wsafe"][0].shape[1] == bun_env.N_MOVES * bun_env.N_BOMB


def test_aux_forward_shapes():
    params = _model(True)
    obs, state = _batch()
    mv, bm, v, v_logits, aux = jax_net.transformer_aux_forward(params, obs, state)
    assert mv.shape == (4, bun_env.N_MOVES)
    assert bm.shape == (4, bun_env.N_BOMB)
    assert aux["safe_action"].shape == (4, bun_env.N_MOVES * bun_env.N_BOMB)
    assert aux["escape"].shape == (4,)
    assert aux["margin"].shape == (4,)


def test_baseline_forward_matches_between_plain_and_aux_paths():
    params = _model(True)
    obs, state = _batch()
    plain = jax_net.transformer_forward(params, obs, state)
    aug = jax_net.transformer_aux_forward(params, obs, state)
    for a, b in zip(plain, aug[:4]):
        assert jnp.allclose(a, b)


def test_aux_bce_gradient_flows_into_backbone():
    params = _model(True)
    obs, state = _batch()
    safe_target = jnp.zeros((4, bun_env.N_MOVES * bun_env.N_BOMB)).at[:, 0].set(1.0)
    esc_target = jnp.ones((4,))
    margin_target = jnp.full((4,), 0.6)

    def aux_only_loss(p):
        _, _, _, _, aux = jax_net.transformer_aux_forward(p, obs, state)
        safe = jnp.mean(_bce(aux["safe_action"], safe_target))
        esc = jnp.mean(_bce(aux["escape"], esc_target))
        margin = jnp.mean((jax.nn.sigmoid(aux["margin"]) - margin_target) ** 2)
        return safe + esc + margin

    grads = jax.grad(aux_only_loss)(params)
    # backbone weights (patch projection + first attention query) must move
    tok_g = jnp.abs(grads["tok"][0]).sum()
    attn_g = jnp.abs(grads["blocks"][0]["q"][0]).sum()
    assert float(tok_g) > 0.0
    assert float(attn_g) > 0.0
    # and the aux head weights themselves get gradient
    assert float(jnp.abs(grads["heads"]["wsafe"][0]).sum()) > 0.0


def test_aux_loss_does_not_touch_policy_head_when_isolated():
    # aux-only loss leaves policy/value head weights untouched (they are not on
    # the aux path), proving the shared signal is the backbone, not the heads.
    params = _model(True)
    obs, state = _batch()
    target = jnp.ones((4,))

    def esc_loss(p):
        _, _, _, _, aux = jax_net.transformer_aux_forward(p, obs, state)
        return jnp.mean(_bce(aux["escape"], target))

    grads = jax.grad(esc_loss)(params)
    assert float(jnp.abs(grads["heads"]["wm"][0]).sum()) == 0.0
    assert float(jnp.abs(grads["heads"]["wesc"][0]).sum()) > 0.0
    assert float(jnp.abs(grads["blocks"][0]["ff1"][0]).sum()) > 0.0


def _bce(logits, target):
    return jnp.maximum(logits, 0) - logits * target + jnp.log1p(jnp.exp(-jnp.abs(logits)))
