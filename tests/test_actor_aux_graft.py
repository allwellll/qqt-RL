"""The production actor path must let danger supervision train the backbone.

The root-cause audit found the danger/escape/safe-action labels only ever
reached the critic or a sidecar NPZ — never the actor's shared transformer
backbone. ``train_bun_separate_ac.ensure_actor_aux_heads`` grafts the three
consequence-prediction heads onto a legacy policy backbone (a NEW actor
lineage: same obs/backbone, three extra heads). These tests pin that (a) the
graft is shape-correct and idempotent, and (b) the aux BCE built exactly as the
actor loss builds it produces non-zero gradient on backbone weights while an
``aux_scale`` of 0 leaves the policy path untouched.
"""

import jax
import jax.numpy as jnp

from jax_bomb import bun_env
from jax_bomb import jax_net
from scripts.train_bun_separate_ac import ensure_actor_aux_heads


def _backbone(aux_heads):
    key = jax.random.PRNGKey(0)
    return jax_net.init_transformer(
        key, c=30, h=bun_env.H, w=bun_env.W, embed=64, depth=2, heads=4,
        patch=3, state_dim=24, aux_heads=aux_heads)


def _batch(n=4, c=30):
    obs = jax.random.normal(jax.random.PRNGKey(1), (n, c, bun_env.H, bun_env.W))
    state = jax.random.normal(jax.random.PRNGKey(2), (n, 24))
    return obs, state


def test_graft_adds_shape_correct_heads_and_is_idempotent():
    legacy = _backbone(False)
    assert "wsafe" not in legacy["heads"]
    grafted = ensure_actor_aux_heads(legacy, jax.random.PRNGKey(7))
    heads = grafted["heads"]
    assert {"wsafe", "wesc", "wmargin"} <= set(heads)
    assert heads["wsafe"][0].shape[1] == bun_env.N_MOVES * bun_env.N_BOMB
    assert heads["wesc"][0].shape[1] == 1
    assert heads["wmargin"][0].shape[1] == 1
    # policy/value heads are preserved untouched
    assert jnp.array_equal(heads["wm"][0], legacy["heads"]["wm"][0])
    # idempotent: re-grafting returns the same object unchanged
    again = ensure_actor_aux_heads(grafted, jax.random.PRNGKey(99))
    assert again is grafted


def test_actor_aux_bce_trains_backbone_and_scale_zero_is_inert():
    params = ensure_actor_aux_heads(_backbone(False), jax.random.PRNGKey(7))
    obs, state = _batch()
    safe_t = jnp.zeros((4, bun_env.N_MOVES * bun_env.N_BOMB)).at[:, 0].set(1.0)
    esc_t = jnp.ones((4,))
    margin_t = jnp.full((4,), 0.6)

    def aux_loss(p, scale):
        _, _, _, _, aux = jax_net.transformer_aux_forward(p, obs, state)
        import optax
        safe = optax.sigmoid_binary_cross_entropy(aux["safe_action"], safe_t).mean()
        esc = optax.sigmoid_binary_cross_entropy(aux["escape"], esc_t).mean()
        margin = ((jax.nn.sigmoid(aux["margin"]) - margin_t) ** 2).mean()
        return scale * (safe + esc + margin)

    grads = jax.grad(lambda p: aux_loss(p, 1.0))(params)
    assert float(jnp.abs(grads["tok"][0]).sum()) > 0.0
    assert float(jnp.abs(grads["blocks"][0]["q"][0]).sum()) > 0.0
    assert float(jnp.abs(grads["heads"]["wsafe"][0]).sum()) > 0.0

    zero_grads = jax.grad(lambda p: aux_loss(p, 0.0))(params)
    assert float(jnp.abs(zero_grads["tok"][0]).sum()) == 0.0
    assert float(jnp.abs(zero_grads["heads"]["wsafe"][0]).sum()) == 0.0
