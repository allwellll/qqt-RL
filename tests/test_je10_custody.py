"""Actual legal stock pickup must keep DAgger active beyond initialized windows."""
import jax
import jax.numpy as jnp
from scripts import je09_bun_scenes as scenes
from scripts import je10_ops as ops


def test_native_stock_pickup_from_reset_window_starts_real_custody(monkeypatch):
    from scripts import je09_stage_h5c_dagger as d
    monkeypatch.setattr(d,"update",ops.update_custody)
    s=scenes.init_batch(jax.random.PRNGKey(11),2,jax.nn.one_hot(1,10))
    r=jnp.arange(2);seat=r%2;enemy=1-seat
    s=s._replace(bun_carried=jnp.full_like(s.bun_carried,-1))
    a=s._replace(bun_carried=s.bun_carried.at[r,seat].set(enemy.astype(s.bun_carried.dtype)),
        bun_stored=s.bun_stored.at[r,enemy,enemy].set(0))
    aux=ops.reward_state(2)
    info=dict(surviving_kill=jnp.zeros((2,2),bool),steal=jnp.zeros((2,2),bool).at[r,seat].set(True),
        pickup_loose=jnp.zeros((2,2),bool),pickup_origin=jnp.full((2,2),-1),native_transfer=jnp.zeros((2,2),bool))
    updated=d.update(s,a,info,aux,aux['dagger'])
    assert bool(updated['custody'][r,seat].all())
