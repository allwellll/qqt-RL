"""Small explicit native states shared by JE08 topology regression tests."""
import jax
import jax.numpy as jnp
from jax_bomb import je07_bun_env as env


def fixture(values):
    env.prepare()
    env.configure_training('full=1', 1)
    state = env._fresh(jax.random.PRNGKey(707))
    core = state.core._replace(
        pos=jnp.asarray(values.get('pos', [[5.5, 5.5], [5.5, 9.5]])),
        brick=jnp.zeros_like(state.brick),
        crate=jnp.zeros_like(state.core.crate),
        invuln=jnp.zeros_like(state.core.invuln))
    carry = jnp.asarray(values.get('carry', [-1, -1]), jnp.int8)
    stored = jnp.asarray([[1, 0], [0, 1]], jnp.int8)
    loose = jnp.zeros_like(state.bun_loose)
    for row, col, origin, count in values.get('loose', []):
        loose = loose.at[row, col, origin].add(count)
        stored = stored.at[origin, origin].add(-count)
    for origin in (0, 1):
        stored = stored.at[origin, origin].add(-jnp.sum(carry == origin).astype(jnp.int8))
    return state._replace(
        core=core, bun_carried=carry, bun_stored=stored, bun_loose=loose,
        trapped=jnp.asarray(values.get('trap', [0, 0]), jnp.int16))
