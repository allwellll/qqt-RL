"""Native JE07 fixture restored from commit 80dd11e for JE08 regressions.

Only the shared fixture is restored; it creates synthetic test states, never
training curriculum or evaluation trajectories.
"""
import jax
import jax.numpy as jnp
from jax_bomb import je07_bun_env as env

def fixture_state(case):
    env.prepare();env.configure_training('full=1',1)
    s=env._fresh(jax.random.PRNGKey(10))
    pos=jnp.array(case.get('pos',[[5.5,5.5],[5.5,9.5]]),jnp.float32)
    alive=jnp.array(case.get('alive',[True,True]))
    core=s.core._replace(pos=pos,brick=jnp.zeros_like(s.brick),crate=jnp.zeros_like(s.core.crate),
        alive=alive,hp=alive.astype(jnp.int32),invuln=jnp.zeros(2,jnp.int32))
    if case.get('bomb'):
        r,c=case['bomb'];core=core._replace(fuse=core.fuse.at[r,c].set(1),owner=core.owner.at[r,c].set(1),bomb_blast=core.bomb_blast.at[r,c].set(2))
    carried=jnp.array(case.get('carry',[-1,-1]),jnp.int8)
    stock=jnp.eye(2,dtype=s.bun_stored.dtype)
    for p in range(2):
        if int(carried[p])>=0:stock=stock.at[int(carried[p]),int(carried[p])].set(0)
    slots=jnp.zeros((2,7),jnp.int8);counts=jnp.zeros((2,7),jnp.int16)
    for player,item in enumerate(case.get('items',[0,0])):
        slots=slots.at[player,0].set(item);counts=counts.at[player,0].set(3 if item==2 else int(item>0))
    return s._replace(item_slots=slots,item_counts=counts,held_item=slots[:,0],core=core,bun_spawn_pos=pos,bun_carried=carried,bun_stored=stock,
        trapped=jnp.array(case.get('trap',[0,0]),jnp.int16),bun_respawn=jnp.array(case.get('respawn',[0,0]),jnp.int16),
        move_status=jnp.array(case.get('slow',[0,0]),jnp.int8),status_ticks=jnp.array([20,20],jnp.int16))


def fixture(case):
    s = fixture_state(case)
    loose = s.bun_loose
    for row, col, origin, count in case.get("loose", []):
        loose = loose.at[row, col, origin].set(count)
    return s._replace(bun_loose=loose)
