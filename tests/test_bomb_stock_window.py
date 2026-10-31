"""User-defined stock rule, applied only by the versioned new runtime."""
import jax
import jax.numpy as j
import pytest
from jax_bomb import je07_bun_env as env
from scripts import je09_bun_scenes as scenes

@pytest.mark.parametrize('seat',[0,1])
@pytest.mark.parametrize('mode',['free','slow','dead','sugar','expired','loose','delivery'])
def test_stock_window_matches_js(seat,mode):
    env.prepare()
    s=scenes.init_batch(jax.random.PRNGKey(333),1,jax.nn.one_hot(8,10))
    enemy=1-seat
    pos=s.pos.at[0,seat].set(env._BUN_BASES[seat if mode=='delivery' else enemy].astype(j.float32)+1.5)
    s=s._replace(core=s.core._replace(pos=pos,alive=j.ones_like(s.alive),
        brick=j.zeros_like(s.brick),pushable=j.zeros_like(s.core.pushable),
        fuse=j.zeros_like(s.fuse),blast_linger=j.zeros_like(s.core.blast_linger)),
        trapped=j.zeros_like(s.trapped),bun_carried=j.full_like(s.bun_carried,-1),
        bun_respawn=j.zeros_like(s.bun_respawn),bun_loose=j.zeros_like(s.bun_loose))
    if mode=='dead':s=s._replace(core=s.core._replace(alive=s.alive.at[0,enemy].set(False)),bun_respawn=s.bun_respawn.at[0,enemy].set(50))
    if mode=='sugar':s=s._replace(trapped=s.trapped.at[0,enemy].set(10))
    if mode=='slow':s=s._replace(move_status=s.move_status.at[0,enemy].set(1),status_ticks=s.status_ticks.at[0,enemy].set(20))
    if mode=='expired':s=s._replace(bun_respawn=s.bun_respawn.at[0,enemy].set(50))
    if mode=='loose':
        r,c=map(int,pos[0,seat]);s=s._replace(bun_loose=s.bun_loose.at[0,r,c,enemy].set(1))
    if mode=='delivery':s=s._replace(bun_carried=s.bun_carried.at[0,seat].set(enemy),bun_stored=s.bun_stored.at[0,enemy,enemy].set(0))
    a=j.array([[[4,0],[4,0]]])
    after,_,info=jax.jit(jax.vmap(lambda s,a:env.step(s,a,jax.random.PRNGKey(1),False,True,require_stock_window=True)))(s,a)
    assert int(after.bun_carried[0,seat])==(enemy if mode in ('dead','sugar','loose') else -1)
    assert bool(info['steal'][0,seat])==(mode in ('dead','sugar'))
    if mode=='delivery':assert int(after.bun_stored[0,seat,enemy])==1
