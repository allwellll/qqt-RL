import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
import jax
import jax.numpy as j
from jax_bomb import je07_bun_env_r1_items as E
from scripts import je09_bun_scenes as scenes


@pytest.mark.parametrize('seat',[0,1])
@pytest.mark.parametrize('window',[False,True])
def test_native_stock_available_against_alive_enemy(seat,window):
    E.prepare()
    s=E.init_batch(jax.random.PRNGKey(333),1,jax.nn.one_hot(8,16))
    pos=s.pos.at[0,seat].set(E._BUN_BASES[1-seat].astype(j.float32)+1.5)
    s=s._replace(core=s.core._replace(pos=pos,alive=j.ones_like(s.alive),brick=j.zeros_like(s.brick),
        pushable=j.zeros_like(s.core.pushable),fuse=j.zeros_like(s.fuse),blast_linger=j.zeros_like(s.core.blast_linger)),
        trapped=j.zeros_like(s.trapped),bun_carried=j.full_like(s.bun_carried,-1),bun_respawn=j.zeros_like(s.bun_respawn),bun_loose=j.zeros_like(s.bun_loose))
    fn=jax.jit(jax.vmap(lambda st:E.step(st,j.array([[4,0],[4,0]]),jax.random.PRNGKey(1),False,True,require_stock_window=window)))
    a,_,info=fn(s)
    assert int(a.bun_carried[0,seat])==(-1 if window else 1-seat)
    assert bool(info['steal'][0,seat]) is (not window)


def test_opt_in_does_not_mutate_reference_process():
    code='''from scripts.bun_native_v2_runtime import install
e=install()
from scripts.bomb_threat_runtime import install as again
assert again() is e
assert e.step._bun_native_v2
assert not getattr(e.step,'_stock_window_v1',False)
assert e.RULE_VERSION=='bun-native-v2'
'''
    subprocess.run([sys.executable,'-c',code],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu'))
    from jax_bomb import je07_bun_env as old
    assert old is not E
    assert not hasattr(E,'RULE_VERSION')
