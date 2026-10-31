"""Forced slide follows production Sim.step's direct movement branch."""
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bomb import je07_bun_env as env
from test_campaign_side_parity_round2 import _configured_state
from test_campaign_side_parity_round3 import clear_core

ROOT = Path(__file__).resolve().parents[1]


def oracle(case):
    source = r"""
const Q=require('./web/sim.js'),level=require('./web/assets/maps/levels.json').find(x=>x.qqt_id===806);
const c=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const s=new Q.Sim(17);s.reset(level,{nativeTrap:true,nativeItems:true});
for(const n of ['wall','brick','bush','cover','crate','fuse','pushable','fieldItem','invuln','spawnProtection'])s[n].fill(0);
s.pos.set([c.y,c.x,10.5,10.5]);
if(c.block){s.wall[4*Q.W+5]=1;s.wall[5*Q.W+5]=1;}
if(c.push){s.wall[4*Q.W+5]=0;s.brick[4*Q.W+5]=1;s.pushable[4*Q.W+5]=1;s.pushBoxAt.fill(-1);s.pushBoxAt[65]=0;s.pushBoxes=[{o:65,cells:[65]}];}
s.movementStatus[0]=c.slide?2:0;s.slideDir[0]=3;
const blocked=new Uint8Array(Q.H*Q.W);for(let i=0;i<blocked.length;i++)blocked[i]=s.wall[i]||s.brick[i]||s.fuse[i]>0;
const dist=.3*s.spdG[0]*s.playerMoveScale(0);
const direct=s._tryMove(c.y,c.x,3,blocked,dist),steer=s._steer(c.y,c.x,3,blocked,dist,0);
s.step([[3,0],[4,0]]);
console.log(JSON.stringify({pos:Array.from(s.pos.slice(0,2)),status:s.movementStatus[0],pushT:s.pushT[65],direct,steer}));
"""
    p = subprocess.run(['node', '-e', source], input=json.dumps(case),
                       cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(p.stdout)


@pytest.mark.parametrize('case', [
    dict(y=4.8,x=4.5,block=False,slide=True,push=False),
    dict(y=4.8,x=4.5,block=True,slide=True,push=False),
    dict(y=4.8,x=4.6399,block=True,slide=True,push=False),
    dict(y=4.8,x=4.5,block=True,slide=False,push=False),
    dict(y=4.8,x=4.6399,block=True,slide=True,push=True),
])
def test_forced_slide_direct_path_partial_stop_and_no_push(case):
    st=_configured_state();core=clear_core(st)
    core=core._replace(pos=jnp.array([[case['y'],case['x']],[10.5,10.5]],jnp.float32))
    if case['block']:
        core=core._replace(wall=core.wall.at[4,5].set(not case['push']).at[5,5].set(True),
                           brick=core.brick.at[4,5].set(case['push']),
                           pushable=core.pushable.at[4,5].set(case['push']))
    st=st._replace(core=core,move_status=jnp.array([2 if case['slide'] else 0,0],jnp.int8),
                   slide_dir=jnp.array([3,2],jnp.int8))
    after,_,_=env.step(st,jnp.array([[3,0],[4,0]]),jax.random.PRNGKey(17),
                        auto_reset=False,return_info=True)
    expected=oracle(case)
    np.testing.assert_allclose(after.pos[0],expected['pos'],atol=2e-5,rtol=0)
    assert int(after.move_status[0])==expected['status']
    np.testing.assert_allclose(after.core.push_t[4,5],expected['pushT'],atol=2e-5)
    if case['slide']:
        np.testing.assert_allclose(after.pos[0],expected['direct'],atol=2e-5,rtol=0)
    if case['block'] and not case['push']:
        assert np.abs(np.asarray(expected['direct'])-expected['steer']).max()>.1
