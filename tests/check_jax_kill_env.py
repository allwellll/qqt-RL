"""Meaningful native JAX rule/input checks; run on eight-free CPU process."""
import json
from pathlib import Path
import subprocess
import jax
import jax.numpy as j
import numpy as np
from qqt_rl.training import jax_kill_env as E


def main():
 E.prepare();fresh=jax.jit(jax.vmap(E.fresh))
 worlds=fresh(jax.random.split(jax.random.PRNGKey(5),32));jax.block_until_ready(worlds)
 crates=np.asarray(worlds.state.core.crate);pos=np.asarray(worlds.state.pos)
 for i in range(32):
  assert (crates[i]>0).sum()==24
  assert all((crates[i]==kind).sum()==8 for kind in (1,2,3))
  np.testing.assert_array_equal(crates[i],crates[i,:,::-1])
  assert pos[i,0,0]==pos[i,1,0] and pos[i,0,1]+pos[i,1,1]==15
  for cell in np.argwhere(crates[i]>0):assert all(max(abs(cell-np.floor(p)))>1 for p in pos[i])
 # Independently encode a web native snapshot, including the generic kill-mode
 # global24 branch (it is deliberately different from bun delivery global24).
 code="""const {create}=require('./scripts/kill_mode_bridge');const I=require('./web/r3_model_inputs');
 const s=create(7);console.log(JSON.stringify({snapshot:s.snapshotReplay(),
 inputs:[0,1].map(p=>{let a=I.inputs(s,p);return {obs:Array.from(a.obs),state:Array.from(a.state)}})}));"""
 fixture=json.loads(subprocess.check_output(['.tools/node/bin/node','-e',code]))
 snap=fixture['snapshot'];w=E.fresh(jax.random.PRNGKey(7));st=w.state;c=st.core
 crate=np.array(snap['crate'],np.int8).reshape(13,15)
 types=np.array(snap['crateType'],np.int8).reshape(13,15)
 c=c._replace(pos=j.array(snap['pos'],j.float32).reshape(2,2),crate=j.array(np.where(crate>0,types+1,0),j.int8))
 w=w._replace(state=st._replace(core=c))
 batch=jax.tree.map(lambda x:j.stack([x,x]),w)
 obs,gv,_,_,_=jax.jit(E.inputs)(batch,j.array([0,1]))
 for p in range(2):
  np.testing.assert_allclose(obs[p].reshape(-1),fixture['inputs'][p]['obs'],rtol=0,atol=1e-6)
  np.testing.assert_allclose(gv[p,:37],fixture['inputs'][p]['state'],rtol=0,atol=1e-6)
  assert gv[p,37]==0
 # Exercise the same compiled transition for actual trap expiry and kill-mode
 # clock. Kill provenance must survive the delay, with 8 typed death drops.
 w=E.fresh(jax.random.PRNGKey(11));st=w.state;c=st.core
 c=c._replace(pos=j.array([[3.5,3.5],[9.5,9.5]]),wall=j.zeros_like(c.wall),crate=j.zeros_like(c.crate))
 st=st._replace(core=c,trapped=j.array([0,1],j.int16),trapped_source=j.array([[False,False],[True,False]]),
  trapped_causal=j.array([[False,False],[True,False]]),native_items=E.I.fresh())
 w=w._replace(state=st)
 step=jax.jit(E.step);idle=j.array([[4,0],[4,0]],j.int32)
 dead,reward,done,metrics=step(w,idle,jax.random.PRNGKey(12));jax.block_until_ready(dead)
 np.testing.assert_array_equal(dead.kills,[1,0]);np.testing.assert_allclose(reward,[5,-5])
 assert not done and not dead.state.alive[1] and dead.respawn[1]==29
 assert (np.asarray(dead.state.core.crate)>0).sum()==8
 assert dead.state.native_items.overflow==0
 last=w._replace(state=w.state._replace(core=w.state.core._replace(t=j.array(1799))))
 after,reward,done,metrics=step(last,idle,jax.random.PRNGKey(13))
 assert done and metrics[0,5]==1 and metrics[1,5]==0
 own=w._replace(state=w.state._replace(trapped_source=j.array([[False,False],[False,True]]),
  trapped_causal=j.array([[False,False],[False,True]])))
 after,reward,done,metrics=step(own,idle,jax.random.PRNGKey(14))
 assert metrics[1,2]==1 and reward[1]==-7
 waiting=dead._replace(respawn=j.array([0,1],j.int16),target=j.array([[-1,-1],[5,5]],j.int32))
 after,reward,done,metrics=step(waiting,idle,jax.random.PRNGKey(15))
 assert after.state.alive[1] and after.state.core.invuln[1]==10 and after.protection[1]==10
 print(json.dumps(dict(status='PASS_NATIVE_JAX_KILL',opening_seeds=32,opening_count=24,
  web_obs25_global37_parity=True,delayed_kill=True,last_tick=True,self_defeat=True,
  death_drops=8,respawn_countdown=True,spawn_protection=True)))

if __name__=='__main__':main()
