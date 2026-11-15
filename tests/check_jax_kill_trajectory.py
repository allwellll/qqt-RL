"""Same-action deterministic native movement/bomb/trap check before random drops."""
import json,subprocess
import jax
import jax.numpy as j
import numpy as np
from qqt_rl.training import jax_kill_env as E

E.prepare()
source="""const {create}=require('./scripts/kill_mode_bridge');const s=create(111);s.crate.fill(0);
const initial=s.snapshotReplay(),rows=[];
for(let t=0;t<60;t++){let a=[[t%5,Number(t===0),0],[(t+2)%5,Number(t===2),0]];
let info=s.step(a);if(info.died.some(Boolean))break;rows.push({actions:a,after:s.snapshotReplay()});}
console.log(JSON.stringify({initial,rows}));"""
fixture=json.loads(subprocess.check_output(['.tools/node/bin/node','-e',source]))
w=E.fresh(jax.random.PRNGKey(111));st=w.state;c=st.core
c=c._replace(pos=j.array(fixture['initial']['pos'],j.float32).reshape(2,2),crate=j.zeros_like(c.crate))
w=w._replace(state=st._replace(core=c))
actions=j.array([[[a[0],a[1]] for a in r['actions']] for r in fixture['rows']],j.int32)
@jax.jit
def simulate(world,actions):
 def body(w,data):
  a,t=data;w,_,done,metrics=E.step(w,a,jax.random.fold_in(jax.random.PRNGKey(111),t))
  st=w.state;c=st.core
  return w,dict(pos=c.pos,fuse=c.fuse,owner=c.owner,trapped=st.trapped,alive=c.alive,
   hp=c.hp,bombBlast=c.bomb_blast,bombsCap=c.bombs_cap,blastCap=c.blast_cap,spdG=c.spd_g)
 return jax.lax.scan(body,world,(actions,j.arange(len(actions))))
_,actual=simulate(w,actions);actual=jax.device_get(actual)
for t,row in enumerate(fixture['rows']):
 for name,value in actual.items():
  expected=np.array(row['after'][name]).reshape(value[t].shape)
  if name in ('pos','spdG'):np.testing.assert_allclose(value[t],expected,atol=3e-4,rtol=0,err_msg=f'{name} tick {t+1}')
  else:np.testing.assert_array_equal(value[t],expected,err_msg=f'{name} tick {t+1}')
print(json.dumps(dict(status='PASS_NATIVE_JAX_TRAJECTORY',ticks=len(actions),fields=list(actual),
 randomized_item_placement='excluded after first death; separate rule tests cover drops/respawn')))
