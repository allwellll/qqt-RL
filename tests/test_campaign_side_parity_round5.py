"""Native field phase ordering and actual consumption against production JS."""
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
s.owner.fill(-1);s.pos.set([4.5,c.pickup?4.9:4.5,c.sharedBomb?4.5:10.5,c.sharedBomb?4.5:10.5]);
s.itemSlots[0]=c.release?[{item:c.item,count:1}]:[];s._syncHeldItem(0);
s.lastMoveDir[0]=2;s.slideDir[0]=3;
if(c.existing){s.fieldItem[64]=c.item;s.fieldOwner[64]=c.release?0:1;s.fieldArmed[64]=c.release?0:1;}
if(c.blast||c.expiringFuse){const i=c.expiringFuse?64:66;s.fuse[i]=1;s.owner[i]=1;s.bombBlast[i]=2;}
if(c.pickup){s.crate[65]=1;s.crateType[65]=c.item+2;s.crateCount[65]=1;}
const out=s.step([[c.pickup?3:4,0,c.release?1:0],[4,c.sharedBomb?1:0,0]]);
console.log(JSON.stringify({pos:Array.from(s.pos),held:s.heldItem[0],counts:s.itemSlots[0].map(x=>x.count),
status:s.movementStatus[0],ticks:s.movementStatusTicks[0],dir:s.slideDir[0],trapped:s.trapped[0],
field:s.fieldItem[64],released:out.itemReleased[0]?c.item:0}));
"""
    proc = subprocess.run(['node', '-e', source], input=json.dumps(case), cwd=ROOT,
                          text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


@pytest.mark.parametrize('item', [1, 2])
@pytest.mark.parametrize('scenario', ['occupied_blast', 'contact_blast', 'release_blast',
                                      'release_pickup', 'expiring_fuse', 'shared_bomb'])
def test_native_field_phase_and_consumption(item, scenario):
    case = dict(item=item, release=scenario!='contact_blast',
                existing=scenario in ['occupied_blast','contact_blast'],
                blast=scenario in ['occupied_blast','contact_blast','release_blast'],
                pickup=scenario=='release_pickup', expiringFuse=scenario=='expiring_fuse',
                sharedBomb=scenario=='shared_bomb')
    st = _configured_state();core = clear_core(st)
    core = core._replace(pos=jnp.array([[4.5,4.9 if case['pickup'] else 4.5],
                        [4.5,4.5] if case['sharedBomb'] else [10.5,10.5]],jnp.float32),
                        owner=jnp.full_like(core.owner,-1),invuln=jnp.zeros_like(core.invuln))
    if case['blast'] or case['expiringFuse']:
        cell=(4,4 if case['expiringFuse'] else 6)
        core=core._replace(fuse=core.fuse.at[cell].set(1),owner=core.owner.at[cell].set(1),
                           bomb_blast=core.bomb_blast.at[cell].set(2))
    if case['pickup']:
        core=core._replace(crate=core.crate.at[4,5].set(0))
        # Native tactical stacks live outside core.crate.
        st=st._replace(tactical_crate=st.tactical_crate.at[4,5].set(item),
            native_items=st.native_items._replace(counts=st.native_items.counts.at[4,5].set(1)))
    st=st._replace(core=core,last_move_dir=jnp.array([2,2],jnp.int8),slide_dir=jnp.array([3,2],jnp.int8),
        item_slots=jnp.zeros_like(st.item_slots).at[0,0].set(item if case['release'] else 0),
        item_counts=jnp.zeros_like(st.item_counts).at[0,0].set(1 if case['release'] else 0),
        held_item=jnp.array([item if case['release'] else 0,0],jnp.int8))
    if case['existing']:
        st=st._replace(field_item=st.field_item.at[4,4].set(item),
            field_owner=st.field_owner.at[4,4].set(0 if case['release'] else 1),
            field_armed=st.field_armed.at[4,4].set(not case['release']))
    action=jnp.array([[3 if case['pickup'] else 4,2 if case['release'] else 0],
                      [4,1 if case['sharedBomb'] else 0]],jnp.int32)
    after,_,info=env.step(st,action,jax.random.PRNGKey(51),auto_reset=False,return_info=True)
    expected=oracle(case)
    np.testing.assert_allclose(np.asarray(after.pos).reshape(-1),expected['pos'],atol=2e-5,rtol=0)
    for name,value in dict(held=after.held_item[0],status=after.move_status[0],ticks=after.status_ticks[0],
                           dir=after.slide_dir[0],trapped=after.trapped[0],field=after.field_item[4,4],
                           released=info['native_item_released'][0]).items():
        assert int(value)==expected[name],(scenario,name,int(value),expected[name])
    assert [int(x) for x in after.item_counts[0] if x]==expected['counts']
    if scenario=='occupied_blast':
        assert int(info['native_item_released'][0])==0 and int(after.item_counts[0,0])==1
    if scenario=='contact_blast':
        assert int(after.move_status[0])==0 and int(after.slide_dir[0])==2
    if scenario=='release_blast':
        assert int(info['native_item_released'][0])==item and int(after.field_item[4,4])==0
    if scenario=='release_pickup':
        assert int(after.item_counts[0,0])==int(st.item_counts[0,0])==1
        assert int(info['native_item_released'][0])==item and bool(info['native_item_pickup'][0])
