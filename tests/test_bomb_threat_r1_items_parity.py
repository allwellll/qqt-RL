"""Native-item per-tick fixtures versus unchanged stock-window JS.

These are diagnostic fixtures, never training data. Movement/control parity
outside the item subsystem is not claimed by these stationary fixtures.
"""
import json,subprocess
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import pytest
from jax_bomb import je07_bun_env_r1_items as E,je07_native_items as I
ROOT=Path(__file__).resolve().parents[1]


def fresh():
    E.prepare();E.configure_training('full=1',5,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=0)
    E._DANGER_TRACKING=False
    s=jax.tree.map(lambda x:x[0],E.init_batch(jax.random.PRNGKey(91),1,jax.nn.one_hot(0,16)))
    c=s.core;z=j.zeros_like(c.wall)
    c=c._replace(wall=z,brick=z,pushable=z,bush=z,crate=j.zeros_like(c.crate),fuse=j.zeros_like(c.fuse),owner=j.full_like(c.owner,-1),
        blast_linger=j.zeros_like(c.blast_linger),pos=j.array([[4.5,4.5],[10.5,10.5]]),invuln=j.zeros_like(c.invuln))
    return s._replace(core=c,tactical_crate=j.zeros_like(s.tactical_crate),native_items=I.fresh())


def oracle(events):
    code=r'''
const Q=require('./web/sim-stock-window-v1'),level=require('./web/assets/maps/levels.json').find(x=>x.qqt_id===806);
const s=new Q.Sim(17);s.reset(level,{nativeTrap:true,nativeItems:true});
for(const k of ['wall','brick','bush','cover','crate','fuse','pushable','fieldItem','invuln','spawnProtection'])s[k].fill(0);
s.owner.fill(-1);s.pos.set([4.5,4.5,10.5,10.5]);s.itemSlots=[[],[]];
const events=JSON.parse(require('fs').readFileSync(0,'utf8')),out=[];
for(const e of events){
 if(e.inventory){s.itemSlots[0]=[{item:1,count:2},{item:2,count:6}];s._syncHeldItem(0);}
 if(e.death)s.trapped[0]=1;
 if(e.code){s.crate[64]=1;s.crateType[64]=(e.code-1)%3;s.superCrate[64]=Number(e.code>=4);s.crateCount[64]=e.qty||1;}
 s.step([[4,0,0],[4,0,0]]);
 const drops=[];for(let i=0;i<195;i++)if(s.crate[i])drops.push([i,s.crateType[i],s.crateCount[i]||1]);
 out.push({stats:[s.bombsCap[0],s.blastCap[0],s.spdG[0]],drops,grave:s.graveyard.map(x=>[x.type,Number(x.isSuper),x.count||1]),alive:s.alive[0]});
}
console.log(JSON.stringify(out));'''
    p=subprocess.run(['node','-e',code],input=json.dumps(events),text=True,capture_output=True,check=True,cwd=ROOT)
    return json.loads(p.stdout)


@pytest.mark.parametrize('events',[
    [{'code':3}]*8,
    [{'code':4},{'code':4},{'code':4}],
    [{'code':5},{'code':5},{'code':5}],
    [{'code':6},{'code':6},{'code':6}],
    [{'code':1},{'code':2},{'code':3},{'death':True},{},{}],
    [{'code':1,'qty':3},{'code':3,'qty':3},{'death':True},{}],
    [{'code':4},{'inventory':True,'death':True},{},{}],
])
def test_per_tick_growth_cap_overflow_death_typed_quantity(events):
    s=fresh();expected=oracle(events)
    fn=jax.jit(lambda st,k:E.step(st,j.array([[4,0],[4,0]]),k,auto_reset=False,return_info=True,require_stock_window=True))
    for tick,(e,want) in enumerate(zip(events,expected)):
        if e.get('inventory'):
            s=s._replace(item_slots=s.item_slots.at[0,:2].set(j.array([1,2],s.item_slots.dtype)),item_counts=s.item_counts.at[0,:2].set(j.array([2,6],s.item_counts.dtype)))
        if e.get('death'):s=s._replace(trapped=s.trapped.at[0].set(1))
        if e.get('code'):
            s=s._replace(core=s.core._replace(crate=s.core.crate.at[4,4].set(e['code'])),native_items=s.native_items._replace(counts=s.native_items.counts.at[4,4].set(e.get('qty',1))))
        s,done,info=fn(s,jax.random.PRNGKey(tick+100))
        np.testing.assert_allclose([s.core.bombs_cap[0],s.core.blast_cap[0],s.core.spd_g[0]],want['stats'],atol=3e-6,rtol=0)
        assert bool(s.alive[0])==want['alive']
        drops=[[i,int(t)+2 if t else (int(c)-1)%3,int(s.native_items.counts.reshape(-1)[i]) or 1] for i,(c,t) in enumerate(zip(np.asarray(s.core.crate).reshape(-1),np.asarray(s.tactical_crate).reshape(-1))) if c or t]
        assert drops==want['drops']
        grave=np.asarray(s.native_items.grave)[:int(s.native_items.grave_size)]
        assert [[int(t),int(sup),int(n) or 1] for t,sup,n in grave]==want['grave']
        assert int(s.native_items.overflow)==0
        assert info['native_growth_before'].shape==(2,3)


def test_legacy_modules_are_unchanged_and_version_is_explicit():
    from jax_bomb import je07_bun_env as old
    assert E is not old and E.BunState is not old.BunState
    assert old.base.GROWTH_SPEED_STEP==.15
    assert E.base.GROWTH_SPEED_STEP==pytest.approx(1.1/7)
    assert 'native_items' not in old.BunState._fields


def test_native_queue_full_serialization_next_step_parity():
    import pickle
    s=fresh();s=s._replace(core=s.core._replace(crate=s.core.crate.at[4,4].set(6)))
    fn=jax.jit(lambda st,k:E.step(st,j.array([[4,0],[4,0]]),k,auto_reset=False,return_info=True,require_stock_window=True))
    s,_,_=fn(s,jax.random.PRNGKey(333))
    s=s._replace(trapped=s.trapped.at[0].set(1));s,_,_=fn(s,jax.random.PRNGKey(334))
    restored=pickle.loads(pickle.dumps(jax.device_get(s)))
    a=fn(s,jax.random.PRNGKey(335));b=fn(restored,jax.random.PRNGKey(335))
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)


def test_new_item_state_fields_never_enter_actor_observation_or_global_vector():
    from jax_bomb import je07_bun_env as old
    s=fresh()
    # Compare encoders on identical physical fields; changed dynamics can of
    # course lead to different future observations, which is intentional.
    for seat in (0,1):
        np.testing.assert_array_equal(E.make_obs(s,seat),old.make_obs(s,seat))
        np.testing.assert_array_equal(E.global_vec(s,seat),old.global_vec(s,seat))
