"""Rule boundary checks against the unmodified production JS oracle."""
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from jax_bomb import je07_bun_env as env
from jax_bomb import je07_native_items as items
from test_campaign_side_parity_round2 import _configured_state

ROOT=Path(__file__).resolve().parents[1]


def js(mode):
    source=r"""
const Q=require('./web/sim.js'),level=require('./web/assets/maps/levels.json').find(x=>x.qqt_id===806);
const s=new Q.Sim(17);s.reset(level,{nativeTrap:true,nativeItems:true});
s.pos.set([4.5,4.5,10.5,10.5]);s.wall.fill(0);s.brick.fill(0);s.bush.fill(0);s.cover.fill(0);s.crate.fill(0);s.fuse.fill(0);s.fieldItem.fill(0);
const mode=process.argv[1];
if(mode==='death'){
 s.bombsCap[0]=s.loBombs[0]+2;s.blastCap[0]=s.loBlast[0]+1;s.spdG[0]=s.loSpeed[0]+2*s.speedStep;
 s.itemSlots[0]=[{item:2,count:6}];s._syncHeldItem(0);s._killPlayer(0);s._releaseDeathDrops();
 const drops=[];for(let i=0;i<Q.H*Q.W;i++)if(s.crate[i])drops.push([i,s.crateType[i],s.crateCount[i]]);
 console.log(JSON.stringify({stats:[s.bombsCap[0],s.blastCap[0],s.spdG[0]],drops,slots:s.itemSlots[0]}));
}else if(mode==='respawn'){
 s.alive[0]=false;s.hp[0]=0;s.bunRespawn[0]=1;s.step([[4,0],[4,0]]);
 console.log(JSON.stringify([s.alive[0],s.invuln[0],s.spawnProtection[0],s.bunRespawn[0]]));
}else if(mode==='air'){
 s.t=269;s.graveyard=[{type:1,isSuper:false,count:3}];s._nativeAirdropStep();
 s.t=270;s._nativeAirdropStep();const total=s.airdropTotal;
 // Keep native scheduling and landing intact; choose the same cell for the draw.
 s.rng=()=>0;
 const result=[];for(let t=271;t<=290;t++){s.t=t;s._nativeAirdropStep();for(let i=0;i<Q.H*Q.W;i++)if(s.crate[i])result.push([t,i,s.crateType[i],s.crateCount[i]]);}
 console.log(JSON.stringify({total,first:result[0]}));
}else if(mode==='dirs'){
 s.alive[0]=false;s.lastMoveDir[0]=3;s.step([[0,0],[4,0]]);console.log(JSON.stringify(s.lastMoveDir[0]));
}
"""
    p=subprocess.run(['node','-e',source,mode],cwd=ROOT,text=True,capture_output=True,check=True)
    return json.loads(p.stdout)


def clear_core(state):
    c=state.core
    return c._replace(pos=jnp.array([[4.5,4.5],[10.5,10.5]],jnp.float32),
        wall=jnp.zeros_like(c.wall),brick=jnp.zeros_like(c.brick),bush=jnp.zeros_like(c.bush),pushable=jnp.zeros_like(c.pushable),
        crate=jnp.zeros_like(c.crate),rec_crate=jnp.zeros_like(c.rec_crate),fuse=jnp.zeros_like(c.fuse),blast_linger=jnp.zeros_like(c.blast_linger))


def test_death_preserves_typed_quantities_and_nearest_safe_cells():
    st=_configured_state();core=clear_core(st);low=env._BUN_INITIAL_STATS;step=env.base._growth_speed_step(core.level_id)
    core=core._replace(bombs_cap=core.bombs_cap.at[0].set(low[0]+2),blast_cap=core.blast_cap.at[0].set(low[1]+1),spd_g=core.spd_g.at[0].set(low[2]+2*step),alive=jnp.array([False,True]))
    slots=st.item_slots.at[0,0].set(2);counts=st.item_counts.at[0,0].set(6)
    s,core,slots,counts=items.death_inventory(items.fresh(),core,jnp.array([True,False]),slots,counts,low,step)
    core,tactical,s=jax.jit(items.release_deaths)(core,jnp.zeros((13,15),jnp.int8),jnp.zeros((13,15),jnp.int8),s)
    drops=[]
    for i in range(items.N):
        code=int(core.crate.reshape(-1)[i]);item=int(tactical.reshape(-1)[i])
        if code or item:drops.append([i,item+2 if item else (code-1)%3,int(s.counts.reshape(-1)[i])])
    expected=js('death')
    assert drops==expected['drops']
    np.testing.assert_allclose([core.bombs_cap[0],core.blast_cap[0],core.spd_g[0]],expected['stats'],atol=2e-5)
    assert not np.asarray(slots[0]).any() and not np.asarray(counts[0]).any()
    assert int(s.pending_size)==0 and int(s.overflow)==0


def test_native_respawn_has_thirty_ticks_protection():
    st=_configured_state();st=st._replace(bun_respawn=jnp.array([1,0],jnp.int16),core=st.core._replace(alive=jnp.array([False,True]),hp=jnp.array([0,5])))
    after,_,_=env.step(st,jnp.array([[4,0],[4,0]]),jax.random.PRNGKey(21),auto_reset=False,return_info=True)
    actual=[bool(after.alive[0]),int(after.core.invuln[0]),int(after.spawn_protection[0]),int(after.bun_respawn[0])]
    assert actual==js('respawn')==[True,30,30,0]


def test_native_airdrop_launch_schedule_and_three_tick_landing():
    st=_configured_state();core=clear_core(st);s=items.fresh()
    s=items.bury(s,jnp.array([1]),jnp.array([False]),jnp.array([3]),jnp.array([True]))
    tactical=jnp.zeros((13,15),jnp.int8);field=jnp.zeros_like(tactical)
    step=jax.jit(items.airdrop);first=None;launch=None
    for t in range(270,291):
        core=core._replace(t=jnp.int32(t));core,tactical,s,cells=step(core,tactical,field,s,jax.random.PRNGKey(400+t))
        if np.asarray(cells>=0).any():launch=(t,int(cells[0]))
        if np.asarray(core.crate>0).any() and first is None:
            cell=int(jnp.argmax(core.crate.reshape(-1)));first=[t,cell,int(core.crate.reshape(-1)[cell])-1,int(s.counts.reshape(-1)[cell])]
    expected=js('air')
    assert expected['total']==1
    assert first[0]==expected['first'][0]==launch[0]+3
    assert first[1]==launch[1] and first[2:]==expected['first'][2:]


def test_dead_player_does_not_change_last_move_direction():
    st=_configured_state();st=st._replace(core=st.core._replace(alive=jnp.array([False,True])))
    after,_,_=env.step(st,jnp.array([[0,0],[4,0]]),jax.random.PRNGKey(22),auto_reset=False,return_info=True)
    assert int(after.last_move_dir[0])==js('dirs')==3


def test_queue_overflow_is_observable():
    s=items.fresh();q,n,o=items.append(s.grave,jnp.int32(items.QUEUE_CAPACITY),jnp.array([0,0,1],jnp.int16),jnp.bool_(True),s.overflow)
    assert int(n)==items.QUEUE_CAPACITY and int(o)==1
    assert not np.asarray(q).any()


def test_bomb_placement_event_survives_same_tick_chain_trigger():
    st=_configured_state();core=clear_core(st)
    core=core._replace(pos=jnp.array([[4.5,4.5],[10.5,10.5]]),fuse=core.fuse.at[4,5].set(1),owner=core.owner.at[4,5].set(1),bomb_blast=core.bomb_blast.at[4,5].set(2))
    st=st._replace(core=core)
    after,_,info=env.step(st,jnp.array([[4,1],[4,0]]),jax.random.PRNGKey(31),auto_reset=False,return_info=True)
    assert bool(info['bomb_placed'][0])
    assert int(after.core.fuse[4,4])==0
    assert bool(info['triggered_early'][4,4])
    source=r"""
const Q=require('./web/sim.js'),level=require('./web/assets/maps/levels.json').find(x=>x.qqt_id===806);
const s=new Q.Sim(17);s.reset(level,{nativeTrap:true,nativeItems:true});
s.wall.fill(0);s.brick.fill(0);s.pos.set([4.5,4.5,10.5,10.5]);s.fuse.fill(0);s.owner.fill(-1);
s.fuse[4*Q.W+5]=1;s.owner[4*Q.W+5]=1;s.bombBlast[4*Q.W+5]=2;
const result=s.step([[4,1],[4,0]]);console.log(JSON.stringify([result.placed[0],s.fuse[4*Q.W+4]]));
"""
    actual=subprocess.run(['node','-e',source],cwd=ROOT,text=True,capture_output=True,check=True)
    assert json.loads(actual.stdout)==[True,0]


def test_unknown_outcome_injection_and_refreshed_brick_expiry(tmp_path):
    # A production oracle module with callbacks injected into source only in RAM.
    source=r"""
const fs=require('node:fs'),Module=require('node:module'),path=require('node:path');
const file=path.resolve('web/sim.js');let text=fs.readFileSync(file,'utf8');
const needle='if (this.pendingCrateType[i] >= 0) {';
if(text.split(needle).length!==2)throw Error('signature');
text=text.replace(needle,'if(this.__expiry){this.pendingCrateType[i]=this.__expiry(i,this.t);}\n          '+needle);
const m=new Module(file,module);m.filename=file;m.paths=module.paths;m._compile(text,file);
const Q=m.exports,level=require('./web/assets/maps/levels.json').find(x=>x.qqt_id===806);
const s=new Q.Sim(17);s.reset(level,{nativeTrap:true,nativeItems:true});
s.wall.fill(0);s.brick.fill(0);s.crate.fill(0);s.fuse.fill(0);s.owner.fill(-1);
const cell=4*Q.W+4;s.brick[cell]=1;s.brickLinger[cell]=1;s.pendingCrateType[cell]=1;s.__expiry=()=>2;
s.step([[4,0],[4,0]]);console.log(JSON.stringify([s.crate[cell],s.crateType[cell]]));
"""
    actual=subprocess.run(['node','-e',source],cwd=ROOT,text=True,capture_output=True,check=True)
    assert json.loads(actual.stdout)==[1,2]


def test_default_off_test_hooks_preserve_production_snapshots():
    actual=subprocess.run(['node','runs/campaign_side_parity/round3/compare_event_parity.js','--validate-disabled'],cwd=ROOT,text=True,capture_output=True,check=True)
    assert json.loads(actual.stdout)['ticks']==2700


def test_slow_records_last_direction_and_slide_records_requested_direction():
    st=_configured_state();core=clear_core(st);cell=(4,4)
    # Slow stores lastMoveDir in slideDir even when the latter is stale.
    st=st._replace(core=core,last_move_dir=jnp.array([2,2],jnp.int8),slide_dir=jnp.array([3,2],jnp.int8),field_item=st.field_item.at[cell].set(2),field_owner=st.field_owner.at[cell].set(1),field_armed=st.field_armed.at[cell].set(True))
    after,_,_=env.step(st,jnp.array([[4,0],[4,0]]),jax.random.PRNGKey(43),auto_reset=False,return_info=True)
    assert int(after.slide_dir[0])==2
    # A new banana hit uses the requested steering direction even during a prior slide.
    st=st._replace(move_status=jnp.array([2,0],jnp.int8),slide_dir=jnp.array([1,2],jnp.int8),field_item=st.field_item.at[cell].set(1))
    after,_,_=env.step(st,jnp.array([[0,0],[4,0]]),jax.random.PRNGKey(44),auto_reset=False,return_info=True)
    assert int(after.last_move_dir[0])==1
    assert int(after.slide_dir[0])==0
    source=r"""
const Q=require('./web/sim.js'),level=require('./web/assets/maps/levels.json').find(x=>x.qqt_id===806);
const s=new Q.Sim(17);s.reset(level,{nativeTrap:true,nativeItems:true});
s.pos.set([4.5,4.5,10.5,10.5]);s.wall.fill(0);s.brick.fill(0);s.fuse.fill(0);s.fieldItem.fill(0);s.invuln.fill(0);s.spawnProtection.fill(0);
s.lastMoveDir[0]=2;s.slideDir[0]=3;s.fieldItem[64]=2;s.fieldOwner[64]=1;s.fieldArmed[64]=1;
s.step([[4,0],[4,0]]);const slow=s.slideDir[0];
s.pos.set([4.5,4.5,10.5,10.5]);s.movementStatus[0]=2;s.slideDir[0]=1;s.fieldItem[64]=1;s.fieldOwner[64]=1;s.fieldArmed[64]=1;
s.step([[0,0],[4,0]]);console.log(JSON.stringify([slow,s.lastMoveDir[0],s.slideDir[0]]));
"""
    actual=subprocess.run(['node','-e',source],cwd=ROOT,text=True,capture_output=True,check=True)
    assert json.loads(actual.stdout)==[2,1,0]


def test_respawn_tick_does_not_contact_field_item_at_spawn():
    st=_configured_state();spawn=st.bun_spawn_pos[0].astype(jnp.int32);r,c=map(int,spawn)
    st=st._replace(bun_respawn=jnp.array([1,0],jnp.int16),core=st.core._replace(alive=jnp.array([False,True]),hp=jnp.array([0,5])),field_item=st.field_item.at[r,c].set(2),field_owner=st.field_owner.at[r,c].set(0),field_armed=st.field_armed.at[r,c].set(True))
    after,_,_=env.step(st,jnp.array([[3,0],[4,0]]),jax.random.PRNGKey(46),auto_reset=False,return_info=True)
    assert int(after.field_item[r,c])==2 and int(after.move_status[0])==0
    source=r"""
const Q=require('./web/sim.js'),level=require('./web/assets/maps/levels.json').find(x=>x.qqt_id===806);
const s=new Q.Sim(17);s.reset(level,{nativeTrap:true,nativeItems:true});
s.alive[0]=false;s.hp[0]=0;s.bunRespawn[0]=1;const [r,c]=s.bunSpawnPos[0].map(Math.floor),i=r*Q.W+c;
s.fieldItem[i]=2;s.fieldOwner[i]=0;s.fieldArmed[i]=1;s.step([[3,0],[4,0]]);
console.log(JSON.stringify([s.fieldItem[i],s.movementStatus[0],s.invuln[0]]));
"""
    actual=subprocess.run(['node','-e',source],cwd=ROOT,text=True,capture_output=True,check=True)
    assert json.loads(actual.stdout)==[2,0,30]
