# 实验中 / EXPERIMENTAL: opt-in only; activation requires frozen gates.
"""Independent tensorized kill arena adapter, using the audited native core.

This file is opt-in. It never installs into a running R3/R2 process.
"""
import inspect
import json
from pathlib import Path
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np

VERSION='kill-arena-sugar-respawn-v1'
CATALOG=Path('jax_bomb/maps/kill_maps.json')
_STEP=None
_TABLE=None

class Arena(NamedTuple):
    state:object
    kills:object
    deaths:object
    self_deaths:object
    life_age:object
    life_spawn:object
    seed:object
    map_id:object

def configure():
    global _STEP,_TABLE
    if _STEP is not None:return
    from jax_bomb import je07_bun_env_r1_items as E,levels
    E.prepare();E.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=0)
    E._DANGER_TRACKING=False
    rows=json.loads(CATALOG.read_text())['maps'];maps=[]
    for i,row in enumerate(rows):
        m=dict(row['map'],id=i)
        maps.append(m)
    path=Path('runs/r4_combat_20261031/kill_levels_runtime.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_text(json.dumps(maps,ensure_ascii=False))
    table=levels.set_active(str(path),weights='')
    _TABLE=table
    # Only this local function uses per-map initial stats instead of map806's.
    source=inspect.getsource(E.step)
    anchor='native_items.death_inventory(native_state,core,death,item_slots,item_counts,_BUN_INITIAL_STATS,base.GROWTH_SPEED_STEP)'
    assert source.count(anchor)==1
    source=source.replace(anchor,'native_items.death_inventory(native_state,core,death,item_slots,item_counts,levels.active().lo[jnp.maximum(core.level_id,0)],base.GROWTH_SPEED_STEP)')
    # Private adapter only: do not change the shared R3 native rule module.
    # Production updates lastMoveDir only when alive and able to move; forced
    # banana motion records the actual slide direction, not a rejected input.
    anchor='(requested_dirs < 4) & ~active_slide,\n                              requested_dirs, state.last_move_dir'
    assert source.count(anchor)==1
    source=source.replace(anchor,'(dirs < 4) & state.alive & (state.trapped == 0),\n                              dirs, state.last_move_dir')
    from types import ModuleType
    native_core=ModuleType('private_kill_arena_native_core')
    native_core.__dict__.update(E.base.__dict__)
    core_source=inspect.getsource(E.base.step)
    for old,new in [('jnp.ceil(y - RADIUS) - 1','jnp.floor(y - RADIUS - EPS * 8)'),
                    ('jnp.ceil(x - RADIUS) - 1','jnp.floor(x - RADIUS - EPS * 8)')]:
        assert core_source.count(old)==1
        core_source=core_source.replace(old,new)
    old='        push_t = push_t.at[pr, pc].set(jnp.where(do_move, 0.0, push_t[pr, pc]))'
    assert core_source.count(old)==1
    # Sim.step leaves the previous box-origin timer in the grid and resets
    # only its NEW origin; later explosions may clear the old cell normally.
    core_source=core_source.replace(old,'        # Preserve JS stale old-origin timer; reset new origin below.')
    old='    # 2.5 推箱子'
    assert core_source.count(old)==1
    core_source=core_source.replace(old,'    movement_blocked = (fuse > 0) | wall | brick\n\n'+old)
    old='    blocked = (fuse > 0) | wall | brick'
    assert core_source.count(old)==1
    # JS builds this once before its sequential player loop. Moving a box
    # changes brick immediately, but that tick still collides with old cells.
    core_source=core_source.replace(old,'    blocked = movement_blocked')
    core_scope=dict(E.base.step.__globals__)
    exec(compile(core_source,__file__,'exec'),core_scope)
    native_core.step=core_scope['step']
    scope=dict(E.step.__globals__,base=native_core)
    exec(compile(source,__file__,'exec'),scope);_STEP=scope['step']

def template(seed,map_id,initial):
    configure()
    from jax_bomb import je07_bun_env_r1_items as E,je07_native_items as I
    st=jax.tree.map(lambda x:x[0],E.init_batch(jax.random.PRNGKey(seed),1,jax.nn.one_hot(0,16)))
    d=initial;n=lambda name,dtype:j.array(d[name],dtype).reshape(13,15)
    lo=_TABLE.lo[map_id]
    c=st.core._replace(wall=n('wall',bool),brick=n('brick',bool),bush=n('bush',bool),
        pushable=n('pushable',bool),crate=n('crate',st.core.crate.dtype),fuse=n('fuse',st.fuse.dtype),
        owner=n('owner',st.core.owner.dtype),bomb_blast=n('bombBlast',st.core.bomb_blast.dtype),
        blast_linger=(n('horzLinger',st.core.blast_linger.dtype)*16+n('vertLinger',st.core.blast_linger.dtype)) if 'horzLinger' in d else n('blastLinger',st.core.blast_linger.dtype),pos=j.array(d['pos'],j.float32).reshape(2,2),
        alive=j.array(d['alive'],bool),hp=j.array(d['hp'],st.hp.dtype),invuln=j.array(d['invuln'],st.core.invuln.dtype),
        bombs_cap=j.array(d['bombsCap'],st.core.bombs_cap.dtype),blast_cap=j.array(d['blastCap'],st.core.blast_cap.dtype),
        spd_g=j.array(d['spdG'],j.float32),level_id=j.array(map_id,st.core.level_id.dtype),t=j.array(0,st.t.dtype))
    if 'brickLinger' in d:
        c=c._replace(brick_linger=n('brickLinger',c.brick_linger.dtype),push_t=n('pushT',j.float32))
    st=st._replace(core=c,bun_stored=j.zeros_like(st.bun_stored),bun_loose=j.zeros_like(st.bun_loose),
        bun_carried=j.full_like(st.bun_carried,-1),bun_respawn=j.zeros_like(st.bun_respawn),
        scene=j.array(0,st.scene.dtype),lesson=j.array(E.LESSON_FULL,st.lesson.dtype),
        tactical_crate=j.zeros_like(st.tactical_crate),native_items=I.fresh(),
        trapped=j.zeros_like(st.trapped),item_slots=j.zeros_like(st.item_slots),item_counts=j.zeros_like(st.item_counts),
        held_item=j.zeros_like(st.held_item),
        slide_dir=j.array(d['slideDir'],st.slide_dir.dtype),
        last_move_dir=j.array(d['lastMoveDir'],st.last_move_dir.dtype),
        move_status=j.array(d['movementStatus'],st.move_status.dtype),
        status_ticks=j.array(d['movementStatusTicks'],st.status_ticks.dtype))
    return Arena(st,j.zeros(2,j.int32),j.zeros(2,j.int32),j.zeros(2,j.int32),j.zeros(2,j.int32),
        j.floor(c.pos).astype(j.int32),j.array(seed,j.uint32),j.array(map_id,j.int32))

def respawn(a):
    from jax_bomb import je07_bun_env_r1_items as E
    st=a.state;core=st.core;spawn=a.life_spawn;age=a.life_age
    danger=E.base._danger_map(st.fuse,st.wall,st.core.bomb_blast,st.brick)
    candidates=_TABLE.spawns[a.map_id];count=_TABLE.cnt[a.map_id]
    for p in range(2):
        q=1-p;enemy=j.floor(core.pos[q]).astype(j.int32);cells=j.maximum(candidates,0)
        at=(cells[:,0],cells[:,1]);valid=j.arange(len(candidates))<count
        valid&=~(core.wall[at]|core.brick[at]|core.pushable[at]|(core.fuse[at]>0)|(core.blast_linger[at]>0))
        valid&=(danger[at]<=0)&(~core.alive[q]|(j.abs(cells-enemy).sum(-1)>=3))
        start=((a.seed^j.uint32(st.t)^j.uint32(p)*j.uint32(0x9e3779b9))*j.uint32(0x85ebca6b))%count.astype(j.uint32)
        order=(j.arange(len(candidates))-start.astype(j.int32)+count)%count
        choice=j.argmin(j.where(valid,order,1<<25));ok=~core.alive[p]&valid.any()
        pos=candidates[choice];lo=_TABLE.lo[a.map_id]
        core=core._replace(pos=core.pos.at[p].set(j.where(ok,pos.astype(j.float32)+.5,core.pos[p])),
            alive=core.alive.at[p].set(core.alive[p]|ok),hp=core.hp.at[p].set(j.where(ok,1,core.hp[p])),
            invuln=core.invuln.at[p].set(j.where(ok,10,core.invuln[p])),
            bombs_cap=core.bombs_cap.at[p].set(j.where(ok,lo[0].astype(core.bombs_cap.dtype),core.bombs_cap[p])),
            blast_cap=core.blast_cap.at[p].set(j.where(ok,lo[1].astype(core.blast_cap.dtype),core.blast_cap[p])),
            spd_g=core.spd_g.at[p].set(j.where(ok,lo[2],core.spd_g[p])))
        spawn=spawn.at[p].set(j.where(ok,pos,spawn[p]));age=age.at[p].set(j.where(ok,0,age[p]))
        st=st._replace(trapped=st.trapped.at[p].set(j.where(ok,0,st.trapped[p])),
            move_status=st.move_status.at[p].set(j.where(ok,0,st.move_status[p])),
            status_ticks=st.status_ticks.at[p].set(j.where(ok,0,st.status_ticks[p])),
            held_item=st.held_item.at[p].set(j.where(ok,0,st.held_item[p])),
            item_slots=st.item_slots.at[p].set(j.where(ok,0,st.item_slots[p])),
            item_counts=st.item_counts.at[p].set(j.where(ok,0,st.item_counts[p])))
    return a._replace(state=st._replace(core=core),life_age=age,life_spawn=spawn)

def step(a,actions,key,items_disabled=True):
    configure();a=respawn(a);before=a.state
    age=a.life_age+before.alive.astype(j.int32)
    after,_,info=_STEP(before,actions,key,auto_reset=False,return_info=True,require_stock_window=False)
    death=info['death'];source=after.trapped_causal
    # Native Sim._killPlayer clears invulnerability; no dead-player30 ticks.
    after=after._replace(core=after.core._replace(invuln=j.where(death,0,after.core.invuln)))
    popper=info['native_trap_popper']
    source=j.where((popper>=0)[:,None],jax.nn.one_hot(j.maximum(popper,0),2,dtype=bool),source)
    enemy=source[j.arange(2),1-j.arange(2)]&death
    own=source[j.arange(2),j.arange(2)]&~enemy&death
    cells=j.floor(after.pos).astype(j.int32)
    protected=(before.core.invuln>0)|(age<15)|(j.abs(cells-a.life_spawn).sum(-1)<=2)
    after=after._replace(bun_respawn=j.zeros_like(after.bun_respawn))
    if items_disabled:
        from jax_bomb import je07_native_items as I
        after=after._replace(core=after.core._replace(crate=j.zeros_like(after.core.crate)),native_items=I.fresh())
    new=a._replace(state=after,kills=a.kills+enemy[::-1].astype(j.int32),
        deaths=a.deaths+death.astype(j.int32),self_deaths=a.self_deaths+own.astype(j.int32),life_age=age)
    done=after.t>=1200
    winner=j.where(new.kills[0]==new.kills[1],-1,j.where(new.kills[0]>new.kills[1],0,1))
    info=dict(info,arena_kill=enemy[::-1],arena_self=own,arena_protected=protected,
        arena_reward_kill=(enemy&~protected)[::-1],arena_reward_death=death&~protected,winner=j.where(done,winner,-2))
    return new,done,info
