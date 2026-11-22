"""Device-only map806 kill-mode adapter over the existing native JAX simulator.

No Node process, host callback or per-tick host transfer is used. Historical
simulator sources remain unchanged; the opt-in adapter captures their audited
native contact/chain fixes and supplies kill resets, drops and observations.
"""
from types import FunctionType,SimpleNamespace
from typing import NamedTuple
import inspect
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np

_READY=False

class World(NamedTuple):
    state:object
    kills:object
    respawn:object
    target:object
    protection:object
    batches:object
    counters:object
    last_placed:object


def prepare():
    global _READY,E,I,C,TEMPLATE,LOWS,CAPS,SPAWNS,PAIRS,VALID,COVER,STEP_CODE,STEP_GLOBALS
    if _READY:return
    from jax_bomb import je07_bun_env_r1_items as E,je07_native_items as I,levels
    from scripts import r4_kill_arena as audited
    audited.configure()
    # Retain the reviewed native core patches, bind to map806's own table.
    E.prepare();E.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=0)
    E._DANGER_TRACKING=False
    C=audited._STEP.__globals__['base']
    table=levels.active();C.step.__globals__['levels']=SimpleNamespace(active=lambda:table)
    STEP_GLOBALS=dict(audited._STEP.__globals__,levels=SimpleNamespace(active=lambda:table))
    source=inspect.getsource(audited._STEP) if False else inspect.getsource(E.step)
    # Reuse the existing private source transformations rather than changing
    # pinned files. The resulting function already lives in an isolated scope.
    STEP_CODE=audited._STEP.__code__
    # Field items respect newly spawned players' protection. Banana speed is
    # the live 480 px/s, while fast shoes retain their 268 px/s rule.
    anchor='trigger = core.alive[player] & (trapped[player]==0) & field_armed[row, column]'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+' & (_SPAWN_PROTECTION[player] <= 0)')
    # Apply the same audited source changes to this private function body.
    for kind in ('owner','causal','trigger'):
        prefix='' if kind=='owner' else kind+'_'
        for axis in (0,1):source=source.replace(f'blast_{kind}_linger[source, {axis}] > 0',f'current_{prefix}sources[source, {axis}]')
    source=source.replace('base._is_hit_by_explosion(','base._native_contact_any(')
    anchor='(requested_dirs < 4) & ~active_slide,\n                              requested_dirs, state.last_move_dir'
    assert source.count(anchor)==1
    source=source.replace(anchor,'(dirs < 4) & state.alive & (state.trapped == 0),\n                              dirs, state.last_move_dir')
    scope=dict(STEP_GLOBALS)
    exec(compile(source,__file__+':private-native-step','exec'),scope)
    STEP_CODE=scope['step'].__code__;STEP_GLOBALS=scope
    STEP_GLOBALS['_movement_scale']=movement_scale
    TEMPLATE=E._fresh(jax.random.PRNGKey(0),jax.nn.one_hot(0,16))
    level=next(x for x in json.loads(Path('web/assets/maps/levels.json').read_text()) if x.get('qqt_id')==806)
    LOWS=E._BUN_INITIAL_STATS;CAPS=table.caps[E._BUN_INDEX]
    SPAWNS=j.array(level['bun_spawns'][0],j.int32)
    wall=np.array(level['wall'],bool).reshape(13,15)
    COVER=j.array((np.array(level['cover'],bool)|np.array(level['overhead'],bool))&~np.array(level['bush'],bool)).reshape(13,15)
    pairs=[]
    for start in np.array(SPAWNS):
        enemy=np.array([start[0],14-start[1]])
        allowed=~wall&~np.array(COVER)
        rr,cc=np.indices((13,15))
        for pos in (start,enemy):allowed&=np.maximum(abs(rr-pos[0]),abs(cc-pos[1]))>1
        def distance(pos):
            d=np.full((13,15),-1,int);d[tuple(pos)]=0;queue=[tuple(pos)]
            for r,c in queue:
                for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    y,x=r+dy,c+dx
                    if 0<=y<13 and 0<=x<15 and not wall[y,x] and d[y,x]<0:
                        d[y,x]=d[r,c]+1;queue.append((y,x))
            return d
        a,b=distance(start),distance(enemy);eligible=[]
        for r,c in np.argwhere(allowed):
            m=14-c
            if c<m and allowed[r,m] and min(a[r,c],a[r,m])==min(b[r,c],b[r,m]) and min(a[r,c],a[r,m],b[r,c],b[r,m])>=0:
                eligible.append([r*15+c,r*15+m])
        if len(eligible)<12:raise RuntimeError('Insufficient symmetric opening pairs')
        pairs.append(eligible)
    width=max(map(len,pairs));pad=np.zeros((len(pairs),width,2),np.int32);valid=np.zeros((len(pairs),width),bool)
    for i,row in enumerate(pairs):pad[i,:len(row)]=row;valid[i,:len(row)]=True
    PAIRS=j.array(pad);VALID=j.array(valid)
    _READY=True


def movement_scale(st):
    absolute=j.where(st.move_status==2,480.,j.where(st.move_status==3,268.,j.where(st.move_status==1,72.,0.)))
    return j.where(absolute>0,absolute/(40*C.TICK_HZ*C.STEP*st.core.spd_g),1.)


def fresh(key):
    prepare();ks,ki=jax.random.split(key)
    choice=jax.random.randint(ks,(),0,len(SPAWNS));start=SPAWNS[choice]
    pos=j.stack([start,j.array([start[0],14-start[1]])]).astype(j.float32)+.5
    ids=j.argsort(j.where(VALID[choice],jax.random.uniform(ki,(PAIRS.shape[1],)),j.inf))[:12]
    cells=PAIRS[choice,ids].reshape(-1)
    codes=j.repeat(j.arange(12)%3+1,2).astype(j.int8)
    core=jax.tree.map(j.zeros_like,TEMPLATE.core)
    core=core._replace(wall=j.array(E._BUN_LEVEL['wall'],bool).reshape(13,15),
        owner=j.full_like(core.owner,-1),pos=pos,alive=j.ones(2,bool),hp=j.full_like(core.hp,C.MAX_HP),
        bombs_cap=j.full_like(core.bombs_cap,LOWS[0]),blast_cap=j.full_like(core.blast_cap,LOWS[1]),
        spd_g=j.full_like(core.spd_g,LOWS[2]),level_id=j.array(E._BUN_INDEX,core.level_id.dtype),
        crate=j.zeros_like(core.crate).reshape(-1).at[cells].set(codes).reshape(13,15),
        is_open=j.array(True),gametype=j.array(0,core.gametype.dtype))
    st=jax.tree.map(j.zeros_like,TEMPLATE)
    st=st._replace(core=core,bun_carried=j.full_like(st.bun_carried,-1),bun_spawn_pos=pos,
        field_owner=j.full_like(st.field_owner,-1),native_items=I.fresh(COVER),
        slide_dir=j.ones_like(st.slide_dir),last_move_dir=j.ones_like(st.last_move_dir))
    return World(st,j.zeros(2,j.int32),j.zeros(2,j.int16),j.full((2,2),-1,j.int32),
        j.zeros(2,j.int16),j.full((13,15),-1,j.int32),j.zeros(2,j.int32),j.full(2,-1000,j.int32))


def available(core,tactical,field,s):
    rr,cc=j.indices((13,15));radius=C.RADIUS;eps=C.EPS
    occupied=j.any(core.alive[:,None,None] & (core.pos[:,0,None,None]+radius>rr+eps)
        & (core.pos[:,0,None,None]-radius<rr+1-eps) & (core.pos[:,1,None,None]+radius>cc+eps)
        & (core.pos[:,1,None,None]-radius<cc+1-eps),axis=0)
    return (~core.wall&~core.brick&~core.pushable&~s.cover&(core.crate==0)&(tactical==0)
        &(field==0)&(core.fuse==0)&(core.blast_linger==0)&~occupied&(s.fall_type.reshape(13,15)<0))


def release_deaths(core,tactical,field,s,key):
    pending,size=s.pending,s.pending_size
    s=s._replace(pending=j.zeros_like(pending),pending_size=j.int32(0))
    def body(i,carry):
        core,tactical,s=carry;drop=pending[i]
        free=available(core,tactical,field,s).reshape(-1)
        random=jax.random.uniform(jax.random.fold_in(key,i),(195,))
        cell=j.argmin(j.where(free,random,j.inf));ok=free.any()
        core,tactical,s=I.spawn(core,tactical,s,cell,drop[0],drop[1].astype(bool),drop[2],ok)
        q,n,o=I.append(s.pending,s.pending_size,drop,~ok,s.overflow)
        return core,tactical,s._replace(pending=q,pending_size=n,overflow=o)
    return jax.lax.fori_loop(0,size,body,(core,tactical,s))


def death_inventory(s,core,dead,slots,counts,lows,speed_step):
    s,core,slots,counts=I.death_inventory(s,core,dead,slots,counts,lows,speed_step)
    for p in range(2):
        cell=j.floor(core.pos[p]).astype(j.int16)
        def add(i,s):
            q,n,o=I.append(s.pending,s.pending_size,j.array([i%3,0,1,cell[0],cell[1]],j.int16),dead[p],s.overflow)
            return s._replace(pending=q,pending_size=n,overflow=o)
        s=jax.lax.fori_loop(0,8,add,s)
    return s,core,slots,counts


def step(world,actions,key):
    prepare();before=world.state
    item_scope={k:getattr(I,k) for k in dir(I) if not k.startswith('__')}
    item_scope.update(death_inventory=death_inventory,
        release_deaths=lambda *args:release_deaths(*args,jax.random.fold_in(key,0xD0)))
    fn=FunctionType(STEP_CODE,dict(STEP_GLOBALS,native_items=SimpleNamespace(**item_scope),
        _SPAWN_PROTECTION=world.protection))
    # Disable the parent's bun respawn; kill-mode owns its countdown/reset.
    source=before._replace(bun_respawn=j.zeros_like(before.bun_respawn))
    after,_,info=fn(source,actions,key,auto_reset=False,return_info=True,require_stock_window=False)
    dead=info['death'];physical=info['damage_source'];causal=info['causal_damage_source'];rows=j.arange(2)
    killed=dead&physical[rows,1-rows]&(physical.sum(-1)==1)
    own=dead&physical[rows,rows]&(physical.sum(-1)==1)&~causal[rows,1-rows]
    kills=world.kills+killed[::-1].astype(j.int32)
    newly=info['native_trap_enter'];trap_credit=(newly&physical[rows,1-rows]&(physical.sum(-1)==1))[::-1]
    done=after.t>=1800;winner=j.where(kills[0]==kills[1],-1,j.where(kills[0]>kills[1],0,1))
    reward=5*killed[::-1].astype(j.float32)-5*dead-2*own+.25*trap_credit-.25*newly
    reward+=j.where(done&(winner>=0),j.where(rows==winner,2.,-2.),0.)
    respawn=j.where(dead,30,world.respawn).astype(j.int16)
    target=world.target;core=after.core._replace(invuln=j.where(dead,0,after.core.invuln))
    protection=j.maximum(world.protection-1,0)
    for p in range(2):
        safe=available(core,after.tactical_crate,after.field_item,after.native_items)
        danger=C._danger_map(core.fuse,core.wall,core.bomb_blast,core.brick)
        safe&=danger<=0
        old=j.clip(target[p],j.array([0,0]),j.array([12,14]));valid=(target[p]>=0).all()&safe[old[0],old[1]]&~dead[p]
        at=j.argmin(j.where(safe.reshape(-1),jax.random.uniform(jax.random.fold_in(key,p+123),(195,)),j.inf))
        selected=j.array([at//15,at%15]);selected=j.where(safe.any(),selected,j.array([-1,-1]))
        target=target.at[p].set(j.where(valid,target[p],selected))
        wait=~core.alive[p]&(respawn[p]>0);time=j.maximum(respawn[p]-wait.astype(j.int16),0)
        ready=wait&(time==0)&safe.any();time=j.where(wait&~safe.any(),j.maximum(time,1),time)
        respawn=respawn.at[p].set(time);protection=protection.at[p].set(j.where(ready,10,protection[p]))
        core=core._replace(pos=core.pos.at[p].set(j.where(ready,target[p].astype(j.float32)+.5,core.pos[p])),
            alive=core.alive.at[p].set(core.alive[p]|ready),hp=core.hp.at[p].set(j.where(ready,C.MAX_HP,core.hp[p])),
            invuln=core.invuln.at[p].set(j.where(ready,10,core.invuln[p])))
        after=after._replace(trapped=after.trapped.at[p].set(j.where(ready,0,after.trapped[p])))
    after=after._replace(core=core,bun_respawn=j.zeros_like(after.bun_respawn))
    added=(after.fuse>0)&~((before.fuse>0)&(before.core.owner==after.core.owner))
    counters=world.counters;last=world.last_placed;batches=world.batches
    for p in range(2):
        placed=info['bomb_placed'][p];counter=counters[p]+(placed&(before.t-last[p]>=8)).astype(j.int32)
        batches=j.where(added&(after.core.owner==p),counter,batches)
        counters=counters.at[p].set(counter);last=last.at[p].set(j.where(placed,before.t,last[p]))
    batches=j.where(after.fuse>0,batches,-1)
    metrics=j.stack([killed[::-1],dead,own,info['bomb_placed'],after.trapped>0,
        j.where(done,j.where(rows==winner,1.,j.where(winner<0,.5,0.)),0.)],-1).astype(j.float32)
    return World(after,kills,respawn,target,protection,batches,counters,last),reward,done,metrics


def inputs(worlds,seats):
    prepare();st=worlds.state;rows=j.arange(len(seats))
    danger=jax.vmap(lambda c:C._danger_map(c.fuse,c.wall,c.bomb_blast,c.brick))(st.core)
    obs=jax.vmap(lambda c,p,d:C.make_obs(c,p,d,channels=14))(st.core,seats,danger)
    obs=obs.at[:,6].set(j.broadcast_to((st.t/1800.)[:,None,None],obs[:,6].shape))
    clock=st.t%300;payload=st.native_items.grave_size+st.native_items.flight_size
    column=j.floor(15.-(18.5/30.)*(clock-270)+.5).astype(j.int32)
    heat=j.where(j.arange(15)[None,:]==column[:,None],1.,j.where(j.abs(j.arange(15)[None,:]-column[:,None])==1,.35,0.))
    heat*=j.where((clock>=270)&(clock<=298),j.minimum(payload/3.,1.),0.)[:,None]
    extra=j.zeros((len(seats),10,13,15),j.float32).at[:,0].set(j.broadcast_to(heat[:,None,:],(len(seats),13,15)))
    obs=j.concatenate([obs,extra],1)
    gv=jax.vmap(C.global_vec)(st.core,seats).at[:,0].set(st.t/1800.)
    from scripts.r3_items_observation import expand_inputs
    obs,gv=expand_inputs(obs,gv,st,seats,dict(bomb_batch=worlds.batches))
    gv=j.concatenate([gv,j.zeros((len(seats),1),j.float32)],-1)
    mm,bm=jax.vmap(E.legal_mask)(st)
    active=st.alive[rows,seats]&(st.trapped[rows,seats]==0)
    mm=j.where(active[:,None],mm[rows,seats],j.array([False,False,False,False,True]))
    bm=j.where(active[:,None],bm[rows,seats],j.array([True,False,False]))
    return obs,gv,mm,bm,active
