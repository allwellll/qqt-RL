"""Versioned R1 event settlement; historical A3S modules stay byte-stable.

The phase/danger gates, deferred settlement and cap are intentionally disclosed
as history-dependent shaping, not an unconditional policy-invariance claim.
"""
from pathlib import Path
from types import ModuleType
import jax
import jax.numpy as j
from scripts import bomb_threat_reward as base
from scripts import bomb_threat_r1_item_progress as items
GAMMA=.995
CAP=.30
# Reuse exact physical-seat ownership/topology evidence without old payment.
_src=Path('scripts/bomb_threat_seat5050_reward.py').read_text()
_anchor="    n,r,m=settle(e,s);n['brick_owned']=owned&after.brick\n    return n,r,m,birth"
assert _src.count(_anchor)==1
_impl=ModuleType('r1_physical_event_evidence')
exec(compile(_src.replace(_anchor,'    return e,birth,owned'),'scripts/bomb_threat_r1_reward.py','exec'),_impl.__dict__)


def state(n):
    return dict(base.state(n),chain_ids=j.full((n,3),-1,j.int32),chain_count=j.zeros(n,j.int32),
        multi_last=j.full(n,-1,j.int32),multi_count=j.zeros(n,j.int32),item_seen=j.zeros((n,3),bool),items=items.state(j.tile(j.asarray(items.INITIAL),(n,1))),
        phi=j.zeros(n),phi_valid=j.zeros(n,bool),r1_scale=j.ones(n),
        item_age=j.full(n,1000,j.int32),item_bombs=j.zeros(n,j.int32),item_self_harm=j.zeros(n,j.int32),
        topo_age=j.full(n,1000,j.int32),topo_origin=j.zeros((n,2),j.int32),topo_goal=j.zeros((n,2),j.int32),topo_seen_progress=j.zeros(n,bool),topo_removed=j.zeros((n,13,15),bool),topo_distance=j.full(n,1000,j.int32),topo_traversed=j.zeros(n,bool))


def weighted_field(wall,brick,push,goal):
    """Shortest traversable path; entering brick costs2, wall/push never open."""
    n=len(goal);rows=j.arange(n);blocked=wall|push
    d=j.full(wall.shape,1000,j.int32).at[rows,goal[:,0],goal[:,1]].set(0)
    # Reverse relaxation charges the neighbor entered on the forward route.
    cost=j.where(brick,2,1)
    def body(c):
        k,old,_=c;z=j.pad(j.minimum(old+cost,1000),((0,0),(1,1),(1,1)),constant_values=1000)
        nxt=j.minimum(old,j.minimum(j.minimum(z[:,:-2,1:-1],z[:,2:,1:-1]),j.minimum(z[:,1:-1,:-2],z[:,1:-1,2:])))
        nxt=j.where(blocked,1000,nxt)
        return k+1,nxt,j.any(nxt!=old)
    return jax.lax.while_loop(lambda c:(c[0]<195)&c[2],body,(0,d,j.array(True)))[1]


def weighted_distance(wall,brick,push,goal,start):
    d=weighted_field(wall,brick,push,goal)
    return d[j.arange(len(goal)),start[:,0],start[:,1]]


def potential(distance):return -.002*j.maximum(j.minimum(distance,1000),3)


def connected_births(before,after,seats,birth):
    """Old source rays actually reach the new bomb, own source only."""
    n=len(seats);rows=j.arange(n);pos=j.clip(j.floor(before.pos[rows,seats]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    rr,cc=j.indices((13,15));r,c=pos[:,0],pos[:,1]
    dr=j.abs(rr[None]-r[:,None,None]);dc=j.abs(cc[None]-c[:,None,None])
    solid=before.wall|before.brick|before.core.pushable
    rp=j.cumsum(solid.astype(j.int32),2);cp=j.cumsum(solid.astype(j.int32),1)
    hi=j.maximum(cc[None],c[:,None,None]);lo=j.minimum(cc[None],c[:,None,None])
    rh=j.take_along_axis(rp,j.maximum(hi-1,0),2)-j.take_along_axis(rp,lo,2)
    hi=j.maximum(rr[None],r[:,None,None]);lo=j.minimum(rr[None],r[:,None,None])
    ch=j.take_along_axis(cp,j.maximum(hi-1,0),1)-j.take_along_axis(cp,lo,1)
    aligned=((dr==0)&(rh<=0))|((dc==0)&(ch<=0))
    mask=(before.fuse>0)&(before.core.owner==seats[:,None,None])&(dr+dc>0)&aligned
    mask&=(dr+dc<=before.core.bomb_blast)&(before.fuse<after.fuse[rows,r,c][:,None,None])
    return j.where(mask,birth,-1).reshape(n,-1)


def multi_time_cover(st,seats,exit_cells=None,safe_exit_mask=None):
    """Effective chain-adjusted detonation times over enemy neighboring exits.

    Exact temporal source propagation uses the same core chain operator as
    bun_safety. Geometric rays exclude intervening walls/bricks/pushables.
    The caller also requires strict native safe-move reduction and own escape.
    """
    from scripts.carrier_mechanism_engine import env
    C=env.base
    n=len(seats);rows=j.arange(n);own=(st.fuse>0)&(st.core.owner==seats[:,None,None])
    bombed=st.fuse>0;blast=j.where(st.core.bomb_blast>0,st.core.bomb_blast,C.BLAST)
    max_time=float(C.FUSE+1)
    score=j.where(bombed,(max_time-st.fuse)/max_time,0.)
    effective=jax.vmap(lambda s,b,bd,w,br:C._stage_a_matrix(s,b.astype(j.float32),bd,w,br,C.MAX_CHAIN))(score,blast,bombed,st.wall,st.brick)
    times=j.rint(max_time*(1-effective)).astype(j.int32)
    enemy=j.clip(j.floor(st.pos[rows,1-seats]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    candidates=enemy[:,None,:]+j.array([[0,0],[-1,0],[1,0],[0,-1],[0,1]])[None] if exit_cells is None else exit_cells
    cells=j.clip(candidates,j.array([0,0]),j.array([12,14]));rr,cc=j.indices((13,15))
    if safe_exit_mask is None:safe_exit_mask=j.ones((n,5),bool)
    solid=st.wall|st.brick|st.core.pushable
    rp=j.cumsum(solid.astype(j.int32),2);cp=j.cumsum(solid.astype(j.int32),1)
    found=j.zeros(n,bool)
    for k in range(5):
        r,c=cells[:,k,0],cells[:,k,1]
        dr=j.abs(rr[None]-r[:,None,None]);dc=j.abs(cc[None]-c[:,None,None])
        hi=j.maximum(cc[None],c[:,None,None]);lo=j.minimum(cc[None],c[:,None,None])
        rh=j.take_along_axis(rp,j.maximum(hi-1,0),2)-j.take_along_axis(rp,lo,2)
        hi=j.maximum(rr[None],r[:,None,None]);lo=j.minimum(rr[None],r[:,None,None])
        ch=j.take_along_axis(cp,j.maximum(hi-1,0),1)-j.take_along_axis(cp,lo,1)
        ray=own&(((dr==0)&(rh<=0))|((dc==0)&(ch<=0)))&(dr+dc<=blast)
        legal=(candidates[:,k]==cells[:,k]).all(-1)&~solid[rows,r,c]&~bombed[rows,r,c]
        early=j.min(j.where(ray,times,10000),(1,2));late=j.max(j.where(ray,times,-10000),(1,2))
        found|=legal&safe_exit_mask[:,k]&(early<10000)&(late-early>=8)
    return found


_NATIVE_SAFETY=None


def native_safety():
    """Bind a PRIVATE safety predictor to current je07 movement/rule module.

    The historical safety module imports bun_env/jax_env; never mutate it in a
    running reference worker. Only new R1 qualification uses this local copy.
    No simulator physics or Actor input changes here.
    """
    global _NATIVE_SAFETY
    if _NATIVE_SAFETY is None:
        m=ModuleType('jax_bomb.r1_native_safety');m.__package__='jax_bomb'
        code=Path('jax_bomb/bun_safety.py').read_text()
        exec(compile(code,'jax_bomb/bun_safety.py','exec'),m.__dict__)
        from scripts.carrier_mechanism_engine import env
        m.env=env;m.base=env.base
        m._RELAX_STEPS=min(env.H*env.W,env.base.FUSE+2)
        _NATIVE_SAFETY=m
    return _NATIVE_SAFETY


def native_safe_exits(st,seats):
    """Current-rule opponent legal first-move escape verdicts and cells."""
    safety=native_safety()
    def one(s,seat):
        enemy=1-seat
        deadlines,fuse,owner=safety._source_deadlines(s.core,s.blast_owner_linger,None)
        deadline=deadlines.min(0);bombed=(owner>=0)&(fuse>0)
        passable=~s.wall&~s.brick&~bombed
        mm,_=safety.env.legal_mask(s)
        flags=[];cells=[]
        for move in range(5):
            pos=safety._candidate_position(s,enemy,move,0)
            flags.append(mm[enemy,move]&safety._can_escape(pos,passable,deadline,safety._movement_scale_for_player(s,enemy),0))
            cells.append(j.clip(j.floor(pos).astype(j.int32),j.array([0,0]),j.array([12,14])))
        return j.stack(flags),j.stack(cells)
    return jax.vmap(one)(st,seats)


def settle(e,s):
    n=len(s['gross']);rows=j.arange(n);valid=e['active']&~e['bad']&~e['terminal']
    threat=e['placed']&e['safe']&e['threat']&~s['enemy_seen'][rows,e['enemy_cell']]&valid
    ids=s['chain_ids'];count=s['chain_count'];candidates=e['old_ids']
    candidate_valid=(candidates>=0)&~(candidates[:,:,None]==ids[:,None,:]).any(-1)
    selected=j.zeros(n,j.int32)
    for _ in range(3):
        oldest=j.min(j.where(candidate_valid,candidates,1<<28),-1)
        ok=(oldest<(1<<28))&(count<3)&e['placed']&e['safe']&valid
        ids=ids.at[rows,j.minimum(count,2)].set(j.where(ok,oldest,ids[rows,j.minimum(count,2)]))
        count+=ok.astype(j.int32);selected+=ok.astype(j.int32)
        candidate_valid&=candidates!=oldest[:,None]
    bricks=e['owned_removed']&~s['brick_seen'];topology=e['improved']&bricks.any((1,2))&valid
    multi=e['placed']&e['safe']&e['multibatch']&(e['batch_id']>s['multi_last'])&(s['multi_count']<6)&valid
    pickups=e['pickup_types']&valid[:,None]
    phi_after=j.where(e['danger_after'],e['phi_before'],e['phi_after'])
    approach=j.where(valid&e['phase_before']&e['phase_after'],GAMMA*phi_after-e['phi_before'],0.)
    raw=.02*threat+.01*selected+.04*topology+.015*multi+approach
    amount=raw*s['r1_scale']
    pending=j.minimum(s['pending']+amount,j.maximum(0.,CAP-s['gross']))
    pending=j.where(valid,pending,0.);pay=j.where(e['settled']&valid,pending,0.)
    out=dict(s,pending=j.where(e['settled'],0.,pending),gross=s['gross']+j.maximum(pay,0),
        enemy_seen=s['enemy_seen'].at[rows,e['enemy_cell']].set(s['enemy_seen'][rows,e['enemy_cell']]|threat),
        chain_ids=ids,chain_count=count,brick_seen=s['brick_seen']|bricks,
        multi_last=j.where(multi,e['batch_id'],s['multi_last']),multi_count=s['multi_count']+multi.astype(j.int32),
        item_seen=s['item_seen']|pickups,phi=j.where(e['phase_after'],phi_after,0.),phi_valid=e['phase_after']&valid,
        brick_pending=j.where(valid,s['brick_pending']+topology.astype(j.int32),0))
    paid_bricks=j.where(e['settled']&valid,out['brick_pending'],0);out['brick_pending']=j.where(e['settled'],0,out['brick_pending'])
    return out,pay,dict(candidate=threat|multi|topology|pickups.any(-1)|(selected>0),settled_count=(pay!=0).astype(j.int32),pay=pay,
        threat_events=threat.astype(j.int32),chain_events=selected,brick_events=topology.astype(j.int32),brick_paid_events=paid_bricks,
        threat_pending=pending,approach=approach,r1_raw=raw,r1_multi=multi.astype(j.int32),r1_pickups=pickups.sum(-1),
        r1_void=j.where(~valid,s['pending'],0.),r1_pending=pending,r1_scale=s['r1_scale'])


def transition(before,after,info,aux,metadata,seats):
    e,birth,owned=_impl.transition(before,after,info,aux,metadata,seats)
    s=metadata['bomb_threat'];rows=j.arange(len(seats))
    def position(st,p):return j.clip(j.floor(st.pos[rows,p]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    start=position(before,seats);end=position(after,seats);goal=position(after,1-seats)
    blocked=after.wall|after.brick|after.core.pushable
    from scripts.carrier_mechanism_engine import env
    danger=jax.vmap(env.base._danger_map)(after.fuse,after.wall,after.core.bomb_blast,after.brick)
    danger=(danger>0)|(after.core.blast_linger>0)
    safe_after=~danger[rows,end[:,0],end[:,1]]
    olddanger=jax.vmap(env.base._danger_map)(before.fuse,before.wall,before.core.bomb_blast,before.brick)
    safe_before=~((olddanger>0)|(before.core.blast_linger>0))[rows,start[:,0],start[:,1]]
    phase=lambda st:st.alive[rows,seats]&st.alive[rows,1-seats]&(st.trapped[rows,seats]==0)&(st.trapped[rows,1-seats]==0)&(st.bun_carried[rows,seats]<0)
    da=weighted_distance(after.wall,after.brick,after.core.pushable,goal,end)
    db=weighted_distance(before.wall,before.brick,before.core.pushable,position(before,1-seats),start)
    phi_before=j.where(s['phi_valid'],s['phi'],potential(db))
    before_stats=j.stack([before.core.bombs_cap[rows,seats],before.core.blast_cap[rows,seats],before.core.spd_g[rows,seats]],-1)
    after_stats=j.stack([after.core.bombs_cap[rows,seats],after.core.blast_cap[rows,seats],after.core.spd_g[rows,seats]],-1)
    grew=after_stats>before_stats+1e-5
    # Genuine pickup must consume a native attribute crate as well as grow stat.
    crate_was=before.core.crate[rows,end[:,0],end[:,1]]
    crate_gone=after.core.crate[rows,end[:,0],end[:,1]]==0
    grew&=((crate_was>0)&crate_gone)[:,None]
    old_ids=connected_births(before,after,seats,metadata['birth'])
    strict_reduction=info['tactical_bomb_enemy_safe_moves_after'][rows,seats]<info['tactical_bomb_enemy_safe_moves_before'][rows,seats]
    def temporal(_):
        safe_exit,cells=native_safe_exits(before,seats)
        safety=native_safety()
        selected=jax.vmap(lambda st,ac:safety.analyze_selected_actions(st,ac).selected_survivable)(before,info['h_joint_actions'])
        own_safe=selected[rows,seats]
        return multi_time_cover(after,seats,cells,safe_exit)&own_safe
    multi=jax.lax.cond((e['placed']&strict_reduction).any(),temporal,lambda _:j.zeros(len(seats),bool),None)&strict_reduction
    e=dict(e,old_ids=old_ids,multibatch=multi,batch_id=before.t,pickup_types=grew,phi_before=phi_before,phi_after=potential(da),
        phase_before=phase(before)&(safe_before|s['phi_valid']),phase_after=phase(after),danger_after=~safe_after,
        terminal=e['terminal']|aux.get('r1_next_terminal',j.zeros(len(seats),bool)))
    n,r,m=settle(e,s);n['brick_owned']=owned&after.brick
    gb=info['native_growth_before'][rows,seats];ga=info['native_growth_after'][rows,seats]
    actual=(ga>gb+1e-6)&info['native_item_pickup'][rows,seats,None]
    ownself=info['own_bomb_defeat'][rows,seats]
    ip,ir,im=items.step(s['items'],gb,ga,actual,active=before.alive[rows,seats]&(before.trapped[rows,seats]==0)&~e['terminal'],
        self_death=ownself,terminal=e['terminal'],control_lost=~after.alive[rows,seats]|(after.trapped[rows,seats]>0))
    n['items']=ip
    terminal_first=~aux['terminal']&aux['r1_next_terminal']
    m.update(r1_attack_pay=r,r1_item_pay=ir,r1_item_candidate=im['candidate'],r1_item_void=im['void'],
        r1_item_pending=im['pending'],r1_item_progress=im['progress'],r1_item_actual_growth=actual.astype(j.int32),
        r1_item_terminal_stats=j.where(terminal_first[:,None],after_stats,0.),r1_item_terminal_count=terminal_first.astype(j.int32),
        r1_item_terminal_full=(terminal_first[:,None]&(items.fraction(after_stats)>=1.-1e-5)).astype(j.int32),
        r1_item_pickup_tick_sum=j.where(actual.any(-1),after.t,0),r1_item_pickup_count=actual.any(-1).astype(j.int32))
    r=r+ir;m['pay']=r

    # Paid topology event -> actual newly-opened cell traversal AND shorter
    # legal path to the event's frozen goal within200 ticks. A later event
    # censors an unfinished predecessor explicitly; no survivor selection.
    cages=j.asarray(env._BUN_BASES,j.int32)+1
    native=~after.alive[rows,1-seats]|(after.trapped[rows,1-seats]>0)
    eventgoal=j.where((after.bun_carried[rows,seats]>=0)[:,None],cages[seats],j.where(native[:,None],cages[1-seats],goal))
    topo=m['brick_events']>0;age=j.where(topo,0,s['topo_age']+1)
    origin=j.where(topo[:,None],start,s['topo_origin']);tgoal=j.where(topo[:,None],eventgoal,s['topo_goal'])
    removed=before.brick&~after.brick&owned
    tracked=j.where(topo[:,None,None],removed,s['topo_removed'])
    traversed=j.where(topo,False,s['topo_traversed'])|tracked[rows,end[:,0],end[:,1]]
    def measure(_):
        field=weighted_field(after.wall,after.brick,after.core.pushable,tgoal)
        return field[rows,start[:,0],start[:,1]],field[rows,end[:,0],end[:,1]]
    dist0,dist1=jax.lax.cond((age<=200).any(),measure,lambda _:(j.full(len(seats),1000),j.full(len(seats),1000)),None)
    olddist=j.where(topo,dist0,s['topo_distance'])
    progress=traversed&(dist1<olddist)&(age<=200)
    prior=j.where(topo,False,s['topo_seen_progress'])
    first_progress=progress&~prior
    m['r1_topology_censored']=(topo&(s['topo_age']<=200)&~s['topo_seen_progress']).astype(j.int32)
    m['r1_topology_failed200']=((age==201)&~s['topo_seen_progress']).astype(j.int32)
    n.update(topo_age=age,topo_origin=origin,topo_goal=tgoal,topo_seen_progress=prior|progress,
        topo_removed=tracked,topo_distance=olddist,topo_traversed=traversed)
    grew=actual
    ia=j.where(grew.any(-1),0,s['item_age']+1);ownself=info.get('own_bomb_defeat',j.zeros_like(info['death']))[rows,seats]
    n.update(item_age=ia,item_bombs=s['item_bombs']+((ia<=100)&e['placed']).astype(j.int32),item_self_harm=s['item_self_harm']+((ia<=100)&ownself).astype(j.int32))
    m.update(r1_empty_bombs=(e['placed']&~(e['threat']|(old_ids>=0).any(-1)|e['owned_removed'].any((1,2)))).astype(j.int32),
        r1_placed=e['placed'].astype(j.int32),r1_close=(j.abs(before.pos[rows,seats]-before.pos[rows,1-seats]).sum(-1)<=3).astype(j.int32),
        r1_postitem_bombs=((ia<=100)&e['placed']).astype(j.int32),r1_postitem_selfharm=((ia<=100)&ownself).astype(j.int32),
        r1_topology_progress=first_progress.astype(j.int32),r1_kills=info.get('causal_kill',j.zeros_like(info['death']))[rows,seats].astype(j.int32),r1_selfharm=ownself.astype(j.int32))
    return n,r,m,birth
