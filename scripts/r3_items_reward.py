"""Independent R3 event ledger. No placement or inventory-tick payments.

All positive event credits pass a 30-tick quarantine; blast/control invalidates
the queue before maturity. Potential differences remain strict raw differences
before clipping/voiding and have no discounted policy-invariance claim.
"""
import jax
import jax.numpy as j

ATTR,ITEM_PATH,GLUE,BANANA,ROAD,CHAIN,PRESSURE,THREAT=range(8)
CAPS=j.array([.36,.60,.36,.20,.20,.30,.10,.45])
VERSION='r3-events-items-route-chain-v1'


def state(n):
    return dict(queue=j.zeros((n,30,8)),gross=j.zeros((n,8)),void=j.zeros((n,8)),
        bomb_batch=j.full((n,13,15),-1,j.int32),batch_counter=j.zeros((n,2),j.int32),
        last_placed=j.full((n,2),-1000,j.int32),road_best=j.full(n,1<<24,j.int32),
        road_ready=j.zeros(n,bool),road_seen=j.zeros((n,13,15),bool),road_owned=j.zeros((n,13,15),bool),
        last_explosion=j.full(n,-1000,j.int32),last_explosion_batch=j.full(n,-1,j.int32),pressure_count=j.zeros(n,j.int32),
        glue_age=j.full((n,6),-1,j.int32),glue_kill=j.zeros((n,6),bool),glue_pickup=j.zeros((n,6),bool),
        glue_bad=j.zeros((n,6),bool),glue_count=j.zeros(n,j.int32),
        slide_active=j.zeros(n,bool),slide_path=j.zeros(n),slide_ticks=j.zeros(n,j.int32),slide_start_home=j.zeros(n),
        slide_count=j.zeros(n,j.int32),slide_bad=j.zeros(n,bool),banana_risk_age=j.full((n,4),-1,j.int32),
        item_phi=j.zeros(n),item_phi_ready=j.zeros(n,bool),item_scale=j.ones(n))


def capped_payment(s,matured,*,attack_gross=0.):
    """Reserve gross monotone credit even after future invalidation.

    Joint item/road .60 and attack .45 include inherited attack payment.
    Stable type order is frozen, avoiding dependence on dict iteration order.
    Negative penalties never consume positive cap or create future allowance.
    """
    gross=s['gross'];out=j.zeros_like(matured)
    item_budget=j.maximum(0.,.60-gross[:,:5].sum(-1))
    attack_budget=j.maximum(0.,.45-gross[:,5:].sum(-1)-j.maximum(attack_gross,0.))
    for k in range(8):
        eligible=j.minimum(j.maximum(matured[:,k],0.),j.maximum(0.,CAPS[k]-gross[:,k]))
        budget=item_budget if k<5 else attack_budget
        paid=j.minimum(eligible,budget)
        out=out.at[:,k].set(paid+j.minimum(matured[:,k],0.))
        if k<5:item_budget-=paid
        else:attack_budget-=paid
    return dict(s,gross=gross+j.maximum(out,0.)),out


def settle(s,candidate,*,bad,terminal,active,attribute_matured=None,attack_gross=0.):
    # No new post-terminal events. Existing factual credit still receives its
    # full30-tick hazard quarantine in the collector's settlement tail.
    invalid=bad
    mature=j.where(invalid[:,None],0.,s['queue'][:,0])
    if attribute_matured is not None:mature=mature.at[:,ATTR].add(j.where(invalid,0.,attribute_matured))
    # New event after tick t is payable no earlier than tick t+30.
    queue=j.concatenate([s['queue'][:,1:],j.where(active[:,None],j.maximum(candidate,0.),0.)[:,None,:]],1)
    void=j.where(invalid[:,None],queue.sum(1)+s['queue'][:,0],0.)
    queue=j.where(invalid[:,None,None],0.,queue)
    matured=mature+j.where(active[:,None]&~terminal[:,None],j.minimum(candidate,0.),0.)
    out,pay=capped_payment(dict(s,queue=queue,void=s['void']+void),matured,attack_gross=attack_gross)
    return out,pay,dict(candidate=candidate,pay=pay,void=void,pending=queue.sum(1),gross=out['gross'])


def glue_reward(kill,pickup):return j.where(kill,.06,j.where(pickup,.04,.01))
def banana_reward(carry,saved_ticks,blast):return j.where(carry,j.where(blast,-.02,j.minimum(.002*j.maximum(saved_ticks,0.),.05)),0.)
def chain_reward(k,*,same_batch=False,bad=False,empty=False,home_stack=False):
    return j.where(bad|empty|home_stack|same_batch,0.,.015*j.minimum(j.maximum(k,0),4))


def road_credit(best,current,without_own,owned_removal,active):
    causal_best=j.minimum(best,without_own)
    improved=owned_removal&(current<causal_best)&(current<10000)&active
    return j.where(improved,.03*(causal_best-current),0.),j.minimum(best,current),improved


def glue_outcomes(age,kills,pickups,invalid,newly,kill_event,pickup_event):
    # One factual kill/pickup cannot upgrade six overlapping glue windows.
    # Attribute it once to the most recent qualifying prior contact.
    eligible=(age>=0)&~invalid&~newly
    chosen=j.argmin(j.where(eligible,age,10000),-1)
    selected=jax.nn.one_hot(chosen,6,dtype=bool)&eligible
    return kills|(selected&kill_event[:,None]),pickups|(selected&pickup_event[:,None])


def batch_update(s,before,after,info):
    n=len(before.t);rows=j.arange(n);batch=s['batch_counter'];last=s['last_placed'];ids=s['bomb_batch']
    added=(after.fuse>0)&~((before.fuse>0)&(before.core.owner==after.core.owner))
    for p in (0,1):
        placed=info['bomb_placed'][:,p];new=batch[:,p]+(placed&(before.t-last[:,p]>=8)).astype(j.int32)
        ids=j.where(added&(after.core.owner==p),new[:,None,None],ids)
        batch=batch.at[:,p].set(new);last=last.at[:,p].set(j.where(placed,before.t,last[:,p]))
    return dict(s,bomb_batch=j.where(after.fuse>0,ids,-1),batch_counter=batch,last_placed=last)


def shortest_fields(st,goals,brick_cost=0):
    from scripts.je08_bun_potential import shortest
    # Existing audited BFS is faster for fully blocked bricks.
    if brick_cost==0:return shortest((st.wall|st.brick|st.core.pushable)[:,None],j.zeros_like(st.brick[:,None]),goals)
    blocked=st.wall|st.core.pushable;n=len(st.t);cost=1+brick_cost*st.brick.astype(j.int32)
    d=j.where(goals&~blocked[:,None],0,10000)
    def body(c):
        k,old,_=c
        z=j.pad(j.minimum(old+cost[:,None],10000),((0,0),(0,0),(1,1),(1,1)),constant_values=10000)
        new=j.minimum(old,j.minimum(j.minimum(z[:,:,:-2,1:-1],z[:,:,2:,1:-1]),j.minimum(z[:,:,1:-1,:-2],z[:,:,1:-1,2:])))
        new=j.where(blocked[:,None],10000,new)
        return k+1,new,j.any(new!=old)
    return jax.lax.while_loop(lambda c:(c[0]<195)&c[2],body,(0,d,j.array(True)))[1]


def road_distance(st):
    from scripts.carrier_mechanism_engine import env
    cages=j.stack([env._base_mask(0),env._base_mask(1)])
    goals=j.broadcast_to(cages,(len(st.t),2,13,15))
    fields=shortest_fields(st,goals,8)
    return j.min(j.where(cages[0][None],fields[:,1],10000),(-2,-1))


def cross_batch_count(before,after,info,s,seats,source_valid=None):
    """Actual native ray edges among resolved own bombs, not placement links."""
    from scripts.carrier_mechanism_engine import env
    C=env.base;n=len(seats);rows=j.arange(n);K=C.MAX_BOMBS*2
    live=before.fuse>1;exists=before.fuse>0
    _,order=jax.lax.top_k(exists.reshape(n,-1).astype(j.int32),K)
    at=lambda x:j.take_along_axis(x.reshape(n,-1),order,1)
    r,c=order//15,order%15
    up,down,left,right=jax.vmap(C._reach_dp)(before.wall|before.brick|live)
    u,d,l,ri=[at(v) for v in (up,down,left,right)]
    dist=j.abs(r[:,:,None]-r[:,None,:])+j.abs(c[:,:,None]-c[:,None,:])
    aligned=(r[:,:,None]==r[:,None,:])|(c[:,:,None]==c[:,None,:])
    reach=aligned&(dist>0)&(dist<=at(before.core.bomb_blast)[:,:,None])&(
        ((r[:,:,None]<r[:,None,:])&(u[:,None,:]>=dist-1))|
        ((r[:,:,None]>r[:,None,:])&(d[:,None,:]>=dist-1))|
        ((c[:,:,None]<c[:,None,:])&(l[:,None,:]>=dist-1))|
        ((c[:,:,None]>c[:,None,:])&(ri[:,None,:]>=dist-1)))
    own=at(exists)&(at(before.core.owner)==seats[:,None]);resolved=own&~(at(after.fuse)>0)
    if source_valid is not None:resolved&=at(source_valid)
    ids=at(s['bomb_batch'])
    eligible_target=at(info['triggered_early'])&(at(info['trigger_actor'])==seats[:,None])&resolved
    linked=(reach&resolved[:,:,None]&eligible_target[:,None,:]&(ids[:,:,None]!=ids[:,None,:])).any(1)
    seen=j.full((n,10),-1,j.int32);count=j.zeros(n,j.int32)
    for i in range(10):
        valid=linked&(ids>=0)&~(ids[:,:,None]==seen[:,None,:]).any(-1)
        selected=j.min(j.where(valid,ids,1<<27),-1);ok=selected<(1<<27)
        seen=seen.at[:,i].set(j.where(ok,selected,-1));count+=ok.astype(j.int32)
    return count


def novel_explosion_cover(before,after,seats,source_valid):
    """Each exploding source's coverage absent from every other live own ray."""
    from scripts.carrier_mechanism_engine import env
    C=env.base;K=C.MAX_BOMBS*2
    def one(st,next_st,seat,source_ok):
        own=(st.fuse>0)&(st.core.owner==seat)
        _,indices=jax.lax.top_k(own.reshape(-1).astype(j.int32),K)
        mask=own.reshape(-1)[indices]
        def ray(at):
            seed=j.zeros((13,15),bool).at[at//15,at%15].set(True)
            return C._rays(seed,st.fuse>1,st.core.bomb_blast,st.wall,st.brick)
        rays=jax.vmap(ray)(indices)&mask[:,None,None]
        count=rays.sum(0);resolved=(next_st.fuse.reshape(-1)[indices]<=0)&mask&source_ok.reshape(-1)[indices]
        return (rays&resolved[:,None,None]&(count[None]==1)).any(0)
    return jax.vmap(one)(before,after,seats,source_valid)


def item_potential(st,seats):
    from scripts.carrier_mechanism_engine import env
    rows=j.arange(len(st.t));cells=j.clip(j.floor(st.pos[rows,seats]).astype(j.int32),j.array([0,0]),j.array([12,14]))
    # A true pickup is visible on the map; a hidden graveyard/airdrop future is
    # never used by Actor or this target. Restrict to reachable within3 steps.
    goals=((st.core.crate>0)|(st.tactical_crate>0))[:,None]
    d=shortest_fields(st,goals)[:,0][rows,cells[:,0],cells[:,1]]
    danger=jax.vmap(env.base._danger_map)(st.fuse,st.wall,st.core.bomb_blast,st.brick)
    safe=(danger[rows,cells[:,0],cells[:,1]]<=0)&(st.core.blast_linger[rows,cells[:,0],cells[:,1]]<=0)&st.alive[rows,seats]&(st.trapped[rows,seats]==0)
    return j.where(safe&(d<=3),-.05*d/20.,0.)


def transition(before,after,info,aux,metadata,seats):
    """Derive factual events from native info and persistent physical batches."""
    s=metadata['bomb_threat']['r3'];n=len(seats);rows=j.arange(n);enemy=1-seats
    active=~aux['terminal'];terminal=aux['r1_next_terminal']
    bad=info['death'][rows,seats]|(after.trapped[rows,seats]>0)|~after.alive[rows,seats]
    candidate=j.zeros((n,8));metrics={}
    from scripts.carrier_mechanism_engine import env
    cages=j.stack([env._base_mask(0),env._base_mask(1)])
    cells=j.clip(j.floor(after.pos).astype(j.int32),j.array([0,0]),j.array([12,14]));oldcells=j.clip(j.floor(before.pos).astype(j.int32),j.array([0,0]),j.array([12,14]))
    homefields=shortest_fields(after,j.broadcast_to(cages,(n,2,13,15)))
    enemycell=cells[rows,enemy]
    enemy_home=homefields[rows,seats,enemycell[:,0],enemycell[:,1]]
    resolved_own=(before.fuse>0)&(before.core.owner==seats[:,None,None])&(after.fuse<=0)
    source_valid=resolved_own&((homefields[rows,seats]>3)|(enemy_home[:,None,None]<=3))
    legal_cover=jax.vmap(lambda seed,st:env.base._rays(seed,st.fuse>1,st.core.bomb_blast,st.wall,st.brick))(source_valid,before)
    # First query at each genuine reset initializes both potentials; no Bank
    # snapshot/reset credit. Same strict state difference thereafter.
    phi0=item_potential(before,seats);phi1=item_potential(after,seats)
    candidate=candidate.at[:,ITEM_PATH].set(j.where(active&~terminal,phi1-phi0,0.))
    owned=(info['blast_owner_h'][rows,seats]|info['blast_owner_v'][rows,seats])&~(info['blast_owner_h'][rows,enemy]|info['blast_owner_v'][rows,enemy])&legal_cover
    # Old threat ledger retains ownership until actual delayed brick removal.
    anycover=info['blast_owner_h'].any(1)|info['blast_owner_v'].any(1)
    brick_owned=j.where(anycover,owned,s['road_owned'])&before.brick
    removed=before.brick&~after.brick&brick_owned&~s['road_seen']
    # Topology is recomputed only on real brick removal. No inventory/tick pay.
    def road(_):return road_distance(before),road_distance(after)
    topology_changed=j.any(before.brick!=after.brick,(-2,-1))
    oldd,newd=jax.lax.cond(topology_changed.any()|~s['road_ready'].all(),road,lambda _:(s['road_best'],s['road_best']),None)
    best=j.where(s['road_ready'],s['road_best'],oldd)
    counter=jax.lax.cond(removed.any(),lambda _:road_distance(after._replace(core=after.core._replace(brick=after.brick|removed))),lambda _:newd,None)
    road_pay,next_best,improve=road_credit(best,newd,counter,removed.any((-2,-1)),active)
    candidate=candidate.at[:,ROAD].set(road_pay)
    s=dict(s,road_best=next_best,road_ready=j.ones(n,bool),road_seen=s['road_seen']|removed,road_owned=brick_owned&after.brick)
    # Critical glue requires a genuine native trigger with exact consumed type.
    own_glue=(info['r3_contact_type'][rows,seats]==2)&(info['trap_owner'][rows,seats]==seats)
    enemy_glue=(info['r3_contact_type'][rows,enemy]==2)&(info['trap_owner'][rows,enemy]==seats)
    targets=j.broadcast_to(cages,(n,2,13,15))[rows,seats]|(after.bun_loose.sum(-1)>0)
    targetdist=shortest_fields(after,targets[:,None])[:,0]
    ec=cells[rows,enemy];oc=oldcells[rows,enemy]
    toward=targetdist[rows,ec[:,0],ec[:,1]]<targetdist[rows,oc[:,0],oc[:,1]]
    owncell=cells[rows,seats];goals=j.zeros((n,1,13,15),bool).at[rows,0,owncell[:,0],owncell[:,1]].set(True)
    chasefield=shortest_fields(after,goals)[:,0];distance=chasefield[rows,ec[:,0],ec[:,1]]
    chase=(distance<=4)&(distance<chasefield[rows,oc[:,0],oc[:,1]])
    critical=enemy_glue&((before.bun_carried[rows,enemy]>=0)|toward|chase)&(s['glue_count']<6)&active
    slot=j.minimum(s['glue_count'],5)
    age=j.where(s['glue_age']>=0,s['glue_age']+1,-1).at[rows,slot].set(j.where(critical,0,s['glue_age'][rows,slot]))
    newly=jax.nn.one_hot(slot,6,dtype=bool)&critical[:,None]
    invalid=(s['glue_bad']|((age>=0)&bad[:,None]))&~newly
    kills,pickups=glue_outcomes(age,s['glue_kill']&~newly,s['glue_pickup']&~newly,invalid,newly,
        info['causal_kill'][rows,seats],info['steal'][rows,seats]|info['pickup_loose'][rows,seats])
    due=(age>=50)|(terminal[:,None]&(age>=0))
    candidate=candidate.at[:,GLUE].set((j.where(due&~invalid,glue_reward(kills,pickups),0.)).sum(-1)-.02*own_glue)
    s=dict(s,glue_age=j.where(due,-1,age),glue_kill=kills,glue_pickup=pickups,glue_bad=invalid,glue_count=s['glue_count']+critical.astype(j.int32))
    # Banana path is fully measured before credit. No distance reward here;
    # only speed difference on the same completed home-progressing polyline.
    trigger=(info['r3_contact_type'][rows,seats]==1)&(info['trap_owner'][rows,seats]==seats)&(before.bun_carried[rows,seats]>=0)&(s['slide_count']<4)&active
    home=homefields[rows,seats,owncell[:,0],owncell[:,1]]
    started=s['slide_active']|trigger
    path=j.where(trigger,0.,s['slide_path'])+j.where(s['slide_active'],j.linalg.norm(after.pos[rows,seats]-before.pos[rows,seats],axis=-1),0.)
    ticks=j.where(trigger,0,s['slide_ticks'])+s['slide_active'].astype(j.int32)
    start_home=j.where(trigger,home,s['slide_start_home']);slidebad=j.where(trigger,False,s['slide_bad'])|bad
    finished=started&((after.move_status[rows,seats]!=2)|terminal)
    saved=j.where((start_home<10000)&(home<start_home),j.maximum(path/(72./40./10.)-ticks,0.),0.)
    # A blast during sliding or next30 ticks voids pending queue, then charges
    # one penalty per eligible slide (not once on every trapped tick).
    risk=j.where(s['banana_risk_age']>=0,s['banana_risk_age']+1,-1)
    slot=j.clip(s['slide_count']+trigger.astype(j.int32)-1,0,3)
    risk=risk.at[rows,slot].set(j.where(finished,0,risk[rows,slot]))
    risky=(risk>=0)&(risk<=30)&bad[:,None]
    candidate=candidate.at[:,BANANA].set(j.where(finished&~slidebad,banana_reward(True,saved,False),0.)-.02*risky.sum(-1))
    risk=j.where(risky|(risk>30),-1,risk)
    s=dict(s,slide_active=started&~finished,slide_path=path,slide_ticks=ticks,slide_start_home=start_home,slide_bad=slidebad,
        slide_count=s['slide_count']+trigger.astype(j.int32),banana_risk_age=risk)
    # Exact core temporal source evidence: only bombs actually resolved early
    # with own cause, never connection-at-placement or live stock counts.
    own=(before.fuse>0)&(before.core.owner==seats[:,None,None])
    resolved=own&(after.fuse<=0)
    ids=s['bomb_batch'];sourcebatch=j.min(j.where(resolved,ids,1<<27),(-2,-1))
    k=cross_batch_count(before,after,info,s,seats,source_valid)
    cover=owned
    rr,cc=j.indices((13,15))
    near=(j.abs(rr[None]-enemycell[:,0,None,None])+j.abs(cc[None]-enemycell[:,1,None,None])<=3)&after.alive[rows,enemy,None,None]
    enemy_exact=j.zeros_like(cover).at[rows,enemycell[:,0],enemycell[:,1]].set(after.alive[rows,enemy])
    useful=(cover&(before.brick|(before.core.crate>0)|(before.tactical_crate>0)|enemy_exact)).any((-2,-1))
    explosion=source_valid.any((-2,-1))&useful&active
    candidate=candidate.at[:,CHAIN].set(chain_reward(k,bad=bad,empty=~explosion))
    pressure=explosion&(cover&near).any((-2,-1))&(before.t-s['last_explosion']<=24)&(sourcebatch!=s['last_explosion_batch'])&(s['pressure_count']<10)
    candidate=candidate.at[:,PRESSURE].set(.01*pressure)
    # Strict actual next-state legal safe-exit reduction; no placement proxy.
    from scripts.bomb_threat_r1_progress_reward import native_safe_exits
    def threat_measure(_):
        b,_=native_safe_exits(before,seats);a,_=native_safe_exits(after,seats)
        return a.sum(-1)<b.sum(-1)
    reduced=jax.lax.cond(explosion.any(),threat_measure,lambda _:j.zeros(n,bool),None)
    novel=jax.lax.cond(explosion.any(),lambda _:novel_explosion_cover(before,after,seats,source_valid),lambda _:j.zeros_like(cover),None)
    candidate=candidate.at[:,THREAT].set(.02*(explosion&reduced&(novel&near).any((-2,-1))))
    s=dict(s,last_explosion=j.where(explosion,before.t,s['last_explosion']),last_explosion_batch=j.where(explosion,sourcebatch,s['last_explosion_batch']),pressure_count=s['pressure_count']+pressure.astype(j.int32))
    s=batch_update(s,before,after,info)
    metrics.update(r3_critical_glue=critical,r3_own_glue=own_glue,r3_banana_start=trigger,r3_banana_saved_ticks=j.where(finished,saved,0.),
        r3_road_improved=improve,r3_actual_cross_batch=k,r3_pressure=pressure,r3_effective_explosion=explosion,
        r3_item_phi_before=phi0,r3_item_phi_after=phi1,r3_event_candidate=candidate)
    return s,candidate,metrics,improve,bad,terminal,active
