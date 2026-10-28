"""Local experiment adapter; previous diagnostic and production source stay unchanged."""
import jax
import jax.numpy as j
from scripts import carrier_mechanism_engine as E,combat_item_events as events
from scripts import je09_stage_g_combat as combat
from scripts.je10_route_oracle import action_set,continuous_distance
from jax_bomb import bun_safety
OLD_TRANSITION=E.transition
OLD_INITIAL=E.initial
BASE_COMPONENTS=E.ops.COMPONENTS


def configure():
    E.configure()
    # Carry is a persistent objective; hazards still filter joint route actions.
    from scripts import je10_route_oracle as oracle
    oracle.route_allowed=events.phase_allowed
    E.safe_teacher=route_teacher


def fresh(keys,bankstates,bankaux,bank_probability=.25):
    """Natural spawn or certified pickup; default preserves the original stream."""
    if not 0 <= bank_probability <= 1:raise ValueError('Invalid Bank probability')
    n=len(keys);split=jax.vmap(lambda k:jax.random.split(k,4))(keys)
    st,ax=OLD_INITIAL(split[:,0],0)
    choice=jax.vmap(lambda k:jax.random.randint(k,(),0,bankstates.t.size))(split[:,1])
    certified=jax.tree.map(lambda a:a[choice],bankstates)
    ca=jax.tree.map(lambda a:a[choice],bankaux)
    use=jax.vmap(lambda k:jax.random.bernoulli(k,bank_probability))(split[:,2])
    item=jax.vmap(lambda k:jax.random.randint(k,(),0,3))(split[:,3])
    slots=st.item_slots.at[:,0,0].set(item.astype(st.item_slots.dtype))
    counts=st.item_counts.at[:,0,0].set((item>0).astype(st.item_counts.dtype))
    st=st._replace(item_slots=slots,item_counts=counts,held_item=slots[:,:,0])
    st=E.select_tree(use,certified,st);ax=E.select_tree(use,ca,ax)
    ax=dict(ax,tactics=events.state(n))
    return st,ax


def route_teacher(st,ax):
    from scripts.je10_route_oracle import labels
    o,g,mm,bm=E.inputs(st);seat=j.zeros(st.t.shape,j.int32)
    target=labels(st,ax,mm,seat,o[:,5]);safe=jax.vmap(bun_safety.analyze_actions)(st).survivable[:,0]
    # Explicit genuine pickup provenance, even for legal free-alive spontaneous theft.
    actual_carry=(st.bun_carried[:,0]==1)&ax['carry_items']['ledger']['custody'][:,0]
    native=~st.alive[:,1]|(st.trapped[:,1]>0)
    # Route labels select stock/loose while empty, persistent home while carrying.
    allow=st.alive[:,0]&(st.trapped[:,0]==0)&~ax['terminal']&j.where(st.bun_carried[:,0]>=0,actual_carry,native)
    return (target[:,:5]>0)[:,:,None]&safe&mm[:,:,None]&bm[:,None,:]&allow[:,None,None]


def combat_teacher(st,ax,keys):
    _,_,mm,bm=E.inputs(st)
    label=combat.labels(st,j.zeros(st.t.shape,j.int32),mm,bm,ax['terminal'],keys[0])
    return (label[:,:5]>0)[:,:,None]&(label[:,5:8]>0)[:,None,:]&(label[:,8]>0)[:,None,None]


def transition(st,ax,actions,keys,arm='control'):
    coreax={k:v for k,v in ax.items() if k!='tactics'}
    after,na,reward,done,info,parts=OLD_TRANSITION(st,coreax,actions,keys)
    old=E.ops.COMPONENTS;get=lambda name:parts[:,:,old.index(name)]
    legacy_positive=get('item_positive')
    legacy_glue=j.minimum(.5*get('glue_attributed_hit'),legacy_positive)
    legacy_banana=legacy_positive-legacy_glue
    n=len(st.t);r=j.arange(n)[:,None];team=j.arange(2)[None];enemy=1-team
    blocked=st.wall|st.brick|st.core.pushable
    fields=na['carry_items']['distance']
    d=j.stack([continuous_distance(after.pos[:,p],fields[:,p],blocked) for p in range(2)],-1)
    db=j.stack([continuous_distance(st.pos[:,p],fields[:,p],blocked) for p in range(2)],-1)
    source=j.where(info['trap_owner']==team,0,1)
    carry=(after.bun_carried==enemy)&after.alive
    bad=info['death']|info['drop']|info['carrier_death']|info['mutual_death'][:,None]|(after.trapped>0)
    changed=j.any((st.wall!=after.wall)|(st.brick!=after.brick)|(st.core.pushable!=after.core.pushable),(-2,-1))
    # A real attempted chase with actual displacement; standing on glue cannot pay.
    delta=after.pos[:,::-1]-st.pos[:,::-1]
    toward=j.sum(delta*(st.pos-st.pos[:,::-1]),-1)>1e-6
    moved=j.linalg.norm(delta,axis=-1)>1e-5
    raw_speed=E.env.base.STEP*st.core.spd_g[:,::-1]
    actual_speed=.18  # native72px/s,40px/cell,10Hz
    speed_before=j.where((st.bun_carried[:,::-1]>=0)|(st.move_status[:,::-1]==1),.18,
                        j.where((st.move_status[:,::-1]==2)|(st.move_status[:,::-1]==3),.67,raw_speed))
    # Once slowed, reconstruct its no-status base speed, only for originally speed-reducible targets.
    delay=j.maximum(raw_speed-actual_speed,0.)*toward*moved*(st.move_status[:,::-1]==1)
    glue=get('glue_attributed_hit')>0
    glue&=(speed_before>.18+1e-6)&(st.bun_carried[:,::-1]<0)
    safe=(get('item_alive_carry')>0)&~done[:,None]
    e=dict(banana_start=get('banana_start')>0,banana_owner=source,glue_hit=glue,
        carry=carry,safe=safe,bad=bad,capture=info['capture'],sliding=after.move_status==2,
        slow=(after.move_status[:,::-1]==1)&after.alive[:,::-1]&(after.trapped[:,::-1]==0),
        topology_changed=changed[:,None],distance=d,walk_budget=j.full((n,2),.18),
        chase_delay=delay,legacy_glue=legacy_glue,legacy_banana=legacy_banana,
        lost=~carry&~info['capture'])
    # Before activation, use precontact home distance as the full-event origin.
    ts=dict(ax['tactics'])
    for f in ['banana','glue']:
        begin=e['banana_start'] if f=='banana' else glue
        ts[f+'_start']=j.where(begin&~ts[f+'_active'],db,ts[f+'_start'])
    # settle starts from distance; correct activation origin via a separate start_distance field.
    e['start_distance']=db
    delta_b,ts,mb=events.settle(e,ts,'banana');delta_g,ts,mg=events.settle(e,ts,'glue')
    extra=delta_b if arm=='banana' else delta_g if arm=='glue' else j.zeros_like(reward)
    na['tactics']=ts
    reward=reward+extra
    obs=j.stack([mb['completed'],mb['benefit'],mb['pay'],mg['completed'],mg['benefit'],mg['pay'],extra],-1)
    parts=j.concatenate([parts[...,:-1],obs,reward[...,None]],-1)
    return after,na,reward,done,info,parts
