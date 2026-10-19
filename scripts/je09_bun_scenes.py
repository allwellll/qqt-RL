"""JE07 isolated map806 topology tables and joint combat/transport curriculum."""
import numpy as np
import jax
import jax.numpy as jnp
from jax_bomb import je07_bun_env as env
from scripts.je05_bun_transport import precompute_regions
NAMES=('combat','vulnerable_steal','carry_route_far','carry_route_mid','carry_route_near',
       'defend_and_steal','escort_under_pressure','route_break_capture','full_bun','drop_chain_return')
BASE=jnp.array([.45,.06,.04,.04,.035,.07,.05,.055,.10,.10],jnp.float32)
from scripts.je10_causal_init import scene_weights
BASE=scene_weights(BASE)
HORIZONS=jnp.array([300,600,600,600,600,900,900,1200,2400,1200],jnp.int32)
wall=np.asarray(env._BUN_LEVEL['wall'],bool).reshape(env.H,env.W)
brick=np.asarray(env._BUN_LEVEL['brick'],bool).reshape(env.H,env.W)
goals=np.zeros((2,env.H,env.W),bool)
for team,(r,c) in enumerate(env._BUN_LEVEL['bun_bases']):goals[team,r:r+3,c:c+3]=True
TABLES=[precompute_regions(wall,brick,goals),precompute_regions(wall,np.zeros_like(brick),goals)]
REGIONS=jnp.asarray(np.stack([t['regions'] for t in TABLES]))
HOME=jnp.asarray(np.stack([t['home'] for t in TABLES]))
DISTANCE=jnp.asarray(np.stack([t['distance'] for t in TABLES]))
OPEN_SCENE=jnp.array([True,False,True,True,True,True,True,False,False,True])
# Every candidate is an actually walkable cell under its frozen geometry.
CELLS=np.indices((env.H,env.W)).reshape(2,-1).T.astype(np.int32)
MEMBERS=np.zeros((len(NAMES),2,len(CELLS)),bool)
for scene in range(len(NAMES)):
    variant=int(OPEN_SCENE[scene]);allowed=~wall & (~brick if not variant else True)
    for team in range(2):
        region=TABLES[variant]['regions'][team]
        if scene==2:mask=region<8
        elif scene==3:mask=(region>=8)&(region<24)
        elif scene in (4,6):mask=(region>=24)&(region<32)
        elif scene==1:mask=(TABLES[variant]['distance'][1-team]<=4)&~goals[1-team]
        elif scene==5:mask=TABLES[variant]['distance'][1-team]<=6
        elif scene==9:mask=(region>=8)&(region<24)
        else:mask=np.ones_like(wall)
        mask &= allowed & (region>=0) & ~goals[team]
        if scene in (0,1,5):mask &= ~goals[1-team]
        MEMBERS[scene,team]=mask.reshape(-1)
        if not mask.any():raise ValueError(f'empty legal region bucket {NAMES[scene]} team{team}')
MEMBERS=jnp.asarray(MEMBERS);CELLS=jnp.asarray(CELLS)
# Frozen reachable carrier/learner pairs on the open native map, with an
# unobstructed legal return path. Geometry is audited before JIT rollout.
from scripts.je08_bun_potential import cpu_shortest
_pairs=[]
DISTANCE_BUCKETS=((8,12),(12,16),(16,24),(24,32))
_long_pairs=[[] for _ in DISTANCE_BUCKETS]
for _r,_c in np.argwhere(~wall & ~goals.any(0) & (np.asarray(TABLES[1]['distance'])<32767).all(0)):
    _goal=np.zeros_like(wall);_goal[_r,_c]=True
    _d=cpu_shortest(wall,np.zeros_like(brick),_goal)
    _pairs.extend((int(_r),int(_c),int(_rr),int(_cc))
                  for _rr,_cc in np.argwhere((_d>=2)&(_d<=8)&~wall))
    for _bucket,(_low,_high) in zip(_long_pairs,DISTANCE_BUCKETS):
        _bucket.extend((int(_r),int(_c),int(_rr),int(_cc))
                      for _rr,_cc in np.argwhere((_d>=_low)&(_d<=_high)&~wall))
BRIDGE_PAIRS=jnp.asarray(_pairs,jnp.int32)
LONG_COUNTS=jnp.asarray([len(p) for p in _long_pairs])
LONG_PAIRS=jnp.asarray([p+[p[0]]*(max(map(len,_long_pairs))-len(p)) for p in _long_pairs],jnp.int32)
def curriculum_weights(counts):
    from scripts.je10_causal_init import scene_weights
    conversion=counts[:,1]/jnp.maximum(1,counts[:,0])
    w=scene_weights(BASE*jnp.clip(1.5-20*conversion,.5,1.5))
    task=w.at[0].set(0.)
    return (task/task.sum()*.55).at[0].set(.45)

def init_batch(key,n,course,distance_stage=None):
    from scripts.je10_causal_init import scene_weights
    key,ks,kc,kw=jax.random.split(key,4)
    states=env.init_batch(key,n,jax.nn.one_hot(0,16))
    scene=jax.random.choice(ks,len(NAMES),(n,),p=scene_weights(course));rows=jnp.arange(n);seat=rows%2
    logits=jnp.where(MEMBERS[scene,seat],0.,-1e9)
    chosen=jax.random.categorical(kc,logits);cell=CELLS[chosen];pos=cell.astype(jnp.float32)+.5
    core=states.core._replace(brick=jnp.where(OPEN_SCENE[scene,None,None],False,states.brick))
    # Reactive opponent starts elsewhere, preserving map legal native spawn.
    core=core._replace(pos=core.pos.at[rows,seat].set(pos))
    full=scene==8
    core=core._replace(pos=jnp.where(full[:,None,None],states.pos,core.pos))
    origin=1-seat
    chain=scene==9
    # Enemy legitimately carries its own recovered object. Learner is empty;
    # no scripted kill/drop/pickup and no post-step teleport. Real control is
    # required to form an attributable drop or direct contact transfer.
    enemy_logits=jnp.where(MEMBERS[9,origin],0.,-1e9)
    enemy_cell=CELLS[jax.random.categorical(kw,enemy_logits)]
    pair=BRIDGE_PAIRS[jax.random.randint(kw,(n,),0,len(BRIDGE_PAIRS))]
    if distance_stage is not None:
        klong,kmix=jax.random.split(kw)
        long=LONG_PAIRS[distance_stage,jax.random.randint(klong,(n,),0,LONG_COUNTS[distance_stage])]
        pair=jnp.where(jax.random.bernoulli(kmix,.5,(n,1)),long,pair)
    enemy_cell=jnp.where(chain[:,None],pair[:,:2],enemy_cell)
    core=core._replace(pos=core.pos.at[rows,seat].set(jnp.where(chain[:,None],pair[:,2:]+.5,core.pos[rows,seat])))
    core=core._replace(pos=core.pos.at[rows,origin].set(jnp.where(chain[:,None],enemy_cell+.5,core.pos[rows,origin])))
    carried=states.bun_carried.at[rows,origin].set(
        jnp.where(chain,origin,states.bun_carried[rows,origin]).astype(jnp.int8))
    # JE09 bridge starts empty-handed for the learner while the enemy carrier
    # is in a real, causally attributed one-tick trap. The next env.step emits
    # an actual death/drop event; no token is teleported or pre-dropped.
    trapped=states.trapped.at[rows,origin].set(jnp.where(chain,1,states.trapped[rows,origin]))
    trap_source=states.trapped_source.at[rows,origin,seat].set(chain)
    trap_causal=states.trapped_causal.at[rows,origin,seat].set(chain)
    # Learners always start empty-handed. Custody may only be created by the
    # native stock/loose/transfer transitions after a real route event.
    stored=states.bun_stored.at[rows,origin,origin].set(jnp.where(chain,0,1).astype(states.bun_stored.dtype))
    window=jax.random.randint(kw,(n,),8,61)
    sugar=(scene==1)&(rows%4<2);dead=(scene==1)&~sugar
    # Explicit legal training start states, never mutate/teleport during play.
    trapped=trapped.at[rows,1-seat].set(jnp.where(chain,1,jnp.where(sugar,window,0)).astype(jnp.int16))
    respawn=states.bun_respawn.at[rows,1-seat].set(jnp.where(dead,window,0).astype(jnp.int16))
    core=core._replace(alive=core.alive.at[rows,1-seat].set(~dead),hp=core.hp.at[rows,1-seat].set(jnp.where(dead,0,core.hp[rows,1-seat])))
    # Curriculum nuisance variants: explicit legal item/chain starts, no free
    # capture/kill. Bombs are enemy-owned at least three cells from both spawns.
    difficulty=jnp.zeros_like(scene,dtype=jnp.bool_)
    valid_cell=~wall & (jnp.where(OPEN_SCENE[scene,None,None],True,~brick))
    yx=jnp.asarray(np.indices((env.H,env.W)).transpose(1,2,0))
    separated=jnp.all(jnp.sum(jnp.abs(yx[None,None]-core.pos[:,:,None,None]),-1)>=3,axis=1)
    available=(valid_cell&separated).reshape(n,-1)
    kb,ki=jax.random.split(kw)
    bubble=jax.random.categorical(kb,jnp.where(available,0.,-1e9))
    br,bc=bubble//env.W,bubble%env.W
    active=difficulty&available.any(-1)
    core=core._replace(fuse=core.fuse.at[rows,br,bc].set(jnp.where(active,20,0)),
        owner=core.owner.at[rows,br,bc].set(jnp.where(active,1-seat,-1).astype(core.owner.dtype)),
        bomb_blast=core.bomb_blast.at[rows,br,bc].set(jnp.where(active,2,0).astype(core.bomb_blast.dtype)))
    # Route-adjacent field item; choose a legal neighbor toward own cage using
    # the frozen distance lookup, never BFS. Field isn't a sugar bubble.
    neighbor=jnp.clip(cell[:,None]+jnp.array([[-1,0],[1,0],[0,-1],[0,1]]),jnp.array([0,0]),jnp.array([env.H-1,env.W-1]))
    legal=~core.wall[rows[:,None],neighbor[:,:,0],neighbor[:,:,1]]&~core.brick[rows[:,None],neighbor[:,:,0],neighbor[:,:,1]]
    dist=DISTANCE[OPEN_SCENE[scene].astype(jnp.int32)[:,None],seat[:,None],neighbor[:,:,0],neighbor[:,:,1]]
    candidate=jnp.argmin(jnp.where(legal,dist,32767),-1);adj=neighbor[rows,candidate]
    hazard=jnp.zeros_like(legal.any(-1))
    field=states.field_item.at[rows,adj[:,0],adj[:,1]].set(jnp.where(hazard,1+rows%2,0).astype(jnp.int8))
    field_owner=states.field_owner.at[rows,adj[:,0],adj[:,1]].set(jnp.where(hazard,1-seat,-1).astype(jnp.int8))
    field_armed=states.field_armed.at[rows,adj[:,0],adj[:,1]].set(hazard)
    # Native FIFO inventory observable through existing held/global channels.
    grant=rows%5==0;item=1+rows%2
    slots=states.item_slots.at[rows,seat,0].set(jnp.where(grant,item,0).astype(jnp.int8))
    counts=states.item_counts.at[rows,seat,0].set(jnp.where(grant,jnp.where(item==2,3,1),0).astype(jnp.int16))
    return states._replace(item_slots=slots,item_counts=counts,held_item=slots[:,:,0],field_item=field,field_owner=field_owner,field_armed=field_armed,
                           core=core,bun_carried=carried,bun_stored=stored,trapped=trapped,trapped_source=trap_source,trapped_causal=trap_causal,bun_respawn=respawn,scene=scene,
                           bun_spawn_pos=core.pos)

def lookup(states):
    variant=OPEN_SCENE[states.scene].astype(jnp.int32)
    cells=jnp.clip(states.pos.astype(jnp.int32),jnp.array([0,0]),jnp.array([env.H-1,env.W-1]))
    teams=jnp.arange(2)[None,:]
    return REGIONS[variant[:,None],teams,cells[:,:,0],cells[:,:,1]],HOME[variant[:,None],teams,cells[:,:,0],cells[:,:,1]]

def bind_training(enabled=True):
    # Process-local bootstrap: JE04's imports and sealed source stay untouched.
    from jax_bomb import jax_train as train,je05_bun_env as legacy
    selected=env if enabled else legacy
    for name in ['make_obs','global_vec','legal_mask','step','init_batch']:
        setattr(train,name,getattr(selected,name))
