"""Reset-only A3SR geometry gate. No reward, policy or production JS changes."""
import json,math
from collections import deque
from pathlib import Path
import numpy as np

H,W=13,15
STEPS=((-1,0),(1,0),(0,-1),(0,1))
_ENABLED=False

def load_level():
    return next(x for x in json.loads(Path('web/assets/maps/levels.json').read_text()) if x.get('qqt_id')==806)

def base_mask(level):
    out=np.zeros((H,W),bool)
    for r,c in level['bun_bases']:out[r:r+3,c:c+3]=True
    return out

def flood(opened,start):
    if not opened[start]:return {}
    d={start:0};q=deque([start])
    while q:
        r,c=q.popleft()
        for dr,dc in STEPS:
            p=(r+dr,c+dc)
            if 0<=p[0]<H and 0<=p[1]<W and opened[p] and p not in d:d[p]=d[(r,c)]+1;q.append(p)
    return d

def blast_mask(wall,brick,cell,blast):
    out=np.zeros((H,W),bool);out[cell]=True
    for dr,dc in STEPS:
        for k in range(1,int(blast)+1):
            p=(cell[0]+dr*k,cell[1]+dc*k)
            if not (0<=p[0]<H and 0<=p[1]<W) or wall[p]:break
            out[p]=True
            if brick[p]:break
    return out

def validate_cell(wall,brick,bases,pushable,cell,blast,speed,fuse,protection=0):
    cell=tuple(map(int,cell))
    if not(0<=cell[0]<H and 0<=cell[1]<W) or wall[cell] or brick[cell] or bases[cell] or pushable[cell]:return dict(valid=False,component_size=0,escape_ticks=None)
    d=flood(~wall&~brick&~pushable,cell);hazard=blast_mask(wall,brick,cell,blast)
    escape=min((steps for p,steps in d.items() if not hazard[p]),default=10**6)
    ticks=math.ceil(escape/max(float(speed),1e-9))
    # Conservative: require actual escape before fuse even when protection lasts
    # longer; protection is recorded, never used to permit a trapped start.
    return dict(valid=len(d)>=6 and ticks<fuse,component_size=len(d),escape_ticks=ticks,protection_ticks=protection)

def wall_targets_reachable(level):
    wall=np.asarray(level['wall'],bool).reshape(H,W);push=np.asarray(level.get('pushable',np.zeros(H*W)),bool).reshape(H,W)
    reach=[]
    for anchor in level['bun_bases']:
        mask=np.zeros((H,W),bool)
        for r in range(anchor[0],anchor[0]+3):
            for c in range(anchor[1],anchor[1]+3):
                for cell in flood(~wall&~push,(r,c)):mask[cell]=True
        reach.append(mask)
    return reach[0]&reach[1]

def sample_numpy(level,seed,count):
    rng=np.random.default_rng(seed);wall=np.asarray(level['wall'],bool).reshape(H,W);native=np.asarray(level['brick'],bool).reshape(H,W);push=np.asarray(level.get('pushable',np.zeros(H*W)),bool).reshape(H,W);bases=base_mask(level);reachable=wall_targets_reachable(level)
    rows={k:[] for k in ['cells','wall','brick','carried','spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','connected']}
    speed=.3*level['initial_stats']['speed'];blast=level['initial_stats']['blast']
    for _ in range(count):
        mode=rng.random()<.5;rate=float(rng.uniform(0,.3)) if mode else 0.;brick=native&~((rng.random((H,W))<rate)&~push&~bases) if mode else native.copy()
        original=np.array([team[rng.integers(len(team))] for team in level['bun_spawns']],np.int32);pair=original.copy();attempt=0;fallback=False
        if mode:
            candidates=np.argwhere(~wall&~brick&~push&~bases&reachable)
            valid={tuple(c):validate_cell(wall,brick,bases,push,tuple(c),blast,speed,30)['valid'] for c in candidates}
            for attempt in range(32):
                p=candidates[rng.integers(len(candidates),size=2)]
                if np.abs(p[0]-p[1]).sum()<6 or not all(valid[tuple(c)] for c in p):continue
                if blast_mask(wall,brick,tuple(p[0]),blast)[tuple(p[1])] or blast_mask(wall,brick,tuple(p[1]),blast)[tuple(p[0])]:continue
                pair=p;break
            else:fallback=True;pair=original
        values=dict(cells=pair,wall=wall,brick=brick,carried=[-1,-1],spawn_mode=int(mode),brick_drop_rate=rate,spawn_resample=attempt if mode else 0,spawn_fallback=int(fallback),connected=tuple(pair[1]) in flood(~wall&~brick&~push,tuple(pair[0])))
        for k,v in values.items():rows[k].append(v)
    return {k:np.asarray(v) for k,v in rows.items()}

def valid_cell_jax(wall,brick,bases,pushable,cell,blast,speed,fuse):
    import jax
    import jax.numpy as j
    opened=~wall&~brick&~pushable;r,c=cell
    legal=opened[r,c]&~bases[r,c]
    dist=j.full((H,W),999,j.int32).at[r,c].set(0)
    def body(_,d):
        pad=j.pad(d,1,constant_values=999)
        near=j.minimum(j.minimum(pad[:-2,1:-1],pad[2:,1:-1]),j.minimum(pad[1:-1,:-2],pad[1:-1,2:]))+1
        return j.where(opened,j.minimum(d,near),999)
    # Every reachable safe cell within the fuse budget, plus six-cell component
    # lower bound. Initial speed is frozen to the map's native initial stats.
    dist=jax.lax.fori_loop(0,max(6,int(math.ceil(float(speed)*fuse))),body,dist)
    rr,cc=j.indices((H,W));hazard=(rr==r)&(cc==c)
    for dr,dc in STEPS:
        active=j.asarray(True)
        for k in range(1,int(blast)+1):
            y=j.clip(r+dr*k,0,H-1);x=j.clip(c+dc*k,0,W-1);inside=(r+dr*k>=0)&(r+dr*k<H)&(c+dc*k>=0)&(c+dc*k<W)
            active=active&inside&~wall[y,x];hazard=hazard.at[y,x].set(hazard[y,x]|active);active=active&~brick[y,x]
    safe=j.min(j.where(~hazard,dist,999));ticks=j.ceil(safe/speed)
    return legal&((dist<999).sum()>=6)&(ticks<fuse)

def configure(enabled=False):
    global _ENABLED
    _ENABLED=bool(enabled)
    return dict(enabled=_ENABLED)

def sample_jax(key,core):
    """Sample the dropped layout first, then validate up to32 directed pairs."""
    import jax
    import jax.numpy as j
    level=load_level();bases=j.asarray(base_mask(level));reachable=j.asarray(wall_targets_reachable(level))
    speed=.3*float(level['initial_stats']['speed']);blast=int(level['initial_stats']['blast'])
    km,kr,kb,kp=jax.random.split(key,4)
    mode=jax.random.bernoulli(km,.5)
    rate=jax.random.uniform(kr,(),maxval=.30)
    brick=core.brick&~((jax.random.uniform(kb,(H,W))<rate)&~core.pushable&~bases)
    mask=~core.wall&~brick&~core.pushable&~bases&reachable
    candidate=jax.random.categorical(kp,j.where(mask.reshape(-1),0.,-j.inf),shape=(32,2))
    pairs=j.stack([candidate//W,candidate%W],-1)
    def check(pair):
        ok=jax.vmap(lambda cell:valid_cell_jax(core.wall,brick,bases,core.pushable,cell,blast,speed,30))(pair).all()
        return ok&(j.abs(pair[0]-pair[1]).sum()>=6)
    valid=jax.vmap(check)(pairs);any_valid=valid.any();index=j.argmax(valid)
    pos=j.where(any_valid,pairs[index].astype(j.float32)+.5,core.pos)
    changed=core._replace(pos=j.where(mode,pos,core.pos),brick=j.where(mode,brick,core.brick))
    meta=dict(spawn_mode=mode.astype(j.int32),brick_drop_rate=j.where(mode,rate,0.),spawn_resample=j.where(mode,j.where(any_valid,index,32),0),spawn_fallback=mode&~any_valid)
    return changed,meta


def apply_batch(keys,states,aux,reset_mask=None,enabled=None):
    """Compact actual eligible resets; never evaluate32pairs per live env/tick."""
    import jax
    import jax.numpy as j
    from scripts.carrier_mechanism_engine import select_tree
    active=_ENABLED if enabled is None else enabled
    n=len(keys)
    meta=dict(spawn_mode=j.zeros(n,j.int32),brick_drop_rate=j.zeros(n,j.float32),
        spawn_resample=j.zeros(n,j.int32),spawn_fallback=j.zeros(n,bool),spawn_eligible=j.zeros(n,bool))
    if not active:return states,meta
    eligible=(aux['provenance_source']<0)&j.all(states.bun_carried<0,axis=-1)
    if reset_mask is not None:eligible=eligible&reset_mask
    meta['spawn_eligible']=eligible
    count=eligible.sum();indices=j.nonzero(eligible,size=n,fill_value=0)[0]
    width=min(n,16)
    # Each tile handles at most16 true reset rows. Excess scatter indices are
    # deliberately out of bounds and dropped, not duplicate index0 writes.
    padded=j.pad(indices,(0,width))
    def body(carry):
        offset,st,stats=carry
        ids=jax.lax.dynamic_slice(padded,(offset,),(width,))
        cores=jax.tree.map(lambda a:a[ids],states.core)
        new,mm=jax.vmap(sample_jax)(keys[ids],cores)
        chosen=jax.tree.map(lambda a:a[ids],states)._replace(core=new,bun_spawn_pos=new.pos)
        dest=j.where(j.arange(width)+offset<count,ids,n)
        st=jax.tree.map(lambda a,v:a.at[dest].set(v,mode='drop'),st,chosen)
        for name,value in mm.items():stats[name]=stats[name].at[dest].set(value,mode='drop')
        return offset+width,st,stats
    _,st,meta=jax.lax.while_loop(lambda c:c[0]<count,body,(j.array(0),states,meta))
    return st,meta
