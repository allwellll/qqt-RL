"""Explicit K4 arena backend with independent maps and native fresh states.

The audited step is captured unchanged. Its private map references point at
the arena table while the live bun backend retains the exact previous table
and configuration. Importing this module installs nothing.
"""
import copy
import json
from types import SimpleNamespace, FunctionType
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np

_READY=False
_FRESH_CORE=None
_TEMPLATE=None
_COVER=None
_SPLITS=None
_SAFETY=None


def prepare():
    global _READY,_FRESH_CORE,_TEMPLATE,_COVER,_SPLITS
    if _READY:return
    from jax_bomb import je07_bun_env_r1_items as env,levels
    from scripts import r4_kill_arena as arena
    # K.configure changes module-level training flags only while constructing
    # its private function. Restore all prior flags and table before returning.
    previous_table=levels.active()
    flags={name:value for name,value in vars(env).items() if name.startswith('_')}
    shared_step=env.step;shared_core_step=env.base.step
    try:
        arena.configure()
        proxy=SimpleNamespace(active=lambda:arena._TABLE)
        arena._STEP.__globals__['levels']=proxy
        native=arena._STEP.__globals__['base']
        native.step.__globals__['levels']=proxy
        original_core=native.step
        original_resolver=original_core.__globals__['_resolve_explosions_matrix']
        def instrumented_core(*args,**kwargs):
            # Trace-local passive inputs: return exactly the same resolver
            # values and record its actual pre-explosion geometry for causal
            # batch evidence. No shared global or physical state is changed.
            recorded={}
            def resolver(fuse,owner,blast,wall,brick=None,*a,**kw):
                recorded.update(k4_resolve_fuse=fuse,k4_resolve_owner=owner,
                    k4_resolve_blast=blast,k4_resolve_wall=wall,k4_resolve_brick=brick)
                return original_resolver(fuse,owner,blast,wall,brick,*a,**kw)
            inner=FunctionType(original_core.__code__,dict(original_core.__globals__,
                _resolve_explosions_matrix=resolver),original_core.__name__,
                original_core.__defaults__,original_core.__closure__)
            result=inner(*args,**kwargs)
            if kwargs.get('return_info',False):
                return (*result[:-1],dict(result[-1],**recorded))
            return result
        native.step=instrumented_core
        native.k4_original_step=original_core
        # Fresh native map sampling reuses the core constructor with an
        # independent namespace. Override its domain-randomized attributes
        # below: kill arena starts with each map's original initial stats.
        _FRESH_CORE=FunctionType(env.base._fresh.__code__,
            dict(env.base._fresh.__globals__,levels=proxy),env.base._fresh.__name__,
            env.base._fresh.__defaults__,env.base._fresh.__closure__)
    finally:
        for name in list(vars(env)):
            if name.startswith('_') and name not in flags:delattr(env,name)
        for name,value in flags.items():setattr(env,name,value)
        levels._ACTIVE=previous_table
    assert env.step is shared_step and env.base.step is shared_core_step
    assert levels.active() is previous_table
    # A schema shell is constructed once. Dynamic arena initialization below
    # never calls the bun curriculum, Bank, spawn bucket or its reset RNG.
    if previous_table is None:env.prepare()
    _TEMPLATE=env._fresh(jax.random.PRNGKey(0x4A4),jax.nn.one_hot(0,16))
    if previous_table is None:levels._ACTIVE=previous_table
    catalog=json.loads(arena.CATALOG.read_text())['maps']
    _SPLITS={split:j.array([i for i,row in enumerate(catalog) if row['split']==split],j.int32)
             for split in ('train','dev','holdout')}
    _COVER=j.asarray(np.stack([(
        np.asarray(row['map'].get('cover',[False]*195),bool)|
        np.asarray(row['map'].get('overhead',[False]*195),bool))&
        ~np.asarray(row['map'].get('bush',[False]*195),bool)
        for row in catalog]).reshape(-1,13,15))
    _READY=True


def fresh(key,*,split='train',clear_fraction=0.,map_id=None):
    """True native fresh arena, sampled from the declared immutable map split.

    No player is fabricated as carrying a package; no arena has a package or
    cage. Two different original spawn points are drawn without side bias.
    clear_fraction is .30 only for the first200 NEW K4 updates, then zero.
    """
    prepare()
    from scripts import r4_kill_arena as arena
    from jax_bomb import je07_native_items as items
    ks,kc,ka,kb=jax.random.split(key,4)
    allowed=_SPLITS[split]
    if map_id is None:map_id=allowed[jax.random.randint(ks,(),0,len(allowed))]
    table=arena._TABLE
    # Sample through an explicit one-hot table, so no dev/holdout index can
    # enter a train reset even when a caller later changes levels.active().
    core=_FRESH_CORE(kc)
    count=table.cnt[map_id]
    a=jax.random.randint(ka,(),0,count);b=jax.random.randint(kb,(),0,count-1)
    b+=b>=a
    pos=j.stack([table.spawns[map_id,a],table.spawns[map_id,b]]).astype(j.float32)+.5
    lo=table.lo[map_id]
    clear=jax.random.uniform(jax.random.fold_in(key,0xC1EA))<clear_fraction
    brick=j.where(clear,j.zeros_like(table.brick[map_id]),table.brick[map_id])
    push=table.pushable[map_id]&brick
    core=core._replace(wall=table.wall[map_id],brick=brick,bush=table.bush[map_id],
        pushable=push,push_t=j.zeros_like(core.push_t),crate=j.where(table.crate[map_id],7,0).astype(core.crate.dtype),
        rec_crate=table.rec[map_id],pos=pos,hp=j.ones(2,core.hp.dtype),
        invuln=j.full(2,10,core.invuln.dtype),bombs_cap=j.full(2,lo[0],core.bombs_cap.dtype),
        blast_cap=j.full(2,lo[1],core.blast_cap.dtype),spd_g=j.full(2,lo[2],core.spd_g.dtype),
        level_id=j.asarray(map_id,core.level_id.dtype),is_open=~brick.any())
    st=_TEMPLATE._replace(core=core,bun_stored=j.zeros_like(_TEMPLATE.bun_stored),
        bun_loose=j.zeros_like(_TEMPLATE.bun_loose),bun_carried=j.full_like(_TEMPLATE.bun_carried,-1),
        bun_respawn=j.zeros_like(_TEMPLATE.bun_respawn),bun_spawn_pos=pos,
        native_items=items.fresh(_COVER[map_id]),scene=j.array(0,_TEMPLATE.scene.dtype),
        lesson=j.array(0,_TEMPLATE.lesson.dtype))
    return arena.Arena(st,j.zeros(2,j.int32),j.zeros(2,j.int32),j.zeros(2,j.int32),
        j.zeros(2,j.int32),j.floor(pos).astype(j.int32),key[1],j.asarray(map_id,j.int32))


def batch_fresh(keys,*,split='train',new_update=1):
    """The course counter is relative, never the absolute parent update4002."""
    fraction=j.where(j.asarray(new_update)<=200,.30,0.)
    return jax.vmap(lambda k:fresh(k,split=split,clear_fraction=fraction))(keys)


def map_split_ids(split):
    prepare();return _SPLITS[split]


def native_safety():
    """Private audited arena movement/rule predictor, never historical env."""
    global _SAFETY
    prepare()
    if _SAFETY is None:
        from types import ModuleType
        from jax_bomb import je07_bun_env_r1_items as env
        from scripts import r4_kill_arena as arena
        module=ModuleType('jax_bomb.k4_native_safety');module.__package__='jax_bomb'
        exec(compile(Path('jax_bomb/bun_safety.py').read_text(),
                     'jax_bomb/bun_safety.py','exec'),module.__dict__)
        proxy=SimpleNamespace(**{name:value for name,value in vars(env).items() if not name.startswith('__')})
        proxy.base=arena._STEP.__globals__['base'];proxy.step=arena._STEP
        module.env=proxy;module.base=proxy.base
        module._RELAX_STEPS=min(env.H*env.W,env.base.FUSE+2)
        _SAFETY=module
    return _SAFETY


def safe_exits(states,seats):
    """Same native first-move exit ABI as K2, bound to arena backend."""
    from scripts.bomb_threat_r1_progress_reward import native_safe_exits
    function=FunctionType(native_safe_exits.__code__,dict(native_safe_exits.__globals__,
        native_safety=native_safety),native_safe_exits.__name__,
        native_safe_exits.__defaults__,native_safe_exits.__closure__)
    return function(states,seats)


def unique_explosion_batches(info,positions,birth):
    """Actual same-wave causal root batches hitting each enemy body.

    Mirrors the audited first-trigger-wave ancestry. Ambiguous roots, enemy
    sources and unknown placement history remain -1; no proximity/request is
    upgraded to a same-batch causal event.
    """
    from scripts import r4_kill_arena as arena
    C=arena._STEP.__globals__['base']
    fuse=info['k4_resolve_fuse'];owner=info['k4_resolve_owner']
    wall=info['k4_resolve_wall'];brick=info['k4_resolve_brick']
    blast=j.where(info['k4_resolve_blast']>0,info['k4_resolve_blast'],C.BLAST)
    live=fuse>0;exists=live|((fuse==0)&(owner>=0));count=C.MAX_BOMBS*2
    _,order=jax.lax.top_k(exists.reshape(-1).astype(j.int32),count)
    mask=exists.reshape(-1)[order];r,c=order//15,order%15
    burst=mask&(fuse.reshape(-1)[order]==0)
    roots=j.eye(count,dtype=bool)&burst[:,None]
    up,down,left,right=C._reach_dp(wall|brick|live)
    u,d,l,ri=[v[r,c] for v in (up,down,left,right)]
    distance=j.abs(r[:,None]-r[None,:])+j.abs(c[:,None]-c[None,:])
    aligned=(r[:,None]==r[None,:])|(c[:,None]==c[None,:])
    ray=((r[:,None]<r[None,:])&(u[None,:]>=distance-1))|((r[:,None]>r[None,:])&(d[None,:]>=distance-1))
    ray|=((c[:,None]<c[None,:])&(l[None,:]>=distance-1))|((c[:,None]>c[None,:])&(ri[None,:]>=distance-1))
    reach=aligned&ray&(distance>0)&(distance<=blast[r,c][:,None])&mask[:,None]&mask[None,:]&live[r,c][None,:]
    def wave(carry,_):
        active,ancestry,current=carry
        edges=reach&current[:,None]&~active[None,:]
        received=edges.any(0)
        parents=(edges[:,:,None]&ancestry[:,None,:]).any(0)
        return (active|received,ancestry|parents,received),None
    (burst,roots,_),_=jax.lax.scan(wave,(burst,roots,burst),None,length=7)
    def source_hits(at,valid):
        seed=j.zeros((13,15),bool).at[at//15,at%15].set(valid)
        _,h,v=C._rays(seed,live,blast,wall,brick,return_dirs=True)
        return arena._native_contact_any(positions,h,v)
    hit=jax.vmap(source_hits)(order,mask&burst)
    own=owner.reshape(-1)[order];ids=birth.reshape(-1)[order]
    result=[]
    for attacker in (0,1):
        causal=(hit[:,1-attacker,None]&roots).any(0)
        candidate=j.min(j.where(causal&(own==attacker)&(ids>=0),ids,1<<28))
        exact=causal.any()&j.all(~causal|((own==attacker)&(ids==candidate)))&(candidate<(1<<28))
        result.append(j.where(exact,candidate,-1))
    return j.stack(result)
