"""Own-side spawn contract; opt-in migration, never rewrites live R1 states."""
import json
from pathlib import Path
import numpy as np

VERSION = 'own-side-native-v1'
MIDLINE = 7

def level():
    return next(x for x in json.loads(Path('web/assets/maps/levels.json').read_text()) if x.get('qqt_id') == 806)

def validate(pair):
    a = np.asarray(pair)
    if a.shape != (2, 2) or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError('expected two integer row/column cells')
    if not ((a >= 0).all() and (a[:, 0] < 13).all() and (a[:, 1] < 15).all()
            and a[0, 1] <= 6 and a[1, 1] >= 8):
        raise ValueError('spawn side must match physical team; center column excluded')
    return a.astype(np.int32)

def native_pairs():
    groups = level()['bun_spawns']
    return np.stack([validate([a, b]) for a in groups[0] for b in groups[1]])

def manifest(seed, games):
    pairs = native_pairs()
    # Balanced native 4x4 combinations; shuffle per block, independent of outcome.
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.permutation(len(pairs)) for _ in range((games+15)//16)])[:games]
    return dict(schema='qqt.spawn-manifest/v2', spawn_side_version=VERSION, seed=seed,
                spawn_cells=pairs[selected].tolist(), games=games, cages=level()['bun_bases'],
                center_column='7 excluded', geometry_count=16,
                sampling='native four cells per physical team; balanced shuffled16 blocks; no player swap')

def side_counts(pairs):
    a=np.asarray(pairs); cols=a[...,1]
    own=np.stack([cols[...,0]<=6,cols[...,1]>=8],-1)
    return dict(samples=int(own.shape[0]), wrong_per_player=(~own).sum(0).tolist(),
                both_own=int(own.all(-1).sum()), center_per_player=(cols==7).sum(0).tolist())

def install():
    """Process-local patch before JIT; old workers and files stay reproducible."""
    import jax
    import jax.numpy as j
    from jax_bomb import je07_bun_env_r1_items as env
    from scripts import bomb_threat_a3sr_spawn as spawn
    if getattr(env, 'SPAWN_SIDE_VERSION', None) == VERSION:
        return
    original=env._fresh
    groups=j.asarray(level()['bun_spawns'],j.int32)
    def native(key):
        ks=jax.random.split(key,13)
        ids=[jax.random.randint(ks[p+1],(),0,4) for p in (0,1)]
        return j.stack([groups[p,ids[p]] for p in (0,1)]).astype(j.float32)+.5
    def fresh(key,*args,**kwargs):
        st=original(key,*args,**kwargs);pos=native(key)
        return st._replace(core=st.core._replace(pos=pos),bun_spawn_pos=pos)
    env._fresh=fresh
    env.SPAWN_SIDE_VERSION=VERSION
    # Keep authorized random brick removal; prefer native four-point groups.
    # The native pair distribution itself satisfies the separation requirement.
    import inspect
    source=inspect.getsource(spawn.sample_jax)
    a='    candidate=jax.random.categorical(kp,j.where(mask.reshape(-1),0.,-j.inf),shape=(32,2))'
    assert source.count(a)==1
    source=source.replace(a,"""    groups=j.asarray(level['bun_spawns'],j.int32)
    side=[j.zeros((H,W),bool).at[groups[p,:,0],groups[p,:,1]].set(True) for p in (0,1)]
    candidate=j.stack([jax.random.categorical(jax.random.fold_in(kp,p),j.where((mask&side[p]).reshape(-1),0.,-j.inf),shape=(32,)) for p in (0,1)],-1)""")
    minimum=int(np.abs(native_pairs()[:,0]-native_pairs()[:,1]).sum(-1).min())
    assert source.count('(j.abs(pair[0]-pair[1]).sum()>=6)')==1
    source=source.replace('(j.abs(pair[0]-pair[1]).sum()>=6)',f'(j.abs(pair[0]-pair[1]).sum()>={minimum})')
    scope=dict(spawn.sample_jax.__globals__)
    exec(compile(source,__file__,'exec'),scope)
    spawn.sample_jax=scope['sample_jax']
    env._own_side_native=native

def install_collector(module):
    """Keep genuine Bank trajectory positions; repair only stored respawn homes."""
    import jax
    import jax.numpy as j
    from jax_bomb import je07_bun_env_r1_items as env
    original=module.fresh
    if getattr(original,'_own_side',False):return
    def fresh(keys,*args,**kwargs):
        st,aux=original(keys,*args,**kwargs)
        homes=jax.vmap(env._own_side_native)(keys)
        bank=aux['provenance_source']>=0
        st=st._replace(bun_spawn_pos=j.where(bank[:,None,None],homes,st.bun_spawn_pos))
        return st,aux
    fresh._own_side=True
    seen=set()
    def replace(m,depth=0):
        if id(m) in seen or depth>4:return
        seen.add(id(m))
        if getattr(m,'fresh',None) is original:m.fresh=fresh
        for name in ('_module','_impl','_base'):
            obj=getattr(m,name,None)
            if obj is not None:replace(obj,depth+1)
    replace(module)
