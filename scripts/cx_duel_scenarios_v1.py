"""Declared artificial, empty-handed native scenarios; disjoint train/dev/holdout."""
from pathlib import Path
import json
import numpy as np

ROOT = Path('runs/cc_train_20261105/training/duel_20261107')
CATALOG = ROOT / 'SCENARIOS.json'
FAMILIES = ('open_chase', 'narrow_block', 'crossroads', 'corner_pillar',
            'brick_break', 'multi_exit', 'staggered_chain', 'items_slide')
SPLIT_SEEDS = dict(train=2026711000, dev=2026715000, holdout=2026719000)


def neighbors(cell):
    r, c = cell
    return [(r+dr,c+dc) for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
            if 0 <= r+dr < 13 and 0 <= c+dc < 15]


def construct(family, seed, variant):
    rng = np.random.default_rng(seed)
    wall = np.zeros((13,15),bool)
    wall[[0,-1],:] = True; wall[:,[0,-1]] = True
    brick = np.zeros_like(wall)
    if family == 'narrow_block':
        wall[3,2:13] = True; wall[9,2:13] = True
        wall[3,[4,10]] = False; wall[9,[4,10]] = False
    elif family == 'crossroads':
        wall[2:11,3] = True; wall[2:11,11] = True
        wall[[4,6,8],3] = False; wall[[4,6,8],11] = False
        wall[3,4:11] = True; wall[9,4:11] = True
        wall[3,[6,8]] = False; wall[9,[6,8]] = False
    elif family == 'corner_pillar':
        wall[4:7,6:9] = True
        wall[8,3:6] = True
    elif family == 'multi_exit':
        wall[4,3:12] = True; wall[8,3:12] = True
        wall[4,[4,7,10]] = False; wall[8,[4,7,10]] = False
    elif family == 'brick_break':
        brick[2:11,7] = True; brick[[4,8],7] = False
    elif family in ('staggered_chain','items_slide'):
        wall[4,6:9] = True; wall[8,6:9] = True
    density = (variant%3)*.035
    brick |= (rng.random((13,15)) < density) & ~wall
    # Vary range/direction; all spawns have at least two unobstructed exits.
    distance = (2,4,6,8)[variant%4]
    r = int(rng.choice([5,6,7])); left = 7-distance//2
    spawn = np.array([[r,left],[r, left+distance]],np.int32)
    for cell in spawn:
        for p in [tuple(cell),*neighbors(cell)]:
            wall[p]=False; brick[p]=False
    if variant&4:
        wall=wall[::-1].copy();brick=brick[::-1].copy();spawn[:,0]=12-spawn[:,0]
    if variant&8:
        wall=wall[:,::-1].copy();brick=brick[:,::-1].copy();spawn[:,1]=14-spawn[:,1]
    if variant&1: spawn=spawn[::-1].copy()
    crate=np.zeros((13,15),np.int8)
    available=np.argwhere(~wall&~brick)
    eligible=[tuple(p) for p in available if min(abs(p-spawn).sum(-1))>=2]
    rng.shuffle(eligible)
    # Production growth pickups: capacity/fire/speed; optional tactical slots.
    for i,cell in enumerate(eligible[:3]):crate[cell]=i+1
    fuse=np.zeros((13,15),np.int16);blast=np.zeros_like(fuse)
    if family=='staggered_chain':
        remote=[p for p in eligible[3:] if min(abs(np.asarray(p)-spawn).sum(-1))>=5]
        for i,cell in enumerate(remote[:2]):fuse[cell]=24+9*i;blast[cell]=2
    caps=np.array([[1+variant%3,1+(variant//2)%3,1.+.1*(variant%3)],
                   [1+(variant+1)%3,1+(variant//3)%3,1.+.1*((variant+1)%3)]],np.float32)
    slots=np.zeros((2,7),np.int8);counts=np.zeros((2,7),np.int16)
    if family=='items_slide':
        slots[:,0]=[1,2] if variant%2 else [2,1];counts[:,0]=1
    heading=np.array([variant%4,(variant+2)%4],np.int8)
    return dict(seed=seed,family=family,variant=variant,artificial=True,
                wall=wall.tolist(),brick=brick.tolist(),crate=crate.tolist(),
                spawn=spawn.tolist(),fuse=fuse.tolist(),blast=blast.tolist(),
                stats=caps.tolist(),slots=slots.tolist(),counts=counts.tolist(),heading=heading.tolist())


def validate(row):
    wall=np.array(row['wall']);brick=np.array(row['brick']);spawn=np.array(row['spawn'])
    assert wall.shape==brick.shape==(13,15) and not (wall&brick).any()
    assert not np.array_equal(*spawn)
    for cell in spawn:
        assert not (wall|brick)[tuple(cell)]
        assert sum(not (wall|brick)[p] for p in neighbors(cell))>=2
    assert np.array(row['stats'])[:,0].min()>=1 and np.array(row['stats'])[:,0].max()<=3
    assert np.array(row['stats'])[:,1].max()<=3
    # Every initial hazard has >20 ticks and is >=5 cells from both spawns.
    for p in np.argwhere(np.array(row['fuse'])>0):
        assert np.array(row['fuse'])[tuple(p)]>=24
        assert min(abs(p-spawn).sum(-1))>=5


def prepare():
    assert not CATALOG.exists(), 'immutable scenario catalog already exists'
    rows=[]
    for split,base in SPLIT_SEEDS.items():
        for k,family in enumerate(FAMILIES):
            for variant in range(16 if split=='train' else 4):
                row=construct(family,base+k*100+variant,variant)
                validate(row);rows.append(dict(row,split=split,id=len(rows)))
    ROOT.mkdir(parents=True,exist_ok=True)
    CATALOG.write_text(json.dumps(dict(version='duel-scenarios-v1',rows=rows,
        difficulty='all ranges retained; reset index progressively favors harder variants during first200 updates'),separators=(',',':'))+'\n')
    return len(rows)

if __name__=='__main__': print(prepare())
