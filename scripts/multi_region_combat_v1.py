"""CPU preparation of legal starts, without altering map or reward rules."""
from collections import deque
import numpy as np

REGIONS=('top_left','top_right','bottom_left','bottom_right','center','edge','choke','open')
DISTANCES=(2,4,6)


def primary_region(row,col):
    if 4<=row<=8 and 5<=col<=9:return 'center'
    return ('top_' if row<6 else 'bottom_')+('left' if col<7 else 'right')


def labels(opened,cell):
    row,col=divmod(int(cell),15)
    exits=sum(0<=r<13 and 0<=c<15 and opened[r*15+c]
              for r,c in ((row-1,col),(row+1,col),(row,col-1),(row,col+1)))
    tags={primary_region(row,col)}
    if row in (1,11) or col in (1,13):tags.add('edge')
    if exits<=2:tags.add('choke')
    if exits>=3:tags.add('open')
    return tags


def legal_pairs(wall,brick,region,distance):
    if region not in REGIONS or distance not in DISTANCES:raise ValueError('unknown frozen stratum')
    opened=~(np.asarray(wall,bool).reshape(195)|np.asarray(brick,bool).reshape(195))
    candidates=[int(c) for c in np.flatnonzero(opened) if region in labels(opened,c)]
    pairs=[]
    for a in candidates:
        queue=deque([(a,0)]);seen={a}
        while queue:
            cell,d=queue.popleft()
            if d==distance:
                if cell in candidates and cell!=a:pairs.append((a,cell))
                continue
            row,col=divmod(cell,15)
            for r,c in ((row-1,col),(row+1,col),(row,col-1),(row,col+1)):
                n=r*15+c
                if 0<=r<13 and 0<=c<15 and opened[n] and n not in seen:
                    seen.add(n);queue.append((n,d+1))
    return pairs


def sample_manifest(wall,brick,seed,games_per_stratum=16):
    if games_per_stratum%4:raise ValueError('four-way role/position balance requires a multiple of4')
    rng=np.random.default_rng(seed);rows=[];missing=[]
    for region in REGIONS:
        for distance in DISTANCES:
            pairs=legal_pairs(wall,brick,region,distance)
            if not pairs:
                missing.append(dict(region=region,distance=distance,reason='no legal reachable pair in unchanged map'))
                continue
            # Four-way teams/positions: each physical point occurs as both
            # learner roles and both displacement directions. No map rotation.
            for i in range(games_per_stratum//4):
                axis=i%2
                choices=[(a,b) for a,b in pairs if abs(a//15-b//15)>abs(a%15-b%15)] if axis==0 else [(a,b) for a,b in pairs if abs(a//15-b//15)<=abs(a%15-b%15)]
                if not choices:choices=pairs
                a,b=choices[int(rng.integers(len(choices)))]
                for flip,seat in ((False,0),(False,1),(True,0),(True,1)):
                    pair=(b,a) if flip else (a,b)
                    own,other=pair[seat],pair[1-seat]
                    dy,dx=other//15-own//15,other%15-own%15
                    heading=('south' if dy>0 else 'north') if abs(dy)>abs(dx) else ('east' if dx>0 else 'west')
                    rows.append(dict(seed=int(seed+len(rows)),region=region,distance=distance,
                        learner_seat=seat,cells=[list(divmod(x,15)) for x in pair],
                        relative_heading=heading,initiative='simultaneous native step',
                        orientation='relative displacement; no invented facing field',
                        no_preplaced_bombs=True,geometry_augmentation=False))
    return dict(schema='multi-region-combat-starts-v1',seed=seed,rows=rows,missing=missing,
        reward_changes=False,rule='bun-native-v2',scope='combat starts; not natural-game capability results')
