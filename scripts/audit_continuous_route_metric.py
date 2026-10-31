"""CPU diagnostic of candidate field signal on collision-valid native positions."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_bomb import je05_core_env as core
from scripts.continuous_route_metric import continuous_distance
from scripts.je08_bun_potential import INF,cpu_shortest


def audit():
    level=next(x for x in json.loads(Path('web/assets/maps/levels.json').read_text()) if x['qqt_id']==806)
    wall=np.asarray(level['wall'],bool).reshape(13,15)
    native_brick=np.asarray(level['brick'],bool).reshape(13,15);results=[]
    for geometry,brick in [('open_bricks',np.zeros_like(wall)),('native_bricks',native_brick)]:
        blocked=wall|brick
        for team in [0,1]:
            goal=np.zeros_like(wall);row,column=level['bun_bases'][team];goal[row:row+3,column:column+3]=True
            distance=cpu_shortest(blocked,np.zeros_like(brick),goal);positions=[]
            for cell in np.argwhere(~blocked&(distance>0)&(distance<INF)):
                for dy in [.14,.25,.38,.5,.62,.75,.86]:
                    for dx in [.14,.25,.38,.5,.62,.75,.86]:
                        pos=cell+[dy,dx]
                        if np.any(pos<core.RADIUS+.001) or np.any(pos>np.array([13,15])-core.RADIUS-.001):continue
                        low=np.floor(pos-core.RADIUS+.00001).astype(int)
                        high=np.floor(pos+core.RADIUS-.00001).astype(int)
                        if blocked[low[0]:high[0]+1,low[1]:high[1]+1].any():continue
                        positions.append(pos)
            pos=jnp.asarray(np.array(positions));n=len(pos)
            maps=jnp.broadcast_to(distance,(n,13,15));barriers=jnp.broadcast_to(blocked,(n,13,15))
            before=continuous_distance(pos,maps,barriers)
            for speed in [.5,.6,.65,1.,1.3]:
                endpoints=jax.jit(jax.vmap(lambda p:jax.vmap(lambda move:
                    core._steer(p,move,True,jnp.array(blocked),speed))(jnp.arange(5))))(pos)
                after=continuous_distance(endpoints.reshape(-1,2),jnp.repeat(maps,5,0),jnp.repeat(barriers,5,0)).reshape(n,5)
                gain=np.asarray(before[:,None]-after);failed=np.flatnonzero(gain.max(-1)<=1e-5)
                results.append(dict(geometry=geometry,team=team,speed=speed,states=n,
                    no_positive_native_step=len(failed),examples=[dict(position=np.asarray(pos[i]).tolist(),
                        gains=gain[i].tolist()) for i in failed[:12]]))
    return dict(diagnostic_only=True,controls_training=False,scope='Bun806 both cages, open/native brick geometry, collision-valid sampled subcell positions; no hazards or abilities; no cross-map claim.',
        construction='connected-corner minimum plus half edge, triangular center field and .125 continuous centering cost',results=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');args=parser.parse_args()
    result=audit();Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
