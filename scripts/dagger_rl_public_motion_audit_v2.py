"""Zero-transition native-env public-state projection audit; never binds a live teacher."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import jax
import jax.numpy as j
from jax_bomb import bun_safety
from scripts import combat_item_engine as T, carrier_mechanism_engine as E
from scripts.je10_route_oracle import continuous_distance
from scripts.population_run_state import atomic_json,sha,lock
from scripts.terminal_checkpoint_restore import digest


@jax.jit
def distances(states,aux,kinds):
    # Native env public-state collision/slide/real-speed projection. No
    # sampled enemy action, RNG, model output or future state is available.
    def project(s):
        fuse=j.where(s.core.fuse>0,s.core.fuse-1,s.core.fuse)
        blocked=(fuse>0)|s.core.wall|s.core.brick
        speed=s.core.spd_g[0]*E.env._movement_scale(s)[0]
        return j.stack([E.env.base._steer(s.pos[0],j.where(s.move_status[0]==E.env.STATUS_SLIDE,s.slide_dir[0],m),
            s.alive[0] & (s.trapped[0]==0),blocked,speed,s.core.pushable) for m in range(5)])
    positions=jax.vmap(project)(states)
    channels=j.where((kinds==1)|(kinds==4),0,j.where(kinds==2,2,4))
    field=aux['dagger']['distance'][j.arange(len(kinds)),channels]
    blocked=aux['dagger']['blocked']
    projected=continuous_distance(positions.reshape(-1,2),j.repeat(field,5,0),j.repeat(blocked,5,0)).reshape(-1,5)
    return positions,projected


def run(base):
    base=Path(base);unique=lock(base/'public_motion_audit.lock')
    output=base/'PUBLIC_MOTION_PROJECTION_AUDIT_v2.json'
    assert not output.exists(),'audit is immutable; do not create repeated diagnostic samples'
    protocol=base/'PUBLIC_MOTION_PROJECTION_PROTOCOL_v2.json'
    atomic_json(protocol,dict(source_sha256=sha(Path(__file__)),
        input_audit_sha256=sha(base/'TRAIN_PROJECTION_AUDIT.json'),
        selection='all322 previously frozen endpoint rows, no outcome selection; ability0 only',
        native_environment_steps=0,optimizer_steps=0,adopted_training_steps=0,
        question='Can the native env collision/slide/absolute-speed projection reproduce the stored real '
                 'ability0 one-step positions/distances without knowing sampled enemy actions or future RNG?',
        limits='Endpoint diagnostic, not adopted-label proof, ability1/2 parity, future safety or Q. '
               'No candidate is bound in R16/R17.'))
    old=json.loads((base/'TRAIN_PROJECTION_AUDIT.json').read_text());T.configure()
    grouped=defaultdict(list)
    for row in old['rows']:grouped[(row['checkpoint'],row['stream'])].append(row)
    rows=[]
    for (checkpoint,stream),group in grouped.items():
        cp=base/checkpoint;m=json.loads((cp/'COMMITTED.json').read_text())
        assert sha(cp/'state.pkl')==m['state_sha256']
        x=pickle.load((cp/'state.pkl').open('rb'));x=x if stream=='PPO' else x['auxiliary']
        flatten=lambda tree:jax.tree.map(lambda a:np.asarray(a).reshape((-1,)+a.shape[2:]),tree)
        states,aux=map(flatten,[x['states'],x['aux']])
        ids=np.array([r['env_index'] for r in group]);pad=np.pad(ids,(0,16-len(ids)),mode='edge')
        take=lambda tree:jax.tree.map(lambda a:a[pad],tree)
        kinds=np.pad(np.array([r['kind'] for r in group],np.int32),(0,16-len(ids)),mode='edge')
        _,projected=map(np.asarray,distances(take(states),take(aux),kinds))
        for i,row in enumerate(group):
            assert digest(jax.tree.map(lambda a:a[ids[i]:ids[i]+1],states))==row['state_hash']
            native=np.asarray(row['native_distance_after']);delta=np.abs(projected[i]-native)
            allowed=np.asarray([m in row['safe_ability0'] for m in range(5)])
            native_best=np.flatnonzero(allowed&(native<=np.min(np.where(allowed,native,np.inf))+1e-5)).tolist()
            prediction_best=np.flatnonzero(allowed&(projected[i]<=np.min(np.where(allowed,projected[i],np.inf))+1e-5)).tolist()
            rows.append(dict(checkpoint=checkpoint,stream=stream,env_index=int(ids[i]),state_hash=row['state_hash'],
                kind=row['kind'],route_adoptable=row['route_adoptable'],
                native_distance=native.tolist(),public_projection_distance=projected[i].tolist(),
                max_absolute_error=float(delta.max()),native_best=native_best,predicted_best=prediction_best,
                best_set_exact=native_best==prediction_best))
    assert len(rows)==len(old['rows'])==322
    atomic_json(output,dict(passed=True,protocol_sha256=sha(protocol),rows=rows,
        max_absolute_distance_error=max(x['max_absolute_error'] for x in rows),
        best_sets_exact=sum(x['best_set_exact'] for x in rows),
        native_environment_steps=0,public_position_projections=len(rows)*5,
        optimizer_steps=0,adopted_training_steps=0,source_sha256=sha(Path(__file__))))
    print(json.dumps(dict(rows=len(rows),max_error=max(x['max_absolute_error'] for x in rows),
        best_sets_exact=sum(x['best_set_exact'] for x in rows))),flush=True)


if __name__=='__main__':run(sys.argv[1])
