"""Read-only legacy return-label compatibility, no native safety claim.

Reconstruct static geometry from saved Actor inputs. These records have no
complete native state, so nominal endpoint comparisons cannot replace the
production native certificate or relabel training data.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path


def audit(run,generation):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scripts.je08_bun_potential import shortest,INF
    from scripts.je10_route_oracle import action_set,continuous_distance,DELTA
    from jax_bomb.jax_net import net_forward
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and jax.default_backend()=='cpu'
    run=Path(run);assert not run.is_absolute()
    assert (run/f'authority_je10/g{generation:06d}/COMPLETE.json').exists()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    replay=run/'inputs/return_replay.pkl';policy=run/f'formal/bun392_bridge_pfsp8/gen{generation:06d}/policy.pt'
    marker=json.loads((policy.parent/'COMMITTED.json').read_text());assert sha(policy)==marker['policy_sha256']
    with replay.open('rb') as f:data=pickle.load(f)
    with policy.open('rb') as f:actor=jax.tree.map(jnp.asarray,pickle.load(f))
    @jax.jit
    def compare(obs,legal):
        coords=jnp.stack(jnp.meshgrid(jnp.arange(13)+.5,jnp.arange(15)+.5,indexing='ij'),-1)
        pos=(obs[:,0,:,:,None]*coords).sum((1,2))
        blocked=(obs[:,4]>0)|(obs[:,13]>0)
        goal=obs[:,14]>0
        distance=shortest(blocked,jnp.zeros_like(blocked),goal)
        danger=obs[:,5];fuse=(obs[:,1]+obs[:,3])>0
        target,active=action_set(pos,distance,legal,danger,blocked,fuse)
        cell=jnp.clip(jnp.floor(pos).astype(jnp.int32),jnp.array([0,0]),jnp.array([12,14]))
        rows=jnp.arange(len(obs));at_goal=(distance[rows,cell[:,0],cell[:,1]]==0)&legal[:,4]
        at_goal&=(danger[rows,cell[:,0],cell[:,1]]==0)&~fuse[rows,cell[:,0],cell[:,1]]
        target=jnp.where(at_goal[:,None],jax.nn.one_hot(jnp.full(len(obs),4),5),target)
        before=continuous_distance(pos,distance,blocked)
        endpoints=pos[:,None]+.18*DELTA
        after=continuous_distance(endpoints.reshape(-1,2),jnp.repeat(distance,5,0),jnp.repeat(blocked,5,0)).reshape(-1,5)
        return target,active|at_goal,before[:,None]-after,pos,blocked
    @jax.jit
    def predict(obs,gv,legal):
        move=net_forward(actor,'transformer',obs,gv)[0]
        return jax.nn.softmax(jnp.where(legal,move,-1e9),-1)
    result=dict(diagnostic_only=True,controls_training=False,generation=generation,
        source_hashes={str(p):sha(p) for p in [Path(__file__),Path('scripts/je10_route_oracle.py'),Path('scripts/continuous_route_metric.py')]},
        checkpoint_sha256=sha(policy),replay_sha256=sha(replay),groups={},
        limits=['Frozen Actor inputs have no full native state or provenance ledger; no certified native endpoint or time-expanded safety comparison.',
            'Current nominal .18 oracle and static topology are descriptive, not a proposed automatic relabel or learned Q.',
            'All immutable legacy rows are analyzed; Actor fit is in-sample on historical train-only data.',
            'Geometry overlap with fixed Review is diagnosis only; no fixed evaluation sample enters training.'])
    exported=run/f'REVIEW_G{generation}_ACTOR_OBSERVATIONS.json'
    reference=json.loads(exported.read_text()) if exported.exists() else dict(rows=[])
    for name,(obs,gv,legal,old) in data.items():
        obs,gv,legal,old=map(np.asarray,(obs,gv,legal,old));n=len(obs)
        assert np.allclose(obs[:,0].sum((1,2)),1.,atol=1e-5) and (gv[:,13]>.5).all()
        target,active,gain,pos,blocked=map(np.asarray,compare(jnp.asarray(obs),jnp.asarray(legal)))
        old_set=(old>0)&legal;new_set=target>0;intersection=(old_set&new_set).any(-1)
        disjoint=active&old_set.any(-1)&~intersection
        negative=active&(old_set&(gain< -1e-5)).any(-1)
        probability=[]
        for start in range(0,n,64):
            stop=min(start+64,n);pad=64-(stop-start)
            block=lambda x:np.pad(x[start:stop],[(0,pad)]+[(0,0)]*(x.ndim-1),mode='edge')
            probability.append(np.asarray(predict(block(obs),block(gv),block(legal)))[:stop-start])
        probability=np.concatenate(probability);greedy=probability.argmax(-1);rows=np.arange(n)
        group=dict(rows=n,current_nominal_active=int(active.sum()),disjoint_sets=int(disjoint.sum()),
            legacy_contains_negative_nominal_move=int(negative.sum()),
            greedy_in_legacy=int(old_set[rows,greedy].sum()),greedy_in_current_active=int((new_set[rows,greedy]&active).sum()),
            mean_legacy_probability=float((probability*old_set).sum(-1).mean()),
            mean_current_probability_on_active=float((probability[active]*new_set[active]).sum(-1).mean()) if active.any() else None,
            disjoint_rows=[dict(index=int(i),position=pos[i].tolist(),legacy_set=np.flatnonzero(old_set[i]).tolist(),
                nominal_set=np.flatnonzero(new_set[i]).tolist(),nominal_gains=gain[i].tolist()) for i in np.flatnonzero(disjoint)],
            fixed_geometry_overlap=[])
        for ref in reference['rows']:
            q=np.array(ref['obs']).reshape(24,13,15)
            same=(blocked==((q[4]>0)|(q[13]>0))).all((1,2))
            same&=(obs[:,14:16]==q[14:16]).all((1,2,3))
            same&=(gv[:,13:15]==np.array(ref['gv'])[13:15]).all(-1)
            same&=np.abs(pos-ref['position']).sum(-1)<=1.
            group['fixed_geometry_overlap'].append(dict(task=ref['task'],tick=ref['tick'],count=int(same.sum()),
                legacy_target_mass=old[same].sum(0).tolist(),nominal_target_mass=target[same].sum(0).tolist(),
                disjoint=int((same&disjoint).sum())))
        result['groups'][name]=group
    target=run/f'FROZEN_RETURN_REPLAY_G{generation}_AUDIT.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'groups':{k:{n:v for n,v in d.items() if n!='disjoint_rows'} for k,d in result['groups'].items()}}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('generation',type=int)
    args=parser.parse_args();audit(args.run,args.generation)
