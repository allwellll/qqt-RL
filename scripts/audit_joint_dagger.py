"""Read-only joint FIFO fit and exact-input conflicts after fixed COMPLETE.

No rollouts, evaluation-to-training import or GPU control. A finite FIFO is
not a whole-trajectory coverage estimate; fit is current-policy in-sample fit.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np


def conflict_groups(obs,gv,move_mask,ability_mask,target,valid,limit=16):
    accepted=target&move_mask[:,:,None]&ability_mask[:,None,:]
    valid=valid&accepted.any((-2,-1));groups={}
    for i in np.flatnonzero(valid):
        digest=hashlib.sha256()
        for value in [obs[i],gv[i],move_mask[i],ability_mask[i]]:
            a=np.ascontiguousarray(value)
            digest.update(str((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
        groups.setdefault(digest.hexdigest(),[]).append(int(i))
    duplicated=conflicts=conflict_rows=0;examples=[]
    for digest,indices in groups.items():
        if len(indices)<2:continue
        duplicated+=1
        if accepted[indices].all(0).any():continue
        conflicts+=1;conflict_rows+=len(indices)
        if len(examples)<limit:examples.append(dict(actor_input_and_masks_sha256=digest,indices=indices,
            joint_sets=[np.argwhere(accepted[i]).tolist() for i in indices]))
    return dict(valid_rows=int(valid.sum()),duplicate_input_groups=duplicated,
        no_common_pair_groups=conflicts,no_common_pair_rows=conflict_rows,examples=examples,
        scope='Exact Actor observation/global/masks; no common accepted joint pair across duplicate labels, not a geometry-only or population conflict claim')


def audit(base,generation):
    import jax
    import jax.numpy as jnp
    from jax_bomb.jax_net import net_forward
    root=Path(__file__).resolve().parents[1];base=Path(base)
    if base.is_absolute():base=base.relative_to(root)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and jax.default_backend()=='cpu'
    out=base/f'authority_je10/g{generation:06d}';proof=json.loads((out/'COMPLETE.json').read_text())
    assert hashlib.sha256((out/'SUMMARY.json').read_bytes()).hexdigest()==proof['shards']['SUMMARY.json']
    cp=base/f'formal/bun392_bridge_pfsp8/gen{generation:06d}'
    marker=json.loads((cp/'COMMITTED.json').read_text())
    with (cp/'state.pkl').open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest();assert digest==marker['state_sha256']
        stream.seek(0);saved=pickle.load(stream)
    actor=jax.tree.map(lambda x:jnp.asarray(x[0]),saved['live']['params']['actor'])
    raw=saved['live']['dagger_buffer'];del saved
    required=['obs','gv','legal','ability_legal','joint_target','valid','kind','preparation','revision','objective','opening']
    buffer={name:np.asarray(raw[name]).reshape((-1,)+raw[name].shape[2:]) for name in required}
    del raw
    conflicts=conflict_groups(buffer['obs'],buffer['gv'],buffer['legal'],buffer['ability_legal'],buffer['joint_target'],buffer['valid'])
    indices=np.flatnonzero(buffer['valid']);fit={}

    @jax.jit
    def predict(obs,gv,mm,am,target):
        move,ability,_,_=net_forward(actor,'transformer',obs,gv)
        pm=jax.nn.softmax(jnp.where(mm,move,-1e9),-1)
        pa=jax.nn.softmax(jnp.where(am,ability,-1e9),-1)
        accepted=target&mm[:,:,None]&am[:,None,:]
        mass=(pm[:,:,None]*pa[:,None,:]*accepted).sum((-2,-1))
        match=accepted[jnp.arange(obs.shape[0]),pm.argmax(-1),pa.argmax(-1)]
        return mass,match

    masses=[];matches=[]
    for start in range(0,len(indices),64):
        ids=indices[start:start+64];size=len(ids);ids=np.pad(ids,(0,64-size),mode='edge')
        values=[buffer[name][ids] for name in ['obs','gv','legal','ability_legal','joint_target']]
        mass,match=predict(*values);masses.extend(np.asarray(mass)[:size].tolist());matches.extend(np.asarray(match)[:size].tolist())
    masses=np.asarray(masses);matches=np.asarray(matches)
    groups=dict(all=np.ones(len(indices),bool),ordinary=~buffer['preparation'][indices],
        preparation_navigation=buffer['preparation'][indices]&~buffer['opening'][indices],opening=buffer['opening'][indices])
    for kind in [1,2,3,4]:groups['kind_'+str(kind)]=buffer['kind'][indices]==kind
    for name,selected in groups.items():fit[name]=dict(count=int(selected.sum()),greedy_in_joint=int(matches[selected].sum()),
        mean_joint_target_probability=float(masses[selected].mean()) if selected.any() else None)
    result=dict(generation=generation,diagnostic_only=True,checkpoint_state_sha256=digest,fit=fit,conflicts=conflicts,
        limitations=['FIFO snapshot is bounded and changes distribution; no full trajectory or same-distribution forgetting claim.',
            'Current Actor fit is in-sample, not query-time policy or holdout performance.',
            'Exact-input no-common-pair groups require native target/safety identity review; no automatic relabeling.'])
    (out/'DAGGER_JOINT_COVERAGE.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('generation',type=int)
    args=parser.parse_args();result=audit(args.run,args.generation)
    print(json.dumps(dict(generation=result['generation'],fit=result['fit'],conflicts=result['conflicts']['no_common_pair_groups'])))
