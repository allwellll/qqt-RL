"""CPU-only proposal coverage on an immutable learner checkpoint, no episodes.

Uses the fixed learner seat (environment index modulo two) and deterministic
device/environment order. This is a rollout-end snapshot, not a time-weighted
population estimate or a counterfactual learned advantage.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path


def audit(run,generation,limit=64,witness_limit=4,native_endpoints=False):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scripts.population_run_state import sha,atomic_json
    from scripts.prestock_route_preparation import native_plan,native_inputs,movement_labels
    from scripts.frontier_native_witness import placement_witness,HORIZON
    from jax_bomb import je07_bun_env as env
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and jax.default_backend()=='cpu'
    run=Path(run);assert not run.is_absolute()
    assert (run/f'authority_je10/g{generation:06d}/COMPLETE.json').exists()
    cp=run/f'formal/bun392_bridge_pfsp8/gen{generation:06d}'
    committed=json.loads((cp/'COMMITTED.json').read_text())
    assert committed['passed'] and committed['generation']==generation
    assert sha(cp/'state.pkl')==committed['state_sha256']
    with (cp/'state.pkl').open('rb') as stream:saved=pickle.load(stream)
    states=saved['live']['states'];aux=saved['live']['aux']
    devices,n=states.alive.shape[:2];del saved
    states=jax.tree.map(lambda x:np.asarray(x).reshape((devices*n,)+x.shape[2:]),states)
    aux=jax.tree.map(lambda x:np.asarray(x).reshape((devices*n,)+x.shape[2:]),aux)
    rows=np.arange(devices*n);seat=rows%n%2;enemy=1-seat
    eligible=(states.scene!=0)&~aux['terminal']&states.alive[rows,seat]
    eligible&=(states.trapped[rows,seat]==0)&(states.bun_carried[rows,seat]<0)
    eligible&=(~states.alive[rows,enemy])|(states.trapped[rows,enemy]>0)
    eligible&=states.bun_stored[rows,enemy,enemy]>0
    eligible&=~aux['dagger']['loose'][rows,seat].any((-2,-1))
    selected=np.flatnonzero(eligible)[:limit]
    output=dict(schema='je10.prestock-snapshot-proposal/v1',generation=generation,
        diagnostic_only=True,controls_training=False,checkpoint=str(cp),
        checkpoint_state_sha256=committed['state_sha256'],learner_states=len(rows),
        eligible_empty_native_stock=int(eligible.sum()),selected_count=len(selected),
        selection='First eligible device/environment indices, fixed learner seat env%2; no successful-state selection',
        source_hashes={name:sha(name) for name in ['scripts/audit_prestock_snapshot.py','scripts/prestock_route_preparation.py',
            'scripts/route_frontier.py','scripts/frontier_native_witness.py']},rows=[],
        limits=['Rollout-end states only; not all learner-visited transitions or an unbiased coverage rate',
            'Capped deterministic geometric labels are not learned policy performance',
            'No new episode or Review chain; native witness is a proposal fork of saved training state',
            'Finite escape search and fixed RNG; not a certificate against an active enemy or all RNG'])
    if len(selected):
        st=jax.tree.map(lambda x:jnp.asarray(x[selected]),states)
        ax=jax.tree.map(lambda x:jnp.asarray(x[selected]),aux)
        seats=jnp.asarray(seat[selected]);result=jax.jit(native_plan)(st,ax,seats)
        mm,_=jax.vmap(env.legal_mask)(st)
        inputs=native_inputs(st,ax,seats)
        hazard=(st.core.blast_linger>0).astype(jnp.int32)
        target,active,_=jax.jit(movement_labels)(*inputs,mm[jnp.arange(len(selected)),seats],hazard,st.fuse)
        native=None
        if native_endpoints:
            from scripts.native_route_certificate import certify_native_positions
            count=len(selected);rr=jnp.arange(count)
            joint=jnp.zeros((count,5,2,2),jnp.int32).at[:,:,:,0].set(4)
            joint=joint.at[rr[:,None],jnp.arange(5)[None,:],seats[:,None],0].set(jnp.arange(5)[None,:])
            repeated=jax.tree.map(lambda x:jnp.repeat(x,5,0),st)
            keys=jax.random.split(jax.random.PRNGKey(6106),count*5)
            step=jax.jit(jax.vmap(lambda s,a,k:env.step(s,a,k,auto_reset=False,return_info=True)))
            after,done,info=step(repeated,joint.reshape(count*5,2,2),keys)
            after_seat=jnp.repeat(seats,5);ids=jnp.arange(count*5)
            safe=(after.alive[ids,after_seat]&(after.trapped[ids,after_seat]==0)
                &~info['own_bomb_defeat'][ids,after_seat]&~done).reshape(count,5)
            # During preparation, a sped-up native step must not accidentally
            # acquire inventory and thereby lose the ability to continue opening.
            empty=(after.bun_carried[ids,after_seat]<0).reshape(count,5)
            safe&=~result.preparing[:,None]|empty
            pos=after.pos[ids,after_seat].reshape(count,5,2)
            before_blocked=inputs[1]|st.brick
            before_blocked|=inputs[4]&(result.objective==2)[:,None,None]
            after_blocked=(after.wall|after.brick|after.core.pushable).reshape(count,5,13,15)
            # Stock avoidance is a planning constraint, not changed terrain.
            physical=inputs[1]|st.brick
            stable=jnp.all(after_blocked==physical[:,None],(-2,-1))
            native_target,native_valid=certify_native_positions(inputs[0],pos,result.distance,
                before_blocked,mm[rr,seats],safe,jnp.zeros(count,bool),stable)
            native=dict(target=native_target,active=native_valid,safe=safe,
                lost_empty=~empty&result.preparing[:,None])
        witness=jax.jit(placement_witness);witness_count=0
        for k,index in enumerate(selected):
            state=jax.tree.map(lambda x:x[k],st);s=int(seats[k]);e=1-s
            digest=hashlib.sha256()
            for leaf in jax.tree.leaves(state):
                array=np.asarray(leaf);digest.update(str((array.shape,array.dtype)).encode());digest.update(array.tobytes())
            cell=np.floor(np.asarray(state.pos[s])).astype(int)
            staged=bool(result.staging[k,cell[0],cell[1]])
            clear=bool((state.core.owner<0).all()&(state.fuse==0).all()
                &(state.core.blast_linger==0).all()&(state.core.brick_linger==0).all())
            long_window=bool((~state.alive[e]&(state.bun_respawn[e]>HORIZON))
                |(state.alive[e]&(state.trapped[e]>HORIZON)))
            row=dict(device=int(index//n),environment=int(index%n),seat=s,scene=int(state.scene),
                state_sha256=digest.hexdigest(),ready=bool(result.ready[k]),preparing=bool(result.preparing[k]),
                objective=int(result.objective[k]),staged=staged,clear=clear,long_window=long_window,
                first_bricks=np.argwhere(np.asarray(result.first_bricks[k])).tolist(),
                movement_active=bool(active[k]),movement_set=np.flatnonzero(np.asarray(target[k])>0).tolist())
            if native is not None:
                row.update(native_movement_active=bool(native['active'][k]),
                    native_movement_set=np.flatnonzero(np.asarray(native['target'][k])>0).tolist(),
                    native_early_pickup_moves=np.flatnonzero(np.asarray(native['lost_empty'][k])).tolist())
            if staged and clear and long_window and witness_count<witness_limit:
                goals=inputs[3][k] if row['objective']==2 else inputs[4][k]
                found=witness(state,jnp.int32(s),goals,result.first_bricks[k])
                row['opening_joint_set']=np.argwhere(np.asarray(found.joint_set)).tolist()
                row['opening_witnesses']=int(found.accepted.sum());witness_count+=1
            output['rows'].append(row)
        from collections import Counter
        output['selected_objectives']=dict(Counter(str(r['objective']) for r in output['rows']))
        output['selected_ready']=sum(r['ready'] for r in output['rows'])
        output['selected_movement_active']=sum(r['movement_active'] for r in output['rows'])
        output['selected_staged_clear_long']=sum(r['staged'] and r['clear'] and r['long_window'] for r in output['rows'])
        output['witness_queries']=witness_count
        output['witness_accepted_states']=sum(r.get('opening_witnesses',0)>0 for r in output['rows'])
        if native is not None:
            output['native_movement_active']=int(native['active'].sum())
            output['native_early_pickup_states']=int(native['lost_empty'].any(-1).sum())
            output['nominal_native_set_disagreements']=sum(r['movement_set']!=r['native_movement_set'] for r in output['rows'])
            output['native_fork_scope']='Five one-step moves from each same fixed snapshot state, opponent waits, ability zero, RNG6106; no new episode, policy performance or failure recording; preparation rejects any premature pickup'
    suffix='NATIVE_AUDIT' if native_endpoints else 'AUDIT'
    target=run/f'PRESTOCK_SNAPSHOT_G{generation}_{suffix}.json'
    atomic_json(target,output)
    print(json.dumps({k:v for k,v in output.items() if k!='rows'}),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('generation',type=int)
    parser.add_argument('--native-endpoints',action='store_true')
    args=parser.parse_args();audit(args.run,args.generation,native_endpoints=args.native_endpoints)
