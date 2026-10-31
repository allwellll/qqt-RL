"""Bounded passive native reward/ability audit from real saved student states.

No optimization, teacher action, reset, fabricated carrier state or reward edit.
Ended episodes are frozen; unfinished episodes and pending credits are censored.
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

from scripts.population_run_state import atomic_json,lock,sha

ROOT=Path(__file__).resolve().parents[1]


def audited_settle(original):
    def settle(e,s):
        import jax.numpy as j
        after,pay,metrics=original(e,s)
        valid=e['active']&~e['bad']&~e['terminal']
        amount=.02*metrics['threat_events']+.01*metrics['chain_events']+.04*metrics['brick_events']
        capped=j.minimum(s['pending']+amount,j.maximum(0.,.20-s['gross']))
        admitted=j.where(valid,capped-s['pending'],0.)
        cancelled=j.where(~valid,s['pending'],0.)
        audit=dict(pending_before=s['pending'],pending_after=after['pending'],admitted=admitted,
            offered=amount,clipped=j.where(valid,j.maximum(0.,s['pending']+amount-capped),0.),
            cancelled=cancelled,cancel_terminal=j.where(e['terminal'],s['pending'],0.),
            cancel_bad=j.where(~e['terminal']&e['bad'],s['pending'],0.),
            cancel_inactive=j.where(~e['terminal']&~e['bad']&~e['active'],s['pending'],0.),
            paid=pay,settled=(e['settled']&valid).astype(j.int32),
            placed=e['placed'].astype(j.int32),safe_placed=(e['placed']&e['safe']).astype(j.int32),
            first_safe_threat=metrics['threat_events'],chain_events=metrics['chain_events'],
            topology_events=metrics['brick_events'],positive_payment=(pay>0).astype(j.int32),
            balance_residual=s['pending']+admitted-cancelled-pay-after['pending'])
        return after,pay,dict(metrics,audit=audit)
    return settle


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('manifest');args=parser.parse_args()
    manifest=Path(args.manifest);config=json.loads(manifest.read_text());out=ROOT/config['output']
    out.mkdir(parents=True,exist_ok=True);guard=lock(out/'worker.lock')
    if (out/'COMPLETE.json').exists():return
    for file,expected in config['source_sha256'].items():assert sha(ROOT/file)==expected,file
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_cpu_multi_thread_eigen=false',JAXBOMB_RULE='bun')
    import jax
    import jax.numpy as j
    import numpy as np
    from scripts import ppo_scale_longrun_train as P,bomb_threat_collector as M,bomb_threat_reward as R
    from scripts.terminal_checkpoint_restore import digest
    from scripts.bomb_threat_runtime import install
    P.T.configure();P.C.install_provenance();install()
    original=R.settle;started=time.time();results=[]
    for item in config['checkpoints']:
        name=item['name'];target=out/name;target.mkdir(exist_ok=True)
        if (target/'RESULT.json').exists():
            results.append(json.loads((target/'RESULT.json').read_text()));continue
        cp=ROOT/item['checkpoint'];commit=json.loads((cp/'COMMITTED.json').read_text())
        assert sha(cp/'state.pkl')==commit['state_sha256'] and sha(cp/'policy.pt')==commit['policy_sha256']
        payload=pickle.load((cp/'state.pkl').open('rb'));n=payload['states'].t.shape[1]
        ids=np.linspace(0,n-1,int(config['environments_per_checkpoint']),dtype=int)
        select=lambda tree:jax.tree.map(lambda a:j.asarray(a[0,ids]),tree)
        st,ax,keys=select(payload['states']),select(payload['aux']),select(payload['keys'])
        params=jax.tree.map(j.asarray,payload['params']);fixed=jax.tree.map(j.asarray,payload['combat_fixed_actor'])
        ring=jax.tree.map(j.asarray,payload['combat_ring'][(payload['update']-160)%4])
        kinds=np.asarray(ax['combat_mix']['kind']);groups={name:np.flatnonzero(kinds==i)
            for i,name in [(3,'current'),(4,'ring'),(5,'fixed')]};groups['bot']=np.flatnonzero(kinds<3)
        protocol=json.loads((ROOT/'runs/bomb_threat_v1/protocols'/(cp.parent.name+'.json')).read_text())
        recipe=protocol['arms'][cp.parent.name];coefficient=recipe['combat_shaping']
        assert not recipe.get('zero_stock_shaping'), 'this diagnostic is for unchanged shaped source ledgers'
        # Exact passive equivalence is tested on the same CPU state/action/RNG,
        # not claimed as CPU/GPU numerical trajectory parity.
        nk,ke,act,data=M.choose(params,ring,fixed,st,ax,keys,groups)
        R.settle=original
        before=jax.jit(lambda s,a,ac,k:M.transition(s,a,ac,k,coefficient))(st,ax,act,ke)
        jax.block_until_ready(before)
        R.settle=audited_settle(original)
        after=jax.jit(lambda s,a,ac,k:M.transition(s,a,ac,k,coefficient))(st,ax,act,ke)
        jax.block_until_ready(after)
        assert digest(before[:6])==digest(after[:6])
        assert digest(before[6])==digest({k:v for k,v in after[6].items() if k!='audit'})
        initial_pending=np.asarray(ax['combat_mix']['bomb_threat']['pending']).copy()
        def body(carry,_):
            states,aux,rng,ended=carry
            nk,ke,actions,data=M.choose(params,ring,fixed,states,aux,rng,groups)
            ns,na,reward,done,info,parts,signal=M.transition(states,aux,actions,ke,coefficient)
            live=~ended;controllable=live&~aux['terminal']&states.alive[:,0]&(states.trapped[:,0]==0)
            carrier=states.bun_carried[:,0]>=0;window=~states.alive[:,1]|(states.trapped[:,1]>0)
            phase=j.where(~controllable,3,j.where(carrier,2,j.where(window,1,0)))
            ability=M.E.prediction(params,states)[1];mask=data[3]
            probabilities=jax.nn.softmax(j.where(mask,ability,-1e9))
            entropy=-j.sum(probabilities*j.log(j.maximum(probabilities,1e-30)),axis=-1)
            metrics=dict(signal['audit'],live=live,phase=phase,legal_bomb=controllable&mask[:,1],
                bomb_probability=probabilities[:,1],ability_entropy=entropy,
                own_action=actions[:,0],actual_place=info['bomb_placed'][:,0],
                death=info['death'][:,0],capture=info['capture'][:,0],
                completed=done&live,new_terminal=na['terminal']&~aux['terminal'],
                raw_reward=reward[:,0],actual_t=states.t)
            ns=M.E.select_tree(ended,states,ns);na=M.E.select_tree(ended,aux,na);nk=M.E.select_tree(ended,rng,nk)
            return (ns,na,nk,ended|done),metrics
        scan=jax.jit(lambda carry:jax.lax.scan(body,carry,None,length=int(config['scan_ticks'])))
        carry=(st,ax,keys,j.zeros(len(ids),bool));traces=[]
        for offset in range(0,int(config['ticks']),int(config['scan_ticks'])):
            tick=time.monotonic();carry,trace=scan(carry);jax.block_until_ready(carry)
            trace=jax.tree.map(lambda a:np.asarray(a).tolist(),jax.device_get(trace))
            file=target/f'ticks{offset:04d}.json';atomic_json(file,trace);traces.append(file)
            atomic_json(out/'PROGRESS.json',dict(checkpoint=name,ticks=offset+int(config['scan_ticks']),
                last_scan_seconds=time.monotonic()-tick,elapsed_seconds=time.time()-started,unix=time.time()))
        joined={k:np.concatenate([np.asarray(json.loads(f.read_text())[k]) for f in traces],axis=0) for k in json.loads(traces[0].read_text())}
        live=joined['live'];sums={k:float(np.asarray(v)[live].sum()) for k,v in joined.items()
            if k not in ('phase','own_action','actual_t')}
        closing=np.asarray(carry[1]['combat_mix']['bomb_threat']['pending']);residual=float(initial_pending.sum()+sums['admitted']-sums['cancelled']-sums['paid']-closing.sum())
        assert abs(residual)<1e-5 and np.abs(joined['balance_residual'][live]).max(initial=0)<1e-6
        phases={}
        for i,label in enumerate(('COMBAT','NATIVE_WINDOW_EMPTY','CARRIER_RETURN','NOT_CONTROLLABLE')):
            phase=live&(joined['phase']==i);legal=phase&joined['legal_bomb'];den=int(legal.sum())
            phases[label]=dict(observed_ticks=int(phase.sum()),legal_bomb_opportunities=den,
                mean_bomb_probability_given_legal=float(joined['bomb_probability'][legal].mean()) if den else None,
                mean_ability_entropy_given_legal=float(joined['ability_entropy'][legal].mean()) if den else None,
                bomb_requests=int(((joined['own_action'][:,:,1]==1)&phase).sum()),
                actual_bombs=int((joined['actual_place']&phase).sum()))
        result=dict(name=name,checkpoint=item['checkpoint'],state_sha256=commit['state_sha256'],policy_sha256=commit['policy_sha256'],
            environment_ids=ids.tolist(),opponent_kinds=kinds.tolist(),source_update=payload['update'],
            source_state_RNG_unchanged=True,passive_same_CPU_transition_digest_exact=True,
            opening_pending=float(initial_pending.sum()),closing_pending=float(closing.sum()),mass_balance_residual=residual,
            actual_observed_transitions=int(live.sum()),completed_source_episodes=int(np.asarray(carry[3]).sum()),
            censored_source_episodes=int((~np.asarray(carry[3])).sum()),ticks_cap=config['ticks'],counters=sums,phases=phases,
            trace_files={str(f.relative_to(ROOT)):sha(f) for f in traces},
            limitations=['8 real saved training states per source; not full natural games from birth',
                'frozen Actor and fixed source current/ring/fixed pool; no optimizer or teacher',
                'CPU backend may change categorical trajectories relative to training GPU',
                'pending at audit end is censored; no per-event terminal causal attribution or capability claim'])
        atomic_json(target/'RESULT.json',result);results.append(result)
    R.settle=original
    atomic_json(out/'COMPLETE.json',dict(status='COMPLETE',manifest_sha256=sha(manifest),new_training_updates=0,
        GPU_used=False,elapsed_seconds=time.time()-started,results=results,unix=time.time()))


if __name__=='__main__':main()
