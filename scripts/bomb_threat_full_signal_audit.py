"""Policy-only, complete natural JAX games; passive reward ledger attribution.

No training, teacher, fabricated carrier or protocol/reward mutation. Per-type
payment is explicitly a proportional shadow accounting convention because the
native reward implementation stores only one aggregate pending balance.
"""
import argparse,json,os,time
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
ROOT=Path(__file__).resolve().parents[1]


def attribution(pending,offers,admitted,cancelled,paid):
    import jax.numpy as j
    opening=pending.sum(-1)
    cancel=pending*j.where(opening>0,cancelled/j.maximum(opening,1e-20),0)[...,None]
    total=offers.sum(-1)
    added=offers*j.where(total>0,admitted/j.maximum(total,1e-20),0)[...,None]
    available=pending-cancel+added
    balance=available.sum(-1)
    payment=available*j.where(balance>0,paid/j.maximum(balance,1e-20),0)[...,None]
    return available-payment,payment,cancel


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('manifest');args=p.parse_args()
    mp=Path(args.manifest);cfg=json.loads(mp.read_text());out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    guard=lock(out/'worker.lock')
    if (out/'COMPLETE.json').exists():return
    for file,h in cfg['source_sha256'].items():assert sha(ROOT/file)==h,file
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JAXBOMB_RULE='bun',
        XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    import jax,jax.numpy as j,numpy as np
    from scripts import expert_reward_alignment_fullmatch as F,bomb_threat_collector as M,bomb_threat_reward as R
    from scripts.bomb_threat_runtime import install
    from scripts.bomb_threat_signal_trace import audited_settle
    from scripts.checkpoint_policy_artifact import verify_policy,load_actor
    from scripts.terminal_checkpoint_restore import digest
    F.T.configure();F.provenance.install_provenance();install();F.SEED_START=cfg['seed']
    original=R.settle;wrapped=audited_settle(original)
    def observer(e,s):
        after,pay,metrics=wrapped(e,s)
        a=metrics['audit']
        a['offers_by_type']=j.stack([.02*metrics['threat_events'],.04*metrics['brick_events'],.01*metrics['chain_events']],-1)
        return after,pay,metrics
    def body(params,carry,categorical):
        s,a,k,ended,shadow=carry
        nk,ek,actions,policy=F.choose(params,s,a,k,categorical)
        ns,na,r,done,info,parts,signal=M.transition(s,a,actions,ek,0.)
        audit=signal['audit'];next_shadow,payment,cancellation=attribution(shadow,audit['offers_by_type'],audit['admitted'],audit['cancelled'],audit['paid'])
        live=~ended;req=actions[:,0,1]==1;placed=info['bomb_placed'][:,0]
        _,_,_,mask=F.E.inputs(s)
        rejection=req&~placed
        # Exclusive observable pre-action reasons; unresolved move/order
        # effects remain unknown rather than invented causal attribution.
        reason=j.zeros(n,j.int32)
        for condition,label in [(~mask[:,1],5),(s.bun_carried[:,0]>=0,4),(s.trapped[:,0]>0,3),(~s.alive[:,0],2),(a['terminal'],1)]:
            reason=j.where(condition,label,reason)
        phase=j.where(~s.alive[:,0]|(s.trapped[:,0]>0)|a['terminal'],3,
            j.where(s.bun_carried[:,0]>=0,2,j.where(~s.alive[:,1]|(s.trapped[:,1]>0),1,0)))
        metrics=dict(audit,live=live,first_terminal=na['terminal']&~a['terminal'],completed=done&live,
            terminal_before=a['terminal'],phase=phase,bomb_request=req,actual_place=placed,
            rejected=rejection,reject_reason=reason,payment_by_type=payment,cancel_by_type=cancellation,
            shadow_balance=next_shadow.sum(-1)-audit['pending_after'],
            death=info['death'][:,0],capture=info['capture'][:,0],actual_t=s.t,raw_reward=r[:,0],
            bomb_probability=policy[1][:,1],ability_entropy=-j.sum(policy[1]*j.log(j.maximum(policy[1],1e-30)),-1))
        return (M.E.select_tree(ended,s,ns),M.E.select_tree(ended,a,na),M.E.select_tree(ended,k,nk),ended|done,
                j.where(ended[:,None],shadow,next_shadow)),metrics
    scan=jax.jit(lambda params,carry,categorical:jax.lax.scan(lambda c,_:body(params,c,categorical),carry,None,length=cfg["scan_ticks"]))
    results=[];started=time.time();n=cfg['games_per_mode']
    fresh=jax.jit(F.initial_batch);choose=jax.jit(F.choose)
    bare_step=jax.jit(lambda s,a,ac,k:M.transition(s,a,ac,k,0.))
    observed_step=jax.jit(lambda s,a,ac,k:M.transition(s,a,ac,k,0.))
    def stage(name,source,mode):
        atomic_json(out/'PROGRESS.json',dict(phase=name,source=source['name'],mode=mode,elapsed_seconds=time.time()-started,unix=time.time()))
    for source in cfg['checkpoints']:
        cp=ROOT/source['checkpoint'];cm=verify_policy(cp);params=jax.tree.map(j.asarray,load_actor(cp))
        for mode in cfg['modes']:
            target=out/(source['name']+'_'+mode);target.mkdir(exist_ok=True)
            if (target/'RESULT.json').exists():
                results.append(json.loads((target/'RESULT.json').read_text()));continue
            stage('INITIALIZING_NATURAL_SPAWNS',source,mode)
            st,ax,keys=fresh(j.arange(n,dtype=j.int32));jax.block_until_ready(st)
            assert np.all(np.asarray(st.bun_carried)<0) and np.all(np.asarray(st.t)==0)
            ax=dict(ax,combat_mix=M.metadata(st,j.ones(n,j.int32)))
            stage('FIRST_POLICY_ACTION',source,mode)
            _,ke,act,_=choose(params,st,ax,keys,mode=='categorical');jax.block_until_ready(act)
            R.settle=original
            stage('PARITY_BASE_TRANSITION',source,mode)
            bare=bare_step(st,ax,act,ke);jax.block_until_ready(bare)
            R.settle=observer
            stage('PARITY_OBSERVED_TRANSITION',source,mode)
            observed=observed_step(st,ax,act,ke);jax.block_until_ready(observed)
            assert digest(bare[:6])==digest(observed[:6])
            assert digest(bare[6])==digest({k:v for k,v in observed[6].items() if k!='audit'})
            carry=(st,ax,keys,j.zeros(n,bool),j.zeros((n,3)));files={}
            stage('SCANNING_COMPLETE_GAMES',source,mode)
            for offset in range(0,cfg['max_ticks'],cfg['scan_ticks']):
                tick=time.monotonic();carry,trace=scan(params,carry,mode=='categorical');jax.block_until_ready(carry)
                file=target/f'ticks{offset:04d}.json';atomic_json(file,jax.tree.map(lambda x:np.asarray(x).tolist(),trace));files[str(file.relative_to(ROOT))]=sha(file)
                atomic_json(out/'PROGRESS.json',dict(source=source['name'],mode=mode,ticks=offset+cfg['scan_ticks'],
                    scan_seconds=time.monotonic()-tick,elapsed_seconds=time.time()-started,unix=time.time()))
                if np.all(np.asarray(carry[3])):break
            chunks=[json.loads((ROOT/f).read_text()) for f in files]
            joined={key:np.concatenate([np.asarray(c[key]) for c in chunks]) for key in chunks[0]}
            live=joined['live'];assert np.max(np.abs(joined['shadow_balance'][live]),initial=0)<1e-6
            episodes=[]
            for i in range(n):
                valid=live[:,i];sumkey=lambda key:float(joined[key][valid,i].sum())
                types=lambda key:joined[key][valid,i].sum(0).tolist()
                rejected=valid&joined['rejected'][:,i]
                opening=0.;paid=sumkey('paid');cancelled=sumkey('cancelled');admitted=sumkey('admitted')
                closing=float(np.asarray(carry[1]['combat_mix']['bomb_threat']['pending'])[i])
                assert abs(opening+admitted-paid-cancelled-closing)<1e-6
                episodes.append(dict(index=i,seed=cfg['seed']+i,complete=bool(np.asarray(carry[3])[i]),ticks=int(valid.sum()),
                    threat_events=sumkey('first_safe_threat'),topology_events=sumkey('topology_events'),chain_events=sumkey('chain_events'),
                    requests=sumkey('bomb_request'),actual_bombs=sumkey('actual_place'),rejected=sumkey('rejected'),
                    reject_reasons={str(v):int((rejected&(joined['reject_reason'][:,i]==v)).sum()) for v in range(6)},
                    paid=paid,paid_by_type_proportional_shadow=types('payment_by_type'),cancelled=cancelled,
                    cancel_bad=sumkey('cancel_bad'),cancel_terminal=sumkey('cancel_terminal'),cancel_inactive=sumkey('cancel_inactive'),
                    admitted=admitted,closing_pending=closing,cancelled_over_admitted=cancelled/admitted if admitted else None,
                    deaths=sumkey('death'),deliveries=sumkey('capture'),net_reward=sumkey('raw_reward'),
                    preterminal_ticks=int((valid&~joined['terminal_before'][:,i]).sum()),tail_ticks=int((valid&joined['terminal_before'][:,i]).sum())))
            value=dict(name=source['name'],checkpoint=source['checkpoint'],policy_sha256=cm['policy_sha256'],mode=mode,
                episodes=episodes,complete_games=sum(e['complete'] for e in episodes),requested_games=n,
                first_transition_passive_exact=True,trace_files=files,critic_used=False,new_training_steps=0)
            atomic_json(target/'RESULT.json',value);results.append(value)
    R.settle=original
    atomic_json(out/'COMPLETE.json',dict(status='COMPLETE' if all(r['complete_games']==n for r in results) else 'INCOMPLETE',
        manifest_sha256=sha(mp),results=results,elapsed_seconds=time.time()-started,new_training_steps=0,GPU_used=False,unix=time.time()))

if __name__=='__main__':main()
