"""Non-adopted, matched-batch update identification after the frozen R6 screen.

Run with ``python -m scripts.dagger_rl_r7_diagnostic``. The only environment
execution regenerates each declared TRAIN u0 batch once, exactly. No branch is
used for a new rollout and no policy is promoted or adopted.
"""
import os
import sys
import time
import json
import pickle
import subprocess
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
import optax
from scripts import dagger_rl_r6_train as T
from scripts.dagger_rl_update_mechanism import make_update, random_displacement,load_committed_update
from scripts.carrier_mechanism_train import preserve_critic
from scripts.carrier_mechanism_analysis import merged
from scripts.population_run_state import atomic_json, sha, lock
from scripts.terminal_checkpoint_restore import digest

R=Path(__file__).resolve().parents[1]
B=R/'runs/dagger_rl_20261024/r7_update_mechanism'
PRIOR=R/'runs/dagger_rl_20261024/r6_opponent_candidate'
read=lambda p:json.loads(p.read_text())


def authorize():
    selection=read(B/'SCIENTIFIC_SELECTION.json')
    assert selection['execute_update_diagnosis'] is True
    assert selection['protocol_sha256']==sha(B/'PROTOCOL.json')
    assert selection['prior_evidence_sha256']==sha(PRIOR/'COMPLETE_EVIDENCE_SYNTHESIS.json')
    assert read(PRIOR/'COMPLETE_FULL_TERMINAL_NOTICE.json')['passed']
    assert read(PRIOR/'GPU_COMPLETE.json')['passed']
    assert (PRIOR/'OWNER_EXIT.json').exists()
    auth=read(B/'SCOPED_GPU_AUTHORIZATION.json')
    assert auth['authorized_by']=='explicit_user_instruction' and not auth['old_continuous_resume']
    assert auth['protocol_sha256']==sha(B/'PROTOCOL.json')
    assert auth['old_global_pause_sha256']==sha(R/'runs/retention_20261021/USER_GPU_PAUSE.json')
    for name,expected in read(B/'SOURCE_SEAL.json')['sources'].items():
        assert sha(R/name)==expected,'sealed R7 diagnostic changed: '+name
    assert not (B/'USER_STOP.json').exists(),'user atomic stop'


def notify(message):
    print(message,flush=True)
    atomic_json(B/'notifications'/f'{time.time_ns()}.json',dict(message=message,unix=time.time()))
    try:subprocess.run(['jdme-notify','send',message],timeout=15,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except (OSError,subprocess.TimeoutExpired):pass


def verified(cp):
    m=read(cp/'COMMITTED.json')
    assert sha(cp/'state.pkl')==m['state_sha256']
    return pickle.load((cp/'state.pkl').open('rb'))


def configure():
    T.authorize=authorize
    T.bind()
    # The original setup checks R6's immutable source seal and certified Bank.
    # Only the local process's authorization callback changes; no source edits.
    T.C.authorize=authorize
    return T.setup()


def flatten(tree):
    return jax.tree.map(lambda a:np.asarray(a).reshape((-1,)+a.shape[2:]),tree)


@jax.jit
def prediction(params,o,g,mm,bm):
    mv,ab,_,_=T.C.E.train.net_forward(params,'transformer',o,g)
    return jax.nn.log_softmax(j.where(mm,mv,-1e9)),jax.nn.log_softmax(j.where(bm,ab,-1e9))


def probe_inputs(seed):
    cp=PRIOR/'formal'/f'control_s{seed}'/'u016'
    x=verified(cp);st,ax=flatten(x['states']),flatten(x['aux'])
    o,g,mm,bm=T.C.E.inputs(st)
    mask,diag=jax.jit(T.C.O.teacher)(st,ax)
    route=jax.jit(T.C.T.route_teacher)(st,ax)
    teacher=T.C.O.combine(route,mask)
    cargo=np.asarray(st.bun_carried[:,0]);live=np.asarray(st.alive[:,0])&(np.asarray(st.trapped[:,0])==0)
    window=~np.asarray(st.alive[:,1])|(np.asarray(st.trapped[:,1])>0)
    phase=np.where(~live,'inactive',np.where(cargo==1,'carry',np.where(window,'window','combat')))
    distance=np.abs(np.asarray(st.pos[:,0])-np.asarray(st.pos[:,1])).sum(-1)
    source=np.asarray(ax['provenance_source'])
    return dict(o=o,g=g,mm=mm,bm=bm,teacher=teacher,item=mask.any((1,2)),
        phase=phase,distance=distance,source=source,
        reference_checkpoint=str(cp.relative_to(R)),reference_sha256=sha(cp/'state.pkl'))


def compare_probe(params,parent,probe):
    args=tuple(probe[k] for k in ['o','g','mm','bm'])
    old=[np.asarray(a) for a in prediction(parent,*args)]
    new=[np.asarray(a) for a in prediction(params,*args)]
    teacher=np.asarray(probe['teacher']);item=np.asarray(probe['item'])
    move=new[0].argmax(-1);ability=new[1].argmax(-1)
    mass=(np.exp(new[0])[:,:,None]*np.exp(new[1])[:,None,:]*teacher).sum((1,2))
    groups={}
    selections={'all':np.ones(64,bool)}
    selections.update({p:probe['phase']==p for p in ['combat','window','carry','inactive']})
    selections.update(dict(near=probe['distance']<=4,mid=(probe['distance']>4)&(probe['distance']<=8),far=probe['distance']>8,
                           natural=probe['source']<0,real_pickup_bank=probe['source']>=0,
                           item_release=item&teacher[:,:,2].any(1),item_contact=item&teacher[:,:,0].any(1)))
    for name,ix in selections.items():
        row=dict(states=int(ix.sum()),teacher_rows=int((teacher.any((1,2))&ix).sum()))
        if ix.any():
            for label,a,c in zip(['move','ability'],old,new):
                row[label+'_parent_KL']=float((np.exp(a[ix])*(a[ix]-c[ix])).sum(-1).mean())
                row[label+'_greedy_changed']=int((a[ix].argmax(-1)!=c[ix].argmax(-1)).sum())
            row['release_probability']=float(np.exp(new[1][ix,2]).mean())
            row['bomb_probability']=float(np.exp(new[1][ix,1]).mean())
            valid=ix&teacher.any((1,2))
            row['teacher_joint_mass']=float(mass[valid].mean()) if valid.any() else None
            row['teacher_greedy_matches']=int(teacher[np.arange(64),move,ability][valid].sum())
        groups[name]=row
    return dict(groups=groups,reference=probe['reference_checkpoint'],reference_sha256=probe['reference_sha256'],
        logits_digest=digest(new),scope='all64 fixed TRAIN learner states, static policy distributions; not Q/GAE/capability')


def component_gradients(params,batch,replay,keys):
    # Exactly the first 8x32 global minibatch from the actual update shuffle.
    def compute(p,b,rb,key):
        b=tuple(a.reshape((-1,)+a.shape[2:]) for a in b)
        ad=b[4];mean=jax.lax.pmean(ad.mean(),'device');square=jax.lax.pmean((ad**2).mean(),'device')
        ad=(ad-mean)/j.sqrt(j.maximum(square-mean**2,0)+1e-8)
        b=b[:4]+(ad,)+b[5:]
        ix=jax.random.permutation(key,len(ad)).reshape(-1,32)[0]
        mb=tuple(a[ix] for a in b);rr=tuple(a[ix] for a in rb)
        def components(q):
            _,m=T.C.objective(q,mb,rr,'anchor')
            return j.stack([m[0],.5*m[1],-.005*m[2],.2*m[3],.05*m[6]])
        values,grads=components(p),jax.jacrev(components)(p)
        return jax.lax.pmean(values,'device'),jax.lax.pmean(grads,'device')
    values,grads=jax.pmap(compute,axis_name='device')(params,batch,replay,keys)
    grads=T.one(grads);values=np.asarray(values)[0]
    labels=['PPO','.5value','-.005entropy','.2DAgger','.05anchor']
    stats={}
    for name,tree in [('actor',grads['actor']),('critic',grads['critic']),('ability',grads['actor']['heads']['wb'])]:
        vectors=np.concatenate([np.asarray(v).reshape(5,-1).astype(float) for v in jax.tree.leaves(tree)],1)
        lengths=np.linalg.norm(vectors,axis=1);denom=lengths[:,None]*lengths[None,:]
        cosine=np.divide(vectors@vectors.T,denom,out=np.zeros((5,5)),where=denom>0)
        stats[name]=dict(norms=dict(zip(labels,lengths.tolist())),cosines=cosine.tolist(),zero_norm_pairs=(denom==0).tolist())
    return dict(weighted_component_losses=dict(zip(labels,values.tolist())),modules=stats,
        scope='first actual shuffled global256 TRAIN minibatch at shared parent; raw gradients before Adam/global clipping; not long-run harm')


def restore(seed):
    authorize()
    T.C.T.configure()
    folder=B/f'seed_{seed}'
    expected=read(folder/'FRESH_EXPECTED.json');path=folder/'joint_anchor/state.pkl'
    assert sha(path)==expected['state_sha256']
    x=pickle.load(path.open('rb'))
    assert digest((x['params'],x['adam'],x['states'],x['aux'],x['keys'],x['history']))==expected['full_digest']
    z=pickle.load((folder/'probe_inputs.pkl').open('rb'))
    args=tuple(z[k] for k in ['o','g','mm','bm'])
    actual=[np.asarray(v) for v in prediction(x['params'],*args)]
    reference=pickle.load((folder/'FRESH_PROBE_LOGS.pkl').open('rb'))
    errors=[]
    for old,new in zip(reference,actual):
        np.testing.assert_array_equal(old,new)
        np.testing.assert_array_equal(old.argmax(-1),new.argmax(-1))
        errors.append(float(np.max(np.abs(old-new))))
    atomic_json(folder/'FRESH_RESTORE.json',dict(passed=True,full_state_exact=True,
        probe_exact=True,greedy_probe_exact=True,same_backend_logprob_max_abs_errors=errors,
        backend=jax.default_backend(),cross_backend_parity_claim=False,
        source_joint_next_update_reproduced=True,extra_environment_steps=0,extra_optimizer_steps=0))


def fresh_check(folder,snapshot,probe,metrics):
    """Check the factual saved update before executing counterfactuals."""
    atomic_json(folder/'FRESH_EXPECTED.json',dict(state_sha256=metrics['state_sha256'],
        full_digest=digest((snapshot['params'],snapshot['adam'],snapshot['states'],snapshot['aux'],snapshot['keys'],snapshot['history']))))
    args=tuple(probe[k] for k in ['o','g','mm','bm'])
    T.C.save(folder/'FRESH_PROBE_LOGS.pkl',[np.asarray(v) for v in prediction(snapshot['params'],*args)])
    # Restore on the same backend/precision. CPU/GPU inference is a separately
    # measured parity question, not proof of saved-state corruption.
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    code=subprocess.run([sys.executable,'-m','scripts.dagger_rl_r7_diagnostic','restore',str(snapshot['seed'])],env=env,
        stdout=(folder/'fresh_restore.log').open('a'),stderr=subprocess.STDOUT).returncode
    assert code==0 and read(folder/'FRESH_RESTORE.json')['passed'],'fresh same-backend state/probe restore failed'
    if not (B/'TECHNICAL_SMOKE.json').exists():
        atomic_json(B/'TECHNICAL_SMOKE.json',dict(passed=True,finite=True,fresh_process_fullstate_exact=True,
            original_params_Adam_env_aux_RNG_history_exact=True,parameter_L2=metrics['parameter_L2'],
            ability_L2=metrics['ability_L2'],state_sha256=metrics['state_sha256'],
            included_in_declared_diagnostic_budget=True,additional_environment_steps=0,additional_optimizer_steps=0,
            adopted_updates=0,capability_gate=False))
        notify('R7首批技术验收：原u1参数/Adam/环境/RNG/记忆逐digest精确，finite/实际非零、fresh完整state精确；只执行声明诊断，无采用更新。')


def run():
    unique=lock(B/'owner.lock');authorize()
    assert (B/'READY_TECHNICAL.json').exists()
    # Imports may have initialized this process's own JAX context already.
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert all(int(pid.strip())==os.getpid() for pid in apps.splitlines() if pid.strip()),'GPU busy, no preemption'
    atomic_json(B/'OWNER.json',dict(unique=True,bounded=True,old_continuous_resume=False))
    try:
        items,source=configure();bs,ba=merged(items,'states'),merged(items,'aux')
        opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
        branches=['joint_anchor','ppo_anchor','dagger_anchor']
        updates={a:make_update(T.C.objective,opt,a,preserve_critic) for a in branches}
        # The factual branch is the production updater itself, not a new
        # implementation whose roundoff could contaminate exact replay.
        updates['joint_anchor']=T.C.updater('anchor',opt)
        collect=T.collector(128)
        new_optimizer_steps=0
        for seed in read(B/'PROTOCOL.json')['source_seeds']:
            authorize();folder=B/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'COMPLETE.json').exists():continue
            start=verified(PRIOR/'formal'/f'control_s{seed}'/'u000')
            target=verified(PRIOR/'formal'/f'control_s{seed}'/'u001')
            assert digest((start['params'],start['adam']))==digest((source['params'],source['adam']))
            p,a=T.rep(start['params']),T.rep(start['adam'])
            payload=folder/'COLLECTED_BATCH.pkl';collected=folder/'COLLECTED_BATCH.json'
            if collected.exists():
                assert sha(payload)==read(collected)['sha256'],'saved on-policy batch SHA mismatch'
                end,batch,history,rb,stats=pickle.load(payload.open('rb'));st,ax,key=end
            else:
                (st,ax,key),batch,diag=collect(p,start['states'],start['aux'],start['keys'],bs,ba)
                history,rb,stats=T.aggregate(batch,start['history'],seed,1)
                T.C.save(payload,jax.device_get(((st,ax,key),batch,history,rb,stats)))
                atomic_json(collected,dict(passed=True,sha256=sha(payload),logical_environment_steps=8192,
                    technical_regeneration=(folder/'JOINT_EXACT.json').exists(),source_batch='same original u0, not new independent data'))
            assert digest((st,ax,key))==digest((target['states'],target['aux'],target['keys']))
            assert digest(history)==digest(target['history'])
            target_stats=read(PRIOR/'formal'/f'control_s{seed}'/'u001/COMMITTED.json')['metrics']['aggregation']
            assert stats==target_stats,'aggregation/source counts changed'
            keys=np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),1),8))
            if (folder/'probe_inputs.pkl').exists():probe=pickle.load((folder/'probe_inputs.pkl').open('rb'))
            else:
                probe=probe_inputs(seed);T.C.save(folder/'probe_inputs.pkl',jax.device_get(probe))
            if not (folder/'COMPONENT_GRADIENTS.json').exists():
                gradient_stats=component_gradients(p,batch,rb,keys)
                atomic_json(folder/'COMPONENT_GRADIENTS.json',gradient_stats)
            outcomes={}
            for branch in branches:
                authorize()
                cp=folder/branch
                reused=(cp/'COMMITTED.json').exists()
                if reused:
                    snapshot,metrics=load_committed_update(cp,expected=target if branch=='joint_anchor' else None)
                    pp,aa=snapshot['params'],snapshot['adam']
                    loss=np.array([0.])
                else:
                    (pn,an),loss=updates[branch](p,a,batch,rb,keys);jax.block_until_ready(pn)
                    pp,aa=T.one(pn),T.one(an)
                    new_optimizer_steps+=32
                assert all(np.isfinite(v).all() for v in jax.tree.leaves((pp,aa))) and np.isfinite(np.asarray(loss)).all()
                assert int(aa[1][0].count)==int(start['adam'][1][0].count)+32
                if branch=='joint_anchor':
                    assert digest((pp,aa))==digest((target['params'],target['adam'])),'actual joint-u1 reproduction mismatch'
                    atomic_json(folder/'JOINT_EXACT.json',dict(passed=True,params_Adam_exact=True,env_aux_RNG_exact=True,
                        history_replay_source_counts_exact=True,source_u1_sha256=sha(PRIOR/'formal'/f'control_s{seed}'/'u001/state.pkl')))
                    joint=pp
                if branch=='dagger_anchor':
                    assert digest(pp['critic'])==digest(start['params']['critic'])
                    for field in ['mu','nu']:
                        assert digest(getattr(aa[1][0],field)['critic'])==digest(getattr(start['adam'][1][0],field)['critic'])
                cp.mkdir(parents=True,exist_ok=True)
                snapshot=dict(params=pp,adam=aa,states=jax.device_get(st),aux=jax.device_get(ax),keys=jax.device_get(key),
                    history=history,parent_params=start['params'],seed=seed,adopted=False)
                if not reused:T.C.save(cp/'state.pkl',snapshot)
                if not reused:metrics=dict(passed=True,branch=branch,seed=seed,finite=True,parameter_L2=T.norms(start['params'],pp),
                    params_digest=digest(pp),Adam_digest=digest(aa),Adam_count=int(aa[1][0].count),
                    ability_L2=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,pp['actor']['heads']['wb'],start['params']['actor']['heads']['wb']))),
                    state_sha256=sha(cp/'state.pkl'),optimizer_steps=32,adopted=False,probe=compare_probe(pp,start['params'],probe))
                if not reused:atomic_json(cp/'COMMITTED.json',metrics)
                outcomes[branch]=metrics
                if branch=='joint_anchor' and not (folder/'FRESH_RESTORE.json').exists():fresh_check(folder,snapshot,probe,metrics)
                atomic_json(B/'STATUS.json',dict(stage='DIAGNOSTIC',seed=seed,branch=branch,finite=True,
                    validated_optimizer_steps_this_seed=32*len(outcomes),new_optimizer_steps_this_process=new_optimizer_steps,
                    new_independent_environment_samples=0,adopted_updates=0))
                action='复用已原子保存的' if reused else '新执行'
                notify(f'R7非采用诊断 {branch}/s{seed} {action}32优化步：ActorΔ{metrics["parameter_L2"]["actor"]:.6g} CriticΔ{metrics["parameter_L2"]["critic"]:.6g}；不是新能力训练。')
            for name,pp in [('no_update',start['params']),('module_norm_matched_random',random_displacement(start['params'],joint,seed=seed))]:
                outcomes[name]=dict(probe=compare_probe(pp,start['params'],probe),parameter_L2=T.norms(start['params'],pp),
                    params_digest=digest(pp),random_draw_seed=seed if name=='module_norm_matched_random' else None,adopted=False)
            atomic_json(folder/'COMPLETE.json',dict(passed=True,seed=seed,branches=outcomes,logical_environment_steps=8192,
                executed_optimizer_steps=96,adopted_updates=0,new_independent_environment_samples=0))
        repair_steps=sum(8192 for f in B.glob('seed_*/COLLECTED_BATCH.json') if read(f)['technical_regeneration'])
        atomic_json(B/'COMPLETE.json',dict(passed=True,logical_environment_steps=24576,technical_regeneration_environment_steps=repair_steps,
            total_executed_environment_steps=24576+repair_steps,executed_optimizer_steps=288,
            adopted_updates=0,new_independent_environment_samples=0,extra_evaluation_rollouts=0,
            seeds=read(B/'PROTOCOL.json')['source_seeds'],old_continuous_resume=False))
        notify('R7三批匹配更新机制诊断完成；24576重放逻辑步/288诊断优化步，0采用更新，等待证据消费选择最小后继。')
    except Exception as e:
        reason=str(e).replace(str(R),'<WORKTREE>')
        atomic_json(B/'TECHNICAL_FAILURE.json',dict(reason=reason,unix=time.time()))
        notify(f'R7技术故障：{reason}；不采用诊断参数，不恢复旧队列。')
        raise
    finally:atomic_json(B/'OWNER_EXIT.json',dict(unix=time.time(),old_continuous_resume=False))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='restore':restore(int(sys.argv[2]))
    else:run()
